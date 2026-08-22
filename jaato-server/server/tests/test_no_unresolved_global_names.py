"""No function may reference a global name the module does not bind.

PR #576 added a call to ``bound_model_for_profile`` inside
``JaatoServer.initialize`` while the only import of that name was
FUNCTION-LOCAL, inside ``_profile_binds_a_model``. So every session created
from a profile died with::

    NameError: name 'bound_model_for_profile' is not defined

Not tiers-only -- the call sits inside ``if self._profile:``, so ANY profile
hit it, and setting ``model:`` did not avoid it.

WHY IT REACHED MAIN. Importing the module succeeds (the name is missing, not
broken). Every test called the OTHER producers, each of which resolves the
name through its own import. And a test that merely *executes* ``initialize()``
does not help either: with a bare double it dies ~900 lines earlier, never
reaching the call site -- I wrote that test first and it passed against the
broken build.

A NameError at an unimported call site is a STATIC property, so check it
statically. ``symtable`` (stdlib) reports, per function, which names are
treated as globals; anything the module does not bind and Python does not
provide is unreachable at runtime.
"""
import builtins
import importlib
import io
import symtable

import pytest

# Modules whose functions must resolve every global they reference.
MODULES = ["server.core", "server.runner_spawn", "server.session_manager"]


def _tables(top):
    yield top
    for child in top.get_children():
        yield from _tables(child)


def _unresolved(module_path: str):
    module = importlib.import_module(module_path)
    source = io.open(module.__file__, encoding="utf-8").read()
    bound = set(dir(module)) | set(dir(builtins))

    found = []
    for table in _tables(symtable.symtable(source, module.__file__, "exec")):
        if table.get_type() != "function":
            continue
        for sym in table.get_symbols():
            # is_global(): not assigned or imported locally, so it must come
            # from the module namespace at call time.
            if (sym.is_global() and sym.is_referenced()
                    and sym.get_name() not in bound):
                found.append((sym.get_name(), table.get_name()))
    return found


@pytest.mark.parametrize("module_path", MODULES)
def test_every_referenced_global_is_bound(module_path):
    unresolved = _unresolved(module_path)
    assert not unresolved, (
        f"{module_path} references names it does not bind: "
        + ", ".join(f"{name}() in {func}()" for name, func in unresolved)
        + ". Each raises NameError when that function RUNS -- import "
        "succeeds, so nothing catches it until a live call. Add a "
        "module-scope import, or import inside the function that uses it."
    )


def test_the_check_would_have_caught_pr_576():
    """Guard the guard: prove the mechanism detects a local-only import.

    Without this, a change that silently weakened ``_unresolved`` would leave
    the suite green and the class unprotected again.
    """
    source = (
        "def helper():\n"
        "    from shared.model_tiers import bound_model_for_profile\n"
        "    return bound_model_for_profile(None)\n"
        "\n"
        "def initialize(self):\n"
        "    return bound_model_for_profile(self._profile)\n"
    )
    bound = set(dir(builtins))
    found = [
        (s.get_name(), t.get_name())
        for t in _tables(symtable.symtable(source, "<synthetic>", "exec"))
        if t.get_type() == "function"
        for s in t.get_symbols()
        if s.is_global() and s.is_referenced() and s.get_name() not in bound
    ]
    assert ("bound_model_for_profile", "initialize") in found
    assert ("bound_model_for_profile", "helper") not in found, (
        "a function-local import must NOT be reported -- that binding is fine"
    )
