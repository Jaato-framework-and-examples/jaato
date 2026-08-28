"""Generated code must be written against the CURRENT SDK contract.

A scaffold is the one form of documentation that gets executed, which is its
value and its hazard: when a contract changes, every template written
against the old one keeps shipping to readers who have no reason to doubt
it.

That happened within a day.  #635 changed ``create_session`` from returning
``Optional[str]`` to returning ``str`` and RAISING — and all four
session-creating templates still carried:

    sid = await client.create_session(...)
    if not sid:                                    # DEAD: cannot be falsy
        print("session.new failed — check provider auth / the daemon log")

Three defects in four lines: a branch that can no longer be taken, no
handler for the exception that now occurs (so a refusal crashes the
generated script with a traceback), and a hardcoded likely-cause — "check
provider auth" — which is one of five causes and the exact sentence removed
from ``convenience.py`` in that same PR.

These tests check the templates against the SDK ITSELF where they can,
rather than against a copy of what it says, because a second statement of a
contract is the thing that rots.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from shared.scaffold._client_templates import TEMPLATES

PLACEHOLDERS = {
    "__TITLE__": "T", "__PROVENANCE__": "p", "__WORKSPACE__": "/ws",
    "__ENV_FILE__": ".env", "__MODEL__": "echo", "__PROVIDER__": "echo",
    "__CLIENT_IMPORT__": "from jaato_sdk import IPCClient, ClientType, EventType",
    "__CONN_CONSTANTS__": "SOCKET='/x'", "__ON_STATUS_DEF__": "",
    "__NEW_CLIENT_CALL__": "IPCClient(socket_path=SOCKET)",
    "__CASCADE_ID__": "cid", "__CLIENT_CLASS__": "IPCClient",
    "__SOCKET__": "/x", "__KEY_ENV__": "K", "__ARCHETYPE__": "a",
}


def _render(name: str) -> str:
    _, tmpl, _ = TEMPLATES[name]
    for k, v in PLACEHOLDERS.items():
        tmpl = tmpl.replace(k, v)
    return tmpl


def _code(rendered: str) -> str:
    """Comments stripped — a template may WARN about a pattern in prose."""
    return "\n".join(l.split("#", 1)[0] for l in rendered.splitlines())


CREATORS = sorted(
    n for n in TEMPLATES if "create_session" in _code(_render(n))
)


def test_there_are_session_creating_archetypes():
    """Guard the guard: if this list empties, every test below vacuously
    passes and the contract goes unchecked."""
    assert CREATORS, "no archetype creates a session; these tests are inert"


@pytest.mark.parametrize("name", CREATORS)
def test_the_generated_script_compiles(name):
    ast.parse(_render(name))


@pytest.mark.parametrize("name", CREATORS)
def test_no_dead_falsy_check_on_create_session(name):
    """``create_session`` returns ``str`` and raises; it never returns None."""
    assert "if not sid" not in _code(_render(name)), (
        f"{name} still tests create_session's result for falsiness — that "
        "branch cannot be taken, and the failure it was written for now "
        "arrives as an exception nothing here catches"
    )


@pytest.mark.parametrize("name", CREATORS)
def test_a_creation_failure_is_handled(name):
    """Unhandled, a refusal crashes a generated script with a traceback."""
    assert "SessionCreateFailed" in _code(_render(name)), (
        f"{name} creates a session and handles no creation failure"
    )


@pytest.mark.parametrize("name", CREATORS)
def test_no_hardcoded_likely_cause(name):
    """"check provider auth" is ONE of five causes.

    A guessed cause sends the reader somewhere specific and wrong, which is
    worse than saying nothing — and the exception now carries the real one.
    """
    assert "check provider auth" not in _render(name), (
        f"{name} guesses the cause of a creation failure; the exception "
        "states it"
    )


@pytest.mark.parametrize("name", CREATORS)
def test_the_persona_is_reachable_from_every_creating_archetype(name):
    """``agent`` is orthogonal to ``profile`` and is half of what a session IS.

    Templates need not PASS one — a single-shot demo reasonably runs without
    a persona — but one that never mentions it leaves a reader unaware the
    axis exists.  ``fire`` and ``host-tools`` did exactly that.
    """
    assert "agent=" in _render(name), (
        f"{name} creates a session and never mentions ``agent``; a reader "
        "learns the framework has capabilities but no identities"
    )


def test_the_exception_name_matches_the_sdk():
    """Checked against the SDK, not against a copy of its name.

    If ``SessionCreateFailed`` is ever renamed, generated code that catches
    the old name compiles and never fires — the worst outcome, since the
    script looks handled.
    """
    import jaato_sdk

    assert hasattr(jaato_sdk, "SessionCreateFailed"), (
        "the templates catch jaato_sdk.SessionCreateFailed and the SDK no "
        "longer exports it; every generated script now has a dead except"
    )


def test_create_session_really_does_raise_rather_than_return_none():
    """The premise of every assertion above, read from the SDK's own source.

    If this ever stops being true, these tests are enforcing the wrong
    contract — and they would go on passing while doing it.
    """
    from jaato_sdk.client.ipc import IPCClient

    src = inspect.getsource(IPCClient.create_session)
    assert "raise SessionNotSent" in src or "SessionCreateFailed" in src, (
        "create_session no longer raises; the templates' try/except is now "
        "the dead branch and these tests are pinned to a contract that moved"
    )


def _imported_names(tree: ast.AST) -> set:
    """Every name an import statement BINDS in the module namespace."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            bound.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
    return bound


def _caught_names(tree: ast.AST) -> set:
    """Every bare name used as an ``except`` type.

    Tuple and attribute handlers are skipped deliberately: this checks the
    simple ``except Foo:`` form, which is the one a template renders.
    """
    return {
        h.type.id for h in ast.walk(tree)
        if isinstance(h, ast.ExceptHandler) and isinstance(h.type, ast.Name)
    }


@pytest.mark.parametrize("name", CREATORS)
def test_every_caught_exception_is_imported(name):
    """A caught name that is never bound is a NameError at the reader's
    first failure — and ONLY at failure.

    ``ast.parse`` cannot see this: the script parses perfectly.  The generated
    file runs fine right up to the moment something goes wrong, and then dies
    with ``NameError`` instead of the handled message — the worst possible
    place to discover the handler was never wired.

    Found because a sabotage that removed the import came back INCONCLUSIVE:
    nothing failed, which meant nothing was checking.
    """
    tree = ast.parse(_render(name))
    missing = (
        _caught_names(tree)
        - _imported_names(tree)
        - set(dir(__import__("builtins")))
    )
    assert not missing, (
        f"{name} catches {sorted(missing)} without importing it; the script "
        "parses and then dies with NameError at the first real failure"
    )
