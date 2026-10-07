"""Provider config-knob + quirk contract guard.

Sibling of ``test_provider_capabilities`` — same AST-only walk (no imports, so
a missing vendor SDK never breaks the guard) but for the config-knob and
model-quirk contracts:

Every provider subdir must declare, in its ``__init__.py``:

- ``PROVIDER_KNOBS = ProviderKnobs(layers=(...))`` — the config knobs an
  author may set under ``plugin_configs.<provider>``, grouped by layer, each
  knob a ``KnobSpec(name, type, default, description)``.
- ``PROVIDER_QUIRKS = frozenset({...})`` — the model-quirk names the provider
  honors (empty for the providers that honor none).

This is the structural gate that stops a new provider from silently shipping
prose-only knobs / a method-local quirk allow-list again (the state the
``explain`` / ``validate`` tooling exists to make introspectable).  It mirrors
``test_provider_capabilities`` exactly so the three contracts (capabilities,
knobs, quirks) read the same way.
"""

import ast
import os
from pathlib import Path
from typing import List, Optional

PROVIDER_DIR = Path(__file__).resolve().parents[1] / "plugins" / "model_provider"

# Same exclusions as the capability guard — test stubs / non-providers.
_EXCLUDE = {"tests", "__pycache__", "bundle_common", "echo"}

# Allowed KnobSpec.type hint strings (kept in sync with KnobSpec's docstring).
_ALLOWED_TYPES = {"str", "int", "float", "bool", "list", "dict"}

# Allowed AuthSource.kind values (kept in sync with base.AUTH_SOURCE_KINDS).
_AUTH_KINDS = {"api_key_param", "env", "stored", "oauth", "adc", "cli", "none"}


def _provider_dirs() -> List[str]:
    out = []
    for entry in sorted(os.listdir(PROVIDER_DIR)):
        d = PROVIDER_DIR / entry
        if not d.is_dir() or entry in _EXCLUDE or entry.startswith("_"):
            continue
        if (d / "__init__.py").exists():
            out.append(entry)
    return out


def _assign_value(provider: str, name: str) -> Optional[ast.expr]:
    """Return the AST value node assigned to module-level ``name``, or None."""
    src = (PROVIDER_DIR / provider / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if name in targets:
                return node.value
    return None


def _is_call_to(node: Optional[ast.expr], func_name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == func_name
    )


# ---------------------------------------------------------------- structural

def test_every_provider_declares_knobs():
    """Every provider must declare PROVIDER_KNOBS = ProviderKnobs(...)."""
    missing = [
        p for p in _provider_dirs()
        if not _is_call_to(_assign_value(p, "PROVIDER_KNOBS"), "ProviderKnobs")
    ]
    assert missing == [], (
        "providers missing a PROVIDER_KNOBS = ProviderKnobs(...) declaration "
        f"in __init__.py: {missing}. Author it from the keys the provider's "
        "code actually reads (config.extra.get / _knob sites), grouped by "
        "layer — see shared/plugins/model_provider/base.py:ProviderKnobs."
    )


def test_every_provider_declares_quirks():
    """Every provider must declare PROVIDER_QUIRKS = frozenset({...})."""
    missing = [
        p for p in _provider_dirs()
        if not _is_call_to(_assign_value(p, "PROVIDER_QUIRKS"), "frozenset")
    ]
    assert missing == [], (
        "providers missing a PROVIDER_QUIRKS = frozenset({...}) declaration "
        f"in __init__.py: {missing}. Declare the quirk names the provider "
        "honors (empty frozenset() if none) so explain/validate can read them."
    )


def test_knobspecs_are_well_formed():
    """Every KnobSpec must have a string name and a valid type literal."""
    problems: List[str] = []
    for p in _provider_dirs():
        knobs = _assign_value(p, "PROVIDER_KNOBS")
        if not _is_call_to(knobs, "ProviderKnobs"):
            continue  # caught by the declaration test
        for n in ast.walk(knobs):
            if not _is_call_to(n, "KnobSpec"):
                continue
            if not n.args or not (
                isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)
            ):
                problems.append(f"{p}: a KnobSpec is missing a string name")
                continue
            kname = n.args[0].value
            # type is the 2nd positional arg or the `type=` keyword
            tnode = n.args[1] if len(n.args) >= 2 else None
            for kw in n.keywords:
                if kw.arg == "type":
                    tnode = kw.value
            if tnode is not None:
                if not (isinstance(tnode, ast.Constant)
                        and tnode.value in _ALLOWED_TYPES):
                    problems.append(
                        f"{p}: KnobSpec {kname!r} type must be a literal in "
                        f"{sorted(_ALLOWED_TYPES)}"
                    )
    assert problems == [], "\n".join(problems)


def test_knoblayers_have_string_layer_names():
    """Every KnobLayer's first arg (the layer name) must be a string literal."""
    problems: List[str] = []
    for p in _provider_dirs():
        knobs = _assign_value(p, "PROVIDER_KNOBS")
        if not _is_call_to(knobs, "ProviderKnobs"):
            continue
        for n in ast.walk(knobs):
            if not _is_call_to(n, "KnobLayer"):
                continue
            if not n.args or not (
                isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)
            ):
                problems.append(f"{p}: a KnobLayer is missing a string layer name")
    assert problems == [], "\n".join(problems)


def test_every_provider_declares_auth_resolution():
    """Every provider must declare PROVIDER_AUTH_RESOLUTION = (AuthSource(...), ...)."""
    missing = []
    for p in _provider_dirs():
        node = _assign_value(p, "PROVIDER_AUTH_RESOLUTION")
        ok = isinstance(node, ast.Tuple) and node.elts and all(
            _is_call_to(e, "AuthSource") for e in node.elts)
        if not ok:
            missing.append(p)
    assert missing == [], (
        "providers missing a PROVIDER_AUTH_RESOLUTION = (AuthSource(...), ...) "
        f"declaration in __init__.py: {missing}. Author the ordered credential "
        "chain from the provider's resolve_api_key/verify_auth code."
    )


def test_auth_sources_have_valid_kinds():
    """Every AuthSource's first arg (kind) must be a literal in AUTH_SOURCE_KINDS."""
    problems = []
    for p in _provider_dirs():
        node = _assign_value(p, "PROVIDER_AUTH_RESOLUTION")
        if not isinstance(node, ast.Tuple):
            continue
        for n in ast.walk(node):
            if not _is_call_to(n, "AuthSource"):
                continue
            if not n.args or not (isinstance(n.args[0], ast.Constant)
                                  and n.args[0].value in _AUTH_KINDS):
                problems.append(
                    f"{p}: AuthSource kind must be a literal in "
                    f"{sorted(_AUTH_KINDS)}")
    assert problems == [], "\n".join(problems)


def test_quirk_names_are_string_literals():
    """Every PROVIDER_QUIRKS element must be a string literal."""
    problems: List[str] = []
    for p in _provider_dirs():
        quirks = _assign_value(p, "PROVIDER_QUIRKS")
        if not _is_call_to(quirks, "frozenset"):
            continue
        for arg in quirks.args:  # frozenset({...}) → one Set arg
            if isinstance(arg, (ast.Set, ast.List, ast.Tuple)):
                for elt in arg.elts:
                    if not (isinstance(elt, ast.Constant)
                            and isinstance(elt.value, str)):
                        problems.append(
                            f"{p}: PROVIDER_QUIRKS must contain only string "
                            "quirk names"
                        )
    assert problems == [], "\n".join(problems)
