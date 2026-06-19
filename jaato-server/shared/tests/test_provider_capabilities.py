"""Provider-capability contract guard + capability-doc drift guard.

Mirrors ``test_plugin_tier_partition`` (AST-only, no imports — so missing
vendor SDKs never break the walk) but for model providers: every provider
subdir must declare a ``PROVIDER_CAPABILITIES = ProviderCapabilities(...)``
constant in its ``__init__.py`` with every :data:`CAPABILITY_FIELDS` key set
to a bool.  This is the structural half of the provider capability contract
(the behavioral conformance half lives in
``test_provider_capability_conformance.py``).

The same AST walk feeds the capability DOC generator: running this module as a
script regenerates ``docs/model-provider-capabilities.md`` from the
declarations, and a test fails CI if the committed doc drifts from the
declarations — so the matrix can never silently go stale (the failure mode
that let the multimodal wire-marshalling rot undetected).
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Optional

import pytest

# Field list is the ONE definition of the capability columns; import it directly
# (base.py is pure — no vendor SDK — so this import is dep-safe).
from shared.plugins.model_provider.base import CAPABILITY_FIELDS

PROVIDER_DIR = Path(__file__).resolve().parents[1] / "plugins" / "model_provider"
DOC_PATH = (
    Path(__file__).resolve().parents[3] / "docs" / "model-provider-capabilities.md"
)

# Test stubs / non-providers excluded from the contract (mirror the tier
# guard's tests/__pycache__ skip).  ``echo`` is the deterministic test provider.
_EXCLUDE = {"tests", "__pycache__", "bundle_common", "echo"}

_DOC_MARKER = "<!-- AUTO-GENERATED from PROVIDER_CAPABILITIES declarations"


def _provider_dirs() -> List[str]:
    out = []
    for entry in sorted(os.listdir(PROVIDER_DIR)):
        d = PROVIDER_DIR / entry
        if not d.is_dir() or entry in _EXCLUDE or entry.startswith("_"):
            continue
        if (d / "__init__.py").exists():
            out.append(entry)
    return out


def _read_declaration(provider: str) -> Optional[Dict[str, bool]]:
    """AST-read ``PROVIDER_CAPABILITIES = ProviderCapabilities(<bool kwargs>)``.

    Returns the kwarg->bool dict, or ``None`` if the constant is absent.
    Raises ValueError if it's present but malformed (non-bool / positional).
    """
    src = (PROVIDER_DIR / provider / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "PROVIDER_CAPABILITIES" not in targets:
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "ProviderCapabilities"):
            raise ValueError(
                f"{provider}: PROVIDER_CAPABILITIES must be a "
                f"ProviderCapabilities(...) literal"
            )
        if call.args:
            raise ValueError(
                f"{provider}: declare every capability as an explicit keyword "
                f"(no positional args) so the contract is self-documenting"
            )
        decl: Dict[str, bool] = {}
        for kw in call.keywords:
            if not (isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, bool)):
                raise ValueError(
                    f"{provider}: capability {kw.arg!r} must be a bool literal"
                )
            decl[kw.arg] = kw.value.value
        return decl
    return None


# ---------------------------------------------------------------- structural

def test_every_provider_declares_capabilities():
    """Every provider must declare PROVIDER_CAPABILITIES (the contract gate)."""
    missing = [p for p in _provider_dirs() if _read_declaration(p) is None]
    assert missing == [], (
        "providers missing a PROVIDER_CAPABILITIES declaration in __init__.py: "
        f"{missing}. Add `PROVIDER_CAPABILITIES = ProviderCapabilities(...)` "
        "declaring every capability field (see docs/model-provider-capabilities.md)."
    )


def test_declarations_cover_every_field_with_bools():
    """Each declaration must set EVERY capability field, all bools (no drift
    when a new capability column is added to CAPABILITY_FIELDS)."""
    problems = []
    for p in _provider_dirs():
        decl = _read_declaration(p)
        if decl is None:
            continue  # caught by the test above
        missing = set(CAPABILITY_FIELDS) - set(decl)
        extra = set(decl) - set(CAPABILITY_FIELDS)
        if missing:
            problems.append(f"{p}: missing fields {sorted(missing)}")
        if extra:
            problems.append(f"{p}: unknown fields {sorted(extra)}")
    assert problems == [], "\n".join(problems)


# ---------------------------------------------------------------- doc drift

def _matrix() -> Dict[str, Dict[str, bool]]:
    return {p: _read_declaration(p) for p in _provider_dirs()}


def render_doc(matrix: Dict[str, Dict[str, bool]]) -> str:
    cols = list(CAPABILITY_FIELDS)
    header = "| Provider | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = []
    for prov in sorted(matrix):
        cells = ["✅" if matrix[prov].get(c) else "—" for c in cols]
        rows.append(f"| `{prov}` | " + " | ".join(cells) + " |")
    legend = (
        "\n**Legend:** ✅ = implemented & verified on the wire by the conformance "
        "guard · — = not implemented (text-only / not forwarded).\n"
    )
    return (
        "# Model-Provider Capability Matrix\n\n"
        f"{_DOC_MARKER} — edit the declarations, not this file. "
        "Regenerate: `python -m shared.tests.test_provider_capabilities`. -->\n\n"
        "Each cell is a **wire-level** behavior the CI conformance guard "
        "asserts, not a label. A provider declares its row via "
        "`PROVIDER_CAPABILITIES = ProviderCapabilities(...)` in its "
        "`__init__.py`; the guard fails the build if a declared capability "
        "isn't actually delivered on the wire.\n\n"
        + header + "\n" + sep + "\n" + "\n".join(rows) + "\n" + legend
    )


def test_committed_doc_matches_declarations():
    """The committed capability doc must match the declarations (no drift)."""
    expected = render_doc(_matrix())
    assert DOC_PATH.exists(), (
        f"{DOC_PATH} missing — regenerate with "
        "`python -m shared.tests.test_provider_capabilities`"
    )
    actual = DOC_PATH.read_text(encoding="utf-8")
    assert actual == expected, (
        "docs/model-provider-capabilities.md is stale vs the PROVIDER_CAPABILITIES "
        "declarations. Regenerate: "
        "`python -m shared.tests.test_provider_capabilities`"
    )


if __name__ == "__main__":
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(render_doc(_matrix()), encoding="utf-8")
    print(f"wrote {DOC_PATH}")
