"""Guard: the comparison and design docs say what the tree says.

WHY THIS EXISTS (jaato #866).  Three documents under ``docs/`` stated things
the tree no longer matched, and each was the kind of claim an evaluator
quotes:

* ``compare-jaato-devin.md`` called jaato "Open-source (MIT)".  ``LICENSE``
  is BUSL-1.1 (Apache-2.0 from 2030-09-01) and every ``pyproject.toml``
  declares ``BUSL-1.1``.  For a corporate reader that is the single most
  consequential sentence in the doc, and it was wrong in the direction that
  invites a licensing mistake.
* ``compare-rbac-profiles-frameworks.md`` called AppArmor confinement
  "(premium)" in five places.  ``server/apparmor.py`` ships in the free
  server, ``IPCClient(apparmor=True)`` is documented in
  ``docs/apparmor-setup.md``, and ``apparmor: true`` is a free profile field.
* ``design/multimodal-model-support.md`` said image input existed in "4/13
  providers" and that breadth was "image-only; no PDF/audio/video".  Fifteen
  wires carry images, three carry PDF, two carry audio, and audio output
  streams.

Both comparison claims had already drifted once, which is the signature of a
claim nobody's test reads.  So this module reads them:

1. **Licence** — the identifier comes from ``jaato-server/pyproject.toml``,
   not from a literal here, and no line a comparison doc attributes to jaato
   may call it MIT or open source.  Attribution matters: a comparison doc may
   legitimately say a *competitor* is MIT, so a line counts as jaato's only
   when it names jaato, sits in jaato's column of a table, or sits under a
   heading that names jaato.
2. **AppArmor** — the premise is checked (the module exists in the free
   package, the setup guide exists) and then no comparison-doc line that
   mentions AppArmor may mention premium.
3. **Multimodal** — the v1 snapshot table is labelled historical, and the
   current-state table names exactly the providers whose
   ``PROVIDER_CAPABILITIES`` declaration says so, read with the same AST
   reader ``test_provider_capabilities`` uses.  A provider gaining PDF input
   fails this guard until the doc says so, which is the drift #866 found.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import pytest

from shared.tests.test_every_guard_detects_its_own_reversion import Reversion
from shared.tests.test_provider_capabilities import (
    _provider_dirs,
    _read_declaration,
)

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
COMPARISON_DOCS = sorted(DOCS.glob("compare-*.md"))
MULTIMODAL_DOC = DOCS / "design" / "multimodal-model-support.md"

#: The heading the historical table lives under, and the one the current
#: table lives under.  Both are asserted on, so a rename must land here too.
HISTORICAL_HEADING = "## Where jaato was when v1 was designed (historical snapshot)"
CURRENT_HEADING = "## Where jaato is now"

#: Claims a BUSL-1.1 tree cannot make about itself.
_FALSE_LICENCE_CLAIM = re.compile(r"\bMIT\b|open[- ]source", re.IGNORECASE)


REVERSIONS = [
    Reversion(
        target="docs/compare-jaato-devin.md",
        find="| **Source** | Source-available (BUSL-1.1; Apache-2.0 from "
             "2030-09-01) | Proprietary (closed-source) |",
        replace="| **Source** | Open-source (MIT) | Proprietary (closed-source) |",
        test="test_no_comparison_doc_calls_jaato_open_source_or_mit",
        because="the summary table calling jaato MIT-licensed, the exact "
                "cell #866 found and the one a corporate reader quotes",
    ),
    Reversion(
        target="docs/compare-rbac-profiles-frameworks.md",
        find="| **AppArmor profiles** (free server, [setup guide]"
             "(apparmor-setup.md)) | Kernel-level Mandatory Access Control",
        replace="| **AppArmor profiles** (premium) | Kernel-level Mandatory "
                "Access Control",
        test="test_apparmor_is_not_described_as_premium",
        because="AppArmor confinement labelled premium when server/apparmor.py "
                "ships in the free package",
    ),
    Reversion(
        target="docs/design/multimodal-model-support.md",
        find=HISTORICAL_HEADING + "\n",
        replace="## Where jaato is today (grounded)\n",
        test="test_the_v1_snapshot_table_is_labelled_historical",
        because="the v1 snapshot presented as the present tense, which is "
                "how '4/13 providers' got quoted as current",
    ),
    Reversion(
        target="docs/design/multimodal-model-support.md",
        find="all but `chrome_ai`, `claude_cli` and `github_models`",
        replace="4/13 providers (anthropic, google_genai, antigravity, "
                "openrouter)",
        test="test_the_current_state_table_names_the_providers_the_tree_declares",
        because="the image-input row naming a provider list the capability "
                "declarations do not agree with",
    ),
]


# --------------------------------------------------------------- licence

def _declared_licence() -> str:
    """The SPDX identifier ``jaato-server/pyproject.toml`` declares."""
    src = (ROOT / "jaato-server" / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^license\s*=\s*"([^"]+)"', src, re.MULTILINE)
    assert m, "jaato-server/pyproject.toml declares no `license = \"...\"`"
    return m.group(1)


def _heading_is_jaatos(heading: str) -> bool:
    """A section heading that names jaato and nothing it is compared with.

    ``# Jaato vs Devin`` names both, so its body is not attributed; ``###
    Jaato`` and ``### Jaato Only`` are.
    """
    h = heading.lower()
    return "jaato" in h and " vs " not in h


def _jaato_column(header: str) -> Optional[int]:
    """Index of the table column whose header names jaato, if any."""
    cells = [c.strip().lower() for c in header.strip().strip("|").split("|")]
    for i, cell in enumerate(cells):
        if "jaato" in cell:
            return i
    return None


def _cell(line: str, index: int) -> str:
    cells = line.strip().strip("|").split("|")
    return cells[index] if index < len(cells) else ""


def _iter_jaato_text(text: str) -> Iterator[Tuple[int, str]]:
    """Yield ``(lineno, fragment)`` for every fragment attributed to jaato.

    A table row contributes jaato's column when the header names one, or
    the whole row when the table sits under a jaato heading.  A prose line
    contributes itself when it names jaato or sits under a jaato heading.
    """
    under_jaato = False
    column: Optional[int] = None
    previous_was_row = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        is_row = stripped.startswith("|")
        if stripped.startswith("#"):
            under_jaato = _heading_is_jaatos(stripped)
        elif is_row and not previous_was_row:
            column = _jaato_column(stripped)
        elif is_row and column is not None:
            yield lineno, _cell(stripped, column)
        elif is_row and under_jaato:
            yield lineno, stripped
        elif not is_row and (under_jaato or "jaato" in stripped.lower()):
            yield lineno, stripped
        previous_was_row = is_row


def _false_licence_claims(doc: Path) -> List[str]:
    text = doc.read_text(encoding="utf-8")
    return [
        f"{doc.relative_to(ROOT)}:{lineno}: {fragment.strip()}"
        for lineno, fragment in _iter_jaato_text(text)
        if _FALSE_LICENCE_CLAIM.search(fragment)
    ]


def test_the_tree_is_busl_licensed():
    """Premise.  If the licence ever changes, the claims below change with it
    and this is the assertion that says so, rather than a silently wrong
    guard."""
    assert _declared_licence().startswith("BUSL"), _declared_licence()
    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Business Source License" in licence


def test_no_comparison_doc_calls_jaato_open_source_or_mit():
    assert COMPARISON_DOCS, "no docs/compare-*.md found; the guard is vacuous"
    offending = [c for doc in COMPARISON_DOCS for c in _false_licence_claims(doc)]
    assert offending == [], (
        f"the tree is {_declared_licence()} (LICENSE: Business Source License "
        f"1.1, Apache-2.0 from the change date), which is source-available, "
        f"not MIT and not OSI open source. These lines say otherwise about "
        f"jaato:\n  " + "\n  ".join(offending)
    )


def test_a_comparison_doc_that_states_the_licence_names_the_declared_one():
    """Wherever a doc talks about jaato's licence at all, the identifier the
    packages declare must appear — a doc that discusses licensing without
    ever naming BUSL-1.1 is one edit away from the #866 claim."""
    licence = _declared_licence()
    for doc in COMPARISON_DOCS:
        text = doc.read_text(encoding="utf-8")
        mentions = [f for _n, f in _iter_jaato_text(text)
                    if re.search(r"licen[cs]e", f, re.IGNORECASE)]
        if mentions:
            assert licence in text, (
                f"{doc.relative_to(ROOT)} discusses jaato's licence "
                f"({mentions[0].strip()!r}) but never names {licence}"
            )


# --------------------------------------------------------------- apparmor

def test_apparmor_ships_in_the_free_server():
    """Premise for the test below: the module, and the guide that documents
    the opt-in surfaces, both live in this tree."""
    assert (ROOT / "jaato-server" / "server" / "apparmor.py").is_file()
    assert (DOCS / "apparmor-setup.md").is_file()


def test_apparmor_is_not_described_as_premium():
    offending = []
    for doc in COMPARISON_DOCS:
        for lineno, line in enumerate(
                doc.read_text(encoding="utf-8").splitlines(), start=1):
            low = line.lower()
            if "apparmor" in low and "premium" in low:
                offending.append(f"{doc.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert offending == [], (
        "AppArmor confinement ships in the free server (server/apparmor.py; "
        "IPCClient(apparmor=True); the profile field `apparmor: true`). "
        "These lines call it premium:\n  " + "\n  ".join(offending)
    )


# ------------------------------------------------------------- multimodal

def _section(text: str, heading: str) -> str:
    """The body of *heading* up to the next ``##`` heading."""
    start = text.index(heading) + len(heading)
    rest = text[start:]
    m = re.search(r"^## ", rest, re.MULTILINE)
    return rest[: m.start()] if m else rest


def _row(section: str, label: str) -> str:
    for line in section.splitlines():
        cells = line.strip().strip("|").split("|")
        if cells and cells[0].strip().startswith(label):
            return line
    raise AssertionError(f"no table row starting with {label!r} under "
                         f"{CURRENT_HEADING!r}")


def _providers_named(row: str) -> Set[str]:
    """Backticked provider package names in a table row."""
    return set(re.findall(r"`([a-z0-9_]+)`", row)) & set(_provider_dirs())


def _declaring(flag: str) -> Set[str]:
    return {p for p in _provider_dirs()
            if (_read_declaration(p) or {}).get(flag)}


def _expected_rows() -> Dict[str, Set[str]]:
    """Row label -> the provider set that row must name, from the tree."""
    everyone = set(_provider_dirs())
    return {
        "Image-input conversion": everyone - _declaring("user_message_images"),
        "Modality breadth": _declaring("pdf_input") | _declaring("audio_input"),
        "Output (model": _declaring("output_media"),
    }


def test_the_v1_snapshot_table_is_labelled_historical():
    text = MULTIMODAL_DOC.read_text(encoding="utf-8")
    assert HISTORICAL_HEADING in text, (
        "the v1 'where jaato is' table must be headed as a historical "
        "snapshot; presented as the present it gets quoted as the present"
    )
    assert "## Where jaato is today" not in text
    assert text.index(HISTORICAL_HEADING) < text.index(CURRENT_HEADING)


def test_the_current_state_table_names_the_providers_the_tree_declares():
    """The image row names the providers that do NOT carry images (the
    exceptions are the short list); the breadth and output rows name the
    providers that DO declare the capability."""
    section = _section(MULTIMODAL_DOC.read_text(encoding="utf-8"),
                       CURRENT_HEADING)
    mismatches = []
    for label, expected in _expected_rows().items():
        named = _providers_named(_row(section, label))
        if named != expected:
            mismatches.append(
                f"{label!r}: doc names {sorted(named)}, declarations say "
                f"{sorted(expected)} (missing {sorted(expected - named)}, "
                f"extra {sorted(named - expected)})"
            )
    assert mismatches == [], (
        "docs/design/multimodal-model-support.md 'Where jaato is now' "
        "disagrees with PROVIDER_CAPABILITIES:\n  " + "\n  ".join(mismatches)
    )


def test_every_reversion_here_names_a_test_in_this_module():
    """A reversion naming a renamed test is BLOCKED in the meta-guard, which
    it reports; this makes the rename fail here first, where it is edited."""
    names = set(globals())
    for rev in REVERSIONS:
        assert rev.test in names, rev.test
