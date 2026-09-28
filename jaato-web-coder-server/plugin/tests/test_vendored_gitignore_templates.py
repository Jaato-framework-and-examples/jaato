"""The vendored gitignore templates match their recorded checksums.

The per-toolchain patterns come from ``github/gitignore`` (CC0-1.0), vendored
under ``gitignore_templates/`` and pinned in ``PROVENANCE.json``.  This guard
fails if a vendored file is edited by hand or a checksum drifts, so the
patterns stay traceable to the upstream commit rather than becoming the
hand-written list this replaced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jaato_web_coder_toolchains import templates

_TDIR = Path(templates._TEMPLATES_DIR)


def _provenance() -> dict:
    return json.loads((_TDIR / "PROVENANCE.json").read_text())


def test_every_template_matches_its_recorded_sha256():
    prov = _provenance()
    for name, want in prov["files"].items():
        got = hashlib.sha256((_TDIR / name).read_bytes()).hexdigest()
        assert got == want, f"{name} drifted from PROVENANCE.json"


def test_every_mapped_template_is_vendored_and_recorded():
    prov = _provenance()
    for tool, base in templates.TEMPLATE_BY_TOOL.items():
        fname = base + ".gitignore"
        assert (_TDIR / fname).is_file(), f"{tool} maps to a missing {fname}"
        assert fname in prov["files"], f"{fname} is not in PROVENANCE.json"


def test_a_ref_is_pinned():
    prov = _provenance()
    assert len(prov["source_ref"]) == 40, "source_ref should be a full commit sha"
    assert prov["license"] == "CC0-1.0"


def test_patterns_come_from_the_template():
    # Maven's headline output, straight from the upstream template.
    assert "target/" in templates.patterns_for_tool("maven")
    # jaato's own jdtls artifact, added on top (not in upstream Java).
    assert ".factorypath" in templates.patterns_for_tool("java")
    # bun reuses the Node template.
    assert templates.patterns_for_tool("bun") == templates.patterns_for_tool("node")
    # comments and blanks are stripped.
    assert all(p and not p.startswith("#") for p in templates.patterns_for_tool("python"))
    # an unknown tool contributes nothing.
    assert templates.patterns_for_tool("cobol") == []
