"""validate tags each finding with its source tier (workspace vs inherited user ~/.jaato).

validate_workspace merges the workspace + user tiers (the daemon's effective set), so
it can surface findings from user-tier profiles the author doesn't own. Each finding
now carries .tier so the author can tell "mine" from "inherited". Requested by the
jaato-tui-driven-tests refactor (the validate scope wart).
"""
import tempfile
from pathlib import Path

from shared.scaffold.validate import validate_workspace, Diagnostic


def test_workspace_profile_findings_tagged_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        pdir = ws / ".jaato" / "profiles"
        pdir.mkdir(parents=True)
        (pdir / "myprof.yaml").write_text("name: myprof\nprovider: openrouter\n")
        diags = validate_workspace(str(ws))
        mine = [d for d in diags if d.profile == "myprof"]
        assert mine, "the workspace profile should produce findings"
        assert all(d.tier == "workspace" for d in mine), [(d.code, d.tier) for d in mine]
        # any inherited user-tier profile finding is tagged "user", never "workspace"
        for d in diags:
            if d.profile and d.profile != "myprof" and d.tier:
                assert d.tier == "user", (d.profile, d.code, d.tier)


def test_diagnostic_tier_in_as_dict():
    d = Diagnostic("warn", "x", "m", profile="p", tier="workspace")
    assert d.as_dict()["tier"] == "workspace"
