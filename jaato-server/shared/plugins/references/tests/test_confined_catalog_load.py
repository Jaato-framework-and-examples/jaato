"""Regression: a denied references tier must be skipped, not raise.

Under AppArmor confinement ``~/.jaato/references`` (and ``~/.config/jaato``) are
*correctly* denied.  Before the fix, ``Path.exists()`` raised ``PermissionError``
(pathlib does not ignore EACCES) out of ``discover_references`` / ``load_config``
-> the references plugin's ``initialize()`` -> the registry disabled the plugin,
so the workspace-tier catalog wired by ``set_workspace_path()`` never loaded.
See the 2026-06-20 LORA cascade bug report.
"""

import pathlib

from shared.plugins.references.config_loader import discover_references, load_config


def test_discover_references_skips_denied_dir(monkeypatch):
    real_exists = pathlib.Path.exists

    def fake_exists(self):
        if "denied-refs" in str(self):
            raise PermissionError(13, "Permission denied")
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)
    # Must NOT raise; a denied tier simply yields no sources.
    assert discover_references("denied-refs") == []


def test_load_config_skips_denied_default_location(monkeypatch):
    real_exists = pathlib.Path.exists

    def fake_exists(self):
        s = str(self)
        if ".config" in s and "jaato" in s:  # ~/.config/jaato/references.json
            raise PermissionError(13, "Permission denied")
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)
    # No workspace, denied HOME default — must fall through to defaults, not raise.
    cfg = load_config(None, workspace_path=None)
    assert cfg is not None
