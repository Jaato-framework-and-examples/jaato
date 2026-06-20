"""Regression: a denied profiles tier must be skipped, not abort discovery.

Under AppArmor confinement ``~/.jaato/profiles`` is *correctly* denied.  Before
the fix, ``_scan_profiles_dir``'s ``is_dir()`` / ``iterdir()`` raised
``PermissionError`` (pathlib does not ignore EACCES), which propagated out of
``discover_profiles`` -> the subagent plugin's ``initialize()`` -> the registry
disabled the whole plugin ("Plugin will not be exposed"), so the workspace-tier
profiles wired by ``set_config_root()`` never loaded.  See the 2026-06-20 LORA
cascade bug report.
"""

from unittest.mock import MagicMock

from shared.plugins.subagent.config import _scan_profiles_dir


def test_scan_profiles_dir_skips_tier_when_is_dir_denied():
    denied = MagicMock()
    denied.is_dir.side_effect = PermissionError(13, "Permission denied")
    profiles, errors = {}, {}
    _scan_profiles_dir(denied, profiles, errors)  # must NOT raise
    assert profiles == {}
    assert errors == {}


def test_scan_profiles_dir_skips_tier_when_iterdir_denied():
    denied = MagicMock()
    denied.is_dir.return_value = True
    denied.iterdir.side_effect = PermissionError(13, "Permission denied")
    profiles, errors = {}, {}
    _scan_profiles_dir(denied, profiles, errors)  # must NOT raise
    assert profiles == {}
