"""discover_bundles DEBUG-skips a tier whose root raises EACCES (confined user tier).

The references plugin's misleading `PermissionError: ~/.jaato/references`: under
AppArmor a confined runner is CORRECTLY denied the USER tier, but pathlib's
is_dir()/iterdir() RAISE PermissionError (they don't return False), and the
generic bundle tier-scanner didn't catch it — surfacing a bug-looking error for
an expected, by-design denial.  discover_bundles now skips the denied tier; other
tiers still load.
"""
from unittest.mock import MagicMock

from shared.plugins.bundle_common.bundle import discover_bundles


def test_is_dir_eacces_tier_skipped_not_raised():
    denied = MagicMock()
    denied.is_dir.side_effect = PermissionError(
        "[Errno 13] Permission denied: '/home/x/.jaato/references'")
    assert discover_bundles([(denied, "user")]) == []   # skipped, no raise


def test_iterdir_eacces_tier_skipped_not_raised():
    denied = MagicMock()
    denied.is_dir.return_value = True
    denied.iterdir.side_effect = PermissionError("[Errno 13] Permission denied")
    assert discover_bundles([(denied, "user")]) == []   # skipped, no raise


def test_denied_tier_does_not_abort_a_readable_tier():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        readable = Path(tmp)             # empty but readable workspace tier
        denied = MagicMock()
        denied.is_dir.side_effect = PermissionError("denied")
        # Must complete (readable tier scanned, denied skipped) without raising.
        assert discover_bundles([(readable, "workspace"), (denied, "user")]) == []
