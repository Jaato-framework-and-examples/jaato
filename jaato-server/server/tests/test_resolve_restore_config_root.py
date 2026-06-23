"""_resolve_restore_config_root: disk-restore config_root fallbacks so a
pre-persistence session saved with config_root=None can't hang the re-spawned
runner (file_edit/auth path resolution). Order: saved → attaching-client →
<workspace>/.jaato default."""
from server.session_manager import SessionManager

R = SessionManager._resolve_restore_config_root


def test_prefers_saved_config_root():
    assert R("/saved/.jaato", "/client/.jaato", "/ws") == "/saved/.jaato"


def test_falls_back_to_attaching_client_when_saved_none():
    assert R(None, "/client/.jaato", "/ws") == "/client/.jaato"


def test_derives_workspace_default_when_saved_and_client_absent():
    # the config_root=None restore-hang case: workspace_path is reliably
    # persisted, so the framework default keeps the runner from hanging.
    assert R(None, None, "/home/u/runtime") == "/home/u/runtime/.jaato"


def test_none_only_when_workspace_also_absent():
    assert R(None, None, None) is None


def test_empty_strings_are_falsy_and_fall_through():
    assert R("", "", "/ws") == "/ws/.jaato"
