"""session_id path-traversal / injection validation.

A client can supply session_id when attaching, and it is interpolated into a
filesystem path (``<sessions>/<id>.json``), the cgroup dir name
(``jaato-ws-<id>``), and the AppArmor profile name.  These tests pin the
validator and prove the persistence sink fails closed on a traversal id.
"""

import pytest

from shared.session_id import is_safe_session_id, validate_session_id


# ---- the validator ----------------------------------------------------------

@pytest.mark.parametrize("sid", [
    "20260704_192600",                       # timestamp (generate_session_id)
    "ephemeral-abcdef012345",                # ephemeral
    "20260704_192600__sub_main.subagent_1",  # isolated subagent (has a dot)
    "A-b_9.x",
])
def test_accepts_server_generated_forms(sid):
    assert is_safe_session_id(sid) is True


@pytest.mark.parametrize("sid", [
    "../../../etc/passwd",   # classic traversal
    "..",                    # parent dir
    "a/b",                   # path separator
    "a\\b",                  # windows separator
    "a..b",                  # embedded parent-dir sequence
    "",                      # empty
    "has space",             # whitespace
    "line\nbreak",           # newline (apparmor-grammar break)
    "brace{inject}",         # apparmor metacharacters
    "x" * 257,               # too long
])
def test_rejects_unsafe(sid):
    assert is_safe_session_id(sid) is False


def test_non_str_is_rejected():
    assert is_safe_session_id(None) is False  # type: ignore[arg-type]


def test_validate_returns_value_or_raises():
    assert validate_session_id("ok_id-1.2") == "ok_id-1.2"
    with pytest.raises(ValueError):
        validate_session_id("../escape")


# ---- the persistence sink fails closed --------------------------------------

def test_file_session_load_refuses_traversal(tmp_path):
    from shared.plugins.session.file_session import FileSessionPlugin
    plugin = FileSessionPlugin()
    # Even though we plant a file at the traversal target, load must refuse to
    # build the path at all (ValueError), never reaching the filesystem.
    with pytest.raises(ValueError):
        plugin.load("../../../etc/hostname", storage_dir=tmp_path)


def test_file_session_load_allows_valid_id(tmp_path):
    from shared.plugins.session.file_session import FileSessionPlugin
    plugin = FileSessionPlugin()
    # A valid id passes the guard and then hits the normal not-found path.
    with pytest.raises(FileNotFoundError):
        plugin.load("20260704_192600", storage_dir=tmp_path)
