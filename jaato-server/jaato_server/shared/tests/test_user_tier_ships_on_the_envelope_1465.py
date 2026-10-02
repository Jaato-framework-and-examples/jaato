"""The user tier a confined runner cannot read reaches it on the envelope (#1465).

With ``~/.jaato/permissions.json`` present, every confined session was
refused: AppArmor grants the runner only the ``~/.jaato`` subtrees plugins
declare, ``load_config`` found the file with ``exists()`` (AppArmor does
not mediate ``stat``) and ``open`` raised ``PermissionError``, which
``PermissionPlugin.initialize`` does not catch.  Reproduced on an
enforcing kernel; see the issue.

The daemon is not confined, so it reads those files and ships them on
``SessionInitEnvelope.user_tier_files``; the runner installs the snapshot
at bootstrap step 1e and every reader asks ``shared.user_tier.path``.

The tests stand in for the kernel with ``chmod 000``: like the AppArmor
denial, ``exists()`` answers True and ``open`` raises ``EACCES``.  That
needs a non-root uid (root ignores the mode), so those cases skip as root.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from jaato_server.shared import user_tier
from jaato_server.shared.plugins.permission.config_loader import load_config
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_USER_TIER = "jaato-server/jaato_server/shared/user_tier.py"
_LOADER = "jaato-server/jaato_server/shared/plugins/permission/config_loader.py"
_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"
_SESSION = "jaato-server/jaato_server/server/runner/session.py"

REVERSIONS = [
    Reversion(
        target=_LOADER,
        find='            _user_tier_path("permissions.json"),',
        replace='            Path.home() / ".jaato" / "permissions.json",',
        test="test_an_unreadable_permissions_file_is_read_from_the_snapshot",
        because="the loader reads the real file, which the runner is not granted",
    ),
    Reversion(
        target=_USER_TIER,
        find="    if is_credential(os.path.basename(rel)):\n        return None\n",
        replace="",
        test="test_credentials_are_never_shipped",
        because="a credential under a shipped directory reaches the runner",
    ),
    Reversion(
        target=_USER_TIER,
        find="os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK",
        replace="os.O_RDONLY | os.O_NONBLOCK",
        test="test_a_symlink_is_not_followed",
        because="a link in ~/.jaato carries another file into the session",
    ),
    Reversion(
        target=_SPAWN,
        find="        user_tier_files=user_tier_snapshot(stashed_runner_user(server)),\n",
        replace="",
        test="test_both_envelope_builders_ship_the_snapshot",
        because="the daemon never ships the snapshot on the main path",
    ),
    Reversion(
        target=_SESSION,
        find="    _install_user_tier(envelope)\n\n    # ---- 2.",
        replace="\n    # ---- 2.",
        test="test_bootstrap_installs_the_snapshot_before_the_runtime",
        because="the runner never installs what the daemon shipped",
    ),
]


@pytest.fixture(autouse=True)
def _no_installed_snapshot():
    user_tier.install(None, "/nonexistent", "unused")
    yield
    user_tier.install(None, "/nonexistent", "unused")


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / ".jaato").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    return h


_POLICY = {"version": "1.0", "defaultPolicy": "allow",
           "whitelist": {"tools": ["readFile"]}}

needs_non_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="root ignores file modes; the denial cannot be simulated")


@needs_non_root
def test_an_unreadable_permissions_file_is_read_from_the_snapshot(home, tmp_path):
    perm = home / ".jaato" / "permissions.json"
    perm.write_text(json.dumps(_POLICY))
    snapshot = user_tier.collect(str(home / ".jaato"))  # the daemon reads it
    perm.chmod(0)                                       # the runner may not
    workspace = tmp_path / "ws"
    workspace.mkdir()
    try:
        with pytest.raises(PermissionError):            # the reported defect
            load_config(base_path=str(workspace))
        user_tier.install(snapshot, str(tmp_path / "tmp"), "s1")
        config = load_config(base_path=str(workspace))
    finally:
        perm.chmod(0o600)
    assert config.default_policy == "allow"
    assert config.whitelist_tools == ["readFile"]


def test_credentials_are_never_shipped(home):
    jaato = home / ".jaato"
    (jaato / "anthropic_auth.json").write_text('{"api_key": "sk-x"}')
    (jaato / "scripts").mkdir()
    (jaato / "scripts" / "leak_auth.json").write_text('{"k": "v"}')
    (jaato / "scripts" / "tok_oauth.json").write_text('{"k": "v"}')
    (jaato / "scripts" / "proc.py").write_text("def validate(p): return []\n")
    snapshot = user_tier.collect(str(jaato))
    assert set(snapshot) == {"scripts/proc.py"}


def test_a_symlink_is_not_followed(home, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("not yours")
    (home / ".jaato" / "permissions.json").symlink_to(outside)
    assert "permissions.json" not in user_tier.collect(str(home / ".jaato"))


def test_only_the_listed_files_and_directories_are_shipped(home):
    jaato = home / ".jaato"
    (jaato / "permissions.json").write_text("{}")
    (jaato / "instructions").mkdir()
    (jaato / "instructions" / "10-base.md").write_text("be terse")
    (jaato / "memories").mkdir()                       # granted, not shipped
    (jaato / "memories" / "curated.jsonl").write_text("{}")
    (jaato / "profiles").mkdir()
    (jaato / "profiles" / "p.yaml").write_text("name: p")
    assert set(user_tier.collect(str(jaato))) == {
        "permissions.json", "instructions/10-base.md"}


def test_an_oversized_file_is_skipped(home):
    (home / ".jaato" / "pricing.json").write_text("x" * (user_tier.MAX_FILE_BYTES + 1))
    assert user_tier.collect(str(home / ".jaato")) == {}


def test_without_a_snapshot_readers_read_the_disk(home):
    (home / ".jaato" / "pricing.json").write_text("{}")
    assert user_tier.path("pricing.json") == home / ".jaato" / "pricing.json"
    assert user_tier.path("pricing.json").read_text() == "{}"


def test_an_empty_snapshot_hides_the_disk(home, tmp_path):
    (home / ".jaato" / "pricing.json").write_text("{}")
    user_tier.install({}, str(tmp_path / "tmp"), "s1")
    assert not user_tier.path("pricing.json").exists()


def test_a_new_install_replaces_the_previous_one(tmp_path):
    first = user_tier.install({"pricing.json": "1"}, str(tmp_path), "a")
    user_tier.install({"pricing.json": "2"}, str(tmp_path), "b")
    assert not first.exists()
    assert user_tier.path("pricing.json").read_text() == "2"


def test_a_snapshot_entry_cannot_leave_its_directory(tmp_path):
    root = user_tier.install({"../escape.json": "x", "ok.json": "y"}, str(tmp_path), "s")
    assert not (tmp_path / "escape.json").exists()
    assert (root / "ok.json").read_text() == "y"


def test_the_envelope_carries_the_snapshot_and_keeps_none_apart_from_empty():
    def roundtrip(value):
        env = SessionInitEnvelope(session_id="s", workspace_path="/w",
                                  profile_name="", provider_name="echo",
                                  model_name="m", user_tier_files=value)
        return SessionInitEnvelope.from_dict(json.loads(json.dumps(env.to_dict())))
    assert roundtrip({"a.json": "x"}).user_tier_files == {"a.json": "x"}
    assert roundtrip({}).user_tier_files == {}
    assert roundtrip(None).user_tier_files is None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _calls_with_kwarg(path: str, func: str, kwarg: str) -> int:
    tree = ast.parse((_repo_root() / path).read_text())
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            for call in ast.walk(node):
                if (isinstance(call, ast.Call)
                        and getattr(call.func, "id", None) == "SessionInitEnvelope"
                        and any(k.arg == kwarg for k in call.keywords)):
                    count += 1
    return count


def test_both_envelope_builders_ship_the_snapshot():
    assert _calls_with_kwarg(_SPAWN, "build_session_envelope", "user_tier_files") == 1
    assert _calls_with_kwarg(
        "jaato-server/jaato_server/server/session_manager.py",
        "_build_isolated_envelope", "user_tier_files") == 1


def test_bootstrap_installs_the_snapshot_before_the_runtime():
    tree = ast.parse((_repo_root() / _SESSION).read_text())
    func = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "bootstrap_session")
    order = [getattr(c.func, "id", None) for c in ast.walk(func)
             if isinstance(c, ast.Call)]
    calls = [n for n in ast.walk(func) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) in ("_install_user_tier", "_pin_session_tmpdir")]
    assert "_install_user_tier" in order
    pin = next(c for c in calls if c.func.id == "_pin_session_tmpdir")
    install = next(c for c in calls if c.func.id == "_install_user_tier")
    assert pin.lineno < install.lineno   # written under the pinned temp dir
