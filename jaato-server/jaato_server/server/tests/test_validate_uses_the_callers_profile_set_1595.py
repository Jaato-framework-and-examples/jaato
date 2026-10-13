"""``scaffold.validate`` validates the CALLER's profile set, and says which (#1595).

Sibling of #1593.  With no ``profile_set`` in the request, the daemon's
validator called ``discover_profiles(force_profile_set=None)``, which read
``JAATO_PROFILE_SET`` through ``get_session_env``.  A daemon-level handler
has no session ContextVar, so that fell back to ``os.environ``: nothing,
on a quiet daemon (a profile that exists only inside a set was never
validated, and the answer still said ``ok=True``), or whatever set
another session's turn had overlaid (the profile validated was a
neighbour's).

Pinned here against the real router, ``SessionManager`` and validator,
with that overlay simulated in ``os.environ``:

- with no set requested, the set the caller's ``.env`` selects is
  validated, and the answer names it;
- a ``.env`` that selects no set means no set, never the overlay's;
- a declared ``env_file`` is the caller's ``.env``;
- a set the request names still wins.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from jaato_sdk.events import ScaffoldValidateEvent
from jaato_server.server.command_router import CommandRouter
from jaato_server.server.session_manager import SessionManager
from jaato_server.shared.tests.reversion import Reversion

_ROUTER = "jaato-server/jaato_server/server/command_router.py"
_VALIDATE = "jaato-server/jaato_server/shared/scaffold/validate.py"


class _Sink:
    def __init__(self):
        self.sent = []

    def send_event(self, client_id, event):
        self.sent.append(event)

    def get_client_user(self, client_id):
        return None

    def get_client_workspace(self, client_id):
        return None

    def visible_workspace_paths(self, client_id):
        return None


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture()
def daemon(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    ws = tmp_path / "ws"
    profiles = ws / ".jaato" / "profiles"
    # ``librarian`` exists only inside sets; each set's copy has a defect
    # the other's has not, so a finding says which copy was validated.
    _write(profiles / "mine_set" / "librarian.yaml",
           "name: librarian\nplugins: [cli]\nmodel: m\nprovider: anthropic\n")
    _write(profiles / "other_set" / "librarian.yaml",
           "name: librarian\ndescription: d\nplugins: [cli, no_such_plugin_x]\n"
           "model: m\nprovider: anthropic\n")
    # What another session's turn leaves in the process environment.
    monkeypatch.setenv("JAATO_PROFILE_SET", "other_set")
    sm = SessionManager()
    sink = _Sink()
    router = CommandRouter(session_manager=sm, event_sink=sink, daemon_plugins={})
    sm._client_config["c1"] = {"working_dir": str(ws)}
    return router, sink, sm, ws


def _validate(daemon, args):
    router, sink, _sm, ws = daemon
    sink.sent.clear()
    router._handle_scaffold_validate("c1", args, str(ws))
    [answer] = [e for e in sink.sent if isinstance(e, ScaffoldValidateEvent)]
    assert answer.ok, answer.error
    codes = {f["code"] for f in answer.findings if f.get("profile") == "librarian"}
    return answer, codes


def test_the_callers_env_selects_the_set_validated(daemon):
    _write(daemon[3] / ".env", "JAATO_PROFILE_SET=mine_set\n")
    _answer, codes = _validate(daemon, [])
    assert "missing_description" in codes
    assert "unknown_plugin" not in codes


def test_the_answer_names_the_set_the_callers_env_selected(daemon):
    _write(daemon[3] / ".env", "JAATO_PROFILE_SET=mine_set\n")
    answer, _codes = _validate(daemon, [])
    assert answer.profile_set == "mine_set"


def test_an_env_with_no_set_is_no_set_not_the_overlays(daemon):
    _write(daemon[3] / ".env", "OTHER=1\n")
    answer, codes = _validate(daemon, [])
    assert answer.profile_set == ""
    assert codes == set()


def test_a_declared_env_file_is_the_callers_env(daemon, tmp_path):
    env_file = tmp_path / "elsewhere.env"
    _write(env_file, "JAATO_PROFILE_SET=mine_set\n")
    _write(daemon[3] / ".env", "OTHER=1\n")
    daemon[2]._client_config["c1"]["env_file"] = str(env_file)
    answer, codes = _validate(daemon, [])
    assert answer.profile_set == "mine_set"
    assert "missing_description" in codes


def test_a_requested_set_wins_over_the_env(daemon):
    _write(daemon[3] / ".env", "JAATO_PROFILE_SET=mine_set\n")
    answer, codes = _validate(daemon, ["other_set"])
    assert answer.profile_set == "other_set"
    assert "unknown_plugin" in codes


REVERSIONS = [
    Reversion(
        target=_ROUTER,
        find="        profile_set = profile_set or _profile_set_of(caller_env)\n",
        replace="",
        test="test_the_answer_names_the_set_the_callers_env_selected",
        because="the answer would not say which set was validated",
    ),
    Reversion(
        target=_ROUTER,
        find="                session_env=caller_env)\n",
        replace="                session_env=None)\n",
        test="test_an_env_with_no_set_is_no_set_not_the_overlays",
        because="the router not handing the caller's env down, so discovery "
                "read the set from another session's os.environ overlay",
    ),
    Reversion(
        target=_VALIDATE,
        find="        session_env=session_env,\n    )\n",
        replace="    )\n",
        test="test_an_env_with_no_set_is_no_set_not_the_overlays",
        because="the validator dropping the caller's env, so discovery fell "
                "back to os.environ",
    ),
    Reversion(
        target=_ROUTER,
        find="        if not isinstance(env_file, str) or not env_file:\n"
             "            env_file = os.path.join(workspace, \".env\")\n",
        replace="        env_file = os.path.join(workspace, \".env\")\n",
        test="test_a_declared_env_file_is_the_callers_env",
        because="the client's declared env_file ignored",
    ),
]
