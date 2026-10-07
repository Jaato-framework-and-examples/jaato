"""``validate`` for an install with only the SDK, answered by the daemon (#1267, tier 3).

``jaato-scaffold`` ships with jaato-sdk, and ``validate`` needs jaato-server's
loader: most checks run on a profile only once it is parsed, merged with its
``inherits:`` and set overlay, and constructed.  The snapshot cannot carry
that (measured on #1267), so an SDK-only install asks the daemon, and the
daemon's own full validator answers.

What this file holds it to:

* the daemon runs ``validate_workspace`` (one validator) on the caller's OWN
  workspace, resolved the way every daemon-level verb resolves it, and passes
  ``--set`` / ``--profile`` through;
* with no workspace it refuses rather than validating the daemon's cwd;
* a validator that raised is a refusal, never an empty pass;
* the SDK refuses a daemon below 1.34, sends both argument positions, and the
  shell prints the daemon's findings the way a local run prints its own, plus
  a line naming whose install answered;
* with no daemon to ask, an SDK-only ``validate`` still refuses (exit 2): a
  validator that did not run never reports a pass.
"""

from __future__ import annotations

import asyncio
import types
from unittest.mock import MagicMock

import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import PROTOCOL_VERSION, ClientType, ScaffoldValidateEvent
from jaato_sdk.scaffold import findings as F
from jaato_sdk.scaffold import remote as R
from jaato_server.server.command_router import CommandRouter
from jaato_server.shared.scaffold import introspection_verbs
from jaato_server.shared.scaffold import validate as V
from jaato_server.shared.tests.reversion import Reversion

_ROUTER = "jaato-server/jaato_server/server/command_router.py"
_REMOTE = "jaato-sdk/jaato_sdk/scaffold/remote.py"

REVERSIONS = [
    Reversion(
        target=_ROUTER,
        find=('''        if not workspace:
            checked = _checked_sources(sources)
            logger.warning("scaffold.validate: client=%s has no resolvable "'''),
        replace=('''        workspace = workspace or "."
        if not workspace:
            checked = _checked_sources(sources)
            logger.warning("scaffold.validate: client=%s has no resolvable "'''),
        because="a connection with no workspace would have the daemon's own "
                "cwd validated and reported as the caller's",
        test="test_no_workspace_is_a_refusal_and_nothing_is_validated",
    ),
    Reversion(
        target=_ROUTER,
        find=('''        except Exception as exc:
            logger.warning("scaffold.validate: client=%s workspace=%s "
                           "raised: %s", client_id, workspace, exc)
            answer(ok=False, workspace=workspace,'''),
        replace=('''        except Exception as exc:
            diags = []
        if False:
            answer(ok=False, workspace=workspace,'''),
        because="a validator that crashed would answer as a clean run with no "
                "findings",
        test="test_a_validator_that_raises_is_a_refusal_not_a_pass",
    ),
    Reversion(
        target=_ROUTER,
        find=('''        if cmd == "scaffold.validate":
            self._handle_scaffold_validate(
                client_id, args, workspace_path, session_id=session_id)
            return True
'''),
        replace="",
        because="the verb exists and nothing routes to it, so every SDK-only "
                "validate waits out its deadline",
        test="test_the_prefixed_dispatcher_routes_the_verb",
    ),
    Reversion(
        target=_REMOTE,
        find='''    return 1 if answer.errors else 0''',
        replace='''    return 0''',
        because="a daemon that found errors would let CI pass",
        test="test_the_shell_prints_the_daemons_findings_and_fails_on_errors",
    ),
    Reversion(
        target=_REMOTE,
        find=('''    if not answer.ok:
        print(answer.error or "the daemon's validator did not run",
              file=sys.stderr)'''),
        replace=('''    if False:
        print(answer.error or "the daemon's validator did not run",
              file=sys.stderr)'''),
        because="a validator that did not run would print as a clean pass",
        test="test_a_validator_that_did_not_run_is_not_a_pass",
    ),
]


# ------------------------------------------------------------------ daemon

def _make_router(session_workspace=None):
    session_manager = MagicMock()
    session = MagicMock() if session_workspace is not None else None
    if session is not None:
        session.workspace_path = session_workspace
    session_manager.get_client_session.return_value = session
    session_manager.get_session.return_value = None
    event_sink = MagicMock()
    router = CommandRouter.__new__(CommandRouter)
    router._session_manager = session_manager
    router._event_sink = event_sink
    return router, event_sink


def _only_answer(sink) -> ScaffoldValidateEvent:
    sent = [call[0][1] for call in sink.send_event.call_args_list]
    assert len(sent) == 1, sent
    assert isinstance(sent[0], ScaffoldValidateEvent)
    return sent[0]


@pytest.fixture
def fake_validate(monkeypatch):
    calls = []
    result = []

    def fake(workspace, *, profile_set=None, only=None, config_root=None):
        calls.append((workspace, profile_set, only))
        return list(result)

    monkeypatch.setattr(V, "validate_workspace", fake)
    return calls, result


def test_the_callers_workspace_is_validated_with_set_and_profile(
        tmp_path, fake_validate):
    calls, result = fake_validate
    result.extend([
        V.Diagnostic("error", "unknown_plugin", "no such plugin", profile="w"),
        V.Diagnostic("warn", "missing_description", "say what it does",
                     profile="w", tier="workspace"),
        V.Diagnostic("warn", "reactor_rule_invalid", "bad rule",
                     source="jaato-premium:reactors"),
    ])
    router, sink = _make_router(session_workspace=str(tmp_path))

    router._handle_scaffold_validate("c1", ["dev", "w"], None)

    assert calls == [(str(tmp_path), "dev", "w")]
    ans = _only_answer(sink)
    assert ans.ok is True
    assert ans.workspace == str(tmp_path)
    assert (ans.profile_set, ans.profile, ans.scope) == ("dev", "w",
                                                         "profile 'w'")
    assert (ans.errors, ans.warnings) == (1, 2)
    assert [f["code"] for f in ans.findings] == [
        "unknown_plugin", "missing_description", "reactor_rule_invalid"]
    # The as_dict shape, `source` only where it was set.
    assert "source" not in ans.findings[0]
    assert ans.findings[2]["source"] == "jaato-premium:reactors"
    assert ans.server_version


def test_a_declared_workspace_is_used_when_no_session_holds_one(
        tmp_path, fake_validate):
    calls, _ = fake_validate
    router, sink = _make_router(session_workspace=None)
    router._handle_scaffold_validate("c1", ["", ""], str(tmp_path))
    assert calls == [(str(tmp_path), None, None)]
    ans = _only_answer(sink)
    assert ans.ok and ans.scope == "all profiles" and ans.findings == []


def test_no_workspace_is_a_refusal_and_nothing_is_validated(fake_validate):
    calls, _ = fake_validate
    router, sink = _make_router(session_workspace=None)
    router._handle_scaffold_validate("c1", [], None)
    ans = _only_answer(sink)
    assert ans.ok is False
    assert "no workspace" in ans.error
    assert calls == []


def test_a_validator_that_raises_is_a_refusal_not_a_pass(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("loader broke")
    monkeypatch.setattr(V, "validate_workspace", boom)
    router, sink = _make_router(session_workspace=str(tmp_path))
    router._handle_scaffold_validate("c1", [], None)
    ans = _only_answer(sink)
    assert ans.ok is False
    assert "RuntimeError: loader broke" in ans.error
    assert ans.findings == []


def test_the_prefixed_dispatcher_routes_the_verb(tmp_path, fake_validate):
    calls, _ = fake_validate
    router, sink = _make_router(session_workspace=str(tmp_path))
    handled = router._dispatch_prefixed_command(
        "scaffold.validate", "c1", ["", "w"], None, None)
    assert handled is True
    assert calls == [(str(tmp_path), None, "w")]


def test_a_real_workspace_is_validated_by_the_real_validator(tmp_path):
    prof = tmp_path / ".jaato" / "profiles"
    prof.mkdir(parents=True)
    (prof / "w.yaml").write_text(
        "name: w\ndescription: x\nplugins: [no_such_plugin_1267]\n",
        encoding="utf-8")
    router, sink = _make_router(session_workspace=str(tmp_path))
    router._handle_scaffold_validate("c1", [], None)
    ans = _only_answer(sink)
    local = [d.as_dict() for d in V.validate_workspace(str(tmp_path))]
    assert ans.ok is True
    assert ans.findings == local
    assert any(f["code"] == "unknown_plugin" for f in ans.findings)


# ------------------------------------------------------------------ client

@pytest.fixture
def sent(monkeypatch):
    out = []

    async def fake_send(self, event):
        out.append(event)
    monkeypatch.setattr(IPCClient, "_send_event", fake_send, raising=True)
    return out


def _client(spoken):
    client = IPCClient("/tmp/does-not-exist.sock", client_type=ClientType.API,
                       auto_start=False)
    client._server_protocol_version = spoken
    return client


def test_the_client_sends_both_positions(sent):
    assert IPCClient.MIN_SCAFFOLD_VALIDATE_PROTOCOL == "1.34"
    # The daemon serves the verb: its protocol is at or past the floor.
    assert tuple(map(int, PROTOCOL_VERSION.split("."))) >= (1, 34)
    asyncio.run(_client("1.34").validate_workspace(profile="w"))
    assert [(e.command, e.args) for e in sent] == [
        ("scaffold.validate", ["", "w"])]


def test_an_older_daemon_is_refused_with_nothing_sent(sent):
    with pytest.raises(ValueError, match="1.34"):
        asyncio.run(_client("1.33").validate_workspace())
    assert sent == []


# ------------------------------------------------------------------ shell

def _answer(**kw):
    base = dict(reached=True, ok=True, socket_path="/tmp/j.sock",
                server_version="9.9.9", scope="all profiles")
    base.update(kw)
    return R.RemoteValidation(**base)


def test_the_shell_prints_the_daemons_findings_and_fails_on_errors(
        tmp_path, monkeypatch, capsys):
    found = [{"severity": "error", "code": "unknown_plugin", "message": "m",
              "profile": "w", "where": None, "tier": None},
             {"severity": "warn", "code": "x", "message": "y", "profile": None,
              "where": "a.json", "tier": "workspace",
              "source": "jaato-premium:reactors"}]
    asked = []
    monkeypatch.setattr(R, "ask_daemon_validate",
                        lambda sock, ws, s, p: asked.append((sock, ws, s, p))
                        or _answer(findings=found, errors=1))
    rc = R.validate_from_daemon("/tmp/j.sock", str(tmp_path), "dev", None,
                                json_out=False, required=True)
    out = capsys.readouterr().out.splitlines()
    assert rc == 1
    assert asked == [("/tmp/j.sock", str(tmp_path.resolve()), "dev", None)]
    # The same line a local run prints for the same finding.
    local = V.Diagnostic("warn", "x", "y", where="a.json", tier="workspace",
                         source="jaato-premium:reactors")
    assert introspection_verbs._format_diagnostic(local) in out
    assert out[0] == "[error] w: unknown_plugin: m"
    assert "jaato-server 9.9.9" in out[-1] and "not this venv" in out[-1]


def test_a_clean_daemon_run_says_so_and_exits_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, "ask_daemon_validate", lambda *a: _answer())
    rc = R.validate_from_daemon(True, str(tmp_path), None, None,
                                json_out=False, required=True)
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert out[0] == F.clean_line("all profiles", None)


def test_a_validator_that_did_not_run_is_not_a_pass(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, "ask_daemon_validate",
                        lambda *a: _answer(ok=False, error="no workspace"))
    rc = R.validate_from_daemon(True, str(tmp_path), None, None,
                                json_out=False, required=True)
    cap = capsys.readouterr()
    assert rc == 2
    assert "valid" not in cap.out
    assert "no workspace" in cap.err


def test_an_unreachable_daemon_is_reported_or_left_to_the_refusal(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, "ask_daemon_validate", lambda *a: R.RemoteValidation(
        socket_path="/tmp/j.sock", unreachable="could not connect"))
    assert R.validate_from_daemon(True, str(tmp_path), None, None,
                                  json_out=False, required=True) == 2
    assert "could not connect" in capsys.readouterr().err
    # Not asked for, and no daemon on the default socket: the caller refuses.
    monkeypatch.setattr(R, "daemon_is_listening", lambda path: False)
    assert R.validate_from_daemon(None, str(tmp_path), None, None,
                                  json_out=False, required=False) is None


def test_a_standalone_profile_file_is_not_sent_to_a_daemon(
        tmp_path, monkeypatch, capsys):
    loose = tmp_path / "example.yaml"
    loose.write_text("name: x\n", encoding="utf-8")
    monkeypatch.setattr(R, "ask_daemon_validate",
                        lambda *a: pytest.fail("a daemon was asked"))
    assert R.validate_from_daemon(True, str(loose), None, None,
                                  json_out=False, required=True) == 2
    assert "standalone profile file" in capsys.readouterr().err


def test_a_profile_file_in_the_layout_names_its_workspace_set_and_profile(
        tmp_path, monkeypatch):
    f = tmp_path / ".jaato" / "profiles" / "dev" / "w.yaml"
    f.parent.mkdir(parents=True)
    f.write_text("name: w\n", encoding="utf-8")
    asked = []
    monkeypatch.setattr(R, "ask_daemon_validate",
                        lambda sock, ws, s, p: asked.append((ws, s, p))
                        or _answer())
    R.validate_from_daemon(True, str(f), None, None, json_out=True,
                           required=True)
    assert asked == [(str(tmp_path.resolve()), "dev", "w")]


def test_server_validate_connect_goes_to_the_daemon(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(R, "validate_from_daemon",
                        lambda *a, **k: seen.append((a, k)) or 0)
    verb = introspection_verbs.ValidateVerb()
    args = types.SimpleNamespace(target=str(tmp_path), set=None, profile=None,
                                 json=False, connect="/tmp/j.sock")
    assert verb.run(args) == 0
    assert seen and seen[0][0][0] == "/tmp/j.sock"
    assert seen[0][1]["required"] is True
