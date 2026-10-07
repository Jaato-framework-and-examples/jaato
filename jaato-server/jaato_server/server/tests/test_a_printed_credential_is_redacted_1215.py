"""A credential a tool prints never leaves the runner verbatim (#1215).

A ``cli_based_tool`` call printed a provider key out of the workspace
``.env`` (the file ``config.update`` writes it into) and the key reached the
web client's live popup.  The #863 scrub was working: it removes secret
NAMES from what a subprocess inherits, and cannot stop a command reading a
value out of a file.  From there the value went, verbatim, to every client,
the model's history, the session record and the traces.

The fix is a value redactor (``shared/secret_redaction.py``) applied at the
two seams every destination is fed from:

* ``ToolExecutor.execute``'s return -- what enters history, so what the
  model and every later request see;
* ``RunnerRPC._write`` -- every stream, notification and response frame the
  daemon fans out to clients, the session record and its traces.

These tests drive the real ``cli`` plugin, the real ``ToolExecutor`` and a
real ``RunnerRPC`` over a socketpair, and read what actually crossed.
"""

from __future__ import annotations

import logging
import socket
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from jaato_server.server.runner.rpc import RunnerRPC
from jaato_server.shared import secret_redaction as sr
from jaato_server.shared.ai_tool_runner import ToolExecutor
from jaato_server.shared.framing import read_frame_sync
from jaato_server.server.runner.json_codec import loads as _json_loads
from jaato_server.shared.tests.reversion import Reversion

_TOOL_RUNNER = "jaato-server/jaato_server/shared/ai_tool_runner.py"
_RPC = "jaato-server/jaato_server/server/runner/rpc.py"
_REDACTION = "jaato-server/jaato_server/shared/secret_redaction.py"

KEY = "sk-zai-5f1c9e27b4d8a6031e2f"
MARKER = "‹redacted:JAATO_ZHIPUAI_API_KEY›"

REVERSIONS = [
    Reversion(
        target=_TOOL_RUNNER,
        find="        result = current_redactor().redact(result)\n",
        replace="",
        test="test_a_cli_command_printing_the_key_enters_history_as_the_marker",
        because=(
            "the tool result is what enters history and is replayed on every "
            "later request: without the redactor at that seam the model reads "
            "the key the command printed out of .env"
        ),
    ),
    Reversion(
        target=_RPC,
        find="""        if payload.get("kind") != KIND_REQUEST:
            payload = current_redactor().redact(payload)
""",
        replace="",
        test="test_every_frame_the_daemon_fans_out_carries_the_marker",
        because=(
            "every client, the persisted session record and the daemon-side "
            "traces are fed from the runner's frames; the history the daemon "
            "persists arrives in a response frame"
        ),
    ),
    Reversion(
        target=_RPC,
        find="""        if redactor.active:
            text = self._carry_stream_text(redactor, request_id, source, text, mode)
            if text is None:
                return
""",
        replace="",
        test="test_a_key_the_model_echoes_token_by_token_is_still_caught",
        because=(
            "model output streams token by token, so an echoed key never "
            "appears whole in one frame: only the carry-over sees it"
        ),
    ),
    Reversion(
        target=_RPC,
        find="""        if redactor.active:
            body = self._carry_notification(redactor, request_id, event_type, body)
            if body is None:
                return
""",
        replace="",
        test="test_a_key_split_across_two_tool_output_chunks_is_still_caught",
        because=(
            "a command's output is read in buffers, so a printed key can "
            "straddle two tool_output chunks"
        ),
    ),
    Reversion(
        target=_REDACTION,
        find="""            if len(value) < MIN_REDACT_LENGTH:
                _warn_short_once(name)
                continue
""",
        replace="",
        test="test_a_value_under_the_floor_is_left_alone_and_warned_about_once",
        because=(
            "a secret-named variable holding 'true' or a port number would "
            "otherwise mangle every occurrence of that text in all output"
        ),
    ),
]


@pytest.fixture(autouse=True)
def _clean_redactor():
    sr.reset_redaction_sources()
    sr._warned_short.clear()
    yield
    sr.reset_redaction_sources()


def _configure(env: Dict[str, str], **kw: Any) -> sr.SecretRedactor:
    return sr.configure_redaction_sources(env, **kw)


# ---------------------------------------------------------------- history


def _cli_executor(workspace: Path) -> ToolExecutor:
    from jaato_server.shared.plugins.registry import PluginRegistry

    registry = PluginRegistry()
    registry.discover()
    registry.expose_tool("cli", {"workspace_path": str(workspace)})
    registry.set_workspace_path(str(workspace))
    executor = ToolExecutor()
    for name, fn in registry.get_exposed_executors().items():
        executor.register(name, fn)
    return executor


def test_a_cli_command_printing_the_key_enters_history_as_the_marker(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(f"JAATO_ZHIPUAI_API_KEY={KEY}\nMODEL_NAME=glm\n")
    monkeypatch.chdir(tmp_path)
    _configure({"JAATO_ZHIPUAI_API_KEY": KEY, "MODEL_NAME": "glm-4.6-long-name"},
               workspace_path=str(tmp_path))

    executor = _cli_executor(tmp_path)
    ok, result = executor.execute("cli_based_tool", {"command": "cat .env"})

    assert ok, result
    text = repr(result)
    assert KEY not in text
    assert MARKER in text
    # The redactor replaces the value only: the rest of the output is intact.
    assert "MODEL_NAME=glm" in result["stdout"]


# ---------------------------------------------------------------- the wire


def _runner() -> Tuple[RunnerRPC, socket.socket]:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.settimeout(2)
    return RunnerRPC(a, lambda name, args: (True, None)), b


def _frames(sock: socket.socket) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    sock.settimeout(0.2)
    while True:
        try:
            raw = read_frame_sync(sock)
        except (socket.timeout, OSError, EOFError):
            return out
        if raw is None:
            return out
        out.append(_json_loads(raw))


def test_every_frame_the_daemon_fans_out_carries_the_marker():
    _configure({"JAATO_ZHIPUAI_API_KEY": KEY})
    rpc, peer = _runner()

    # tool_call_start args (the display path), a whole tool_output chunk,
    # the model's own text, and the history the daemon persists.
    rpc.emit_notification(7, rpc._NOTIF_TOOL_CALL_START, {
        "agent_id": "main", "tool_name": "cli_based_tool",
        "tool_args": {"command": f"curl -H 'Authorization: {KEY}' x"},
        "call_id": "c1",
    })
    rpc.emit_notification(7, rpc._NOTIF_TOOL_OUTPUT, {
        "agent_id": "main", "call_id": "c1", "chunk": f"key={KEY}\n",
    })
    rpc._emit_stream(7, "model", f"The key is {KEY}.", "write")
    rpc._emit_response(8, True, {"history": [
        {"role": "tool", "parts": [{"type": "function_response",
                                    "response": {"stdout": f"{KEY}\n"}}]},
    ]})

    wire = repr(_frames(peer))
    assert KEY not in wire
    assert wire.count(MARKER) == 4


def test_a_key_the_model_echoes_token_by_token_is_still_caught():
    _configure({"JAATO_ZHIPUAI_API_KEY": KEY})
    rpc, peer = _runner()

    rpc._emit_stream(3, "model", "Here it is: ", "write")
    for i in range(0, len(KEY), 3):
        rpc._emit_stream(3, "model", KEY[i:i + 3], "append")
    rpc._emit_stream(3, "model", " done", "append")
    rpc._emit_response(3, True, {})

    frames = _frames(peer)
    text = "".join(f.get("text", "") for f in frames if f.get("kind") == "stream")
    assert text == f"Here it is: {MARKER} done"
    # A partial prefix never crossed on its own either.
    assert not any(KEY[:6] in f.get("text", "") for f in frames)


def test_a_key_split_across_two_tool_output_chunks_is_still_caught():
    _configure({"JAATO_ZHIPUAI_API_KEY": KEY})
    rpc, peer = _runner()

    half = len(KEY) // 2
    rpc.emit_notification(5, rpc._NOTIF_TOOL_OUTPUT,
                          {"agent_id": "main", "call_id": "c9", "chunk": "A=" + KEY[:half]})
    rpc.emit_notification(5, rpc._NOTIF_TOOL_OUTPUT,
                          {"agent_id": "main", "call_id": "c9", "chunk": KEY[half:] + "\n"})
    rpc.emit_notification(5, rpc._NOTIF_TOOL_CALL_END,
                          {"agent_id": "main", "call_id": "c9", "tool_name": "x"})

    frames = _frames(peer)
    chunks = "".join(f["payload"].get("chunk", "") for f in frames
                     if f.get("event_type") == "tool_output")
    assert chunks == f"A={MARKER}\n"
    # The tail is released before the row closes, not after.
    assert frames[-1]["event_type"] == "tool_call_end"


def test_held_text_that_was_not_a_key_is_released_before_the_response():
    _configure({"JAATO_ZHIPUAI_API_KEY": KEY})
    rpc, peer = _runner()

    rpc.emit_notification(2, rpc._NOTIF_TOOL_OUTPUT,
                          {"agent_id": "main", "call_id": "c1", "chunk": "prefix " + KEY[:5]})
    rpc._emit_response(2, True, {"ok": 1})

    frames = _frames(peer)
    chunks = "".join(f["payload"].get("chunk", "") for f in frames
                     if f.get("event_type") == "tool_output")
    assert chunks == "prefix " + KEY[:5]
    assert frames[-1]["kind"] == "response"


def test_binary_payloads_are_not_touched():
    red = sr.SecretRedactor([("K", KEY)])
    payload = {"chunk": KEY, "data_b64": KEY, "part": {"mime_type": "a/b", "data": KEY}}
    out = red.redact(payload)
    assert out["chunk"] == "‹redacted:K›"
    assert out["data_b64"] == KEY
    assert out["part"]["data"] == KEY


def test_an_unaffected_structure_is_returned_unchanged():
    red = sr.SecretRedactor([("K", KEY)])
    payload = {"a": ["x", {"b": "y"}]}
    assert red.redact(payload) is payload


# ------------------------------------------------------------ the secret set


def test_a_value_under_the_floor_is_left_alone_and_warned_about_once(caplog):
    with caplog.at_level(logging.WARNING, logger="jaato_server.shared.secret_redaction"):
        red = _configure({"FEATURE_TOKEN": "true", "JAATO_ZHIPUAI_API_KEY": KEY})
        _configure({"FEATURE_TOKEN": "true", "JAATO_ZHIPUAI_API_KEY": KEY})

    assert red.redact_text("enabled=true") == "enabled=true"
    warnings = [r for r in caplog.records if "FEATURE_TOKEN" in r.getMessage()]
    assert len(warnings) == 1
    # Named, never shown.
    assert "true" not in warnings[0].getMessage().replace("FEATURE_TOKEN", "")


def test_a_granted_or_exempted_name_is_still_redacted():
    # ``!GH_TOKEN`` keeps the value in a subprocess's environment; that is
    # not permission to print it.  Same for an app:// grant (#1228), which
    # the redactor never consults.
    token = "ghp_abcdefghijklmnopqrstuvwxyz0123"
    red = _configure({"GH_TOKEN": token},
                     plugin_configs={"cli": {"scrub_secret_env": ["default", "!GH_TOKEN"]}})
    assert red.redact_text(token) == "‹redacted:GH_TOKEN›"


def test_a_profile_glob_widens_the_set():
    value = "internal-service-credential-9"
    assert not _configure({"CORP_SVC_CRED": value}).active
    red = _configure({"CORP_SVC_CRED": value},
                     plugin_configs={"cli": {"scrub_secret_env": ["default", "CORP_*"]}})
    assert red.redact_text(value) == "‹redacted:CORP_SVC_CRED›"


def test_a_stored_auth_file_and_the_live_provider_are_covered(tmp_path):
    jaato = tmp_path / ".jaato"
    jaato.mkdir()
    stored = "stored-zhipuai-key-0123456789"
    (jaato / "zhipuai_auth.json").write_text('{"api_key": "%s", "created": "2026"}' % stored)
    red = _configure({}, workspace_path=str(tmp_path))
    assert red.redact_text(stored) == "‹redacted:zhipuai_auth.json:api_key›"

    class _Provider:
        _api_key = "provider-resolved-key-abcdef"

    sr.note_provider_credential(_Provider(), "zhipuai")
    assert sr.current_redactor().redact_text(_Provider._api_key) == \
        "‹redacted:zhipuai_api_key›"


def test_a_provider_outside_a_runner_adds_nothing():
    class _Provider:
        _api_key = "provider-resolved-key-abcdef"

    sr.note_provider_credential(_Provider(), "zhipuai")
    assert not sr.current_redactor().active


def test_session_reload_env_rebuilds_the_redactor():
    """A key supplied by ``session.reload_env`` is redacted from then on."""
    from types import SimpleNamespace

    _configure({})
    rpc, _peer = _runner()
    session = SimpleNamespace(
        is_running=False,
        reload_provider=lambda: {"provider": "p", "model": "m", "auth_info": ""},
    )
    rpc._require_ready_session = lambda: (True, None, session)  # type: ignore

    ok, _ = rpc._handle_session_reload_env({"session_env": {"JAATO_ZHIPUAI_API_KEY": KEY}})

    assert ok
    assert sr.current_redactor().redact_text(KEY) == MARKER
