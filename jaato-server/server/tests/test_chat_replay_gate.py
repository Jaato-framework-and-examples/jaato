"""Chat-client replay-on-attach gate: emit_current_state skips the conversation
replay for chat-type clients (persistent history) but replays for terminal/web
(ephemeral redraw) and when the presentation context is unknown."""
from types import SimpleNamespace
from unittest.mock import MagicMock
from server.core import JaatoServer
from jaato_sdk.events import ClientType


def _bare_server(client_type):
    s = JaatoServer.__new__(JaatoServer)        # skip heavy __init__
    s._agents = {}
    s._runner_rpc = None
    s._on_event = lambda e: None
    s._model_provider = "p"
    s._model_name = "m"
    s._session_id = "sid"
    s._presentation_context = (
        None if client_type is None else SimpleNamespace(client_type=client_type)
    )
    s._emit_conversation_replay = MagicMock()
    return s


def _run_gate(server):
    # Post-gate code (registry / budget) needs a fuller server; the replay gate
    # runs first, so its decision is recorded before any downstream error.
    try:
        server.emit_current_state(skip_session_info=True)
    except Exception:
        pass


def test_chat_client_skips_replay():
    s = _bare_server(ClientType.CHAT)
    _run_gate(s)
    s._emit_conversation_replay.assert_not_called()


def test_terminal_client_replays():
    s = _bare_server(ClientType.TERMINAL)
    _run_gate(s)
    s._emit_conversation_replay.assert_called_once()


def test_web_client_replays():
    s = _bare_server(ClientType.WEB)
    _run_gate(s)
    s._emit_conversation_replay.assert_called_once()


def test_unknown_presentation_defaults_to_replay():
    s = _bare_server(None)
    _run_gate(s)
    s._emit_conversation_replay.assert_called_once()
