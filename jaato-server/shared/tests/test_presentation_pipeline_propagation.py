"""Tests for PresentationContext propagation to formatter pipelines.

Regression coverage for the bug where ``client_type`` from
``PresentationContext`` was not propagated to lazily-created per-agent
formatter pipelines — the main pipeline and any already-existing agent
pipelines got the setting, but subagent pipelines created on first
output after the client connected silently kept their default, causing
``table_formatter`` to emit box-drawing for web clients.

The fix centralises the propagation in
``JaatoServer._apply_presentation_to_pipeline`` and calls it from both
``set_presentation_context`` (for existing pipelines) and
``_get_agent_pipeline`` (for lazily-created ones).
"""

from unittest.mock import MagicMock

from jaato_sdk.events import PresentationContext, ClientType
from server.core import JaatoServer, AgentState


def _make_server() -> JaatoServer:
    """Build a minimal server suitable for exercising the propagation helper."""
    server = JaatoServer.__new__(JaatoServer)
    server._presentation_context = None
    server._formatter_pipeline = None
    server._agents = {}
    server._workspace_path = None
    server._terminal_width = 80
    server._disable_formatter_truncation = False
    # _trace is called by _get_agent_pipeline; give it a no-op.
    server._trace = lambda *_args, **_kwargs: None
    server.registry = None
    return server


class TestApplyPresentationToPipeline:

    def test_noop_when_no_context(self):
        server = _make_server()
        pipeline = MagicMock()
        server._apply_presentation_to_pipeline(pipeline)
        pipeline.set_client_type.assert_not_called()

    def test_noop_when_no_pipeline(self):
        server = _make_server()
        server._presentation_context = PresentationContext(client_type=ClientType.WEB)
        # Should not raise
        server._apply_presentation_to_pipeline(None)

    def test_propagates_client_type_to_pipeline(self):
        server = _make_server()
        server._presentation_context = PresentationContext(client_type=ClientType.WEB)
        pipeline = MagicMock()
        server._apply_presentation_to_pipeline(pipeline)
        pipeline.set_client_type.assert_called_once()
        # Formatters compare against literal "web" / "terminal" / etc., so
        # the contract IS the exact enum value string — not the enum's repr.
        call_arg = pipeline.set_client_type.call_args[0][0]
        assert call_arg == "web", (
            f"expected exact value 'web', got {call_arg!r} — "
            f"likely str(enum) sneaking in instead of enum.value"
        )

    def test_terminal_client_routed_to_terminal_string(self):
        """Regression test: TUI must receive 'terminal', not 'ClientType.TERMINAL'.

        The code_block_formatter and table_formatter check
        ``self._client_type != "terminal"`` to decide between Rich/ANSI
        rendering and semantic ``<j-code>`` / ``<j-table>`` markup.  If the
        propagation passes ``str(ClientType.TERMINAL)`` (which produces
        ``"ClientType.TERMINAL"`` because ClientType is a (str, Enum)
        mixin), the comparison falsely returns True and the TUI gets
        unrendered semantic markup instead of syntax-highlighted code.
        """
        server = _make_server()
        server._presentation_context = PresentationContext(
            client_type=ClientType.TERMINAL
        )
        pipeline = MagicMock()
        server._apply_presentation_to_pipeline(pipeline)
        call_arg = pipeline.set_client_type.call_args[0][0]
        assert call_arg == "terminal", (
            f"TUI got {call_arg!r} — formatters will treat this as "
            f"non-terminal and emit <j-code> markup instead of ANSI"
        )

    def test_skips_pipeline_without_set_client_type(self):
        """Some pipeline implementations may not expose set_client_type."""
        server = _make_server()
        server._presentation_context = PresentationContext(client_type=ClientType.WEB)

        class PipelineWithoutClientType:
            pass

        pipeline = PipelineWithoutClientType()
        # Should not raise
        server._apply_presentation_to_pipeline(pipeline)


class TestLazyAgentPipelineInheritsClientType:
    """The exact scenario from the bug report: presentation set before
    the agent pipeline is lazily created."""

    def test_lazy_pipeline_receives_client_type(self, monkeypatch):
        server = _make_server()
        # Presentation is set before the agent pipeline exists.
        server._presentation_context = PresentationContext(client_type=ClientType.WEB)
        # A server-level pipeline exists (guard in _get_agent_pipeline
        # requires it before lazy creation happens).
        server._formatter_pipeline = MagicMock()
        server._agents["subagent_1"] = AgentState(
            agent_id="subagent_1",
            name="sub",
            agent_type="subagent",
        )

        created_pipeline = MagicMock()

        # Stub the formatter_registry factory path: replace create_registry
        # so we don't need real plugin discovery in a unit test.
        def fake_create_registry():
            reg = MagicMock()
            reg.create_pipeline.return_value = created_pipeline
            reg.load_config.return_value = False
            return reg

        import shared.plugins.formatter_pipeline as fp_mod
        monkeypatch.setattr(fp_mod, "create_registry", fake_create_registry)

        result = server._get_agent_pipeline("subagent_1")

        assert result is created_pipeline
        # The lazily-created pipeline must have received client_type as
        # the exact enum value string the formatters compare against.
        created_pipeline.set_client_type.assert_called_once()
        call_arg = created_pipeline.set_client_type.call_args[0][0]
        assert call_arg == "web"
