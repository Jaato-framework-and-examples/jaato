"""A prompt keeps every ``@`` no enricher resolved (#1429).

``JaatoSession._enrich_and_clean_prompt`` used to end every prompt with a
blanket ``AT_REFERENCE_PATTERN.sub(r'\\1', ...)``.  Written for the
``@photo.png`` / ``@ref-id`` mentions the multimodal and references plugins
handle, it removed the ``@`` from EVERY word: ``@jaato/sdk`` reached the
model as ``jaato/sdk``, ``dani@example.com`` as ``daniexample.com``,
``@dataclass`` as ``dataclass``.  A completion-gated release judge ran out
of budget because its processor required a package name the model had never
been shown.

Now each prompt enricher reports the mentions it resolved
(``RESOLVED_MENTIONS_METADATA_KEY``) and the session removes the ``@`` from
exactly those.  Driven through ``send_message`` with the REAL multimodal and
references plugins on a real ``PluginRegistry``, asserting on the user
message the session stored in its history.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    Message,
    Part,
    ProviderResponse,
    Role,
    TokenUsage,
    TurnResult,
)

from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.plugins.multimodal.plugin import MultimodalPlugin
from jaato_server.shared.plugins.references.models import (
    InjectionMode,
    ReferenceSource,
    SourceType,
)
from jaato_server.shared.plugins.references.plugin import ReferencesPlugin
from jaato_server.shared.plugins.registry import PluginRegistry
from jaato_server.shared.prompt_mentions import strip_resolved_mentions
from jaato_server.shared.tests.reversion import Reversion


_SESSION = "jaato-server/jaato_server/shared/jaato_session.py"
_MULTIMODAL = "jaato-server/jaato_server/shared/plugins/multimodal/plugin.py"
_REFERENCES = "jaato-server/jaato_server/shared/plugins/references/plugin.py"

REVERSIONS = [
    Reversion(
        target=_SESSION,
        find="        return strip_resolved_mentions(enriched_prompt, resolved)\n",
        replace=(
            "        return __import__('re').sub(r'@([\\w./\\-]+(?:\\.\\w+)?)', "
            "r'\\1', enriched_prompt)\n"
        ),
        because="the blanket strip back: every @word loses its @ again",
        test="test_unhandled_at_signs_reach_the_history_byte_identical",
    ),
    Reversion(
        target=_MULTIMODAL,
        find="                RESOLVED_MENTIONS_METADATA_KEY: list(detected_images),\n",
        replace="",
        because="the multimodal plugin stops reporting the image mention it resolved",
        test="test_a_resolved_image_mention_still_loses_its_at",
    ),
    Reversion(
        target=_REFERENCES,
        find="            all_metadata[RESOLVED_MENTIONS_METADATA_KEY] = sorted(set(mentioned_ids))\n",
        replace="",
        because="the references plugin stops reporting the @ref-id mentions it resolved",
        test="test_a_resolved_reference_mention_still_loses_its_at",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/prompt_mentions.py",
        find=r"            r'(?<![\w@])@(' + re.escape(token) + r')(?![\w\-]|[./][\w\-])'",
        replace=r"            r'@(' + re.escape(token) + r')'",
        because="a resolved token would also strip the @ inside an email or a longer mention",
        test="test_a_resolved_token_does_not_reach_into_other_words",
    ),
]


UNHANDLED = (
    "Bump @jaato/sdk (npm); dependents: @jaato/web-coder-server (^0.19.0).\n"
    "Mail dani@example.com about it.\n"
    "```python\n@dataclass\nclass X:\n    pass\n```\n"
    "Also @nosuch.png, which does not exist."
)


def _provider():
    p = MagicMock()
    p.name = "fake"
    p.model_name = "m"
    p.supports_streaming.return_value = True
    p.get_retry_after.return_value = None
    p.get_context_limit.return_value = 100_000

    def complete(messages, **kwargs):
        return TurnResult.from_provider_response(ProviderResponse(
            parts=[Part(text="ok")],
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=1, output_tokens=1, total_tokens=2),
        ))
    p.complete.side_effect = complete
    return p


@pytest.fixture
def session(tmp_path: Path):
    (tmp_path / "photo.png").write_bytes(b"\x89PNG")
    multimodal = MultimodalPlugin()
    multimodal.initialize({"base_path": str(tmp_path)})
    references = ReferencesPlugin()
    references._sources = [ReferenceSource(
        id="ref-id", name="Ref", description="a reference",
        type=SourceType.INLINE, mode=InjectionMode.SELECTABLE,
        content="reference body",
    )]
    registry = PluginRegistry()
    registry.register_plugin(multimodal, enrichment_only=True)
    registry.register_plugin(references, enrichment_only=True)

    provider = _provider()
    runtime = MagicMock()
    runtime.provider_name = "fake"
    runtime.create_provider.return_value = provider
    runtime.get_tool_schemas.return_value = []
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.permission_plugin = None
    runtime.ledger = None
    runtime.reliability_plugin = None
    runtime.registry = registry

    s = JaatoSession(runtime, "m")
    s.configure()
    return s


def _stored_user_text(session: JaatoSession) -> str:
    users = [m for m in session.get_history() if m.role == Role.USER]
    assert users, "no user message was stored"
    return "".join(p.text or "" for p in (users[0].parts or []))


def test_unhandled_at_signs_reach_the_history_byte_identical(session):
    session.send_message(UNHANDLED, lambda *a, **k: None)
    stored = _stored_user_text(session)
    assert stored.startswith(UNHANDLED), (
        "an @ no enricher resolved was changed before the model saw it.\n"
        f"sent:   {UNHANDLED!r}\nstored: {stored[:len(UNHANDLED) + 20]!r}"
    )


def test_a_resolved_image_mention_still_loses_its_at(session):
    session.send_message("What is in @photo.png? See @jaato/sdk.", lambda *a, **k: None)
    stored = _stored_user_text(session)
    assert stored.startswith("What is in photo.png? See @jaato/sdk."), stored


def test_a_resolved_reference_mention_still_loses_its_at(session):
    session.send_message("Follow @ref-id for @dataclass usage.", lambda *a, **k: None)
    stored = _stored_user_text(session)
    assert stored.startswith("Follow ref-id for @dataclass usage."), stored
    assert "reference body" in stored, "the references plugin did not run"


def test_a_resolved_token_does_not_reach_into_other_words():
    text = "@ref and a@ref and @ref-id and @ref/x and @ref."
    assert strip_resolved_mentions(text, ["ref"]) == (
        "ref and a@ref and @ref-id and @ref/x and ref."
    )
