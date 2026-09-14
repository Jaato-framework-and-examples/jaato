"""Chat-Completions converters for the native OpenAI provider.

This module exists to state ONE thing the shared converter cannot state
for everybody: OpenAI's own endpoint carries more than images.

``_openai_compat/converters.py`` is shared by nine gateways that all
declare ``pdf_input=False`` and ``audio_input=False``, so its wire policy
defaults to images-only.  OpenAI's Chat Completions API carries, in
addition:

- **PDFs**, as a ``file`` content block (``{"filename", "file_data"}``
  with a ``data:application/pdf;base64,`` URL) — the same shape
  ``_attachments.attachment_content_block`` already emits for OpenRouter;
- **audio input**, as an ``input_audio`` block (``{"data", "format"}``)
  for the audio-capable models.

So the functions here are the shared ones with the wire policy pinned on.
They are re-exported rather than reimplemented: a second copy of
``message_to_openai`` is precisely the drift that produced #829, where
two converters for one wire family held opposite policies.

The module is importable **as a file**, with no package context and no
vendor SDK, because that is how ``test_provider_capability_conformance``
and ``test_tool_id_wire_conformance`` load a provider's converter to run
a real conversion against it — hence the absolute ``shared.`` imports.
"""

from __future__ import annotations

from typing import Any, Dict, List

from jaato_sdk.plugins.model_provider.types import Message

from shared.plugins.model_provider._openai_compat.converters import (  # noqa: F401
    deserialize_history,
    deserialize_message,
    extract_finish_reason,
    extract_parts_from_response,
    extract_usage,
    get_original_tool_name,
    map_finish_reason,
    message_from_openai,
    response_from_openai,
    sanitize_tool_name,
    serialize_history,
    serialize_message,
    tool_schema_to_openai,
    tool_schemas_to_openai,
)
from shared.plugins.model_provider._openai_compat.converters import (
    history_to_openai as _shared_history_to_openai,
    message_to_openai as _shared_message_to_openai,
)

#: The wire policy of ``api.openai.com``.  Kept beside the two wrappers
#: so the pair cannot disagree, and named so the provider's
#: ``WIRE_PDF_AS_FILE`` / ``WIRE_AUDIO_AS_INPUT_AUDIO`` class attributes
#: and ``PROVIDER_CAPABILITIES`` can be read against one source.
PDF_AS_FILE = True
AUDIO_AS_INPUT_AUDIO = True


def message_to_openai(message: Message) -> List[Dict[str, Any]]:
    """Convert one internal ``Message`` to OpenAI chat message dict(s).

    The shared converter with OpenAI's own wire policy applied: PDFs ride
    as ``file`` blocks and audio as ``input_audio`` blocks, instead of
    being withheld as they are on the gateways that do not carry them.

    Args:
        message: Internal message.

    Returns:
        List of dicts in OpenAI chat message format (1 per tool result).
    """
    return _shared_message_to_openai(
        message,
        pdf_as_file=PDF_AS_FILE,
        audio_as_input_audio=AUDIO_AS_INPUT_AUDIO,
    )


def history_to_openai(history: List[Message]) -> List[Dict[str, Any]]:
    """Convert internal history to an OpenAI chat message list.

    Args:
        history: List of internal messages.

    Returns:
        List of OpenAI message dicts.
    """
    return _shared_history_to_openai(
        history,
        pdf_as_file=PDF_AS_FILE,
        audio_as_input_audio=AUDIO_AS_INPUT_AUDIO,
    )
