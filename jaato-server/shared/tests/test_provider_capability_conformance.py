"""Behavioral conformance half of the provider capability contract.

The structural guard (``test_provider_capabilities``) only checks that each
provider DECLARES its capabilities. This guard checks they're TRUE: a provider
that declares an image capability must actually put the image on the wire. It
is the guard that would have caught the multimodal rot — every broken provider
happily declared ``image: yes`` while its converter silently dropped the bytes.

Dep-safe: each provider's message converter uses only absolute
``jaato_sdk``/``shared`` imports (no vendor SDK), so we ``importlib``-load the
converter FILE directly — bypassing the package ``__init__`` that would pull
``anthropic``/``openai``/``google`` — and run a real Message→wire conversion.
The image is detected universally by scanning the serialized wire output for
the image's base64 (works for OpenAI ``image_url`` data-URLs and Anthropic
``source.data`` blocks alike).

Providers whose converter emits SDK OBJECTS rather than plain dicts
(google_genai, antigravity via the google SDK; github_models via Azure;
claude_cli has no image transport) can't be scanned this way; they are listed
in ``_CONFORMANCE_PENDING`` and covered structurally only until a full-suite
behavioral test is added. A test pins that list so a NEW provider can't slip
into "unverified" silently.
"""

import base64
import importlib.util
import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pytest

from jaato_sdk.plugins.model_provider.types import (
    Attachment,
    Message,
    Part,
    Role,
    ToolResult,
)
from shared.plugins.model_provider.base import CAPABILITY_FIELDS  # noqa: F401

PROVIDER_DIR = Path(__file__).resolve().parents[1] / "plugins" / "model_provider"

# Reuse the structural guard's AST reader so the two halves share one source.
from shared.tests.test_provider_capabilities import (  # noqa: E402
    _provider_dirs,
    _read_declaration,
)

# provider -> (converter file relative to model_provider/, message-conversion fn).
# Providers without their own converters.py inherit another's (the value points
# at the file they actually use at runtime).
_CONVERTERS: Dict[str, Tuple[str, str]] = {
    "nim":            ("_openai_compat/converters.py",        "message_to_openai"),
    "nebius":         ("nebius/converters.py",     "message_to_openai"),
    "openrouter":     ("openrouter/converters.py", "message_to_openai"),
    "vllm":           ("_openai_compat/converters.py",        "message_to_openai"),
    "lmstudio":       ("_openai_compat/converters.py",        "message_to_openai"),
    "tensorrt_llm":   ("_openai_compat/converters.py",        "message_to_openai"),
    "zhipuai_openai": ("_openai_compat/converters.py",        "message_to_openai"),
    "triton":         ("_openai_compat/converters.py",        "message_to_openai"),
    "ovhcloud":       ("_openai_compat/converters.py",        "message_to_openai"),
    "doubleword":     ("_openai_compat/converters.py",        "message_to_openai"),
    "helmcode":       ("_openai_compat/converters.py",        "message_to_openai"),
    "anthropic":      ("anthropic/converters.py",  "message_to_anthropic"),
    "chrome_ai":      ("chrome_ai/converters.py",  "message_to_prompt_api"),
    "ollama":         ("anthropic/converters.py",  "message_to_anthropic"),
    "zhipuai":        ("anthropic/converters.py",  "message_to_anthropic"),
}

# SDK-object converters (not plain-dict scannable) — behavioral conformance is a
# full-suite follow-up; pinned so a new provider can't join silently.
_CONFORMANCE_PENDING = {"google_genai", "antigravity", "github_models", "claude_cli"}

_PNG = b"\x89PNG\r\n\x1a\nCONFORMANCE-IMAGE-PAYLOAD-1234567890"
_B64 = base64.b64encode(_PNG).decode("utf-8")
_PDF = b"%PDF-1.4 CONFORMANCE-PDF-PAYLOAD-1234567890 %%EOF"
_PDF_B64 = base64.b64encode(_PDF).decode("utf-8")


def _load_converter(relpath: str, fn: str) -> Callable:
    path = PROVIDER_DIR / relpath
    spec = importlib.util.spec_from_file_location(f"_conv_{relpath}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn)


def _wire_has_image(wire) -> bool:
    """Did the image base64 reach the wire (any role/block shape)?"""
    return _B64 in json.dumps(wire, default=str)


def _user_image_msg() -> Message:
    return Message(role=Role.USER, parts=[
        Part(text="what is in this image?"),
        Part(inline_data={"mime_type": "image/png", "data": _PNG}),
    ])


def _tool_image_msg() -> Message:
    return Message(role=Role.TOOL, parts=[Part(function_response=ToolResult(
        call_id="c1", name="readFile",
        result={"path": "x.png", "type": "image"},
        attachments=[Attachment(mime_type="image/png", data=_PNG,
                                display_name="x.png")],
    ))])


def _user_pdf_msg() -> Message:
    return Message(role=Role.USER, parts=[
        Part(text="summarize this document"),
        Part(inline_data={"mime_type": "application/pdf", "data": _PDF,
                          "display_name": "doc.pdf"}),
    ])


def _tool_pdf_msg() -> Message:
    return Message(role=Role.TOOL, parts=[Part(function_response=ToolResult(
        call_id="c1", name="readFile",
        result={"path": "doc.pdf", "type": "file"},
        attachments=[Attachment(mime_type="application/pdf", data=_PDF,
                                display_name="doc.pdf")],
    ))])


@pytest.mark.parametrize("provider", sorted(_CONVERTERS))
def test_declared_pdf_input_user_message_is_marshalled(provider):
    if not _read_declaration(provider).get("pdf_input"):
        pytest.skip(f"{provider} does not declare pdf_input")
    relpath, fn = _CONVERTERS[provider]
    convert = _load_converter(relpath, fn)
    wire = convert(_user_pdf_msg())
    assert _PDF_B64 in json.dumps(wire, default=str), (
        f"{provider} declares pdf_input=True but {fn} did NOT put the PDF on the "
        f"wire. Fix the converter or set pdf_input=False."
    )


@pytest.mark.parametrize("provider", sorted(_CONVERTERS))
def test_declared_pdf_input_tool_result_is_marshalled(provider):
    if not _read_declaration(provider).get("pdf_input"):
        pytest.skip(f"{provider} does not declare pdf_input")
    relpath, fn = _CONVERTERS[provider]
    convert = _load_converter(relpath, fn)
    wire = convert(_tool_pdf_msg())
    assert _PDF_B64 in json.dumps(wire, default=str), (
        f"{provider} declares pdf_input=True but {fn} did NOT surface the "
        f"tool-result PDF to the model. Fix the converter or set pdf_input=False."
    )


@pytest.mark.parametrize("provider", sorted(_CONVERTERS))
def test_declared_user_image_is_marshalled(provider):
    if not _read_declaration(provider).get("user_message_images"):
        pytest.skip(f"{provider} does not declare user_message_images")
    relpath, fn = _CONVERTERS[provider]
    convert = _load_converter(relpath, fn)
    wire = convert(_user_image_msg())
    assert _wire_has_image(wire), (
        f"{provider} declares user_message_images=True but {fn} did NOT put the "
        f"image on the wire (the multimodal-rot bug). Either fix the converter "
        f"or set user_message_images=False."
    )


@pytest.mark.parametrize("provider", sorted(_CONVERTERS))
def test_declared_tool_result_image_is_marshalled(provider):
    if not _read_declaration(provider).get("tool_result_images"):
        pytest.skip(f"{provider} does not declare tool_result_images")
    relpath, fn = _CONVERTERS[provider]
    convert = _load_converter(relpath, fn)
    wire = convert(_tool_image_msg())
    assert _wire_has_image(wire), (
        f"{provider} declares tool_result_images=True but {fn} did NOT surface "
        f"the tool-result image to the model. Either fix the converter or set "
        f"tool_result_images=False."
    )


def test_every_provider_is_either_conformance_tested_or_explicitly_pending():
    """No provider may silently escape behavioral conformance — it's either in
    the testable converter registry or the explicitly-pinned pending set."""
    covered = set(_CONVERTERS) | _CONFORMANCE_PENDING
    uncovered = [p for p in _provider_dirs() if p not in covered]
    assert uncovered == [], (
        f"providers neither behaviorally tested nor in _CONFORMANCE_PENDING: "
        f"{uncovered}. Add a converter entry or pin it as pending (with reason)."
    )
