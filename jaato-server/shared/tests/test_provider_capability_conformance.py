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
import importlib
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
from shared.tests.test_every_guard_detects_its_own_reversion import (  # noqa: E402
    Reversion,
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
    "mimo":           ("_openai_compat/converters.py",        "message_to_openai"),
    "kimi":           ("_openai_compat/converters.py",        "message_to_openai"),
    "minimax":        ("_openai_compat/converters.py",        "message_to_openai"),
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
_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt CONFORMANCE-AUDIO-PAYLOAD-1234567890"
_WAV_B64 = base64.b64encode(_WAV).decode("utf-8")


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


def _user_audio_msg() -> Message:
    return Message(role=Role.USER, parts=[
        Part(text="what did I say?"),
        Part(inline_data={"mime_type": "audio/wav", "data": _WAV,
                          "display_name": "clip.wav"}),
    ])


def _tool_audio_msg() -> Message:
    return Message(role=Role.TOOL, parts=[Part(function_response=ToolResult(
        call_id="c1", name="readFile",
        result={"path": "clip.wav", "type": "audio"},
        attachments=[Attachment(mime_type="audio/wav", data=_WAV,
                                display_name="clip.wav")],
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


# Providers whose converter puts a PDF on the wire even though the provider
# declares ``pdf_input=False`` — the declaration and the code disagree, which
# is exactly the defect class of #829.
#
# A RATCHET, not a permission list: an entry may only be REMOVED (by fixing
# the provider), never added.  A listed provider that stops violating fails as
# stale, so the set shrinks to empty and cannot quietly grow.  Adding a new
# provider here instead of fixing it is the thing this guard exists to stop.
#
# Both current entries share one cause: ``ollama`` and ``zhipuai`` reuse
# ``anthropic/converters.py``, whose ``_anthropic_media_block`` emits a
# ``document`` block for ``application/pdf`` unconditionally.  That is correct
# for ``anthropic`` (which declares ``pdf_input=True``) and wrong for these
# two, which declare ``False``.  Fixing it means threading wire capability
# through the Anthropic converter the way ``pdf_as_file`` threads it through
# ``model_provider/_attachments`` — out of scope for #829, which fixed the
# OpenAI-shaped family.
_PDF_DECLARATION_VIOLATIONS = {
    "ollama": "shares anthropic/converters.py; emits a `document` block "
              "for PDFs though ollama declares pdf_input=False",
    "zhipuai": "shares anthropic/converters.py; emits a `document` block "
               "for PDFs though zhipuai declares pdf_input=False",
}


def _assert_undeclared_pdf_absent(provider, wire, what):
    """Assert the PDF stayed off the wire — or that a listed violator still violates.

    The two branches keep the ratchet honest in both directions: an unlisted
    provider must not marshal a PDF it never declared, and a listed one must
    still be marshalling it, so a fix surfaces as a stale entry instead of
    lingering as permanent permission.
    """
    on_wire = _PDF_B64 in json.dumps(wire, default=str)
    if provider in _PDF_DECLARATION_VIOLATIONS:
        assert on_wire, (
            f"{provider} is listed in _PDF_DECLARATION_VIOLATIONS but no longer "
            f"puts the {what} PDF on the wire — the entry is stale, remove it."
        )
        return
    assert not on_wire, (
        f"{provider} declares pdf_input=False but its converter put the {what} "
        f"PDF on the wire anyway — the declaration and the code disagree "
        f"(#829). Withhold it, or set pdf_input=True if the wire really "
        f"carries it."
    )


@pytest.mark.parametrize("provider", sorted(_CONVERTERS))
def test_undeclared_pdf_input_user_message_is_not_marshalled(provider):
    """The negative half of the pdf_input contract.

    #829 was nine providers declaring ``pdf_input=False`` whose converter sent
    PDFs regardless — mislabelled as ``image_url``, at that.  The positive test
    above could not see it: it skips every provider declaring ``False``, which
    was all of them.  A capability registry that only checks one direction
    cannot catch a converter doing MORE than it declared.
    """
    if _read_declaration(provider).get("pdf_input"):
        pytest.skip(f"{provider} declares pdf_input — covered by the positive test")
    relpath, fn = _CONVERTERS[provider]
    convert = _load_converter(relpath, fn)
    _assert_undeclared_pdf_absent(provider, convert(_user_pdf_msg()), "user-message")


@pytest.mark.parametrize("provider", sorted(_CONVERTERS))
def test_undeclared_pdf_input_tool_result_is_not_marshalled(provider):
    """Same contract on the tool-result path, which has its own marshalling."""
    if _read_declaration(provider).get("pdf_input"):
        pytest.skip(f"{provider} declares pdf_input — covered by the positive test")
    relpath, fn = _CONVERTERS[provider]
    convert = _load_converter(relpath, fn)
    _assert_undeclared_pdf_absent(provider, convert(_tool_pdf_msg()), "tool-result")


def test_pdf_violation_ratchet_names_only_non_declaring_providers():
    """A provider that DECLARES pdf_input has nothing to be excused from.

    Without this, parking a ``pdf_input=True`` provider in the ratchet would
    silently disable the positive test's counterpart for it.
    """
    for provider in _PDF_DECLARATION_VIOLATIONS:
        assert provider in _CONVERTERS, (
            f"{provider} is in _PDF_DECLARATION_VIOLATIONS but not in the "
            f"converter registry — remove the stale entry."
        )
        assert not _read_declaration(provider).get("pdf_input"), (
            f"{provider} declares pdf_input=True, so it is not violating "
            f"anything — remove it from _PDF_DECLARATION_VIOLATIONS."
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


# ------------------------------------------------------------- audio_input
#
# #830: the framework could speak (#824/#828 deliver model-emitted audio to a
# client) and could not be spoken to — ``input_audio``, the content-block form
# for audio INPUT, appeared nowhere in the tree.  Both directions of the
# contract are guarded from the start, because the pdf_input experience was
# that a one-directional guard cannot see a converter doing MORE than it
# declared, which is the defect that actually shipped (#829).


@pytest.mark.parametrize("provider", sorted(_CONVERTERS))
def test_declared_audio_input_user_message_is_marshalled(provider):
    if not _read_declaration(provider).get("audio_input"):
        pytest.skip(f"{provider} does not declare audio_input")
    relpath, fn = _CONVERTERS[provider]
    convert = _load_converter(relpath, fn)
    wire = convert(_user_audio_msg())
    assert _WAV_B64 in json.dumps(wire, default=str), (
        f"{provider} declares audio_input=True but {fn} did NOT put the audio "
        f"on the wire. Fix the converter or set audio_input=False."
    )


@pytest.mark.parametrize("provider", sorted(_CONVERTERS))
def test_declared_audio_input_tool_result_is_marshalled(provider):
    if not _read_declaration(provider).get("audio_input"):
        pytest.skip(f"{provider} does not declare audio_input")
    relpath, fn = _CONVERTERS[provider]
    convert = _load_converter(relpath, fn)
    wire = convert(_tool_audio_msg())
    assert _WAV_B64 in json.dumps(wire, default=str), (
        f"{provider} declares audio_input=True but {fn} did NOT surface the "
        f"tool-result audio to the model. Fix the converter or set "
        f"audio_input=False."
    )


@pytest.mark.parametrize("provider", sorted(_CONVERTERS))
@pytest.mark.parametrize("build,what", [
    (_user_audio_msg, "user-message"),
    (_tool_audio_msg, "tool-result"),
])
def test_undeclared_audio_input_is_not_marshalled(provider, build, what):
    """The negative half: a wire that never declared audio must not send it.

    There is no ratchet here and there must never need to be one — the audio
    path was built with the declaration and the converter agreeing, rather
    than reconciled afterwards the way ``pdf_input`` had to be.
    """
    if _read_declaration(provider).get("audio_input"):
        pytest.skip(f"{provider} declares audio_input — covered by the positive test")
    relpath, fn = _CONVERTERS[provider]
    convert = _load_converter(relpath, fn)
    assert _WAV_B64 not in json.dumps(convert(build()), default=str), (
        f"{provider} declares audio_input=False but its converter put the "
        f"{what} audio on the wire anyway — the declaration and the code "
        f"disagree. Withhold it, or set audio_input=True if the wire really "
        f"carries it."
    )


def _audio_mislabelled_as_image(node) -> bool:
    """Is audio riding inside an IMAGE block anywhere in ``node``?

    Walks the converted wire structure rather than one known shape, because
    the two block families that could hide it look nothing alike: OpenAI's
    ``image_url`` carries the mime inside a data URL, Anthropic's ``image``
    carries it as ``source.media_type``.  Both are checked; a converter that
    grows a third shape is caught the same way.
    """
    if isinstance(node, dict):
        if node.get("type") == "image_url":
            url = (node.get("image_url") or {}).get("url", "")
            if isinstance(url, str) and url.startswith("data:audio/"):
                return True
        if node.get("type") == "image":
            media = (node.get("source") or {}).get("media_type", "")
            if isinstance(media, str) and media.startswith("audio/"):
                return True
        return any(_audio_mislabelled_as_image(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return any(_audio_mislabelled_as_image(v) for v in node)
    return False


@pytest.mark.parametrize("provider", sorted(_CONVERTERS))
def test_audio_never_travels_inside_an_image_block(provider):
    """#829's invariant, re-asserted for the mime family #830 added.

    Carrying audio and mislabelling audio are different things, and the
    second is the one that shipped last time: an ``image_url`` whose data URL
    said ``audio/wav``.  A provider that now genuinely carries audio must do
    it in an audio-shaped block, not by widening the image branch.
    """
    relpath, fn = _CONVERTERS[provider]
    convert = _load_converter(relpath, fn)
    for build in (_user_audio_msg, _tool_audio_msg):
        assert not _audio_mislabelled_as_image(convert(build())), (
            f"{provider}: audio reached the wire inside an image block — "
            f"that is #829, not audio support."
        )


OPENROUTER_FIXED = (
    "    return user_message_with_attachments(\n"
    "        content, message.parts, pdf_as_file=True, audio_as_input_audio=True\n"
    "    )"
)
OPENROUTER_BROKEN = (
    "    return user_message_with_attachments(\n"
    "        content, message.parts, pdf_as_file=True\n"
    "    )"
)
DISPATCH_FIXED = (
    '    mime = mime or ""\n'
    '    if mime.startswith("image/"):'
)
DISPATCH_BROKEN = (
    '    mime = mime or ""\n'
    '    if mime.startswith("image/") or mime.startswith("audio/"):'
)


#: The defect, put back.
#
# The audio guards are the ones declared here because they are the ones
# written from scratch (#830); the image / PDF guards predate the
# meta-guard and their reversions belong with whoever revisits them.  Both
# halves of the audio contract are covered, because they fail differently:
# the first is the wire staying silent, the second is the wire lying.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/plugins/model_provider/openrouter/converters.py",
        find=OPENROUTER_FIXED,
        replace=OPENROUTER_BROKEN,
        test="test_declared_audio_input_user_message_is_marshalled",
        because="a provider declaring audio_input=True whose converter "
                "withholds the audio anyway -- the model is told it can "
                "listen and then hears nothing",
    ),
    Reversion(
        target="jaato-server/shared/plugins/model_provider/_attachments.py",
        find=DISPATCH_FIXED,
        replace=DISPATCH_BROKEN,
        test="test_audio_never_travels_inside_an_image_block",
        because="audio reaching the wire inside an image_url block, which "
                "is #829 wearing #830's clothes: carried, but mislabelled",
    ),
]


# ---------------------------------------------------------------- reasoning replay

_REASONING = "CONFORMANCE-REASONING-PAYLOAD-1234567890"


def _reasoned_tool_turn() -> Message:
    """An assistant turn that thought, then called a tool — the shape every
    replaying vendor's multi-turn rule is about."""
    from jaato_sdk.plugins.model_provider.types import FunctionCall
    return Message(role=Role.MODEL, parts=[
        Part(thought=_REASONING),
        Part(function_call=FunctionCall(id="c1", name="readFile",
                                        args={"path": "x"})),
    ])


def _load_provider_instance(provider: str):
    """``create_provider()`` for ``provider``, or skip when its SDK is absent
    in this environment (the contract-guards job installs every extra)."""
    try:
        mod = importlib.import_module(f"shared.plugins.model_provider.{provider}")
    except ImportError as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"{provider} SDK not installed here: {exc}")
    factory = getattr(mod, "create_provider", None)
    if factory is None:
        pytest.skip(f"{provider} has no create_provider()")
    return factory()


@pytest.mark.parametrize("provider", sorted(_provider_dirs()))
def test_declared_reasoning_replay_puts_reasoning_on_the_wire(provider):
    """A provider declaring ``reasoning_replay`` must (1) opt in through
    ``replay_reasoning`` — the attribute the session gates history on —
    and (2) map a thought part to wire fields carrying the reasoning text,
    through the converter it actually uses at runtime."""
    if not _read_declaration(provider).get("reasoning_replay"):
        pytest.skip(f"{provider} does not declare reasoning_replay")
    inst = _load_provider_instance(provider)
    assert getattr(inst, "replay_reasoning", False) is True, (
        f"{provider} declares reasoning_replay=True but its provider class "
        "does not set replay_reasoning = True, so the session would drop "
        "the thought part before it could ever be replayed."
    )
    relpath, fn = _CONVERTERS[provider]
    history_to_wire = _load_converter(relpath, "history_to_openai")
    wire = history_to_wire([_reasoned_tool_turn()],
                           reasoning_fields=inst._reasoning_replay_fields)
    assert _REASONING in json.dumps(wire, default=str), (
        f"{provider} declares reasoning_replay=True but {fn} did NOT put the "
        "thought part's text on the assistant message."
    )
    assistant = [m for m in wire if m.get("role") == "assistant"]
    assert assistant and assistant[0].get("content") == "", (
        f"{provider}: a replayed assistant turn with tool calls and no text "
        "must carry content \"\" (MiMo rejects null next to tool_calls)."
    )


@pytest.mark.parametrize("provider", sorted(_provider_dirs()))
def test_undeclared_reasoning_replay_is_not_opted_in(provider):
    """The declaration and the opt-in attribute must agree in the other
    direction too: a provider that replays without declaring it would be
    a wire behaviour the capability doc denies."""
    if _read_declaration(provider).get("reasoning_replay"):
        pytest.skip(f"{provider} declares reasoning_replay")
    inst = _load_provider_instance(provider)
    assert getattr(inst, "replay_reasoning", False) is not True, (
        f"{provider} sets replay_reasoning = True but declares "
        "reasoning_replay=False."
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
