"""Pure tool-result building helpers extracted from ``JaatoSession``.

These functions cover the side-effect-free shape-normalization steps that
turn raw executor output into the dict/string a provider converter will
send to the model:

- :func:`split_executor_result` — unwrap the ``(ok, data)`` tuple form.
- :func:`extract_multimodal_attachments` — pull image attachments out of a
  ``_multimodal`` result dict.
- :func:`normalize_result_dict` — wrap non-dicts, surface the permission
  advisory note, strip internal ``_``-prefixed scaffolding keys, and
  collapse single-key error dicts to a bare string.
- :func:`tool_result_text_view` / :func:`apply_text_view_enrichment` — the
  two halves of handing a **dict** result to the tool-result enrichment
  chain, which speaks strings: render the dict's scalar content as text,
  then write back what enrichment returned (#922).

They read no session state, so they are unit-testable in isolation.
``JaatoSession._build_tool_result`` keeps the stateful orchestration —
the registry-driven tool-result enrichment, reference pinning, and
telemetry — and calls these helpers for the pure transforms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from jaato_sdk.plugins.model_provider.types import Attachment


def split_executor_result(executor_result: Any) -> Tuple[bool, Any]:
    """Split executor output into an ``(ok, result_data)`` pair.

    Executors return either an explicit ``(ok, result_data)`` 2-tuple or a
    bare value (dict / string / other), which is treated as success.

    Args:
        executor_result: The raw value an executor returned.

    Returns:
        ``(ok, result_data)`` — ``ok`` is the success flag, ``result_data``
        the payload.
    """
    if isinstance(executor_result, tuple) and len(executor_result) == 2:
        ok, result_data = executor_result
        return ok, result_data
    return True, executor_result


def extract_multimodal_attachments(
    result: Dict[str, Any],
) -> Optional[List[Attachment]]:
    """Extract multimodal attachments from a ``_multimodal`` result dict.

    Three shapes:

    * ``_multimodal_type == 'image'`` (the default) reads ``image_data``;
    * ``'file'`` (PDFs/documents) reads ``file_data``;
    * ``'attachments'`` reads ``_multimodal_attachments``, a LIST of
      canonical wire dicts ``{mime_type, data, display_name}`` -- the
      only shape that can carry more than one payload, which is what a
      clarification BATCH needs (#989: N questions, any of them answered
      with media).  The single-payload forms stay exactly as they were;
      they are what every ``_multimodal`` tool in the tree emits today.

    Returns ``None`` when there is nothing to attach.
    """
    multimodal_type = result.get('_multimodal_type', 'image')

    if multimodal_type == 'attachments':
        return _attachments_from_entries(
            result.get('_multimodal_attachments') or []
        )

    if multimodal_type == 'image':
        image_data = result.get('image_data')
        if not image_data:
            return None

        mime_type = result.get('mime_type', 'image/png')
        display_name = result.get('display_name', 'image')

        return [Attachment(
            mime_type=mime_type,
            data=image_data,
            display_name=display_name,
        )]

    if multimodal_type == 'file':
        file_data = result.get('file_data')
        if not file_data:
            return None

        mime_type = result.get('mime_type', 'application/octet-stream')
        display_name = result.get('display_name', 'file')

        return [Attachment(
            mime_type=mime_type,
            data=file_data,
            display_name=display_name,
        )]

    return None


def _attachments_from_entries(
    entries: List[Any],
) -> Optional[List[Attachment]]:
    """Build attachments from the list form of ``_multimodal``.

    Accepts an already-built :class:`Attachment` (an in-process producer)
    and the canonical wire dict with a raw-``bytes`` or base64-``str``
    ``data`` (anything that crossed a transport).  An entry whose payload
    will not decode is DROPPED rather than turned into an empty
    attachment: a zero-byte payload on the wire is a provider error, and
    an attachment that is silently empty is worse than one that is
    absent.

    Returns ``None`` when nothing usable is left, which is the "no
    attachments" answer every caller of
    :func:`extract_multimodal_attachments` already handles.
    """
    import base64

    out: List[Attachment] = []
    for entry in entries:
        if isinstance(entry, Attachment):
            out.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        data = entry.get('data')
        if isinstance(data, str):
            try:
                data = base64.b64decode(data, validate=True)
            except Exception:  # noqa: BLE001 - undecodable means unusable
                continue
        if not isinstance(data, (bytes, bytearray)):
            continue
        mime_type = (entry.get('mime_type') or '').strip()
        if not mime_type:
            continue
        out.append(Attachment(
            mime_type=mime_type,
            data=bytes(data),
            display_name=entry.get('display_name') or None,
        ))
    return out or None


def normalize_result_dict(result_data: Any, *, ok: bool) -> Any:
    """Normalize a non-string executor payload into the model-facing form.

    Steps (all preserving the original ``_build_tool_result`` semantics):

    1. Wrap non-dict payloads as ``{"result": <value>}``.
    2. Surface a permission advisory comment (``_permission.comment``) as a
       visible ``permission_note`` field.
    3. Strip internal ``_``-prefixed scaffolding keys (``_permission``,
       ``_multimodal`` flags, etc.) that aren't meaningful to the model.
    4. For error results whose only remaining key is ``error``, collapse to
       the bare error string so converters don't JSON-encode a single-key
       dict.

    Args:
        result_data: The executor payload (dict or other non-string value).
        ok: Whether the tool call succeeded.

    Returns:
        The normalized result — a dict, or a bare error string for the
        single-key error case.
    """
    # Build result dict
    if isinstance(result_data, dict):
        result_dict = result_data
    else:
        result_dict = {"result": result_data}

    # Inject advisory comment from permission evaluator (ALLOW_WITH_COMMENT)
    # before stripping internal metadata.  The comment becomes a visible
    # field so the model sees the feedback alongside the tool result.
    perm_meta = result_dict.get('_permission')
    if isinstance(perm_meta, dict) and perm_meta.get('comment'):
        result_dict['_permission_note'] = perm_meta['comment']

    # Strip internal metadata keys (prefixed with '_') before sending
    # to the model.  These carry scaffolding like _permission, _multimodal
    # flags, etc. that are not meaningful to the model.  The
    # _permission_note is intentionally kept (renamed below).
    permission_note = result_dict.pop('_permission_note', None)
    result_dict = {
        k: v for k, v in result_dict.items()
        if not k.startswith('_')
    }
    if permission_note:
        result_dict['permission_note'] = permission_note

    # For error results, extract a clean error string so provider
    # converters don't double-wrap a dict inside {"error": str(dict)}.
    # This ensures the model receives a readable message (e.g.,
    # "Tool not executed. User comment: ...") rather than a repr of
    # internal scaffolding.
    if not ok and 'error' in result_dict:
        error_msg = result_dict['error']
        # If 'error' is the only remaining key, pass the string directly
        # so converters don't JSON-encode a single-key dict.
        if len(result_dict) == 1:
            result_dict = error_msg

    return result_dict


# --------------------------------------------------------------------------
# Tool-result text view (#922)
#
# A dict tool result has to become a *string* before the tool-result
# enrichment chain can look at it, and until #922 the session picked that
# string by guessing: it enriched fields named one of six well-known names
# (``result``, ``content``, ``stdout``, ``output``, ``text``, ``data``) and
# only when the value ran to at least 100 characters.  Both filters were
# invisible.  ``store_memory`` names its text ``message`` and the observed
# message was 83 characters, so the two plugins that implement tool-result
# enrichment — ``memory`` and ``references`` — never ran on the pairing
# they exist for ("the agent just wrote down something about X; surface
# what we know about X").  No error, no warning, not a single trace line:
# the plugins looked correctly written and simply never fired, and every
# future dict-returning tool had to guess six field names correctly or be
# silently skipped forever.
#
# The session no longer guesses.  It renders the dict's *scalar* content
# as a read-only text view and hands that to the chain.  The view is split
# into a ``header`` (one ``key: value`` line per remaining field) and a
# ``body`` (the anchor field's raw value), so what enrichment returns can
# be written back precisely: anything still carrying the header is the
# anchor field's new value, whether the enricher appended a hint block or
# rewrote the body in place.
#
# Nested payloads are deliberately NOT rendered — ``TRAIT_FILE_WRITER`` /
# ``TRAIT_GREPPABLE_CONTENT`` are what hand an enricher the whole JSON.
# --------------------------------------------------------------------------

#: Field names conventionally carrying a tool's prose output.  They no
#: longer decide WHETHER enrichment runs — every dict result is enriched
#: now — only WHICH field an enriched body is written back to when several
#: strings are present, so hints keep landing where they always did for
#: the tools that use these names.
TEXT_FIELD_PREFERENCE: Tuple[str, ...] = (
    'result', 'content', 'stdout', 'output', 'text', 'data',
)

#: Where enrichment lands when a dict result has no string field at all to
#: write into (e.g. ``{"count": 3, "tags": [...]}``).
ENRICHMENT_FALLBACK_KEY = '_enrichment'

#: Per-field cap on the header lines.  The header is context for matching,
#: never written back to a field, so truncating it costs nothing and keeps
#: one oversized sibling field from dwarfing the text being enriched.
_MAX_CONTEXT_FIELD_CHARS = 500


def _render_context_value(value: Any) -> Optional[str]:
    """Render one non-anchor field as a single line of matching context.

    Args:
        value: The field's value.

    Returns:
        The rendered line, or ``None`` for values carrying no text worth
        matching on — ``None``, empty strings, empty sequences — and for
        nested structures (dicts, lists of dicts), which are what the
        trait path exists to hand enrichers whole.
    """
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        rendered = value.strip()
    elif isinstance(value, (list, tuple, set)):
        parts = [
            str(item) for item in value
            if isinstance(item, (str, int, float, bool))
        ]
        rendered = ", ".join(parts)
    else:
        return None
    return rendered[:_MAX_CONTEXT_FIELD_CHARS] or None


def pick_anchor_field(result_dict: Dict[str, Any]) -> Optional[str]:
    """Choose the field an enriched text view is written back to.

    Preference order:

    1. the first present name from :data:`TEXT_FIELD_PREFERENCE` holding a
       non-blank string — so tools already using a conventional name keep
       receiving their hints exactly where they used to;
    2. otherwise the wordiest non-blank string field — most whitespace-
       separated tokens, ties broken by length.  That is what "the text of
       this result" means for a tool that named it anything else
       (``message``, ``summary``, ``body``, ...), and counting words rather
       than characters is what keeps a one-word status field from
       out-weighing a short sentence: a result pairing
       ``status: "success"`` with ``message: "one two three"`` anchors
       on ``message``, though ``"success"`` is the longer string;
    3. ``None`` when the dict holds no string at all.

    Args:
        result_dict: The model-facing result dict.

    Returns:
        The anchor field's key, or ``None``.
    """
    for name in TEXT_FIELD_PREFERENCE:
        value = result_dict.get(name)
        if isinstance(value, str) and value.strip():
            return name

    anchor: Optional[str] = None
    best: Tuple[int, int] = (0, 0)
    for key, value in result_dict.items():
        if key.startswith('_') or not isinstance(value, str):
            continue
        text = value.strip()
        score = (len(text.split()), len(text))
        if text and score > best:
            anchor, best = key, score
    return anchor


def tool_result_text_view(
    result_dict: Dict[str, Any],
) -> Tuple[str, str, Optional[str]]:
    """Render a dict tool result as the text the enrichment chain sees.

    Each non-anchor field becomes its own ``key: value`` line, because the
    enrichers match on line- and sentence-scoped segments: joining fields
    onto one line would manufacture co-occurrences between unrelated
    values and surface memories that were never mentioned.

    Args:
        result_dict: The model-facing result dict.

    Returns:
        ``(header, body, anchor)``.  The text view is ``header + body``;
        ``header`` ends with a blank-line separator when non-empty, and
        ``body`` is the anchor field's raw value (``""`` when the dict has
        no string field).
    """
    anchor = pick_anchor_field(result_dict)

    lines: List[str] = []
    for key, value in result_dict.items():
        if key.startswith('_') or key == anchor:
            continue
        rendered = _render_context_value(value)
        if rendered:
            lines.append(f"{key}: {rendered}")

    header = "\n".join(lines) + "\n\n" if lines else ""
    body = result_dict.get(anchor, "") if anchor else ""
    return header, body, anchor


def apply_text_view_enrichment(
    result_dict: Dict[str, Any],
    header: str,
    body: str,
    anchor: Optional[str],
    enriched: str,
) -> bool:
    """Write an enriched text view back onto ``result_dict`` in place.

    Three cases, in order:

    1. nothing changed — the dict is left alone;
    2. the header survived — everything after it is the anchor field's new
       value.  This covers both shapes enrichers use: a hint block appended
       to the end (``memory``, ``lsp``) and an in-place rewrite of the body
       (``references`` expanding an ``@ref-id`` mention);
    3. the header did not survive — the enricher replaced the whole view
       (``result_grep`` returns a JSON envelope), so the returned string
       becomes the anchor field's value outright.

    With no anchor field to write into, whatever enrichment *added* lands
    under :data:`ENRICHMENT_FALLBACK_KEY` instead, leaving the tool's own
    fields untouched.

    Args:
        result_dict: The dict to mutate.
        header: The header half of the text view handed to the chain.
        body: The body half (the anchor field's original value).
        anchor: The anchor field's key, or ``None``.
        enriched: What the enrichment chain returned.

    Returns:
        True when the dict was changed.
    """
    text_view = header + body
    if enriched == text_view:
        return False

    if anchor is not None:
        result_dict[anchor] = (
            enriched[len(header):] if enriched.startswith(header) else enriched
        )
        return True

    addition = (
        enriched[len(text_view):] if enriched.startswith(text_view) else enriched
    ).strip("\n")
    if not addition:
        return False
    result_dict[ENRICHMENT_FALLBACK_KEY] = addition
    return True
