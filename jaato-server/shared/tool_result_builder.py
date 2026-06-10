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

    Currently supports ``_multimodal_type == 'image'`` (the default),
    reading ``image_data`` / ``mime_type`` / ``display_name``. Returns
    ``None`` when there is nothing to attach.
    """
    multimodal_type = result.get('_multimodal_type', 'image')

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

    return None


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
