"""Pytest helpers shared across model_provider plugin test suites.

The factored helper here is ``assert_no_likely_cause_prose`` — a
falsification guard for the rule in
``feedback_no_hardcoded_likely_cause_in_error_messages.md``: when the
framework cannot extract an upstream's actual failure reason, the
surfaced error message must be GENERIC.  No enumeration of "likely
causes", no "common causes" lists, no "almost always X" prose, no
knob-tuning suggestions for failure modes the framework didn't
actually detect.

Three providers (tensorrt_llm, triton, vllm) all wrap an
OpenAI-compatible self-hosted inference server and surface the same
mid-stream connection-drop failure mode (HTTP 200 committed before the
engine validates → wire-level ``RemoteProtocolError`` with no
extractable cause).  Each provider's ``*MidStreamError`` message MUST
pass this assertion.  When a fourth such provider lands, point its
mid-stream test at the same helper.
"""

from __future__ import annotations

from typing import Iterable


# Keywords that signal hardcoded likely-cause guessing.  Lower-case for
# case-insensitive comparison.
#
# This list is conservative — it catches the actual prose patterns
# from the pre-fix trtllm/triton/vllm messages.  When a new
# likely-cause-style phrasing emerges, add it here rather than
# silently tolerating it in one provider.
FORBIDDEN_LIKELY_CAUSE_KEYWORDS: tuple[str, ...] = (
    # Enumeration framings.
    "almost always",
    "common causes",
    "fixes by cause",
    "fix tree",
    "this is almost always",
    "one of:",                # "Common causes: 1. ...  Almost always one of:"
    # Specific causes the framework did NOT detect from the wire.
    "prompt-too-long",
    "prompt exceeds",
    "kv-cache",
    "kv cache exhaustion",
    "cuda oom",
    "out of memory",
    "engine-internal exception",
    "max_input_length",
    "max_input_len",
    "max_num_tokens",
    "max_batch_size",
    "kv_cache_free_gpu_mem_fraction",
    "suppress_base_instructions",  # Fix suggestion, not observation.
)


def assert_no_likely_cause_prose(
    message: str,
    *,
    extra_forbidden: Iterable[str] = (),
) -> None:
    """Assert ``message`` contains no hardcoded likely-cause prose.

    Used by mid-stream error tests across self-hosted-inference
    providers (tensorrt_llm, triton, vllm).  Fails with a precise
    pointer to the offending keyword so the operator sees which rule
    was broken, not just "test failed".

    Args:
        message: The error message text (typically ``str(err)``).
        extra_forbidden: Additional keywords to forbid for this caller
            (per-provider extensions when a specific provider has a
            past-incident pattern that shouldn't recur).
    """
    msg_lower = message.lower()
    forbidden = (*FORBIDDEN_LIKELY_CAUSE_KEYWORDS, *(k.lower() for k in extra_forbidden))
    hits = [kw for kw in forbidden if kw in msg_lower]
    assert not hits, (
        "Mid-stream error message contains hardcoded likely-cause "
        f"prose (forbidden keywords found: {hits!r}).  Per "
        "feedback_no_hardcoded_likely_cause_in_error_messages.md, "
        "the message must surface 'what we observed' (HTTP 200 "
        "committed, then connection drop) and point at the server "
        "log — NOT enumerate which engine-level failure was 'likely'.  "
        f"Offending message:\n\n{message}"
    )
