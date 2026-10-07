"""Pytest helpers shared across model_provider plugin test suites.

The factored helper here is ``assert_no_likely_cause_prose`` — a
falsification guard for the rule in
``feedback_no_hardcoded_likely_cause_in_error_messages.md``: when the
framework cannot extract an upstream's actual failure reason, the
surfaced error message must be GENERIC.  No enumeration of "likely
causes", no "common causes" lists, no "almost always X" prose, no
knob-tuning suggestions for failure modes the framework didn't
actually detect.

The two fixtures here isolate the tiers of credential resolution that
``jaato-server/conftest.py`` cannot close for every test on its own
(issue #721: a mis-isolated assertion read a developer's real
credential, and pytest printed the key into the failure message).

``fake_home`` is a writable stand-in for ``~/.jaato``, so a test can
exercise the HOME tier ON PURPOSE.  The global isolation points every
test's ``HOME`` at an empty directory, which makes "no credential is
configured" true — and also makes the home tier unreachable, so a
suite that only ever asserts the empty case can no longer tell "the
tier is empty" from "the tier is never consulted".  ``fake_home``
restores the second half.

``empty_project_tier`` closes the tier the global fixture deliberately
leaves alone: the working directory, which the project tier falls back
to when no workspace is passed.

Three providers (tensorrt_llm, triton, vllm) all wrap an
OpenAI-compatible self-hosted inference server and surface the same
mid-stream connection-drop failure mode (HTTP 200 committed before the
engine validates → wire-level ``RemoteProtocolError`` with no
extractable cause).  Each provider's ``*MidStreamError`` message MUST
pass this assertion.  When a fourth such provider lands, point its
mid-stream test at the same helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest


@pytest.fixture
def fake_home(tmp_path, monkeypatch) -> Path:
    """Point ``Path.home()`` at a writable directory this test owns.

    Overrides the session-wide isolation from ``jaato-server/
    conftest.py`` for one test — function-scoped monkeypatching runs
    after the autouse fixture, so this wins — and pre-creates
    ``.jaato/`` so a caller can drop a credential file straight in::

        (fake_home / ".jaato" / "openrouter_auth.json").write_text(...)

    Use it for the companion to every "no credential file" test: that
    one proves the loader returns ``None`` when no tier answers, this
    one proves the home tier is a tier at all.  Both are needed —
    together they say the loader consults exactly the tiers it claims.

    Returns:
        The directory ``HOME`` (and ``USERPROFILE``) now point at.
    """
    home = tmp_path / "home"
    (home / ".jaato").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


@pytest.fixture
def empty_project_tier(tmp_path, monkeypatch) -> Path:
    """Neutralise the PROJECT tier of credential resolution.

    Every provider resolves ``workspace_path or get_workspace_root() or
    os.getcwd()``, so a test that passes no explicit workspace lands on
    the current working directory — and pytest runs from the repo root,
    where a developer running the daemon in their checkout has real
    stored credentials in ``.jaato/``.  Moving cwd somewhere empty and
    clearing ``JAATO_WORKSPACE_ROOT`` closes that tier.

    Unlike the ``HOME`` tier — isolated for every test by
    ``jaato-server/conftest.py`` — this one cannot be neutralised
    globally: the working directory is state a great many tests
    legitimately derive paths from, and a suite-wide ``chdir`` would
    move them all.  So it is opt-in, for the suites whose subject is
    credential resolution.

    Returns:
        The empty directory that is now the working directory.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("JAATO_CONFIG_ROOT", raising=False)
    return workspace


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
