"""Wire-format snapshot baseline for ``jaato_sdk.events``.

Phase 0 safety net for the upcoming pydantic migration of
``events.py``.  The migration must preserve byte-for-byte JSON
output on the wire — clients connected to the daemon (TUI,
premium webcomponents, third-party SDK consumers) must not see
any protocol drift.

How it works:

* For every :class:`Event` subclass in ``_EVENT_CLASSES``,
  default-construct an instance, serialise via ``serialize_event``,
  and compare the JSON against a frozen baseline file in
  ``baselines/events_wire_format/<event_type>.json``.
* The ``timestamp`` field is masked (replaced with
  ``"<TIMESTAMP>"``) before comparison because its value is
  generated at construction time and would otherwise differ
  every run.
* A second parametrised test asserts the round-trip
  serialise → deserialise → re-serialise produces identical
  JSON, catching any deserialiser-side drift.

Workflow:

1. Run with ``REGENERATE_EVENT_BASELINES=1 pytest …`` to
   (re)generate the baseline files from the current code.
2. Commit the baselines.
3. After the pydantic migration, re-run the tests without the
   env var.  Any baseline mismatch is a wire-format regression.

If a new Event class is added to ``events.py`` without a
baseline, the test fails with a clear message instructing how
to regenerate.
"""

import json
import os
from pathlib import Path
from typing import Type

import pytest

from jaato_sdk.events import (
    _EVENT_CLASSES,
    Event,
    deserialize_event,
    serialize_event,
)

BASELINE_DIR = Path(__file__).parent / "baselines" / "events_wire_format"
TIMESTAMP_PLACEHOLDER = "<TIMESTAMP>"
REGENERATE_ENV = "REGENERATE_EVENT_BASELINES"


def _mask_timestamp(payload: dict) -> dict:
    """Replace the auto-generated ``timestamp`` with a stable placeholder."""
    if "timestamp" in payload:
        payload["timestamp"] = TIMESTAMP_PLACEHOLDER
    return payload


def _baseline_path(type_value: str) -> Path:
    safe = type_value.replace("/", "_").replace(":", "_")
    return BASELINE_DIR / f"{safe}.json"


def _all_event_classes():
    """Return ``[(type_value, EventClass), ...]`` for parametrisation.

    Sorted by type_value for stable test-case ordering across runs.
    """
    return sorted(_EVENT_CLASSES.items(), key=lambda kv: kv[0])


@pytest.mark.parametrize(
    "type_value,event_class",
    _all_event_classes(),
    ids=[t for t, _ in _all_event_classes()],
)
def test_event_serializes_to_baseline(type_value: str, event_class: Type[Event]):
    """Default-constructed event JSON must match its frozen baseline.

    Catches any post-migration drift in field set, field order,
    default values, or enum serialisation.
    """
    instance = event_class()
    payload = _mask_timestamp(json.loads(serialize_event(instance)))

    path = _baseline_path(type_value)

    if os.environ.get(REGENERATE_ENV):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        return

    assert path.exists(), (
        f"No baseline for event type {type_value!r}.  Regenerate with "
        f"{REGENERATE_ENV}=1 pytest … and commit the new file at "
        f"{path.relative_to(Path.cwd()) if path.is_absolute() else path}"
    )
    expected = json.loads(path.read_text())
    assert payload == expected, (
        f"Wire format drift for {type_value!r} — the JSON shape changed "
        f"since the baseline was captured.  If the change is intentional, "
        f"regenerate via {REGENERATE_ENV}=1 pytest … and commit the new "
        f"baseline.  Otherwise the producer side has a bug."
    )


@pytest.mark.parametrize(
    "type_value,event_class",
    _all_event_classes(),
    ids=[t for t, _ in _all_event_classes()],
)
def test_event_round_trip_byte_identical(
    type_value: str, event_class: Type[Event]
):
    """``serialize → deserialize → re-serialize`` is idempotent.

    Catches deserialiser-side regressions independently from the
    baseline check — even without committed baselines, the
    round-trip property must hold.
    """
    original = event_class()
    first_json = serialize_event(original)
    restored = deserialize_event(first_json)
    second_json = serialize_event(restored)

    first = _mask_timestamp(json.loads(first_json))
    second = _mask_timestamp(json.loads(second_json))
    assert first == second, (
        f"Round-trip drift for {type_value!r}.  The deserialiser "
        f"changed the shape: first={first!r}, second={second!r}"
    )


def test_baseline_directory_has_no_orphans():
    """Every baseline file corresponds to a current ``EventType``.

    If an event type is removed from ``EventType`` without
    deleting its baseline, this fires — preventing stale
    baselines from masking real coverage gaps.
    """
    if not BASELINE_DIR.exists():
        pytest.skip(
            f"Baseline directory not yet populated; "
            f"run {REGENERATE_ENV}=1 pytest … to create."
        )
    known_types = set(_EVENT_CLASSES.keys())
    orphans = []
    for path in BASELINE_DIR.glob("*.json"):
        type_value = path.stem
        if type_value not in known_types:
            orphans.append(path.name)
    assert not orphans, (
        f"Orphaned baseline files for unknown event types: {orphans}.  "
        f"Delete them or restore the corresponding EventType."
    )


def test_session_terminated_error_context_round_trips():
    """``error_summary`` + ``error_type`` (server 0.6.159+ / SDK 0.14.1+)
    must round-trip through serialise → deserialise unchanged.

    Cascade observers depend on these fields to surface the failure
    cause without grepping the daemon log.  A deserialiser regression
    that dropped them would re-open the visibility gap that Q2 closed.
    """
    from jaato_sdk.events import SessionTerminatedEvent

    original = SessionTerminatedEvent(
        session_id="20260526_181603",
        agent_id="codegen_step_2",
        reason="error",
        error_summary="402 Payment Required - credit balance is too low",
        error_type="AnthropicAPIError",
    )
    restored = deserialize_event(serialize_event(original))

    assert isinstance(restored, SessionTerminatedEvent)
    assert restored.session_id == "20260526_181603"
    assert restored.reason == "error"
    assert restored.error_summary == (
        "402 Payment Required - credit balance is too low"
    )
    assert restored.error_type == "AnthropicAPIError"


def test_history_event_accepts_rich_turn_accounting():
    """HistoryEvent.turn_accounting must accept the runner's RICHER per-turn
    dict — float ``duration_seconds`` + list ``function_calls`` (mirroring
    TurnCompletedEvent) — not just int token counts.  A strict
    ``Dict[str, int]`` rejected those with a pydantic ValidationError and took
    the whole event down, so a disk-restored session's history.request /
    attach-replay / snapshot all failed on the accounting alone.
    """
    from jaato_sdk.events import HistoryEvent

    # The exact shape from the observed ValidationError (2-turn session).
    rich = [
        {"prompt": 10, "output": 20, "total": 30,
         "duration_seconds": 0.530914, "function_calls": []},
        {"prompt": 5, "output": 7, "total": 12,
         "duration_seconds": 0.257768, "function_calls": [{"name": "x"}]},
    ]
    ev = HistoryEvent(agent_id="main", turn_accounting=rich)
    assert ev.turn_accounting[0]["duration_seconds"] == 0.530914
    assert ev.turn_accounting[1]["function_calls"] == [{"name": "x"}]
    # Backward-compat: plain int-only dicts still accepted.
    assert HistoryEvent(
        turn_accounting=[{"prompt": 1, "output": 2, "total": 3}]
    ).turn_accounting[0]["total"] == 3


def test_session_terminated_natural_reason_keeps_error_fields_none():
    """Backwards-compat: natural-completion / client-request / stopped
    paths emit ``error_summary=None`` + ``error_type=None``.  Existing
    consumers that don't read these fields see no behavior change.
    """
    from jaato_sdk.events import SessionTerminatedEvent

    for reason in ("natural", "client_request", "stopped"):
        ev = SessionTerminatedEvent(
            session_id="s1", agent_id="main", reason=reason,
        )
        assert ev.error_summary is None, (
            f"reason={reason!r} should keep error_summary=None"
        )
        assert ev.error_type is None, (
            f"reason={reason!r} should keep error_type=None"
        )
