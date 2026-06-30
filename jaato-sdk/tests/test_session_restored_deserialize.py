"""Regression: session.restored is a first-class event and must deserialize.

EventType.SESSION_RESTORED + SessionRestoredEvent existed, but the class was
never registered in deserialize_event's dispatch table — so the background
drain logged "Unknown event type: session.restored" on every re-attach to a
restored session (harmless catch+continue, but noisy, and clients couldn't
receive restored-attach UX / pending-tool-call info).
"""
from jaato_sdk.events import (
    deserialize_event, serialize_event, SessionRestoredEvent, EventType,
)


def test_session_restored_round_trips():
    ev = SessionRestoredEvent(session_id="sid-1")
    back = deserialize_event(serialize_event(ev))
    assert isinstance(back, SessionRestoredEvent)
    assert back.type == EventType.SESSION_RESTORED
    assert back.session_id == "sid-1"


def test_session_restored_in_dispatch_table():
    from jaato_sdk import events as _e
    table = next(v for v in vars(_e).values()
                 if isinstance(v, dict)
                 and v.get(EventType.SESSION_TERMINATED.value) is not None)
    assert EventType.SESSION_RESTORED.value in table
