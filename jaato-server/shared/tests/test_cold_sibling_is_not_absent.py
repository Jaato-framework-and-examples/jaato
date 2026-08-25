"""A resting sibling must report ``sibling_cold``, never ``no_such_sibling``.

Design §11 Q2 went the conservative way deliberately: cold peers are not
woken, and ``sibling_cold`` says *the address is real and the peer is
resting* — a different fact from ``no_such_sibling``, needing a different
response from the sender.

Both sibling paths resolved on-disk sessions with
``_get_persisted_sessions()`` and **no workspace**.  That does not mean "every
workspace": ``file_session.list_sessions`` does
``target_dir = storage_dir or self._storage_path``, so ``None`` falls through
to the plugin's DEFAULT storage directory.  For a workspace-scoped daemon that
is a different directory, the listing came back empty, and:

  - ``send_to_sibling`` reported ``no_such_sibling`` for a live address
  - ``build_sibling_roster`` omitted the cold rows entirely

so cold and deleted became the same answer — the collapse the C4 split exists
to prevent.

Absent-and-empty one layer down: an omitted ``workspace_path`` was
indistinguishable from a workspace with no sessions in it.

Reported by the cascade-coordination probe, made cold by attach-away.
"""

import threading

import pytest

from server.session_manager import SessionManager


WS = "/tmp/ws-alpha"


class _Persisted:
    def __init__(self, sid, cid, name):
        self.session_id, self.cascade_driver_id, self.sibling_name = sid, cid, name
        self.description = None
        self.profile_name = None


def _session(sid, cid="cid-1", name=None, workspace=WS):
    s = type("S", (), {})()
    s.session_id, s.cascade_driver_id, s.sibling_name = sid, cid, name
    s.server = type("V", (), {"_model_running": False})()
    s.attached_clients, s.description, s.workspace_path = [], None, workspace
    return s


def _sm(*sessions, on_disk=(), disk_by_workspace=None):
    """A manager whose disk listing is SCOPED, like the real one.

    ``disk_by_workspace`` mirrors production: sessions live under their
    workspace's storage dir, and asking with the wrong (or no) workspace
    returns nothing.  A fixture that ignored the argument would pass with the
    bug present — the whole defect was that the argument was omitted.
    """
    disk = disk_by_workspace if disk_by_workspace is not None else {WS: list(on_disk)}
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {s.session_id: s for s in sessions}
    sm._lock = threading.RLock()
    sm._sibling_pending, sm._sibling_exchanges = {}, {}
    sm.delivered = []
    # Both mechanisms recorded: busy peers are queued (inject), idle peers
    # are DRIVEN (send_message_to_session).  Stubbing only the injector
    # would make "the idle peer ran" look like "nothing was delivered".
    sm.inject_prompt_to_session = (
        lambda sid, t, source_id=None, source_type=None:
        sm.delivered.append(sid) or True
    )
    sm.send_message_to_session = (
        lambda sid, t: sm.delivered.append(sid) or True
    )
    sm._get_persisted_sessions = lambda workspace_path=None: list(
        disk.get(workspace_path, []))
    sm._roster_profile_name = lambda s: None
    return sm


# ----------------------------------------------------------------------

def test_a_resting_sibling_is_cold_not_absent():
    """The reported failure, end to end."""
    sm = _sm(_session("s-a", name="alice"),
             on_disk=[_Persisted("s-b", "cid-1", "resting")])
    r = sm.deliver_sibling_message("s-a", "resting", "hello")
    assert r["status"] == "sibling_cold"
    assert sm.delivered == [], "a cold sibling must not be driven"


def test_an_address_that_never_existed_is_still_absent():
    """The contrast that gives ``sibling_cold`` its meaning.

    Without this, returning ``sibling_cold`` unconditionally would pass the
    test above while destroying the distinction it exists to protect.
    """
    sm = _sm(_session("s-a", name="alice"),
             on_disk=[_Persisted("s-b", "cid-1", "resting")])
    assert sm.deliver_sibling_message("s-a", "ghost", "hi")["status"] == "no_such_sibling"


def test_an_idle_sibling_still_takes_its_turn():
    """The half that already worked must keep working."""
    sm = _sm(_session("s-a", name="alice"), _session("s-b", name="awake"))
    assert sm.deliver_sibling_message("s-a", "awake", "hello")["status"] == "accepted"
    assert sm.delivered == ["s-b"]


def test_the_roster_lists_the_cold_sibling_it_can_be_sent_to():
    """The roster and the sender must agree.

    The error text tells the caller to "use list_siblings for the roster" — a
    roster that omits the sibling just refused would send them in a circle.
    """
    sm = _sm(_session("s-a", name="alice"),
             on_disk=[_Persisted("s-b", "cid-1", "resting")])
    roster = sm.build_sibling_roster("s-a")
    rows = {r["sibling_name"]: r["status"] for r in roster["siblings"]}
    assert rows == {"resting": "cold"}


def test_the_lookup_is_scoped_to_the_cascade_workspace():
    """A same-named session in ANOTHER workspace must not be found.

    Deriving the workspace must not become "search everywhere" — that would
    resolve one cascade's address to another workspace's session.
    """
    sm = _sm(_session("s-a", name="alice"),
             disk_by_workspace={
                 WS: [],
                 "/tmp/ws-other": [_Persisted("s-x", "cid-1", "resting")],
             })
    assert sm.deliver_sibling_message("s-a", "resting", "hi")["status"] == "no_such_sibling"


def test_the_workspace_is_derived_when_the_viewer_is_not_loaded():
    """The client path (``session.send``) has a cid but no viewer session."""
    sm = _sm(_session("s-b", name="awake"),
             on_disk=[_Persisted("s-c", "cid-1", "resting")])
    assert sm._cascade_storage_workspace(None, "cid-1") == WS


def test_an_explicit_workspace_still_wins():
    """Derivation is a default, not an override — existing callers pass one."""
    sm = _sm(_session("s-a", name="alice", workspace="/tmp/ws-wrong"),
             disk_by_workspace={WS: [_Persisted("s-b", "cid-1", "resting")]})
    roster = sm.build_sibling_roster("s-a", workspace_path=WS)
    assert [r["sibling_name"] for r in roster["siblings"]] == ["resting"]
