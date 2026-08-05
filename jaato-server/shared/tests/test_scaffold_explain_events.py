"""scaffold ``explain events`` / ``explain event <NAME>`` — the event protocol.

Mirrors ``explain env``: the SDK's ``EventType`` enum in ``jaato_sdk/events.py``
is the single source of truth, AST-scanned offline (no import side-effects).
Each member carries its wire value, DIRECTION (from the section header or a
trailing per-member comment), DOMAIN (the section header), and — where a
``…Event(Event)`` class defaults its ``type:`` to that member — the class name,
docstring, and typed fields.

These tests pin the parse against known events + the two structural rules that
are easy to get wrong:
  * a member's own multi-line doc-comment (directly after a member, no blank
    line) must NOT reset the domain (AGENT_ERROR stays under "Agent lifecycle");
  * a blank-line-delimited header DOES start a section (the permission-policy
    verbs are their own domain, not the preceding "Wake primitive" one).
"""

from shared.scaffold import explain, introspect


# --------------------------------------------------------- introspect core

def test_events_catalog_is_populated_from_the_sdk():
    ev = introspect.events()
    # The protocol is large; a floor guards against an empty/partial parse.
    assert len(ev) > 50
    # Known members with their wire values.
    assert ev["AGENT_OUTPUT"].wire == "agent.output"
    assert ev["TOOL_EXECUTE_REQUEST"].wire == "tool.execute_request"
    assert ev["SESSION_WOKEN"].wire == "session.woken"


def test_events_direction_parsing():
    ev = introspect.events()
    # Section-level direction (Server -> Client) inherited by a member.
    assert ev["AGENT_OUTPUT"].direction == "Server → Client"
    # Bidirectional section (Server <-> Client).
    assert ev["PERMISSION_REQUESTED"].direction == "Server ↔ Client"
    # A trailing per-member comment overrides the section direction.
    assert ev["PERMISSION_RESPONSE"].direction == "Client → Server"


def test_events_join_to_their_event_class_fields():
    ev = introspect.events()
    woken = ev["SESSION_WOKEN"]
    assert woken.event_class == "SessionWokenEvent"
    assert [f.name for f in woken.fields] == ["session_id", "wake_ref", "source"]
    assert woken.doc  # class docstring first line is carried


def test_member_doc_comment_does_not_reset_domain():
    # AGENT_ERROR is preceded by its OWN multi-line doc-comment (directly after
    # AGENT_COMPLETED, no blank line) — that block is NOT a section header, so
    # AGENT_ERROR stays under the "Agent lifecycle" section.
    ev = introspect.events()
    assert ev["AGENT_ERROR"].domain == "Agent lifecycle"


def test_blank_delimited_header_starts_a_new_section():
    # The permission-policy verbs sit under a blank-line-delimited header, so
    # they form their own domain — they must NOT inherit the preceding
    # "Wake primitive" section.
    ev = introspect.events()
    dom = ev["PERMISSION_ADD_WHITELIST_REQUEST"].domain
    assert dom != "Wake primitive"
    assert "parity" in dom.lower() or "permission" in dom.lower()


def test_non_direction_parenthetical_kept_in_domain():
    # ``# Peer channel events (server-to-server gossip)`` — the parenthetical is
    # NOT a direction, so it stays part of the label.
    ev = introspect.events()
    peers = [e for e in ev.values() if e.domain.startswith("Peer channel events")]
    assert peers and "(server-to-server gossip)" in peers[0].domain


# --------------------------------------------------------------- rendering

def test_explain_events_overview_groups_by_domain():
    data, text = explain.events()
    assert "Agent lifecycle" in data
    assert "SESSION_WOKEN" in data["Wake primitive"]
    assert "event protocol" in text and "[S→C]" in text


def test_explain_events_filter_matches_domain_member_and_wire():
    _, text = explain.events("permission")
    assert "PERMISSION_REQUESTED" in text
    # filtering narrows the set
    _, all_text = explain.events()
    assert len(text) < len(all_text)


def test_explain_event_detail_by_member_and_by_wire():
    d1, t1 = explain.event("SESSION_WOKEN")
    d2, t2 = explain.event("session.woken")          # wire value also resolves
    assert d1["event_class"] == d2["event_class"] == "SessionWokenEvent"
    assert "wake_ref" in t1 and "Server → Client" in t1


def test_explain_event_unknown_gives_separator_insensitive_hint():
    data, text = explain.event("sessionwoke")        # no separators
    assert "error" in data
    assert "SESSION_WOKEN" in text
