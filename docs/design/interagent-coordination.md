# Design: peer-to-peer agent coordination

**Status:** design. Prompted by the Prime Agent comparison
(`docs/compare-jaato-prime-agent.md` §3.1), which found jaato's only
model-facing messaging path is parent→child.

## 1. What is missing, precisely

jaato routes agent traffic through the parent. `send_to_subagent` addresses a
child by opaque id; `share_context` pushes structured findings child→parent;
`list_active_subagents` filters `_active_sessions` by
`owner_id = id(self._parent_session)`, so it returns own direct children only.
There is no sibling edge at the model tier.

A roster *does* exist one tier up — `session.list` is a client command and
`cascade_events(cid)` observes every session under a cascade — so an
orchestrating client can already enumerate and route. The gap is specifically
that **a stage cannot reach a peer stage without the driver in the loop.**

## 2. The scope is the cascade, not the tree

Prime Agent's "family" is its RLM parent/child tree. jaato has a better-fitting
scope already stamped on sessions: **`cascade_driver_id`**. Sessions carrying the
same cid are exactly the set that should be able to address one another, and
`cascade_events(cid)` proves the daemon can already resolve that set.

This makes sibling addressing a *filter over existing state*, not a new topology.
It also gives a natural authority boundary — the cid is the blast radius — and a
natural uniqueness scope for names (§4).

## 3. Reuse the injection primitives; build addressing only

Two injection paths already exist:

| Primitive | Shape |
|---|---|
| `send_to_subagent` | queue injection into a running child, processed at its next yield point |
| `wake_session` | revives a cold session and drives a USER turn, with `wrap_untrusted_content` and `event_id` dedup |

Peer messaging is a **third addressing mode over these**, not new transport. All
traffic stays daemon-mediated, so per-session confined runners never talk to each
other directly — the confinement boundary is untouched.

## 4. Named addressing needs a uniqueness scope

Sessions already carry `session_name`, but it auto-generates as
`Session <timestamp>` — useless as an address. Peer addressing needs names that
are **distinct within a cid**, checked at `session.new`. The cascade is small, so
uniqueness is cheap to enforce, and a name scoped to a cid cannot collide with an
unrelated cascade's.

Named addressing is what makes everything else usable: an id you never saw cannot
be typed by a human, put in a profile, or written into a persona.

## 5. Messages should be typed, not prose

Prime Agent's agent messages are strings; its own tool description tells the
model to use messages for conversation and files for data. jaato should not copy
that limitation — it already has `completion_payload_schema` and the schema
loader. A peer message carrying a **validated payload** delivers the receiving
agent a typed object instead of prose to re-parse, and lets the operator declare
what one stage may say to another.

Prose messages stay available for course correction; typed ones are for handoffs.

## 6. Inbound peer messages are untrusted content

A message from a peer is content the receiving model did not author. It must be
wrapped with `wrap_untrusted_content`, exactly as `wake_session` already wraps
external wake text and for the same reason — so the receiver treats it as **data
to weigh, never as instructions**.

Prime Agent does not do this. It matters more in jaato, because jaato agents hold
permissioned tools that a confused or hostile peer would otherwise be able to
drive by writing imperative prose.

## 7. The authority boundary — the one thing that must not leak

`send_to_subagent` today carries more than conversation. Its own system
instructions document sending `<permission_response request_id="…">` and
`<clarification_response …>` through the same channel: **a parent answering its
child's permission request**.

That is parent authority, and it must not travel sideways. A sibling edge that
reuses the channel naively would let any peer grant permissions to any other
peer, defeating the permission system entirely.

**Required:** peer messages are parsed under a restricted grammar with
`permission_response` and `clarification_response` rejected, and the daemon —
not the sender — stamps the sender relationship, so a peer cannot claim to be a
parent. Answering a permission request stays a parent-only, tier-checked path.

## 8. Bounding the loop

Prime Agent bounds messaging with a token bucket (3/s), a 20-message pending cap
and a 16 KiB size cap. jaato has a better terminator already: **`budget_control`**.
A ping-pong between peers burns turns, and cascade budgets count turns, seconds,
tool calls and spend across every session under the cid — so a runaway
conversation hits a ceiling that already exists and degrades before it stops.

Still worth taking from Prime Agent: a **per-session pending cap** (backpressure,
so a busy peer cannot accumulate an unbounded queue) and a **size cap** (a peer
should not be able to blow another's context in one message).

Keep one Prime Agent discipline exactly: **fire-and-forget with a receipt, never
a blocking request/response.** Their `rlm()` deliberately never returns a child's
answer, which is what makes deadlock unrepresentable. A peer send must return
`delivered | queued` and nothing else; two agents awaiting each other must not be
expressible.

## 9. Proposed surface

Model-facing, in the subagent plugin (or a new `peer` plugin):

| Tool | Approval | Behaviour |
|---|---|---|
| `list_peers` | auto | Sessions sharing my `cascade_driver_id`: name, role (`parent`/`sibling`/`child`), status, profile |
| `send_to_peer` | **gated** | `(peer_name, message \| payload, mode)` → receipt `{delivered\|queued, at}`. Permission display names the target |

`mode` mirrors Prime Agent's `auto` / `steer` / `follow_up`, which jaato lacks
entirely today — `send_to_subagent` has one implicit behaviour ("next yield
point" ≈ `follow_up`). A course correction wants `steer`; a handoff wants
`follow_up`.

Client-tier, for parity with `prime-agent send`: a `session.send` command so a
human or script can nudge a named session in a cascade without the model.

## 10. Non-goals

- **No cross-cascade addressing.** The cid is the boundary. Reaching another
  cascade goes through its driver, or through `wake_session` with its signature
  checks.
- **No sibling authority.** §7. Peers coordinate; they do not approve, grant, or
  cancel one another.
- **No blocking request/response.** §8.
- **Not a replacement for the driver.** Completion payloads and reactors remain
  the way stages hand off *results*. Peer messaging is for coordination the
  driver should not have to mediate — "are you done with the file I need?",
  "I found the config you were looking for" — not for pipeline control flow.

## 11. Open questions

1. Should `share_context` gain a peer target, or should typed peer messages
   subsume it? They overlap: both move structured data between agents.
2. Does a peer message wake a **cold** peer (via `wake_session`) or only reach
   running ones? Waking is more capable and more surprising; the safer default is
   to reach running peers and queue for idle ones, and require an explicit flag
   to revive a cold peer.
3. Does the permission prompt for `send_to_peer` need the message body, or only
   the target? Body-in-prompt is more informative and much noisier.
