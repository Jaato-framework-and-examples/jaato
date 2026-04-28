# Task-Graph Orchestrator

A functional specification for a DAG-based task scheduler implemented as
an **SDK application** that drives jaato sessions over the public IPC /
WebSocket surface.

## Where this lives

The orchestrator is **not** part of the daemon. It is a long-running
SDK consumer that connects to a daemon via `IPCClient` /
`WebSocketClient`, holds the DAG state in its own process, and uses
public surfaces (`create_session`, `attach_session`,
`send_message`, `stop`) to dispatch and observe work.

| Component | Package | Reason |
|---|---|---|
| Orchestrator runtime (DAG store, scheduler, dispatcher, watchdog) | Standalone app or library on top of `jaato-sdk` | The orchestrator is a client, not a daemon component |
| `task_graph` tool plugin (model-callable `task_decompose`, `task_assign`, `task_status`) | Standalone or `jaato-premium` | Lets the supervising agent drive the graph from inside its own session via tool calls |
| Per-session reactor reflexes (memory consolidation, budget guard) | `jaato-premium` reactor framework | Operate alongside the orchestrator, not in place of it |
| Discovery of reactor-spawned sessions | Public SDK gate-event subscription (`docs/design/handoff-gate-api.md`) | The orchestrator subscribes to gate events to fold reactor-managed sessions into its DAG |

The daemon stays oblivious to the DAG. Multiple orchestrators can run
against one daemon (one per project) — each owns its own state, all
coexist with whatever in-daemon reactors premium provides.

## 1. Background

The OneManCompany paper (*From Skills to Talent: Organising Heterogeneous
Agents as a Real-World Company*, arXiv:2604.22446) proposes a six-interface
"organisational layer" for agent orchestration. Five of the six interfaces map
directly onto existing jaato surfaces:

| OMC interface | jaato equivalent |
|---|---|
| `Execution` | `JaatoSession.send_message` + provider `send_message` |
| `Event` | EventBus, typed events in `jaato_sdk/events.py` |
| `Storage` | session history, `TokenLedger`, `.jaato/logs/` |
| `Context` | system instructions, GC plugins, `PresentationContext` |
| `Lifecycle` | plugin auto-wiring, `LifecycleTools.signal_completion`, enrichment plugins |

The remaining interface — `Task` — is what this document specifies. It
sits **above** the daemon: a task graph is project-level state owned by
the orchestrator, separate from any one session's lifecycle. The daemon
knows about employees (sessions); the orchestrator knows how those
employees' work composes into a project.

## 2. Goals & Non-Goals

### Goals

- A persistent, crash-recoverable DAG of tasks with explicit dependencies.
- An FSM-governed lifecycle per node with a supervisor accept/reject gate.
- AND-tree completion semantics: a parent is resolved iff all children are.
- Bounded retry, task timeouts, deadlock detection.
- Built entirely on public SDK primitives — no daemon changes.
- Tenant-aware: the DAG is owned by a tenant; cross-tenant dispatch
  follows the multi-tenancy authz model.

### Non-goals (initial release)

- Multi-daemon / peer-clustered DAG state (single-daemon, single-orchestrator-process at first).
- An RL-trained decomposition policy (the supervising agent is the policy).
- A community marketplace of task templates.
- A separate UI server for remote DAG inspection — v1 assumes the
  orchestrator process is co-located with whatever renders it.

## 3. Concepts

| Term | Definition |
|---|---|
| **Node** | A single task. Owns a description, an assigned employee, an FSM state, a result, and a cost. |
| **Employee** | A **top-level jaato session** that executes assigned tasks. The orchestrator creates each employee with `IPCClient.create_session(profile=...)`. One session per employee — the orchestrator owns the session for the lifetime of the project (or until the employee retires). Subagents (intra-session nested work) are an employee's internal mechanism and not visible to the DAG. |
| **Decomposition edge** | "Parent was decomposed into child." Forms a strict tree. |
| **Dependency edge** | "Child cannot start until parent is accepted." Can cross sibling branches. |
| **Reviewer** | A jaato session that produces an `accept` or `reject` decision on a completed node. The supervisor picks the reviewer profile per decomposition (e.g. a `code-reviewer` profile for code tasks, a `design-reviewer` profile for design tasks). |
| **Supervisor** | The agent that decides decomposition, employee assignment, and reviewer choice for a node — typically the parent's owner, often a "COO" employee. The supervisor calls the orchestrator's tool plugin to mutate the DAG. |
| **Project** | A root node and its subgraph. Owned by exactly one tenant; the project's `tenant_id` propagates to every node and is the authz scope for dispatched sessions. |

## 4. Data Model

### 4.1 Node schema

```python
@dataclass
class TaskNode:
    id: str                              # uuid
    project_id: str                      # references the project's root id
    tenant_id: str                       # propagated from project; never changes
    parent_id: Optional[str]             # None for root
    description: str
    employee_session_id: Optional[str]   # None for unassigned / interior nodes
    reviewer_profile: Optional[str]      # supervisor's choice at decompose time
    state: TaskState                     # see FSM below
    result: Optional[Dict[str, Any]]     # typed completion payload
    token_usage: Dict[str, int]          # from AgentCompletedEvent.token_usage
    cost_usd: float                      # derived: from token_usage * provider rate
    idempotent: bool                     # supervisor declares; controls crash retry policy
    created_at: datetime
    updated_at: datetime
    review_count: int = 0                # bounded retry counter
    deps: List[str] = field(default_factory=list)        # node IDs
    children: List[str] = field(default_factory=list)    # node IDs
    timeout_at: Optional[datetime] = None
```

`token_usage` and `cost_usd` are kept distinct: token counts come
directly from the daemon's `AgentCompletedEvent.token_usage` (already
stamped by `LifecycleTools.signal_completion`); `cost_usd` is the
orchestrator's locally-applied rate calculation per provider/model.
This avoids ambiguity about what `cost: float` meant.

### 4.2 FSM

```
                 ┌──────────────────────────────┐
                 │                              │ retry (review_count < k_retry)
                 ▼                              │
   pending ──► processing ──► completed ──review pass──► accepted ──► finished
      │            │              │
      │            │              └──review fail──► processing (with feedback prepended)
      │            │
      │            ▼
      │         failed (timeout / k_retry exhausted)
      │
      └──► cancelled (cascade from a failed dep, or supervisor cancel)
```

Terminal states: `finished`, `failed`, `cancelled`. Only `finished`
counts as "resolved" for AND-semantics.

The `holding` state from earlier drafts is **removed** — it added a
state without a clear trigger or consumer.  Pause-and-resume is
modelled at the orchestrator level (the scheduler simply doesn't
dispatch `pending` nodes during pause), not as a per-node FSM state.

### 4.3 Persistence

- Single JSON file per project: `<orchestrator_state>/<project_id>.json`.
  The orchestrator's state directory is the orchestrator app's
  responsibility, not the daemon's. A reasonable default is
  `~/.jaato/orchestrator/<orchestrator_name>/`.
- Format: `{ "nodes": {id: TaskNode}, "edges": {...}, "tenant_id": str, "version": 1 }`.
- Writes go through a `TaskGraphStore` that holds an in-memory copy plus a
  `threading.RLock`. Each public mutator method:
  1. Acquires the lock.
  2. Mutates the in-memory dict.
  3. Atomically rewrites the JSON file (`tmp + os.replace`).
  4. Releases the lock.
- The lock is the only concurrency primitive needed inside the
  orchestrator process (see §7).

## 5. Architecture

The orchestrator is an **SDK application**. It connects to a jaato
daemon over IPC or WebSocket, holds the DAG in its own process,
subscribes to events, and dispatches by sending messages to specific
sessions.

```
┌──────────────────────────────────────────────────────────┐
│  task-graph orchestrator (SDK app process)              │
│                                                          │
│   ┌─────────────┐   ┌──────────────┐   ┌─────────────┐  │
│   │ TaskGraph   │◀──│  Scheduler   │   │  Watchdog   │  │
│   │ Store       │   │              │   │ (bg thread) │  │
│   │ (locked)    │   │              │   │             │  │
│   └─────────────┘   └──────┬───────┘   └──────┬──────┘  │
│                            │                  │         │
│                            ▼                  │         │
│                     ┌────────────┐             │         │
│                     │ Dispatcher │             │         │
│                     └─────┬──────┘             │         │
│                           │                    │         │
│                           ▼                    │         │
│                     ┌──────────────┐           │         │
│                     │  IPCClient   │◀──────────┘         │
│                     │  (multi-     │                     │
│                     │   session)   │                     │
│                     └──────┬───────┘                     │
└──────────────────────────  │  ─────────────────────────  ┘
                             │ IPC / WS
                             ▼
┌──────────────────────────────────────────────────────────┐
│  jaato-server (unchanged)                                │
│                                                          │
│   SessionManager        EventBus       LifecycleTools    │
│                                                          │
│   employees (sessions)                                   │
│   reviewers (sessions)                                   │
│   reactor-spawned helpers (memory-advisor, etc.)         │
│   premium reactor framework (incl. HandoffGate)          │
└──────────────────────────────────────────────────────────┘
```

### 5.1 Components

- **`TaskGraphStore`** — owns the in-memory DAG and the lock; persists to disk.
- **`Scheduler`** — pure logic: given an incoming event, decide which
  nodes transition state, which dependents are now ready, which to
  dispatch. Idempotent and lock-protected (see §7).
- **`Dispatcher`** — given a ready node, calls
  `client.create_session(profile=...)` (new employee) or
  `client.send_message(session_id, ...)` (existing employee) via the
  SDK. Owns the supervising agent's authz — every dispatch carries the
  orchestrator's identity.
- **`Watchdog`** — background thread; scans the graph every N seconds for
  timeouts and deadlocks. Synthesises FSM transitions on expiry.
- **`ReviewerCoordinator`** — for each `completed` node, calls
  `client.create_session(profile=node.reviewer_profile)` with a
  reviewer prompt; subscribes to that session's
  `AgentCompletedEvent` and routes the typed `decision` payload back
  to the scheduler.
- **`IPCClient` (multi-session)** — the orchestrator either holds N
  client connections (one per attached session, today's surface) or
  uses the multi-attached-sessions extension noted in the multi-tenancy
  doc. Either works.

### 5.2 Optional: model-callable tool plugin

For supervising agents that drive the DAG by calling tools (e.g. a COO
employee whose system instruction is "decompose your project, assign
tasks"), the orchestrator exposes its API as a **tool plugin** loaded
into the supervisor's session. The plugin's tools (`task_decompose`,
`task_assign`, `task_status`, `task_cancel`) call into the
orchestrator's `TaskGraphAPI` (§11) over IPC, so the supervisor's
session can mutate the DAG it lives inside.

The plugin is independent of the orchestrator process — it just needs
the orchestrator's IPC address. Multiple supervisor agents can share
one orchestrator.

## 6. Event Subscriptions

The orchestrator subscribes to the daemon's typed events via the SDK.
Each subscription is a per-session attach (`client.attach_session(id)`)
plus event-type filtering on the client side.

### 6.1 Session attach lifecycle

When the dispatcher creates an employee or reviewer session, it
attaches its client to the new session id immediately so it sees
every event from creation onward:

```python
session_id = await client.create_session(
    profile=node.assigned_profile,
    tenant_id=project.tenant_id,
)
await client.attach_session(session_id)
self._session_to_node[session_id] = node.id
```

The orchestrator either holds N `IPCClient` instances (one per
attached session, today's surface) or uses the multi-session-per-client
extension flagged in the multi-tenancy doc.  Either works; pick one
based on connection-count budget.

### 6.2 Handling `AgentCompletedEvent`

The core transition.  Fired when an employee invokes
`signal_completion` (`shared/lifecycle_tools.py:295`).  The handler:

1. Looks up the node by `event.session_id → self._session_to_node`.
2. CAS-transitions `processing → completed` under the store lock.
3. Persists the typed `payload` as the node's `result`, copies
   `event.token_usage` into `node.token_usage`, computes `cost_usd`
   from the provider rate.
4. Calls `ReviewerCoordinator.review(node)`.

The `signal_completion` payload schema for employee sessions is the
project's per-task schema (declared by the supervisor at decompose
time).  Reviewer sessions use a separate schema (§6.4).

### 6.3 Handling reviewer completion

Reviewer sessions complete with a typed payload
`{decision: "accept"|"reject", feedback: str}`.  The same
`AgentCompletedEvent` handler routes by session role (kept in
`self._session_to_role` alongside `self._session_to_node`):

- `decision == accept`: CAS `completed → accepted`, then run forward
  dependency resolution (§7.2) and AND-propagation (§7.3).
- `decision == reject`:
  - If `review_count < k_retry` (default 3): CAS `completed → processing`,
    increment counter, `client.send_message(employee_session_id, feedback_prompt)`
    where `feedback_prompt` includes the reviewer's feedback prepended
    to the original task description.
  - Else: CAS `completed → failed`, cascade cancel (§7.4).

### 6.4 Reviewer dispatch

```python
async def review(node: TaskNode) -> None:
    initial_prompt = textwrap.dedent(f"""
        Review this work and respond by calling signal_completion with
        a payload of {{ decision: "accept" | "reject", feedback: str }}.

        Original task: {node.description}
        Result: {json.dumps(node.result, indent=2)}
    """)
    session_id = await client.create_session(
        profile=node.reviewer_profile,         # supervisor's choice
        tenant_id=project.tenant_id,           # same tenant as the project
        completion_payload_schema=REVIEW_SCHEMA,
    )
    await client.attach_session(session_id)
    self._session_to_node[session_id] = node.id
    self._session_to_role[session_id] = "reviewer"
    await client.send_message(session_id, initial_prompt)
```

Reviewer profile choice is per-decomposition (a `code-reviewer` for
code tasks, a `design-reviewer` for design, etc.) — the supervisor
declares it via `task_decompose`.  This is not a system constraint;
it's the supervisor's policy.

### 6.5 Optional: HandoffGate event subscription

If premium is installed and reactor-spawned sessions exist (e.g. a
memory-advisor running on an employee), the orchestrator can subscribe
to gate events to fold those sessions into its accounting:

```python
attacher = GateAttacher(client, kind_filter={"memory-advisor"})
# Now reactor-spawned advisor sessions auto-attach; their token usage
# rolls into the project's cost via the same AgentCompletedEvent path.
```

This is purely additive — the orchestrator works with or without
premium reactors.  When a gate event signals that a reactor spawned a
session related to one of the orchestrator's employees (visible via
`intent.started_by_session`), the orchestrator can attribute the
reactor session's cost to the originating node's project.

## 7. Concurrency Model

### 7.1 Lock + CAS

All FSM transitions are CAS under the store lock:

```python
def transition(node_id: str, expected: TaskState, target: TaskState) -> bool:
    with self._lock:
        node = self._nodes[node_id]
        if node.state != expected:
            return False                       # someone else won
        node.state = target
        node.updated_at = datetime.now()
        self._persist_locked()
        return True
```

Callers that race (e.g. two siblings both trying to promote the parent on
near-simultaneous completion) are serialised by the lock; only the first
sees `expected == pending_promotion` and the rest no-op. This is the
"Schedule Idempotency" invariant from OMC.

### 7.2 Forward dependency resolution

When a node enters `accepted`:

```python
def resolve_forward(accepted_id: str) -> List[str]:
    with self._lock:
        ready = []
        for dependent in self._reverse_deps[accepted_id]:
            n = self._nodes[dependent]
            if n.state == TaskState.pending and all(
                self._nodes[d].state in (TaskState.accepted, TaskState.finished)
                for d in n.deps
            ):
                ready.append(dependent)
        return ready
# outside the lock — calls inject_prompt / create_headless_session
for nid in ready:
    dispatcher.dispatch(nid)
```

### 7.3 AND-propagation

After every `accepted` transition, the scheduler (not the store) walks
up the parent chain.  No synthetic events: the scheduler's
`accept_node` method calls `maybe_promote_parent` directly under the
lock.

```python
def accept_node(self, node_id: str) -> None:
    with self._lock:
        if not self._cas_transition(node_id, "completed", "accepted"):
            return
        # While holding the lock, recursively check parent promotion.
        nid = node_id
        while True:
            n = self._nodes[nid]
            if n.parent_id is None:
                break
            parent = self._nodes[n.parent_id]
            if all(
                self._nodes[c].state in ("accepted", "finished")
                for c in parent.children
            ):
                # Interior node: skip review (no result of its own,
                # the AND of children IS the result).  Promote
                # straight through completed → accepted.
                self._cas_transition(parent.id, "processing", "accepted")
                nid = parent.id
            else:
                break
        # Collect dependents to dispatch outside the lock.
        ready = self._collect_ready_dependents(node_id)
    for nid in ready:
        self._dispatcher.dispatch(nid)
```

Interior nodes don't get reviewed because their "result" is just the
conjunction of accepted children — there's nothing new for a reviewer
to evaluate.  Only leaf-node accepts go through the reviewer.

### 7.4 Cascade cancel

When a node enters `failed`, every transitive dependent is cancelled:

```python
def cascade_cancel(failed_id: str) -> List[str]:
    """Returns session IDs the orchestrator should stop, outside the lock."""
    sessions_to_stop = []
    with self._lock:
        stack = list(self._reverse_deps[failed_id])
        while stack:
            nid = stack.pop()
            n = self._nodes[nid]
            if n.state in TERMINAL_STATES:
                continue
            n.state = TaskState.cancelled
            stack.extend(self._reverse_deps[nid])
            if n.employee_session_id and n.state == TaskState.processing:
                sessions_to_stop.append(n.employee_session_id)
        self._persist_locked()
    return sessions_to_stop
# Outside the lock:
for sid in cascade_cancel(failed_id):
    await client.stop(sid)
```

### 7.5 Why no singleton

A previous draft assumed this needed singleton coordination.  It does
not.  Singletons matter when a daemon-singleton reactor's handler can
fire concurrently from multiple events and dispatch overlapping work;
the gate doc covers that case (`docs/design/handoff-gate-api.md`).

The orchestrator is a single process holding a single in-memory DAG
behind a single `threading.RLock`.  Concurrent event arrivals
serialise through the lock; CAS at every FSM transition guarantees
idempotency.  No cross-process coordination is needed.

Multiple orchestrator processes against one daemon are fine — each
owns a *different* DAG, so they don't share state and don't need to
coordinate.

## 8. Invariants

The seven OMC invariants, mapped to enforcement points:

| # | Invariant | Enforced by |
|---|---|---|
| 1 | DAG acyclic | `add_dependency` runs incremental DFS from the new edge's target back to its source — O(V+E) per insertion, fine at project scale |
| 2 | Mutual exclusion (1 task per employee) | The orchestrator dispatches at most one `processing` task per employee session at a time, and `client.send_message` queues mid-turn anyway |
| 3 | Schedule idempotency | CAS in `_cas_transition()` |
| 4 | Review termination | `review_count >= k_retry` ⇒ `failed` |
| 5 | Cascade completeness | `cascade_cancel()` walks transitive closure of `_reverse_deps` |
| 6 | Dependency completeness | Forward resolve runs after every `accepted` |
| 7 | Recovery correctness | Crash recovery (§10) reattaches surviving sessions instead of re-dispatching |

## 9. Failure Handling

### 9.1 Task timeout

Watchdog thread (interval = 30s) scans for nodes where
`state == processing and now() > timeout_at`.  Transition to `failed`,
cascade cancel, call `client.stop(employee_session_id)`.

### 9.2 Bounded retry

Tracked per node in `review_count`. Default `k_retry = 3`. Configurable per
node via the decomposition payload.

### 9.3 Deadlock detection

After every transition: if no node is in `processing`, no node is in
`pending` with all deps resolved, and the root is not `finished`, the
project is **deadlocked**. Mark the root `failed` and emit a structured
`TaskGraphDeadlocked` event on the orchestrator's local event bus
(consumed by whatever rendering or reporting the orchestrator pairs
with).

### 9.4 Cost budget

Optional. The scheduler accumulates `cost_usd` (computed from
`AgentCompletedEvent.token_usage` and the provider rate). When
`sum(costs) > budget`, the watchdog pauses dispatch (no new
`pending → processing` transitions) until a human resumes via the
orchestrator's API.

## 10. Crash Recovery

On orchestrator startup, the recovery routine reads every persisted
project file and reconciles in-memory state with what the daemon
reports:

1. Load every `<orchestrator_state>/*.json`.
2. For each project, query `client.list_sessions(tenant_id=project.tenant_id)`
   to learn which employee/reviewer sessions still exist.
3. For each node in `processing`:
   - **If the employee session still exists**: `attach_session` to it
     and resume observation. Don't re-dispatch — the model may have
     already done partial work.
   - **If the session is gone**:
     - If `node.idempotent`: transition back to `pending` for re-dispatch.
     - Else: transition to `failed` with reason `"session_lost"` and
       cascade cancel. The supervisor's job to decide if a manual
       redo is appropriate.
4. For each node in `completed` whose reviewer session is gone:
   re-spawn a reviewer (review work is always idempotent — same
   `(task, result)` pair always gets the same decision distribution).
5. For each node in `pending` whose deps are all `accepted`/`finished`,
   dispatch normally.

This eliminates the "naive re-dispatch causes double execution"
concern from the v1 sketch.  The default for `idempotent` is **False**
so opt-in, not silent assumption.

## 11. Public API

The orchestrator exposes a typed API consumed by:
- The model-callable tool plugin (`task_decompose`, `task_assign`,
  `task_status`, `task_cancel`) loaded into the supervisor's session.
- An optional CLI for human operators.
- Whatever rendering layer is co-located with the orchestrator process.

```python
class TaskGraphAPI:
    def create_project(
        self,
        root_description: str,
        tenant_id: str,
        supervisor_session_id: str,
        budget_usd: Optional[float] = None,
    ) -> str:
        """Create a project. The orchestrator's identity must hold
        (write, session) on tenant_id."""
    
    def decompose(
        self,
        parent_id: str,
        children: List[ChildSpec],   # description, profile, reviewer_profile, deps, idempotent
    ) -> List[str]: ...
    
    def assign(self, node_id: str, employee_session_id: str) -> None:
        """Bind a node to an existing session. Used when the supervisor
        wants to reuse an employee across multiple tasks rather than
        spawning a new session per task."""
    
    def get_node(self, node_id: str) -> TaskNode: ...
    def get_subtree(self, root_id: str) -> Dict[str, TaskNode]: ...
    def cancel(self, node_id: str) -> None: ...
    def status(self, project_id: str) -> ProjectStatus: ...
    def pause(self, project_id: str) -> None: ...
    def resume(self, project_id: str) -> None: ...
```

`ChildSpec` carries the supervisor's per-decomposition policy:

```python
@dataclass
class ChildSpec:
    description: str
    profile: str                          # employee profile to spawn
    reviewer_profile: str                 # reviewer profile for this task
    deps: List[str] = field(default_factory=list)
    idempotent: bool = False              # affects crash recovery (§10)
    timeout_seconds: Optional[int] = None
    completion_payload_schema: Optional[Dict[str, Any]] = None
```

## 12. Events (orchestrator-local)

The orchestrator publishes events on its own internal event bus
(consumed by co-located rendering code) — not on the daemon's
EventBus.  The daemon doesn't know about the DAG.

| Event | When |
|---|---|
| `TaskNodeCreated` | New node inserted |
| `TaskNodeStateChanged` | Any FSM transition |
| `TaskNodeReviewed` | Reviewer returned a decision |
| `TaskGraphDeadlocked` | Watchdog detected stall |
| `TaskGraphProjectFinished` | Root reached `finished` |
| `TaskGraphBudgetExhausted` | Project hit its cost budget; dispatch paused |

For multi-process consumers (separate dashboard reading a remote
orchestrator's state), the orchestrator can optionally expose an
HTTP/WS endpoint that streams these events.  This is **out of scope
for v1** — see open question 1.

## 13. Configuration

The orchestrator reads `<orchestrator_state>/orchestrator.json`:

```jsonc
{
  "daemon_socket": "/tmp/jaato.sock",
  "service_token_file": "~/.jaato/service_tokens.json",
  "state_dir": "~/.jaato/orchestrator/default/",
  "k_retry": 3,
  "task_timeout_seconds": 3600,
  "watchdog_interval_seconds": 30,
  "default_cost_budget_usd": null,
  "subscribe_gates": true                     // §6.5
}
```

## 14. Worked Example

Three-node project: build a feature.

```
root: "Add login API"   [supervisor: COO,          tenant: acme]
 ├─ A: "Design schema"      [profile: architect,  reviewer: design-reviewer]
 ├─ B: "Implement endpoint" [profile: backend,    reviewer: code-reviewer, deps: A]
 └─ C: "Write tests"        [profile: qa,         reviewer: code-reviewer, deps: B]
```

Flow:

1. COO (running in a session attached to the orchestrator) calls
   `task_decompose(root, [A_spec, B_spec, C_spec])` via the tool
   plugin.  A, B, C inserted as `pending`.
2. Dispatcher sees A ready.  Calls
   `client.create_session(profile="architect", tenant_id="acme")`,
   attaches, sends the prompt.  A → `processing`.
3. Architect calls `signal_completion(payload=...)`.
   `AgentCompletedEvent` fires; orchestrator transitions A →
   `completed`, copies token usage, computes cost.
4. ReviewerCoordinator spawns a `design-reviewer` session, attaches,
   sends the review prompt.  Reviewer signals
   `{decision: "accept", feedback: "..."}`.  A → `accepted`.
5. Forward dependency resolution finds B ready.  Dispatcher creates a
   backend session, sends the prompt.  B → `processing`.
6. Backend's first attempt fails the `code-reviewer`'s review.
   `review_count=1`.  Orchestrator calls
   `client.send_message(backend_session_id, feedback_prompt)`.  Second
   attempt accepted.
7. C runs identically, accepted.
8. AND-propagation: A, B, C all accepted ⇒ root promotes to
   `accepted` (interior, no review needed) → `finished`.
   `TaskGraphProjectFinished` emitted on the orchestrator's event
   bus.

## 15. Open Questions

1. **Multi-client DAG visibility.** v1 assumes the rendering layer is
   co-located with the orchestrator process.  A separate dashboard
   would need either (a) the orchestrator exposing an HTTP/WS endpoint
   that streams `TaskGraph*Events`, or (b) the orchestrator publishing
   selected events through a session it owns onto the daemon's
   EventBus so attached clients see them.  Pick one in v2 when the
   need arises; (a) is more decoupled, (b) requires no extra ports.

2. **Cross-session memory on retry.** When a child fails review and
   gets re-prompted, how much of the failed attempt's context
   propagates?  Suggest: re-prompt via `send_message` to the same
   session (history naturally preserved).  If retry is in a *fresh*
   session, optionally seed `initial_history` with the failed turn.

3. **Recruitment integration.** When `task_decompose` requests an
   employee profile that doesn't exist, should the orchestrator spawn
   a hiring subagent (OMC's `α_recruit`)?  v2 hook.

4. **Streaming partial results.** Should `accepted` require the full
   payload, or can it accept partial deliverables?  Defer to reviewer
   prompt and the per-task completion schema.

5. **Orchestrator restart while supervisor is mid-decompose.**  A
   supervisor that was inside a `task_decompose` tool call when the
   orchestrator died will see a tool error on next turn.  The
   supervisor needs guidance in its system prompt to retry safely.
   Document the retry pattern in the tool plugin's docstring.

## 16. Out of Scope

- Multi-daemon DAG sharing (would require a distributed coordination
  service — separate design).
- Replacing `TodoWrite` — the two coexist; `TodoWrite` is per-session
  scratchpad, the task graph is multi-session orchestration.
- A Talent Market for reusable task templates.
- RL-trained decomposition policies.
- Building the orchestrator's HTTP/WS dashboard endpoint (open
  question 1).

## 17. References

- OneManCompany paper, §2.2.4 *DAG-based Task Decomposition and Execution*
- `docs/design/multi-tenancy.md` — tenant scoping, RBAC, service
  identities consumed throughout this doc
- `docs/design/handoff-gate-api.md` — gate event subscription used by
  the optional reactor-cost folding in §6.5
- `docs/design/daemon-extensions.md` — public extension surface for
  reference (the orchestrator does not use it directly)
- `jaato-sdk/jaato_sdk/client/ipc.py:937` — `attach_session`
- `jaato-sdk/jaato_sdk/events.py:293` — `AgentCompletedEvent`
- `jaato-server/shared/lifecycle_tools.py` — `signal_completion` typed
  payload contract
- `jaato-server/server/session_manager.py:446` — `attached_clients`
  fan-out (the mechanism that makes orchestrator observation work)
