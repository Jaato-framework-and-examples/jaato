# Task-Graph Reactor

A functional specification for a DAG-based task scheduler implemented as a
jaato daemon extension on top of the existing reactor surface.

## 1. Background

The OneManCompany paper (*From Skills to Talent: Organising Heterogeneous
Agents as a Real-World Company*, arXiv:2604.22446) proposes a six-interface
"organisational layer" for agent orchestration. Five of the six interfaces map
directly onto existing jaato surfaces:

| OMC interface | jaato equivalent |
|---|---|
| `Execution` | `JaatoSession.send_message` + provider `send_message` |
| `Event` | `EventBus`, typed events in `jaato_sdk/events.py` |
| `Storage` | session history, `TokenLedger`, `.jaato/logs/` |
| `Context` | system instructions, GC plugins, `PresentationContext` |
| `Lifecycle` | plugin auto-wiring, `LifecycleTools.signal_completion`, enrichment plugins |

The remaining interface — `Task` — is the one jaato does not yet have. This
document specifies how to build it as a reactor extension, without modifying
`jaato-server` itself.

## 2. Goals & Non-Goals

### Goals

- A persistent, crash-recoverable DAG of tasks with explicit dependencies.
- An FSM-governed lifecycle per node with a supervisor accept/reject gate.
- AND-tree completion semantics: a parent is resolved iff all children are.
- Bounded retry, task timeouts, deadlock detection.
- Driven entirely from EventBus events; no changes to `jaato-server` public APIs.

### Non-goals (initial release)

- Multi-daemon / peer-clustered DAG state (single-daemon only at first).
- An RL-trained decomposition policy (the supervising agent is the policy).
- A community marketplace of task templates.
- UI for visualising the DAG (events are emitted; presentation is the client's job).

## 3. Concepts

| Term | Definition |
|---|---|
| **Node** | A single task. Owns a description, an assigned employee, an FSM state, a result, and a cost. |
| **Employee** | A `JaatoSession` that executes assigned tasks. One session per employee. |
| **Decomposition edge** | "Parent was decomposed into child." Forms a strict tree. |
| **Dependency edge** | "Child cannot start until parent is accepted." Can cross sibling branches. |
| **Reviewer** | A jaato session (typically loaded from a `reviewer` profile) that produces an `accept` or `reject` decision on a completed node. |
| **Supervisor** | The agent that decides decomposition and assignment for a node — typically the parent's owner. |

## 4. Data Model

### 4.1 Node schema

```python
@dataclass
class TaskNode:
    id: str                              # uuid
    parent_id: Optional[str]             # None for root
    description: str
    employee_session_id: Optional[str]   # None for unassigned / interior
    state: TaskState                     # see FSM below
    result: Optional[Dict[str, Any]]     # typed completion payload
    cost: float                          # accumulated tokens / dollars
    created_at: datetime
    updated_at: datetime
    review_count: int = 0                # bounded retry counter
    deps: List[str] = field(default_factory=list)        # node IDs
    children: List[str] = field(default_factory=list)    # node IDs
    timeout_at: Optional[datetime] = None
```

### 4.2 FSM

```
                 ┌──────────────┐
                 ▼              │  retry (review_count < k_retry)
   pending ──► processing ──► completed ──review pass──► accepted ──► finished
      │            │              │
      │            │              └──review fail──► (back to processing)
      │            │
      │            └──► holding (paused, e.g. blocked on dep)
      │
      ├──► failed (timeout / k_retry exhausted)
      └──► cancelled (cascade from a failed dep)
```

Terminal states: `finished`, `failed`, `cancelled`. Only `finished` counts as
"resolved" for AND-semantics.

### 4.3 Persistence

- Single JSON file per project: `.jaato/task_graphs/<project_id>.json`.
- Format: `{ "nodes": {id: TaskNode}, "edges": {...}, "version": 1 }`.
- Writes go through a `TaskGraphStore` that holds an in-memory copy plus a
  `threading.RLock`. Each public mutator method:
  1. Acquires the lock.
  2. Mutates the in-memory dict.
  3. Atomically rewrites the JSON file (`tmp + os.replace`).
  4. Releases the lock.
- The lock is the only concurrency primitive needed in single-daemon mode (see §7).

## 5. Architecture

The task-graph scheduler is a **daemon extension** registered via the
`jaato.extensions` entry point (see `docs/design/daemon-extensions.md`). It
ships as its own package — it does not require changes to `jaato-server`.

```
┌─────────────────────────────────────────────────────────────┐
│  jaato-server (unchanged)                                   │
│                                                             │
│   SessionManager        EventBus       LifecycleTools       │
│      │                    │               │                 │
│      │ create_headless_session()          │ on_agent_       │
│      │ inject_prompt_to_session()         │ completed       │
└──────┼────────────────────┼────────────────┼────────────────┘
       │                    │                │
       ▼                    ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│  jaato-task-graph (extension)                               │
│                                                             │
│   ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│   │ TaskGraph   │   │  Scheduler   │   │  Watchdog       │  │
│   │ Store       │◀──│              │   │  (bg thread)    │  │
│   │ (locked)    │   │              │   │                 │  │
│   └─────────────┘   └──────┬───────┘   └────────┬────────┘  │
│                            │                    │           │
│                  ┌─────────┴────────┐           │           │
│                  ▼                  ▼           ▼           │
│           Reactor handlers     Reviewer        Timeouts,    │
│           (per-event)          spawning        deadlocks    │
└─────────────────────────────────────────────────────────────┘
```

### 5.1 Components

- **`TaskGraphStore`** — owns the in-memory DAG and the lock; persists to disk.
- **`Scheduler`** — pure logic: given an event, decide which nodes transition
  state, which dependents are now ready, which to dispatch.
- **`Dispatcher`** — calls `inject_prompt_to_session` (existing employee) or
  `create_headless_session` (new employee) to start work.
- **`Watchdog`** — background thread; scans the graph every N seconds for
  timeouts and deadlocks.
- **`ReviewerCoordinator`** — spawns a reviewer session for each `completed`
  node and consumes its `accept`/`reject` payload.

## 6. Reactor Handlers

The extension registers a single session hook (see `daemon-extensions.md`
§3) that fires for every newly-initialised session. The hook attaches
per-session callbacks on the session's `_ui_hooks` so the extension observes
completion events from every session in the daemon.

### 6.1 `on_session_ready(server)`

Triggered when a new session is initialised.

```python
def on_session_ready(server: JaatoServer) -> None:
    session = server.get_session()
    # Wrap the existing hooks object so on_agent_completed forwards
    # both to the original target (TUI/IPC client) and to our scheduler.
    original = session._ui_hooks
    session._ui_hooks = _ChainedHooks(original, scheduler)
```

### 6.2 `on_agent_completed(agent_id, payload, ...)`

The core transition. Called by `LifecycleTools._execute_signal_completion`
(`shared/lifecycle_tools.py:295`) when an employee invokes
`signal_completion`. The reactor:

1. Looks up the node owned by this employee session.
2. CAS-transitions `processing → completed` under the store lock.
3. Persists `payload` as the node's `result` and updates accumulated `cost`.
4. Spawns a reviewer (§6.4) for this node.

### 6.3 `on_review_completed(node_id, decision, feedback)`

Triggered when a reviewer session signals completion.

- `decision == accept`: CAS `completed → accepted`, then run
  **forward dependency resolution** (§7.2) and **AND-propagation** (§7.3).
- `decision == reject`:
  - If `review_count < k_retry` (default 3): CAS `completed → processing`,
    increment counter, re-inject the prompt with reviewer feedback prepended.
  - Else: CAS `completed → failed`, run **cascade cancel** (§7.4).

### 6.4 Reviewer dispatch

```python
def spawn_reviewer(node: TaskNode) -> None:
    initial_prompt = textwrap.dedent(f"""
        Review this work and respond by calling signal_completion with
        a payload of {{ decision: "accept" | "reject", feedback: str }}.

        Original task: {node.description}
        Result: {json.dumps(node.result, indent=2)}
    """)
    session_id = session_manager.create_headless_session(
        agent_name="reviewer",
        profile_name="reviewer",
        initial_prompt=initial_prompt,
        session_name=f"review-{node.id}",
    )
    scheduler.bind_reviewer(node.id, session_id)
```

The reviewer profile must declare a `completion_payload_schema` so its
`signal_completion` is typed:

```jsonc
// .jaato/completion_schemas/review.json
{
  "type": "object",
  "required": ["decision"],
  "properties": {
    "decision": { "enum": ["accept", "reject"] },
    "feedback": { "type": "string" }
  }
}
```

The scheduler subscribes to the reviewer's `on_agent_completed` exactly as
it does for any other session; the only difference is that completed
reviewer sessions route to `on_review_completed` instead of being treated
as employee work.

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

After every `accepted` transition, walk up the parent chain:

```python
def maybe_promote_parent(child_id: str) -> None:
    with self._lock:
        child = self._nodes[child_id]
        if child.parent_id is None:
            return
        parent = self._nodes[child.parent_id]
        if all(
            self._nodes[c].state in (TaskState.accepted, TaskState.finished)
            for c in parent.children
        ):
            if parent.state == TaskState.processing:
                parent.state = TaskState.completed
                # cascading review will fire via on_agent_completed equivalent
```

Because parent nodes have no employee session, a synthetic
"interior-node-completed" event is enqueued onto the scheduler from inside
the store; the reviewer coordinator handles it identically to a leaf.

### 7.4 Cascade cancel

When a node enters `failed`, every transitive dependent is cancelled:

```python
def cascade_cancel(failed_id: str) -> None:
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
                # Fire-and-forget stop request on the employee.
                request_stop_outside_lock.append(n.employee_session_id)
```

### 7.5 Why no singleton

A previous draft assumed this needed singleton coordination by analogy with
the `memory-advisor` reactor. It does not. The advisor's singleton exists
because a long-running worker mutates curated memory based on a queue of
raw entries, and concurrent workers can produce conflicting writes. The
task graph instead uses CAS at the FSM-transition boundary: any number of
concurrent reactor handlers can fire, but only one wins each transition,
and the rest no-op. Idempotent inserts (keyed by node id) handle the same
concern at insertion time.

Cross-process serialisation only becomes necessary if you later want
multiple daemons to share one DAG file. That is out of scope for v1.

## 8. Invariants

The seven OMC invariants, mapped to enforcement points:

| # | Invariant | Enforced by |
|---|---|---|
| 1 | DAG acyclic | `add_dependency` runs DFS cycle check before insertion |
| 2 | Mutual exclusion (1 task per employee) | `JaatoSession.inject_prompt` already queues mid-turn |
| 3 | Schedule idempotency | CAS in `transition()` |
| 4 | Review termination | `review_count >= k_retry` ⇒ `failed` |
| 5 | Cascade completeness | `cascade_cancel()` walks transitive closure |
| 6 | Dependency completeness | `resolve_forward()` runs after every `accepted` |
| 7 | Recovery correctness | Startup reset (§10) |

## 9. Failure Handling

### 9.1 Task timeout

`Watchdog` thread (interval = 30s) scans for nodes where
`state == processing and now() > timeout_at`. Transition to `failed`,
fire `cascade_cancel`, request stop on the employee session.

### 9.2 Bounded retry

Tracked per node in `review_count`. Default `k_retry = 3`. Configurable per
node via the decomposition payload.

### 9.3 Deadlock detection

After every transition: if no node is in `processing`, no node is in
`pending` with all deps resolved, and the root is not `finished`, the
project is **deadlocked**. Mark the root `failed` and emit a structured
`TaskGraphDeadlocked` event for the supervising agent or human to inspect.

### 9.4 Cost budget

Optional. The scheduler accumulates `node.cost` from completion-payload
token counts. When `sum(costs) > budget`, the watchdog pauses dispatch
(no new `pending → processing` transitions) until a human resumes.

## 10. Crash Recovery

On daemon startup the extension's `start()`:

1. Loads every `.jaato/task_graphs/*.json`.
2. For each node in `processing`, transitions back to `pending` (the
   employee's session was killed; re-dispatch is safe because tools are
   idempotent at the user's discretion — same contract jaato already
   assumes).
3. For each node in `pending` whose deps are all `accepted`/`finished`,
   dispatches via `Dispatcher`.
4. Reviewer sessions in flight at crash time are dropped; the scheduler
   re-enqueues a fresh review for any node still in `completed`.

## 11. Public API (extension surface)

The extension exposes a small typed API to callers (e.g., the supervising
agent's tool plugin, or a CLI command):

```python
class TaskGraphAPI:
    def create_project(self, root_description: str, owner_session_id: str) -> str: ...
    def decompose(
        self,
        parent_id: str,
        children: List[ChildSpec],   # description, employee, deps
    ) -> List[str]: ...
    def assign(self, node_id: str, employee_session_id: str) -> None: ...
    def get_node(self, node_id: str) -> TaskNode: ...
    def get_subtree(self, root_id: str) -> Dict[str, TaskNode]: ...
    def cancel(self, node_id: str) -> None: ...
    def status(self, project_id: str) -> ProjectStatus: ...
```

A companion `task_graph` tool plugin can wrap these as model-callable
tools (`task_decompose`, `task_assign`, `task_status`) so a COO-style
supervising agent can drive the graph from inside a regular jaato session.

## 12. Events Emitted

The extension publishes typed events on the EventBus so clients (TUI,
dashboards) can render the DAG live:

| Event | When |
|---|---|
| `TaskNodeCreated` | New node inserted |
| `TaskNodeStateChanged` | Any FSM transition |
| `TaskNodeReviewed` | Reviewer returned a decision |
| `TaskGraphDeadlocked` | Watchdog detected stall |
| `TaskGraphProjectFinished` | Root reached `finished` |

These follow the existing pattern in `jaato_sdk/events.py` (pydantic
dataclasses, registered in `EVENT_TYPE_MAP`).

## 13. Configuration

`.jaato/task_graph.json`:

```jsonc
{
  "k_retry": 3,
  "task_timeout_seconds": 3600,
  "watchdog_interval_seconds": 30,
  "cost_budget_usd": null,
  "reviewer_profile": "reviewer",
  "default_completion_schema": "completion_schemas/task_result.json"
}
```

## 14. Worked Example

Three-node project: build a feature.

```
root: "Add login API"  [owner: COO]
 ├─ A: "Design schema"      [owner: architect]
 ├─ B: "Implement endpoint" [owner: backend, deps: A]
 └─ C: "Write tests"        [owner: qa,      deps: B]
```

Flow:

1. `create_project("Add login API", coo_session_id)` creates root.
2. COO calls `task_decompose(root, [A, B, C])`. A, B, C inserted as
   `pending`. A has no deps, B depends on A, C depends on B.
3. `Dispatcher` sees A is ready. `inject_prompt_to_session(architect, A.desc)`.
   A → `processing`.
4. Architect calls `signal_completion`. `on_agent_completed` fires.
   A → `completed`. Reviewer spawned.
5. Reviewer says `accept`. A → `accepted`. `resolve_forward(A)` finds B
   ready. `inject_prompt_to_session(backend, B.desc)`. B → `processing`.
6. Backend's first attempt fails review. `review_count = 1`. B re-injected
   with feedback. Eventually accepted.
7. C runs, accepted.
8. AND-propagation: A, B, C all accepted ⇒ root → `completed` →
   reviewer → `accepted` → `finished`. `TaskGraphProjectFinished` emitted.

## 15. Open Questions

1. **Interior-node owners.** Does the COO own the root forever, or does
   ownership shift on decomposition? Affects who reviews children.
2. **Reviewer model.** Single reviewer profile vs per-domain reviewers
   (code-reviewer, design-reviewer)? Lean toward per-domain, selectable
   from the decomposition payload.
3. **Cross-session memory.** When a child fails and gets re-decomposed,
   how much of the failed attempt's context propagates? Suggest:
   `initial_history` carries the last N turns of the failed employee.
4. **Recruitment integration.** When `task_decompose` requests an employee
   profile that doesn't exist, should the extension auto-spawn a hiring
   subagent (OMC's `α_recruit`)? Out of scope for v1; can be a v2 hook.
5. **Streaming partial results.** Should `accepted` require the full
   payload, or can it accept partial deliverables? Defer to reviewer.

## 16. Out of Scope

- Multi-daemon DAG sharing (would require `flock` or moving to SQLite).
- Replacing `TodoWrite` — the two coexist; `TodoWrite` is per-session
  scratchpad, the task graph is multi-session orchestration.
- A Talent Market for reusable task templates.
- RL-trained decomposition policies.

## 17. References

- OneManCompany paper, §2.2.4 *DAG-based Task Decomposition and Execution*
- `docs/design/daemon-extensions.md` — extension points used here
- `jaato-server/server/session_manager.py:1048` — `create_headless_session`
- `jaato-server/server/session_manager.py:1149` — `inject_prompt_to_session`
- `jaato-server/shared/lifecycle_tools.py` — `signal_completion` and the
  `on_agent_completed` hook surface
- `jaato-sdk/jaato_sdk/events.py:293` — `AgentCompletedEvent`
