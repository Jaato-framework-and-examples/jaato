# Multi-Server Gossip, Environment Aspect, and Reliability Policies

## 1. Problem Statement

Jaato servers today operate in isolation. A single daemon holds all sessions,
all provider credentials, all tool plugins, and all subagents. This creates
several scaling bottlenecks:

1. **Provider rate limits** — A single API key hits rate limits as agent
   concurrency grows. Different servers can hold different keys/accounts,
   multiplying effective throughput.
2. **Provider diversity** — Server A has Anthropic credentials, Server B has
   Google, Server C runs local Ollama. Subagents should route to the server
   that has the right provider for the task.
3. **Resource isolation** — Heavy tool execution (interactive shells, MCP
   servers, CLI builds) consumes CPU/memory. Isolating tool-heavy subagents on
   dedicated servers prevents interference with LLM-bound work.
4. **Concurrent sessions** — One daemon has practical limits. Distributing
   sessions across N daemons scales linearly.
5. **Geographic** — Servers in different regions can route to the
   lowest-latency provider endpoint.

**The gap:** No mechanism exists for servers to discover each other, exchange
health data, or delegate work. The main agent has no visibility into the cluster
and cannot make informed decisions about where to run subagents.

## 2. Design Decisions

Five key decisions shape this design, each chosen from alternatives explored
during brainstorming:

| Decision | Chosen | Alternatives Considered | Rationale |
|---|---|---|---|
| Inter-server communication | Gossip (peer-to-peer) | Shared registry (Redis/etcd), hub-and-spoke | No external dependencies, reuses WebSocket, fits 2-10 server scale |
| Peer authentication | Mutual TLS | Shared secret, open | Production-grade, certificate-based identity per server |
| Workspace replication | Git-based sync | Shared filesystem (NFS/SSHFS), none | Git is already present, handles file diffing, no infra dependency |
| Server selection | Model-driven (agent chooses) | System auto-routes, hybrid | Consistent with jaato's philosophy of keeping the agent in control |
| Affinity scoring | Configurable formula | Fixed formula, no scoring | Same pattern as existing `EscalationRule` — no hardcoded thresholds |

## 3. Architecture Overview

### 3.1 Peer Mesh Topology

```
                        ┌─────────────┐
                        │   Server A   │
                        │   (local)    │
                        │  anthropic   │
                        │  3 sessions  │
                        └──┬───────┬──┘
                  mTLS WS  │       │  mTLS WS
                  heartbeat│       │  heartbeat
                           │       │
              ┌────────────┘       └────────────┐
              │                                  │
              ▼                                  ▼
    ┌─────────────┐                    ┌─────────────┐
    │   Server B   │◄── mTLS WS ────►│   Server C   │
    │  (gpu-box)   │    heartbeat     │   (cloud)    │
    │    ollama    │                   │  anthropic   │
    │  1 session   │                   │  google      │
    └─────────────┘                    │  8 sessions  │
                                       └─────────────┘
```

Every server maintains persistent WebSocket connections to all known peers.
Heartbeats flow bidirectionally at a configurable interval (default: 5s).
Each server holds a complete `PeerRegistry` of last-known state for all peers.

### 3.2 Data Flow: Environment Aspect Query

```
Agent Session (Server A)                   PeerRegistry (Server A)
────────────────────────                   ────────────────────────

1. Model calls                             Gossip heartbeats arrive
   get_environment(                        continuously from B, C
     aspect="jaato_agentic_servers"        updating PeerRegistry
   )                                       cache in real time
       │
       ▼
2. Environment plugin
   reads local health
   (psutil: CPU, mem)
       │
       ▼
3. Environment plugin
   queries PeerRegistry
   for peer snapshots
       │
       ▼
4. Constructs JSON with
   self + peers +
   cluster_summary
       │
       ▼
5. Returns to model
   as tool result
       │
       ▼
6. Model reasons about
   server health, picks
   target for subagent
       │
       ▼
7. spawn_subagent(
     profile="researcher",
     server="gpu-box"
   )
```

### 3.3 Remote Subagent Delegation

```
Server A (parent)                      Server B (delegate)
─────────────────                      ─────────────────

spawn_subagent(
  server="gpu-box",
  profile="researcher",
  task="Analyze the logs"
)
    │
    ├─ Validate: Server B
    │  is TRUSTED or RECOVERING
    │
    ├─ Serialize spawn request ────►  Receive spawn request
    │  (profile, task, context)        │
    │                                  ├─ git pull workspace
    │                                  │
    │                                  ├─ Create local session
    │                                  │  with profile config
    │                                  │
    │  ◄── AgentOutputEvent ──────────├─ Agent runs turns
    │  ◄── AgentOutputEvent ──────────├─ Streaming output
    │  ◄── AgentOutputEvent ──────────│  forwarded back
    │                                  │
    │  ◄── SubagentResult ────────────├─ Agent completes
    │                                  │
    ├─ Forward output to               └─ Clean up session
    │  parent session queue
    │
    └─ Resume parent turn
       with subagent result
```

## 4. Gossip Protocol

### 4.1 Transport

The gossip protocol reuses the existing WebSocket infrastructure
(`server/websocket.py`). Each server exposes a **peer channel** on its
WebSocket endpoint, distinct from client connections. Peer connections are
identified during the WebSocket handshake by a `X-Jaato-Peer: true` header
and authenticated via mutual TLS.

### 4.2 Heartbeat Schema

```python
@dataclass
class PeerHeartbeat:
    """Sent by each server to all peers at a configurable interval."""
    server_id: str                      # Stable UUID, generated on first start
    server_name: str                    # Human-readable name from servers.json
    server_version: str                 # Package version (importlib.metadata)
    timestamp: str                      # ISO 8601 UTC

    # Capacity
    active_sessions: int
    active_agents: int                  # Including subagents
    available_providers: List[str]      # Provider names with valid credentials
    available_models: List[str]         # Models the server can serve
    tags: List[str]                     # From servers.json config

    # Health
    cpu_percent: float                  # psutil.cpu_percent()
    memory_percent: float               # psutil.virtual_memory().percent
    uptime_seconds: float

    # Reliability (self-reported)
    trust_state: str                    # "trusted", "escalated", etc.
    success_rate_1h: float              # 0.0–1.0
    escalated_tools: int                # Count of tools in ESCALATED state
```

### 4.3 Peer Lifecycle States

Each peer in the `PeerRegistry` progresses through these states based on
heartbeat reception:

```
                    heartbeat received
    ┌──────────────────────────────────────────┐
    │                                          │
    ▼                                          │
┌────────┐   no heartbeat for    ┌──────────┐ │
│HEALTHY │──(N × interval)──────►│ DEGRADED │─┘
└────────┘                       └──────────┘
    ▲                                │
    │   heartbeat received           │ no heartbeat for
    │                                │ (M × interval)
    │                                ▼
    │                          ┌──────────────┐
    └──────────────────────────│ UNREACHABLE  │
         heartbeat received    └──────────────┘
```

| State | Condition | Effect |
|---|---|---|
| `HEALTHY` | Heartbeat received within `degraded_after_missed × interval` | Full delegation allowed |
| `DEGRADED` | Heartbeat missed for `degraded_after_missed` intervals (default: 3 = 15s) | Model sees `status: "degraded"`, reliability plugin records `HEARTBEAT_MISSED` |
| `UNREACHABLE` | Heartbeat missed for `unreachable_after_missed` intervals (default: 5 = 25s) | Server removed from active peers, delegation blocked |

When a heartbeat is received from an `UNREACHABLE` peer, it transitions back
to `HEALTHY` and the reliability plugin records a recovery event.

### 4.4 mTLS Configuration

Each server needs a certificate, private key, and a shared CA certificate:

```
.jaato/certs/
├── ca.pem            # Certificate authority (shared by all servers)
├── server.pem        # This server's certificate
└── server-key.pem    # This server's private key
```

The `tls` section in `servers.json` points to these paths. Certificate
generation is outside jaato's scope — operators use `openssl`, `mkcert`, or
their organization's PKI.

## 5. Server Health Collection

### 5.1 Local Metrics

Each server collects its own health via `psutil` (already a common Python
dependency). The `ServerHealthCollector` runs as a background task, sampling
at the heartbeat interval:

```python
@dataclass
class ServerHealthSnapshot:
    """Local health metrics collected via psutil."""
    cpu_percent: float          # psutil.cpu_percent(interval=1)
    memory_percent: float       # psutil.virtual_memory().percent
    active_sessions: int        # len(session_manager.sessions)
    active_agents: int          # Sum of subagents across all sessions
    uptime_seconds: float       # time.monotonic() - start_time
    available_providers: List[str]
    available_models: List[str]
    tags: List[str]
```

### 5.2 Peer Health (from Gossip)

For peer servers, health data comes exclusively from the gossip heartbeat —
the local server never queries peers directly. The `PeerRegistry` stores
the most recent `PeerHeartbeat` for each peer along with a receive timestamp
for freshness calculation.

## 6. Environment Aspect: `jaato_agentic_servers`

### 6.1 Integration with Existing Environment Plugin

The environment plugin (`shared/plugins/environment/plugin.py`) dispatches
aspects via the `VALID_ASPECTS` list and `if aspect in (name, "all")` pattern.
The new aspect follows the same pattern with one exception:
**it is excluded from `aspect="all"`** because it requires network data
(peer registry queries) and is only relevant when the model is planning
subagent delegation.

```python
VALID_ASPECTS = [
    "os", "shell", "arch", "cwd", "terminal", "context",
    "session", "datetime", "network", "jaato_agentic_servers", "all"
]

def _get_environment(self, args):
    aspect = args.get("aspect", "all")
    result = {}

    # ... existing aspects ...

    # Excluded from "all" — model queries this explicitly
    if aspect == "jaato_agentic_servers":
        result["jaato_agentic_servers"] = self._get_agentic_servers()

    # Flatten for single-aspect requests
    if aspect != "all" and len(result) == 1:
        result = result[aspect]
    return json.dumps(result, indent=2)
```

The `_get_agentic_servers()` method queries the `PeerRegistry` (injected via
the plugin's session reference — using the existing thread-local
`_thread_local.session` pattern to avoid cross-agent contamination).

### 6.2 Full JSON Response Schema

```json
{
  "self": {
    "name": "local",
    "server_id": "srv_a1b2c3",
    "transport": "ipc",
    "address": "/tmp/jaato.sock",
    "providers": ["anthropic", "google_genai"],
    "models": ["claude-sonnet-4-5", "gemini-2.5-flash"],
    "tags": ["local", "tools", "gpu"],
    "health": {
      "status": "healthy",
      "cpu_percent": 23.5,
      "memory_percent": 41.2,
      "active_sessions": 2,
      "active_agents": 5,
      "uptime_seconds": 86400
    },
    "reliability": {
      "trust_state": "trusted",
      "success_rate_1h": 0.97,
      "escalated_tools": 0,
      "affinity_score": 0.92
    }
  },
  "peers": [
    {
      "name": "gpu-box",
      "server_id": "srv_d4e5f6",
      "transport": "ws",
      "address": "ws://gpu-box.local:8080",
      "providers": ["ollama"],
      "models": ["qwen3:32b", "llama3.3:70b"],
      "tags": ["gpu", "local-models"],
      "health": {
        "status": "healthy",
        "cpu_percent": 65.0,
        "memory_percent": 78.3,
        "active_sessions": 1,
        "active_agents": 2,
        "uptime_seconds": 3600
      },
      "reliability": {
        "trust_state": "trusted",
        "success_rate_1h": 0.92,
        "escalated_tools": 1,
        "affinity_score": 0.95
      },
      "last_heartbeat_seconds_ago": 3.2
    },
    {
      "name": "cloud",
      "server_id": "srv_g7h8i9",
      "transport": "ws",
      "address": "ws://jaato.mycompany.com:8080",
      "providers": ["anthropic", "google_genai"],
      "models": ["claude-sonnet-4-5", "gemini-2.5-flash", "gemini-2.5-pro"],
      "tags": ["cloud", "high-capacity"],
      "health": {
        "status": "degraded",
        "cpu_percent": 91.0,
        "memory_percent": 85.7,
        "active_sessions": 8,
        "active_agents": 23,
        "uptime_seconds": 259200
      },
      "reliability": {
        "trust_state": "escalated",
        "success_rate_1h": 0.71,
        "escalated_tools": 4,
        "escalation_reason": "High failure rate on anthropic provider (rate limit)",
        "affinity_score": 0.52
      },
      "last_heartbeat_seconds_ago": 2.1
    }
  ],
  "cluster_summary": {
    "total_servers": 3,
    "healthy": 2,
    "degraded": 1,
    "unreachable": 0,
    "total_active_sessions": 11,
    "total_active_agents": 30
  }
}
```

**Field semantics:**

| Field | Type | Description |
|---|---|---|
| `self` | object | The server this agent is running on (always present) |
| `peers` | array | Live data from gossip protocol (empty if no peers configured) |
| `cluster_summary` | object | Aggregate stats for quick model reasoning |
| `health.status` | string | Computed enum: `healthy` / `degraded` / `unreachable` |
| `reliability.trust_state` | string | Matches `TrustState` enum values from `reliability/types.py` |
| `reliability.affinity_score` | float | 0.0–1.0, computed from configurable formula |
| `reliability.escalation_reason` | string | Present only when `trust_state != "trusted"` |
| `last_heartbeat_seconds_ago` | float | Peers only — lets the model judge data freshness |

### 6.3 When It's Computed

The aspect is computed **on demand** when the model calls
`get_environment(aspect="jaato_agentic_servers")`. It is not cached or
pre-computed because:

- Health data changes continuously (CPU, memory, session counts)
- The gossip protocol already maintains an up-to-date `PeerRegistry` in memory
- The cost of reading local `psutil` + iterating the peer registry is negligible

## 7. Subagent Plugin Extension

### 7.1 New `server` Parameter

The `spawn_subagent` tool schema gains an optional `server` parameter:

```python
{
    "name": "server",
    "type": "string",
    "description": (
        "Name of the server to delegate this subagent to. "
        "If omitted, runs on current server. Use get_environment "
        "aspect='jaato_agentic_servers' to see available servers."
    )
}
```

### 7.2 Remote Spawn Flow

When `server` is specified and refers to a peer:

1. **Validation** — Check the peer exists in `PeerRegistry` and its
   `trust_state` is not `BLOCKED`. If `ESCALATED`, inject a reliability nudge
   but allow the operation (the model made an informed choice).

2. **Serialization** — The spawn request is serialized as a
   `RemoteSpawnRequest` containing the `SubagentProfile`, task prompt, context,
   and inline config overrides. This is sent over the peer WebSocket channel.

3. **Workspace sync** — The remote server performs a `git pull` (or
   `git clone` on first delegation) before creating the session. See
   Section 9 for details.

4. **Session creation** — The remote server creates a local `JaatoSession`
   with the received profile configuration, as if the subagent were local.

5. **Output forwarding** — `AgentOutputEvent`s from the remote session are
   forwarded back to the parent server over the peer channel. The parent
   server injects them into the parent session's message queue with the
   subagent's `agent_id`, maintaining the existing output routing.

6. **Completion** — The remote server sends a `SubagentResult` back. The
   parent session resumes its turn with the result.

### 7.3 Cross-Server Cancellation

When the parent session calls `request_stop()`:

1. Parent server checks `_active_sessions` for any remote subagents
2. For each remote subagent, sends a `StopRequest` over the peer channel
3. The remote server cancels the session via its `CancelToken`
4. The remote server acknowledges cancellation and returns a partial result

### 7.4 Peer Event Types

New event types for the peer channel (not client-visible):

```python
class PeerEventType(str, Enum):
    HEARTBEAT           = "peer.heartbeat"
    SPAWN_REQUEST       = "peer.spawn_request"
    SPAWN_ACCEPTED      = "peer.spawn_accepted"
    SPAWN_REJECTED      = "peer.spawn_rejected"
    AGENT_OUTPUT        = "peer.agent_output"       # Forwarded AgentOutputEvent
    AGENT_COMPLETED     = "peer.agent_completed"    # SubagentResult
    STOP_REQUEST        = "peer.stop_request"
    STOP_ACKNOWLEDGED   = "peer.stop_acknowledged"
```

These are internal to the server mesh and never exposed to clients.

## 8. Reliability Plugin Extension

### 8.1 Server-Level Trust States

The existing trust state machine from `reliability/types.py` applies unchanged
at the server level:

```
TRUSTED ──[N failures in window]──► ESCALATED
ESCALATED ──[cooldown + successes]──► RECOVERING
RECOVERING ──[enough successes]──► TRUSTED
TRUSTED/ESCALATED ──[critical failure]──► BLOCKED
```

| State | Effect on Delegation |
|---|---|
| `TRUSTED` | Model can freely delegate subagents |
| `ESCALATED` | Model sees `trust_state: "escalated"` in environment aspect; reliability plugin injects a nudge suggesting alternatives |
| `RECOVERING` | Delegation allowed, successes tracked for recovery |
| `BLOCKED` | Delegation rejected by validation step; server shown with `status: "blocked"` in environment aspect |

### 8.2 Server Failure Types

| Type | Trigger | Severity |
|---|---|---|
| `HEARTBEAT_MISSED` | No heartbeat for N intervals | `LOW` |
| `DELEGATION_FAILED` | Subagent spawn on peer rejected or errored | `HIGH` |
| `DELEGATION_TIMEOUT` | Subagent on peer exceeded time limit | `MEDIUM` |
| `PROVIDER_EXHAUSTED` | Peer reports all rate limits hit | `MEDIUM` |
| `HIGH_ERROR_RATE` | Peer's self-reported `success_rate_1h` drops below threshold | `HIGH` |
| `RESOURCE_EXHAUSTION` | Peer CPU or memory above critical threshold | `MEDIUM` |

These follow the `FailureSeverity` enum from `reliability/types.py`. The
`FailureKey` for server failures uses `tool_name="__server__"` and
`parameter_signature=server_name` to reuse the existing tracking infrastructure.

### 8.3 Server Escalation Rules

Server escalation rules follow the same `EscalationRule` dataclass pattern:

```python
@dataclass
class ServerEscalationRule:
    """Configurable thresholds for server trust state transitions.

    Same fields as EscalationRule but applied at server granularity.
    """
    count_threshold: int = 3
    consecutive_threshold: Optional[int] = None
    rate_threshold: Optional[float] = None
    window_seconds: int = 300
    severity_filter: Set[FailureSeverity] = field(
        default_factory=lambda: {FailureSeverity.MEDIUM, FailureSeverity.HIGH, FailureSeverity.CRITICAL}
    )
    escalation_duration_seconds: int = 600
    cooldown_seconds: int = 120
    success_count_to_recover: int = 2
```

Rule resolution priority: `server_rules[name]` > `default_rule` (same layered
pattern as tool reliability).

### 8.4 Affinity Scoring

Each server gets a computed affinity score (0.0–1.0) exposed in the
environment aspect. The formula is configurable through `servers.json`:

```json
{
  "server_policies": {
    "affinity_weights": {
      "success_rate": 0.4,
      "response_time": 0.3,
      "availability": 0.3
    }
  }
}
```

Component definitions:

| Component | Calculation | Range |
|---|---|---|
| `success_rate` | Peer's `success_rate_1h` directly | 0.0–1.0 |
| `response_time` | `1.0 - (avg_response_ms / max_response_ms)` across recent delegations | 0.0–1.0 |
| `availability` | `1.0 - (active_agents / max_capacity)` where `max_capacity` is configurable | 0.0–1.0 |

```
affinity = (success_rate × W₁) + (response_time × W₂) + (availability × W₃)
```

Weights must sum to 1.0. If not configured, defaults to equal weighting (0.33
each).

### 8.5 Nudges

The reliability plugin generates nudges that appear in the model's context:

**Negative nudge** (server escalated):
> Server 'cloud' is experiencing reliability issues (71% success rate, trust
> state: escalated). Consider using 'local' or 'gpu-box' for new subagents.

**Positive nudge** (server excelling):
> Server 'gpu-box' has been performing excellently (95% success rate, affinity
> 0.95). Consider it for compute-intensive subagents.

Nudges are injected via the same mechanism the reliability plugin uses for
tool-level nudges — through the session's system instruction enrichment.

## 9. Workspace Replication

### 9.1 Git-Based Sync

Remote servers access the workspace via git. This leverages the fact that
jaato workspaces are already git repositories.

**First delegation to a peer:**
```
Parent Server                          Peer Server
─────────────                          ────────────
                                       git clone <repo_url> <workspace_path>
                                       git checkout <branch>
```

**Subsequent delegations:**
```
Parent Server                          Peer Server
─────────────                          ────────────
                                       cd <workspace_path>
                                       git pull --ff-only
```

### 9.2 Uncommitted Changes

If the parent workspace has uncommitted changes that the subagent needs:

1. Parent server creates a temporary branch: `jaato/delegation/<server>/<timestamp>`
2. Parent commits uncommitted changes to the temporary branch
3. Parent pushes the temporary branch
4. Peer server checks out the temporary branch before spawning the subagent
5. After the subagent completes, the temporary branch is deleted

### 9.3 Subagent Modifications

If a remote subagent modifies files (via `file_edit`, `cli`, etc.):

1. The peer server commits changes to the temporary branch
2. After subagent completion, the parent server pulls the temporary branch
3. The parent server cherry-picks or merges the changes into the working tree
4. The temporary branch is deleted on both sides

### 9.4 Limitations

- **Large workspaces** — Initial `git clone` can be slow. Shallow clones
  (`--depth 1`) mitigate this for workspaces with deep history.
- **Binary files** — Git is not ideal for large binary assets. Workspaces
  with large binaries should use `.gitignore` or Git LFS.
- **Concurrent modifications** — If both the parent and a remote subagent
  modify the same files, merge conflicts may occur. The parent agent resolves
  these like any other git conflict.

## 10. Configuration

### 10.1 `servers.json` Schema

```json
{
  "servers": [
    {
      "name": "local",
      "transport": "ipc",
      "address": "/tmp/jaato.sock",
      "tags": ["local", "tools"],
      "auto_start": true
    },
    {
      "name": "gpu-box",
      "transport": "ws",
      "address": "ws://gpu-box.local:8080",
      "tags": ["gpu", "ollama"],
      "auto_start": false
    },
    {
      "name": "cloud",
      "transport": "ws",
      "address": "ws://jaato.mycompany.com:8080",
      "tags": ["cloud", "high-capacity"],
      "auto_start": false
    }
  ],
  "gossip": {
    "heartbeat_interval_seconds": 5,
    "degraded_after_missed": 3,
    "unreachable_after_missed": 5
  },
  "tls": {
    "ca": ".jaato/certs/ca.pem",
    "cert": ".jaato/certs/server.pem",
    "key": ".jaato/certs/server-key.pem"
  },
  "server_policies": {
    "default_rule": {
      "count_threshold": 3,
      "window_seconds": 300,
      "escalation_duration_seconds": 600,
      "cooldown_seconds": 120,
      "success_count_to_recover": 2
    },
    "server_rules": {
      "cloud": {
        "count_threshold": 5,
        "window_seconds": 600
      }
    },
    "affinity_weights": {
      "success_rate": 0.4,
      "response_time": 0.3,
      "availability": 0.3
    }
  }
}
```

### 10.2 Field Reference

#### `servers[]`

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Human-readable identifier, used in `spawn_subagent(server=...)` |
| `transport` | `"ipc"` \| `"ws"` | yes | Connection type |
| `address` | string | yes | Socket path (IPC) or `ws://host:port` (WebSocket) |
| `tags` | array of string | no | Semantic tags for model reasoning (e.g., `"gpu"`, `"local"`, `"high-capacity"`) |
| `auto_start` | bool | no | If `true`, server daemon is started automatically when referenced |

#### `gossip`

| Field | Type | Default | Description |
|---|---|---|---|
| `heartbeat_interval_seconds` | int | 5 | Seconds between heartbeat broadcasts |
| `degraded_after_missed` | int | 3 | Missed heartbeats before marking peer `DEGRADED` |
| `unreachable_after_missed` | int | 5 | Missed heartbeats before marking peer `UNREACHABLE` |

#### `tls`

| Field | Type | Description |
|---|---|---|
| `ca` | string | Path to CA certificate (shared across all servers) |
| `cert` | string | Path to this server's certificate |
| `key` | string | Path to this server's private key |

All paths are relative to the workspace root or absolute.

#### `server_policies`

| Field | Type | Description |
|---|---|---|
| `default_rule` | object | Default `ServerEscalationRule` for all servers |
| `server_rules` | object | Per-server overrides (server name → partial rule). Missing fields inherit from `default_rule` |
| `affinity_weights` | object | Weights for affinity score components. Keys: `success_rate`, `response_time`, `availability`. Must sum to 1.0 |

**Every field is optional.** Missing fields retain their built-in defaults.

### 10.3 File Location Precedence

The server loads the first file it finds:

1. `.jaato/servers.json` (workspace — per-project)
2. `~/.jaato/servers.json` (user home — global default)

Same precedence pattern as `reliability-policies.json`.

## 11. Implementation Phases

### Phase 1: Gossip Infrastructure (backward-compatible)

Server-to-server communication and health collection. No changes to the
agent-facing API.

**Changes:**
- `jaato-server/server/peers.py` — `PeerRegistry`, gossip protocol, WebSocket peer channel
- `jaato-server/server/health.py` — `ServerHealthCollector` using `psutil`
- `jaato-server/server/__main__.py` — Load `servers.json`, initialize peer connections on startup
- `jaato-server/server/websocket.py` — Peer handshake detection (`X-Jaato-Peer` header), route to `PeerRegistry`
- `jaato-sdk/jaato_sdk/events.py` — `PeerEventType` enum (internal, not client-visible)

**Tests:** Heartbeat exchange, peer lifecycle transitions, degraded/unreachable detection, mTLS handshake

### Phase 2: Environment Aspect

Expose the peer mesh to the agent via the existing environment plugin.

**Changes:**
- `jaato-server/shared/plugins/environment/plugin.py` — Add `jaato_agentic_servers` to `VALID_ASPECTS`, implement `_get_agentic_servers()`
- `jaato-server/server/core.py` — Inject `PeerRegistry` reference into environment plugin via session

**Tests:** Aspect response schema validation, `self` always present, peer data freshness, excluded from `all`

### Phase 3: Remote Subagent Delegation

The model can choose which server runs a subagent.

**Changes:**
- `jaato-server/shared/plugins/subagent/plugin.py` — Add `server` parameter to `spawn_subagent`, remote spawn logic
- `jaato-server/shared/plugins/subagent/remote.py` — `RemoteSpawnRequest`, cross-server output forwarding, cancellation
- `jaato-server/server/peers.py` — Handle `peer.spawn_request` / `peer.agent_output` / `peer.agent_completed` events

**Tests:** Local spawn unchanged (no regression), remote spawn with mock peer, output forwarding, cancellation propagation, delegation to unreachable peer rejected

### Phase 4: Server-Level Reliability

Extend the reliability plugin to track server trust states.

**Changes:**
- `jaato-server/shared/plugins/reliability/server_tracking.py` — `ServerReliabilityState`, `ServerEscalationRule`, failure recording, affinity scoring
- `jaato-server/shared/plugins/reliability/plugin.py` — Integrate server tracking, generate server-level nudges
- `servers.json` schema — `server_policies` section

**Tests:** Server trust state transitions, escalation thresholds, affinity score computation, nudge generation, recovery flow

### Phase 5: Workspace Replication

Git-based workspace sync for remote subagents.

**Changes:**
- `jaato-server/shared/plugins/subagent/workspace.py` — Git clone/pull, temporary branch management, modification pull-back
- `jaato-server/shared/plugins/subagent/remote.py` — Workspace sync before session creation

**Tests:** Initial clone, subsequent pull, uncommitted change handling, modification pull-back, shallow clone for large repos

## 12. Source File Reference

| Component | File | Description |
|---|---|---|
| Peer registry + gossip | `jaato-server/server/peers.py` | `PeerRegistry`, heartbeat send/receive, peer lifecycle |
| Health collector | `jaato-server/server/health.py` | `ServerHealthCollector`, `ServerHealthSnapshot` via `psutil` |
| Peer event types | `jaato-sdk/jaato_sdk/events.py` | `PeerEventType` enum (internal) |
| Environment aspect | `jaato-server/shared/plugins/environment/plugin.py` | `_get_agentic_servers()`, `VALID_ASPECTS` extension |
| Server reliability | `jaato-server/shared/plugins/reliability/server_tracking.py` | `ServerReliabilityState`, `ServerEscalationRule` |
| Reliability plugin | `jaato-server/shared/plugins/reliability/plugin.py` | Server nudge generation, trust state integration |
| Subagent remote spawn | `jaato-server/shared/plugins/subagent/remote.py` | `RemoteSpawnRequest`, output forwarding |
| Subagent plugin | `jaato-server/shared/plugins/subagent/plugin.py` | `server` parameter on `spawn_subagent` |
| Workspace sync | `jaato-server/shared/plugins/subagent/workspace.py` | Git clone/pull, temp branch management |
| WebSocket (peer channel) | `jaato-server/server/websocket.py` | Peer handshake, routing |
| Server config | `jaato-server/server/config.py` | `servers.json` loading, validation |
| Existing reliability types | `jaato-server/shared/plugins/reliability/types.py` | `TrustState`, `EscalationRule`, `FailureKey` |
| Existing subagent config | `jaato-server/shared/plugins/subagent/config.py` | `SubagentProfile`, `SubagentResult` |
| Existing environment plugin | `jaato-server/shared/plugins/environment/plugin.py` | `VALID_ASPECTS`, aspect dispatch pattern |
