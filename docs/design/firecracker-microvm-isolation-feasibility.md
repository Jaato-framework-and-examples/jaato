# Feasibility: Firecracker microVM as an alternative confinement backend

**Status:** Assessment / no code committed beyond this document.
**Question:** can jaato *allow* an operator to choose a Firecracker microVM
instead of AppArmor as the per-session isolation mechanism — coexisting with
AppArmor rather than replacing it?

**Short answer:** Yes, and the architecture is unusually well-positioned for
it — but not as a swap of `server/apparmor.py`. The work is roughly
**20% "boot a VM"** and **80% "make the ~756 non-test references to AppArmor
across 31 modules speak through a backend-neutral interface, and rebuild the
five things AppArmor gives us for free that a VM does not."** Estimated
**8–12 weeks** to a usable single-tenant backend, **4–6 months** to
feature-parity with the AppArmor path.

---

## 1. Why this is more tractable than it sounds

The single most important finding: **the isolation boundary already exists as
a process boundary.** Phases 2–5 of `per_session_confined_runner.md` did the
expensive part of the work, for AppArmor's benefit, but the shape is
VM-agnostic:

| Existing property | Where | Why it matters for a VM backend |
|---|---|---|
| Session runs in a **separate runner process** | `server/runner_spawner.py`, `server/runner/session.py` | The VM guest is a drop-in for that process. |
| Daemon↔runner speaks a **framed JSON RPC** over `AF_UNIX` socketpair fd 3 | `server/runner_rpc_client.py`, `shared/framing.py` | Swap the socketpair for a **vsock** (`AF_VSOCK`) — the framing, dispatch, streaming and cancel logic are transport-agnostic. |
| The handshake is a **versioned, JSON-only envelope** | `shared/session_envelope.py` (`SESSION_ENVELOPE_VERSION = 5`) | Explicitly documents *"Anything richer (callable references, file descriptors, etc.) is NOT permitted in the envelope"* — i.e. it already survives a machine boundary, not just a process boundary. |
| Plugins are **partitioned by tier** (`daemon` / `runner` / `daemon_callable`) with enforced build gates | `shared/plugins/CLAUDE.md`, `test_plugin_tier_partition.py` | The "what runs inside the sandbox" question is already answered and test-enforced. |
| Cross-tier calls already round-trip through RPC in **both directions** | `RunnerForwardingMixin`, `DaemonForwardingMixin`, `runner_rpc_handlers/` | Clarification prompts, permission asks, operator prompts, subagent spawn all already work over a wire. |
| Confinement is applied by the **guest itself**, not the host | `server/runner/bootstrap.py` step 1c (`aa_change_profile`) | The "self-confine at bootstrap" contract maps cleanly onto "the guest kernel is the boundary; nothing to do at bootstrap". |
| The `server → shared` boundary already passes confinement as an **injected callable** | `make_confine_context()` in `server/apparmor.py`, consumed by `ToolExecutor` | A ready-made seam for a `ConfinementBackend` protocol. |

A Firecracker backend is therefore *not* a rewrite of the session lifecycle.
It is a new implementation of **spawn + transport + resource limits + policy
projection**, behind an interface that mostly already has a shape.

---

## 2. What "confinement" actually means today

AppArmor is one of five overlapping layers. A VM backend replaces two of them
outright, subsumes a third, and must **re-implement or drop** the other two.

| Layer | Mechanism today | Under Firecracker |
|---|---|---|
| **MAC / path confinement** | `server/apparmor.py` per-session profile `jaato-ws-{id}`, ~2 999 LoC, template v23+ | **Replaced.** Guest sees only what is mounted. |
| **Escape-vector containment** | `//child` sub-profile; known-exploitable `change_profile -> unconfined` documented in the template itself | **Subsumed, and strictly better** — see §4.1. |
| **Resource limits** | `server/cgroups.py` + `shared/runtime_limits.py` (memory, pids, cpu) | **Replaced** by VM `mem_size_mib` / `vcpu_count`, but the *cgroup-attach callback plumbing* (`preexec_fn`, `cgroup_attach`) must be reworked, not deleted (nested limits inside the guest still want cgroups). |
| **Egress control** | Not shipped. `phase5_5_11_egress_proxy_spike.md` recommends a proxy; profile grants blanket `network inet stream` | **Improved** — a microVM has exactly one tap device, so a per-session host firewall rule is trivially enforceable and unbypassable, versus AppArmor's per-*socket-family* granularity. This is arguably the strongest argument for the VM. |
| **App-layer path validation** | `shared/plugins/sandbox_utils.py`, `check_path_with_jaato_containment` | **Unchanged** — still wanted as defense-in-depth, and still the only layer on non-Linux hosts. |

---

## 3. The hard parts (what does *not* map)

These are ordered by how much they will hurt.

### 3.1 The workspace is shared read-write with the daemon, at runtime

This is the biggest single problem. AppArmor confines a process that shares
the host's page cache; a VM does not.

Concretely, the daemon touches the live workspace *while the session runs*:

- `server/workspace_monitor.py` watches the workspace with inotify to produce
  change events for the TUI workspace panel (and watches sandbox-added paths
  too).
- `server/session_logging.py` writes per-session logs under
  `{workspace}/.jaato/logs`.
- `server/workspace_command.py` / staged-files handling reads workspace files
  daemon-side.
- Profile / agent / reference / schema discovery reads `config_root` and
  `{workspace}/.jaato/**` daemon-side, *before and during* the session.

So the guest needs **virtio-fs** (not a block device) exporting the workspace,
the `config_root`, and the read-only host trees. That brings:

- **Coherence**: virtio-fs with `cache=none` is coherent but slow; `cache=auto`
  is fast but the host's inotify no longer reliably reflects guest writes in
  the ordering the monitor assumes. Expect the workspace panel and the
  artifact-tracker/LSP enrichment paths to need rework or a guest→host change
  notification channel over the same RPC.
- **A running `virtiofsd` per session** — one more supervised process per
  session, with its own crash/teardown semantics.
- **uid/gid mapping** between guest and host.
- **Locking**: the profile template deliberately grants `k` (file lock) and
  reasons about hardlink aliasing (`l`). Those semantics do not survive a
  naive virtio-fs export unchanged.

### 3.2 Policy is mutated *mid-session* — a VM has no equivalent

AppArmor gives jaato dynamic, in-flight policy edits that a booted VM
fundamentally does not:

- `AppArmorManager.add_reference_fragment(session_id, ref_id, path)` writes a
  rule file into `{profile}.refs.d/` and runs `apparmor_parser -r` **while the
  session is live**, so `selectReferences` can grant read access to a newly
  chosen reference (`apparmor.py:1697`).
- The `sandbox_manager` plugin's `sandbox add` / `sandbox remove` user commands
  grant or revoke paths at runtime, per session.
- `~/.jaato/apparmor-fragments/*.rules` lets daemon extensions (premium's
  reactor `handoff_gates.json`) splice grants in.

Under Firecracker each of these becomes **"hot-add a virtio-fs mount / a new
shared directory to a running guest"**, which Firecracker supports poorly
(device hotplug is limited; there is no dynamic `fs` device add today). The
realistic answers are all compromises:

1. Export one broad "grantable" directory and enforce selection in-guest
   (weakens the boundary to app-layer for that subtree),
2. Pre-declare the full grantable set at boot (loses dynamism — breaks
   `selectReferences` UX), or
3. Route those reads through the daemon over RPC as a file-fetch API (clean,
   but a new API surface and a latency hit on reference-heavy sessions).

**This is the design decision that most needs to be made before any code.**

### 3.3 Ten plugins contribute AppArmor *syntax*

`get_apparmor_rules()` is a classmethod on `cli`, `interactive_shell`,
`file_edit`, `lsp`, `notebook`, `memory`, `references`, `prompt_library`,
`service_connector`, and `subagent` — it returns literal AppArmor rule text,
spliced into the template (`docs/design/plugin-apparmor-contribution.md`).

For a second backend these must become **declarative resource requests**
(`{path, access, kind}` tuples) that each backend *renders* — AppArmor to rule
text, Firecracker to a mount list. That is a mechanical but wide refactor
touching every one of those plugins plus the rendering site and its tests.

### 3.4 Host paths the runner needs are numerous and path-shaped

The profile template enumerates what a runner genuinely needs beyond the
workspace: the **venv** (`{venv_path}` incl. `*.so` mmap-exec), the **jaato
source tree** (editable installs), the **premium package**, `config_root`,
`env_file`, `~/.jaato/` (credentials `*_auth.json`, `services/`, `references/`,
memories, prompts, Claude Code interop), `~/.cache/huggingface`,
`~/.cache/torch`, `/tmp/jaato-{id}-**`, plus `/usr/bin`, `/lib`, `/etc/passwd`,
`/dev/{null,urandom,pts}`.

Under a VM this becomes a **guest image composition problem**: build and
maintain a kernel + rootfs that contains a compatible Python, the venv, and
the jaato source — and keep it in lockstep with the host install on every
upgrade. Editable installs (`pip install -e`) and out-of-tree plugins make
"just bake an image" harder than it looks. Expect a build pipeline
(`scripts/build-guest-rootfs.sh`) and a version-skew check analogous to the
existing `SESSION_ENVELOPE_VERSION` guard.

### 3.5 The runner holds provider credentials and makes the LLM calls

Post-seat-flip the model loop runs **runner-side**: the runner builds its own
`JaatoRuntime`, loads the provider plugin, and calls the API
(`server/runner/session.py`; confirmed by the egress spike's framing — *"a
confined runner that has legitimate network access (to call the model
provider's API ...)"*).

So a microVM must either (a) receive API keys inside the guest — moving the
credential blast radius *into* the sandbox that exists because we distrust its
contents — or (b) gain a host-side provider broker so keys never enter the
guest. (b) is strictly better security and is a **new subsystem**, roughly the
egress-proxy work plus a credential-injection proxy. Worth noting: this
weakness exists today under AppArmor too (the profile grants read of
`~/.jaato/*_auth.json`), so a VM backend is an opportunity to fix it, not a
regression — but it is scope.

### 3.6 The pre-warm pool must become snapshot/restore

`JAATO_RUNNER_POOL_ENABLED` (default on) cuts session bootstrap from ~30 s to
~7 s by **forking pool slots from a warm template** that has already imported
every runner-tier plugin (`server/runner_template.py`, `runner_pool.py`).

Firecracker's analogue is **snapshot/restore**: boot one VM, warm the imports,
snapshot, then restore N clones. This works and is fast (Firecracker boots in
~125 ms cold; restore is faster still), but it is a genuinely different
mechanism with its own hazards that jaato's fork model does not have:

- **Entropy duplication** across restored clones (needs the virtio-rng /
  `PostVMResume` handling; a classic snapshot-clone CVE class).
- **Clock skew** on resume.
- **Uniqueness**: MAC addresses, vsock CIDs, hostnames must be re-stamped
  per clone.
- **Memory footprint**: fork gives copy-on-write sharing for free; VM clones
  need `--memory-backend`/UFFD tricks to approximate it, or you pay full RAM
  per idle slot (the pool's current ~150–300 MiB idle footprint becomes
  ~N × guest RAM).

Also note the current pool routing gate is `cgroup_attach is None` — a
Firecracker backend changes the meaning of that gate entirely.

### 3.7 Nested / isolated subagents

`runner_rpc_handlers/spawn_isolated_runner.py` (Phase 4 §4.3, still a stub)
plans `jaato-ws-{parent}//{subagent_id}` sub-profiles + sub-cgroups for
`agent_params.isolated=true`. Under a VM there is no sub-profile: the choices
are **one VM per isolated subagent** (clean, but multiplies VM count and boot
cost, and the parent must proxy its child's RPC or the daemon must address the
child directly) or **in-guest AppArmor/namespaces**, which reintroduces the
thing we were replacing. Worth deciding early since the stub is not yet wired.

### 3.8 "Am I confined?" is currently spelled `/proc/self/attr/current`

Two fail-closed gates read the AppArmor label directly:

- `shared/plugins/notebook/backends/local.py:_apparmor_enforced_profile()` —
  in-process cell `exec()` refuses to run unless an *enforce*-mode profile is
  active (or `JAATO_NOTEBOOK_ALLOW_INPROCESS_EXEC` is set).
- `shared/runtime_limits.py` + `test_confinement_guard.py` —
  `assert_inprocess_can_honor` / `profile_requires_kernel_confinement` reject
  in-process sessions carrying kernel limits.

Both must become a backend-neutral predicate (`confinement.is_enforced()`),
otherwise every VM-hosted session **silently fails closed** on notebooks and
on any profile declaring `runtime_limits`. Cheap to fix, easy to miss.

### 3.9 AppArmor knowledge has leaked into `shared/`

756 non-test occurrences across 31 modules, and the tier that is *supposed* to
be backend-agnostic is the worst offender:

```
44  shared/plugins/subagent/config.py
41  shared/plugins/lsp/plugin.py
23  shared/plugins/interactive_shell/plugin.py
23  shared/plugins/cli/plugin.py
21  shared/ai_tool_runner.py
 8  shared/safe_pool.py
 8  shared/jaato_session.py
 7  shared/plugins/file_edit/plugin.py
 …
```

Much of that is comments/docstrings (which the repo's docstring policy makes
load-bearing), but the live parts are real: `preexec_fn` callbacks writing
`changeprofile {profile}//child`, `apparmor_confine` context managers around
in-process tools, `safe_pool` worker-thread confinement restoration. Each is a
behaviour a VM backend implements as **a no-op** — but they must be reachable
through an interface, not `if apparmor_available()`.

### 3.10 Operational prerequisites are a step change

| | AppArmor | Firecracker |
|---|---|---|
| Host requirement | AppArmor LSM + `apparmor_parser` | **KVM** (`/dev/kvm`, r/w for the server user) |
| Works in a container? | Yes (with host policy) | Only with **nested virtualisation** — unavailable on many managed/cloud runners and most CI |
| Works on WSL2 / macOS? | Degrades to soft sandbox | Degrades to soft sandbox (same fallback path) |
| Privileged setup | sudoers rule for `apparmor_parser`, writable `/etc/apparmor.d/jaato` | `jailer` (recommended), tap-device creation (`CAP_NET_ADMIN`), cgroup + seccomp setup |
| Artifacts to maintain | one profile template | **kernel image + rootfs image + build pipeline**, versioned against the server |
| Failure mode today | logged warning, degrade to soft sandbox (`JAATO_REQUIRE_APPARMOR=1` to fail closed) | same contract reusable — `--isolation=firecracker` + a `JAATO_REQUIRE_ISOLATION` promotion |

The existing three-state config (`--apparmor` / `--no-apparmor` / auto, plus
`JAATO_REQUIRE_APPARMOR`, plus `IPCClient(apparmor=True)`) is a good model to
generalise to `--isolation={none,apparmor,firecracker}` while keeping the old
flags as aliases.

---

## 4. Security and performance: is it actually better?

### 4.1 Security — yes, materially, in two specific places

The AppArmor template documents its own **verified escape vector** at length:

> ```
> sudo aa-exec -p jaato-ws-... -- python3 -c '
>     open("/proc/self/attr/current","w").write("changeprofile unconfined")'
> ```
> *"write_ok=True, opendir succeeded … cli / interactive_shell subprocesses
> still inherit the rules. This is a known gap."*

The `//child` sub-profile mitigates it for subprocesses; the parent profile
must keep the rules for `apparmor_confine.__exit__`. A microVM removes the
class of bug entirely — there is no in-guest write that promotes to host
access short of a VMM/KVM escape, and Firecracker's attack surface (minimal
device model, Rust, seccomp-filtered VMM, `jailer`) is far smaller than the
full Linux syscall surface that AppArmor leaves exposed.

Second: **egress**. AppArmor cannot express "deny TCP to `evil.example.com`"
(the spike says so explicitly). A microVM's single tap device makes per-session
`nftables` egress policy trivial and unbypassable — closing §5.11's gap as a
side effect rather than as a proxy-shaped project.

Third, less obvious: a VM boundary means **kernel-level bugs in the guest do
not escalate on the host**, which is the actual multi-tenant story. AppArmor
confines a process sharing the host kernel; that is a mitigation, not a
tenancy boundary. If jaato's WS multi-tenant deployment story is meant to hold
against genuinely untrusted tenants, this is the honest upgrade.

### 4.2 Performance — worse on latency, better on blast radius

- **Cold start**: AppArmor adds ~0 ms; Firecracker adds ~125 ms boot plus
  guest userspace + Python import (the expensive part today is *already* ~30 s
  of plugin imports, so relatively the VM boot is noise — *provided* the
  snapshot-restore pool lands, otherwise you pay the imports again per VM).
- **Filesystem**: virtio-fs is materially slower than native page-cache access.
  Tool-heavy sessions (`file_edit`, `glob_files`, LSP indexing, `references`
  embedding over HF caches) will feel it. Worth benchmarking before commitment.
- **Memory**: fork-CoW → full guest RAM per session, unless UFFD/snapshot
  sharing is implemented. For a pool of idle slots this is the dominant cost.
- **Teardown**: strictly better and simpler — kill the VM, everything inside
  dies. No cgroup leak audits (`phase5_5_3_cgroup_leak_audit`), no straggler
  reaping, no profile unload races.

---

## 5. Proposed shape

### 5.1 Introduce a `ConfinementBackend` protocol

Not a class hierarchy over AppArmor — a **spawn-and-supervise** interface,
because that is where the two designs actually differ:

```python
class ConfinementBackend(Protocol):
    name: str                                   # "apparmor" | "firecracker" | "none"

    def is_available(self) -> tuple[bool, str]: ...          # (ok, reason)

    def provision(self, spec: SessionIsolationSpec) -> IsolationHandle: ...
        # apparmor  : render + apparmor_parser -r  (+ cgroup provision)
        # firecracker: compose mounts, start virtiofsd, boot/restore VM, open vsock

    def spawn_runner(self, handle, envelope) -> RunnerChannel: ...
        # returns a framed-RPC channel; socketpair fd3 vs AF_VSOCK — the
        # existing RunnerRPCClient consumes either

    def grant(self, handle, req: ResourceRequest) -> bool: ...  # §3.2 dynamic
    def revoke(self, handle, req: ResourceRequest) -> bool: ...

    def is_enforced_in_guest(self) -> bool: ...  # replaces /proc/self/attr reads

    def teardown(self, handle) -> None: ...
```

with plugins moving from `get_apparmor_rules() -> str` to
`get_resource_requests() -> list[ResourceRequest]`, and the AppArmor backend
keeping today's renderer as one consumer of that list. (Keep
`get_apparmor_rules` as a deprecated shim for out-of-tree plugins, mirroring
how `tools=`/`plugins=` was handled in #292.)

### 5.2 What stays completely untouched

The session envelope, the RPC dispatch/streaming/cancel machinery, the tier
partition and its build gates, `RunnerForwardingMixin` /
`DaemonForwardingMixin`, every plugin body, `sandbox_utils` app-layer checks,
the SDK event protocol, and the degrade-with-a-`SystemMessageEvent` contract
(`[apparmor] …` messages generalise to `[isolation] …`).

### 5.3 Phasing

| Phase | Content | Est. |
|---|---|---|
| **0. Backend seam** | `ConfinementBackend` protocol; port AppArmor behind it with zero behaviour change; `--isolation=` flag with old flags as aliases; generalise `is_enforced_in_guest()` (§3.8). Pure refactor, fully testable today. | 2–3 wk |
| **1. Plugin resource requests** | `get_resource_requests()` across 10 plugins + renderer + tests; AppArmor renders from the new type. Still zero behaviour change. | 1–2 wk |
| **2. vsock transport** | `RunnerRPCClient` / `server/runner/rpc.py` accept an `AF_VSOCK` channel; prove it with a **non-VM** vsock loopback or a plain container first. | 1 wk |
| **3. Guest image** | kernel + rootfs build pipeline, venv/source baking, version-skew guard. This is where "it works on my machine" goes to die. | 2–3 wk |
| **4. Firecracker MVP** | boot per session, virtio-fs workspace + config_root, VM-level mem/vcpu limits, teardown, degrade path. **Single-tenant, no pool, no dynamic grants.** First end-to-end session. | 3–4 wk |
| **5. Egress** | per-VM tap + nftables allowlist; closes §5.11 for this backend. | 1–2 wk |
| **6. Dynamic grants** | resolve §3.2 (recommend: daemon-mediated file-fetch RPC for references; keep `sandbox add` host-side for the workspace subtree). | 2–4 wk |
| **7. Snapshot pool** | snapshot/restore parity with the fork pool, incl. entropy/clock/uniqueness; keep `JAATO_RUNNER_POOL_*` semantics. | 3–4 wk |
| **8. Nested subagents** | decide §3.7; likely one VM per isolated subagent, daemon-addressed. | 2–3 wk |
| **9. Credential broker** | host-side provider proxy so keys never enter the guest (§3.5). Optional but strongly recommended, and independently valuable to the AppArmor path. | 3–4 wk |

**Phases 0–4 ≈ 9–13 weeks to a demonstrable, honestly-secure single-tenant
Firecracker session.** Phases 5–9 ≈ another 11–17 weeks to parity. Phases 0–2
are worth doing **regardless of whether Firecracker ever ships** — they pay
down the `shared/`-tier AppArmor leakage documented in §3.9.

---

## 6. Cheaper alternatives worth pricing first

If the goal is "another choice", Firecracker is the most expensive point on
the curve. Two nearer options reuse Phase 0–2 entirely:

- **OCI container backend** (runc/podman, optionally **gVisor** or **Kata**).
  Gets you namespace + seccomp isolation, per-container network policy, and a
  *filesystem view* story (bind mounts) with far less operational burden than
  a VM, and — unlike Firecracker — works without nested virt on most cloud
  runners. gVisor closes much of the kernel-surface gap; Kata is a VM under an
  OCI interface, i.e. a softer landing to the same place. **This is the best
  effort/benefit ratio and I would build it first**, since Phases 0–3 are
  shared and Phase 4 becomes ~1 week instead of ~4.
- **Landlock + seccomp + user namespaces (bubblewrap-style)**. Unprivileged,
  no daemon setup, no images, cross-distro. Weaker than AppArmor on some axes
  (no `mmap`-exec granularity, no capability rules of comparable expressivity)
  but *unprivileged* — attractive for the IPC opt-in path where the operator
  cannot edit `/etc/apparmor.d`.

A useful framing: AppArmor = "cheap, in-kernel, path-shaped, leaky";
containers = "portable, namespace-shaped, shared kernel"; Firecracker =
"expensive, hardware-shaped, genuinely multi-tenant".

---

## 7. Recommendation

1. **Do Phase 0–1 unconditionally.** The `ConfinementBackend` seam and the
   plugin `get_resource_requests()` refactor are net-positive today: they
   remove AppArmor vocabulary from `shared/`, make the notebook/runtime-limit
   fail-closed gates backend-neutral, and are testable without any new
   infrastructure.
2. **Ship an OCI/gVisor backend as the second choice** before Firecracker.
   It validates the seam against a genuinely different mechanism, works in
   containers and CI, and reuses ~80% of what Firecracker needs.
3. **Treat Firecracker as the multi-tenant/hostile-tenant tier**, justified by
   §4.1 (escape vector + egress + kernel blast radius), and gate it on the two
   decisions that cannot be deferred: **dynamic grants (§3.2)** and
   **credentials in-guest vs. brokered (§3.5)**.
4. **Do not frame it as replacing AppArmor.** They compose: AppArmor (or
   seccomp) *inside* the guest is still worth having, and AppArmor alone
   remains the right default for the single-user IPC/TUI path where the VM's
   cost buys nothing.

## 8. Open questions for the operator/product side

- Is the target "harden the existing single-user path" or "safely host
  untrusted tenants"? Only the second justifies Firecracker's ops cost.
- Are deployments allowed nested virtualisation? If jaato's WS server is itself
  expected to run in a container/managed runner, Firecracker is often simply
  unavailable and the OCI backend is the only reachable answer.
- Is per-session RAM (full guest, no CoW) acceptable for the pool, or is
  snapshot+UFFD sharing a hard requirement from day one?
- Should the provider credential broker (§3.5, §9 of the phasing) be pulled
  forward as its own project — it improves the AppArmor path too.
