# An SELinux Confinement Backend

Status: **proposed**. Nothing here is implemented. Written against the tree
at the time of the RHEL compatibility question; AppArmor template v43.

## 1. Why

Every kernel-enforced boundary jaato has is AppArmor: the per-session
profile, the `//child` sub-profile for model-driven subprocesses, the
isolated sub-runner profile, the per-thread verification (#1023), the mode
check (#1014), the grant record (#1326), the private `/tmp` (#1381).
`AppArmorManager.is_available()` needs `apparmor_parser`, `aa-exec` and
`/sys/kernel/security/apparmor`, none of which exist on RHEL, Fedora, Rocky,
Alma or CentOS Stream. There the daemon logs "AppArmor confinement NOT
available" and falls back to the application-layer heuristics:

| Surface | Without a kernel boundary |
|---|---|
| runner process | unconfined; `cli` containment is string-level |
| model-driven subprocesses | unconfined; `interactive_shell` warns, `require_confinement` refuses every spawn |
| notebook | audit-hook tier: `import ctypes` refused, so no numpy / pandas in a cell |
| private `/tmp` | off (only the confined path provisions it) |
| `JAATO_REQUIRE_APPARMOR` | the daemon refuses to start |

Those hosts ship SELinux, enforcing by default. This design adds an SELinux
backend that is selected when AppArmor is not, and gives those hosts the
same guarantees, with the differences stated where the two LSMs genuinely
differ.

The two cannot both be the active major LSM on one kernel, so "instead of"
is the only mode: there is no host on which both are available.

## 2. The central difference: paths vs labels

AppArmor rules name **paths**. A profile rendered per session can say
"`/srv/ws/abc/**` rwkl, `.jaato/agents/**` write-denied, `/usr/bin/git`
executable", and a fresh profile is loaded per boundary.

SELinux rules name **types** carried on inodes (xattrs) and processes. A
policy is compiled and loaded as a module, which is slow (seconds) and
global. Loading one module per session, the literal translation of the
AppArmor design, is wrong: `semodule -i` takes 5–30 s, serialises on the
policy store, and grows the policy with every workspace.

So the backend does what container runtimes do (sVirt, `container-selinux`):

* **One policy module, installed once**, defining a small fixed set of
  domains and file types (section 4).
* **Per-boundary isolation by MCS categories.** Each workspace gets a
  stable category pair (`s0:c123,c456`). The runner runs at that level and
  the workspace's files are labelled at that level. The MCS constraint then
  keeps a runner out of every other workspace even though all of them share
  the same types, exactly as it keeps one container out of another.

Consequences that shape the rest of this document:

1. **Nothing is loaded per session.** Provisioning is: allocate or look up
   the category pair, make sure the workspace tree carries it, and compute
   the label to transition to. No privileged tool runs per session once the
   tree is labelled.
2. **Labelling has a cost AppArmor does not have.** A workspace must be
   relabelled once (a recursive `setfilecon`). Files created afterwards
   inherit the right type from their parent and the level from the creating
   process, so the relabel is one-time per workspace, not per session
   (section 6).
3. **Per-path grants outside the workspace do not translate.** A fragment
   saying "read `/srv/corpus`" has no SELinux equivalent short of labelling
   `/srv/corpus`. Section 8 says what each AppArmor feature becomes, and
   which ones become "unsupported, said so".

## 3. The seam: a `ConfinementBackend`

Today the call sites import `AppArmorManager` and the runner-side helpers
directly (`server/websocket.py`, `server/session_manager.py`,
`server/runner_spawn.py`, `server/runner_pool.py`, `server/runner/bootstrap.py`,
`server/runner/session.py`, `server/runner/rpc.py`, `server/core.py`,
`server/diagnostics_verbs.py`, `server/runner_rpc_handlers/*`,
`shared/jaato_session.py`, `shared/safe_pool.py`). The first step is a
refactor with **no behaviour change**: put a protocol between them and
AppArmor.

```
server/confinement/
  __init__.py        select_backend() -> ConfinementBackend
  base.py            ConfinementBackend (Protocol), Boundary, ConfinementHandle
  apparmor.py        AppArmorBackend: wraps today's AppArmorManager, unchanged
  selinux.py         SELinuxBackend (daemon side)
shared/lsm_label.py  LSM-neutral label parsing (generalises apparmor_label.py)
server/runner/lsm_confine.py   runner-side transition, per LSM
```

### 3.1 Daemon side

```python
@dataclass(frozen=True)
class Boundary:
    workspace_path: str            # realpath
    config_root: Optional[str]
    env_file: Optional[str]
    managed: bool                  # under the WS workspace_root
    private_tmp_dir: Optional[str]
    requested_fragments: tuple[str, ...]
    plugin_rules: tuple[str, ...]

@dataclass(frozen=True)
class ConfinementHandle:
    backend: str                   # "apparmor" | "selinux"
    label: str                     # what the runner transitions to
    confinement_id: str            # slot-key component (#1033)
    child_label: str               # what model-driven subprocesses exec into
    grants: dict                   # the #1326 record, backend-shaped

class ConfinementBackend(Protocol):
    name: str
    def is_available(self) -> bool: ...
    @property
    def unavailable_reason(self) -> Optional[str]: ...
    def confinement_id_for_boundary(self, b: Boundary) -> str: ...
    def provision(self, session_id: str, b: Boundary) -> Optional[ConfinementHandle]: ...
    def release(self, handle: ConfinementHandle) -> None: ...   # per-slot, as today
    def provision_isolated(self, parent: ConfinementHandle, subagent_id: str,
                           workspace_path: str) -> Optional[ConfinementHandle]: ...
    def add_reference_grant(self, handle, ref_id: str, path: str) -> bool: ...
    def remove_reference_grant(self, handle, ref_id: str) -> bool: ...
    def record(self, handle) -> dict: ...                       # diagnostics
```

`select_backend()` order: an explicit `JAATO_CONFINEMENT=apparmor|selinux|none`
wins; otherwise AppArmor if available, else SELinux if available, else none.
Because the two cannot coexist, "else" only ever matters for the reason
string: on a SELinux host the log says "AppArmor unavailable (no
securityfs entry); SELinux backend selected" rather than the current
degraded-isolation warning.

`JAATO_REQUIRE_APPARMOR` / `--apparmor` keep their meaning (AppArmor
specifically). A new `JAATO_REQUIRE_CONFINEMENT` / `--require-confinement`
means "some kernel backend, or refuse to start", and is what RHEL
deployments set. `--apparmor` on an SELinux host keeps failing, with a
message naming the new flag.

### 3.2 Envelope

`SessionInitEnvelope.profile_name` stays (older runners read it) and gains
a sibling `confinement: {backend, label, child_label}`. A runner that does
not understand `backend: selinux` must refuse the bootstrap rather than run
unconfined, so this is an **envelope version bump**, not an additive field.
Both ends ship in `jaato-server`, so there is no cross-version pairing in
practice; the bump is what makes a mismatched pool template fail loudly.

### 3.3 Runner side

`shared/apparmor_label.py` becomes a special case of `shared/lsm_label.py`,
stdlib-only as now:

```python
@dataclass(frozen=True)
class LsmLabel:
    backend: str          # "apparmor" | "selinux" | "none"
    raw: str
    identity: str         # AppArmor profile name | SELinux "type:level"
    enforced: bool
    mode: str             # enforce/complain | enforcing/permissive
```

The two questions #1014 keeps apart stay apart: `identity_ignoring_mode`
(which boundary is this task in) and `enforced` (is the kernel applying it).
For SELinux, `enforced` is true only when **both** `/sys/fs/selinux/enforce`
reads `1` **and** the domain is not permissive. Per-domain permissive is
the SELinux analogue of AppArmor complain mode; it is visible through
`libselinux` (`security_compute_av_flags` returns `SELINUX_AVD_FLAGS_PERMISSIVE`
for a permissive source domain). An unreadable answer is "not enforced", as
today.

`sandbox_mode` gains `selinux` and `selinux-permissive`. The two existing
predicates (`sandbox_mode_is_apparmor`, `sandbox_mode_is_enforced`) are
renamed `sandbox_mode_is_kernel` / `sandbox_mode_is_enforced` with the old
names kept as aliases; an older reader comparing `== "apparmor"` reads an
SELinux session as unconfined, the safe direction, as #1014 already argued
for `apparmor-complain`.

## 4. The policy module

Shipped as source under `jaato-server/selinux/` (`jaato.te`, `jaato.fc`,
`jaato.if`), built with `make -f /usr/share/selinux/devel/Makefile`, and
installed once by the operator (later: a `jaato-server-selinux` RPM, the
`container-selinux` pattern). The module carries a version; the backend
refuses (`unavailable_reason`) when the loaded version is older than the
one the daemon was built for, the way `TEMPLATE_VERSION` gates AppArmor
profiles today.

### 4.1 Domains

| Domain | AppArmor equivalent | Entered by |
|---|---|---|
| `jaato_runner_t` | base profile `jaato-ws-<id>` | cold spawn: `setexeccon` in the daemon's fork before `execve`; pool slot: `setcon` (section 7) |
| `jaato_child_t` | `//child` | `setexeccon` in the `preexec_fn` of every model-driven subprocess |
| `jaato_isolated_t` | isolated sub-runner profile | exec transition at sub-runner spawn (cold, like the runner) |
| `jaato_isolated_ro_t` | isolated sub-runner profile with `isolated_read_only_workspace` | same |
| `jaato_template_t` | (unconfined pool template) | only with pool support, section 7 |

There is no `tool_hat` domain, and there cannot be one of the same shape.
A per-call hat would need `setcon` on a worker thread: into a tool domain
that is `typebounds`-bounded by `jaato_runner_t` (phase 0, 5.7, showed the
kernel refuses an unbounded one in a threaded process), and then back, which
the bounded check refuses because `jaato_runner_t` is not bounded by the tool
domain. The runner is also given no `dyntransition` by design. Tool bodies
therefore run in `jaato_runner_t`, so `jaato_runner_t` never gets write on
the reference catalog: catalog writes (a promotion, a links edit, a bundle
reconcile) stay daemon-side, the #1420 path. The AppArmor `tool_hat`
sub-profile is still never entered on the current tree; draft PR #1443
proposed entering it and moving those writes back into the runner, and was
not merged.

The daemon itself stays in whatever domain systemd gives it
(`unconfined_service_t` on RHEL), as it stays unconfined under AppArmor.
The module grants `unconfined_service_t` (and `unconfined_t`, for a daemon
started from a login shell) `process transition` into `jaato_runner_t` and
`jaato_isolated_t`, plus `entrypoint` on the interpreter's file type.
The runner keeps the daemon's SELinux user and role (`setexeccon` sets
type and level), so the module authorizes both `system_r` and
`unconfined_r` for the jaato domains.

The module must also grant each jaato domain `fd use` on the daemon's
domains and read/write on their pipes and sockets. A runner exec'd without
it loses the inherited stdio silently: the kernel closes descriptors the
new domain may not use, the denial is `dontaudit`-ed, and the symptom is
empty output with exit status 0 (phase 0).

### 4.2 File types

| Type | Labels | `jaato_runner_t` | `jaato_child_t` |
|---|---|---|---|
| `jaato_workspace_t` | the workspace tree | read, write, create, unlink, rename, lock, link | same |
| `jaato_managed_ws_t` | a managed workspace's tree (under `workspace_root`) | as above, **plus execute** | same |
| `jaato_authored_t` | `.jaato/{services/*/,references,templates,plans}`, `template_routing.yaml` and the other authored entries | read, search | read, search |
| `jaato_agent_config_t` | `.jaato/{agents,profiles,scripts,completion_schemas,spawn_schemas,instructions}` and `reactors.json` (phase 3) | read, search | read, search |
| `jaato_prompts_t` | `.jaato/prompts/` (phase 3) | read, write | read, write |
| `jaato_claims_t` | `.jaato/references-claims/` | read, write | read only (template v43's `//child` deny) |
| `jaato_tmp_t` | session tmpdir and private `/tmp` | read, write, create | same |
| `jaato_runner_log_t` | an isolated sub-runner's `.jaato/logs/runner-<id>.log` (phase 3, created by the daemon) | none | none (the isolated domains: open, append) |
| `jaato_devpts_t` | a pty either domain opens (`type_transition` from `devpts_t`) | read, write, ioctl, setattr | same |

`jaato_agent_config_t` and `jaato_prompts_t` exist only so the isolated
domains (§5.3) can be refused them: they are what the AppArmor isolated
sub-profile read-denies (`selinux_labels.ISOLATED_UNREADABLE` plus
`prompts`, kept equal to that body by a test). For the runner and the child
they behave as `jaato_authored_t` and as workspace files.

The authored set is the one `jaato_sdk/scaffold/gitignore.py` `AUTHORED`
already declares and `test_gitignore_authored_set_tracks_apparmor.py`
already checks against the AppArmor write-denies; that guard gains the
`.fc`/relabel list as a third party to agree with.

**Execution from the workspace** follows the AppArmor rule: a managed
workspace (#1273/#1274: `node_modules/.bin`, the tool venv, `.home/.local/bin`)
is executable, a user's own checkout is not. Two types rather than a
boolean, because a boolean would be global and the distinction is per
workspace.

### 4.3 Everything else, as type rules

| AppArmor rule | SELinux rule |
|---|---|
| `/usr/bin/** ix`, `/usr/lib/** rm`, `/etc/ld.so.cache r`, … | `corecmd_exec_bin`, `libs_use_ld_so`, `files_read_etc_files` (refpolicy interfaces) |
| `network inet stream/dgram` | `corenet_tcp_connect_all_ports`, `corenet_udp_*`, `sysnet_dns_name_resolve` |
| `deny network raw` | no `rawip_socket create` |
| `deny ptrace`, `deny capability sys_admin/net_admin/sys_ptrace` | none of those permissions granted; `dontaudit` where noisy |
| `deny mount` | no `mount`/`mounton`; no `filesystem mount` |
| `change_profile -> …//child` | `allow jaato_runner_t jaato_child_t:process transition` + `setexec` |
| `change_profile -> unconfined` | **nothing.** See 4.4 |
| `audit deny /proc/*/environ …` (other processes) | no `file read` on other domains' `/proc` entries; `jaato_child_t` gets no read on `jaato_runner_t:file`, so a subprocess cannot read the runner's environ, mem, cmdline |
| `/proc/self/** r`, `owner /proc/*/limits r` | `allow jaato_runner_t self:file read`, `self:dir search` |
| `/dev/null rw`, `/dev/urandom r`, `/dev/pts/* rw` | `dev_rw_null`, `dev_read_urand`; ptys through a type of jaato's own, `jaato_devpts_t` (`term_create_pty`), never `term_use_all_ptys`, which reaches login terminals (§12, the 2a kernel run) |

Home directories (`user_home_t`), `/etc/shadow`, other services' data and
other workspaces are unreachable by type or by level without any rule
saying so. This is broader coverage than the AppArmor template, which is
allow-listed by path and so only as good as its list.

### 4.4 An escape AppArmor needs and SELinux does not

The AppArmor base profile keeps `change_profile -> unconfined` and write
access to `/proc/self/attr/current`, because the framework historically
restored its own threads (see the `apparmor_confine` docstring). That is
why `//child` exists: a subprocess must not inherit a profile it can leave
(#1323). Since phase 2 the runner never restores itself, so under SELinux
`jaato_runner_t` gets **no** `dyntransition` and **no** `setcurrent`. Code
running in the runner cannot change its own domain, and `setexec` is
constrained by `process transition` to `jaato_child_t` only. The notebook
kernel's check (#1323: "a kernel that finds itself in a profile it could
leave does not count it") therefore passes for the runner domain too, not
only for `jaato_child_t`.

## 5. MCS levels

### 5.1 Allocation

One category pair per **workspace**, not per session. This matches the
AppArmor boundary (#1033: sessions of one workspace and config root share a
profile), and it is what makes relabelling one-time.

* Derived from `sha256(realpath(workspace))` into a pair `cA,cB` with
  `A < B` in `c0..c1023`, then collision-checked against the daemon's
  allocation table; on collision, probe the next pair.
* Recorded daemon-side, never in the workspace (the runner can write the
  workspace): the WS workspace registry row gains `selinux_level`; IPC
  workspaces go in `~/.jaato/selinux_levels.json` (daemon-owned, 0600).
* The pool of pairs is 523,776; container runtimes allocate from the same
  space at random, but a collision with a container is harmless because
  type enforcement (`container_t` vs `jaato_runner_t`) already separates
  them. MCS only separates processes of one type.

### 5.2 What the level separates

| Two processes | Type | Level | Result |
|---|---|---|---|
| runner A, workspace A files | same | same | allowed |
| runner A, workspace B files | same | different | **denied by the MCS constraint** |
| runner A, runner B `/proc` | same | different | denied |
| runner A, session tmpdir of B | same | different | denied |
| two sessions of workspace A | same | same | share, as under AppArmor |

`confinement_id_for_boundary` becomes, for SELinux, a slug plus a digest of
`(policy_version, level, managed, private_tmp_dir)`. The rendered-body
digest it uses for AppArmor has no analogue (nothing is rendered), and the
inputs that change what the runner may do are exactly those four.

### 5.3 Isolated sub-runners

Phase 3. The AppArmor isolated sub-runner works in its **parent's**
workspace (`_spawn_isolated_runner` passes the parent's `workspace_path`);
what isolates it is what its flat sub-profile denies, not a separate tree.
So the SELinux sub-runner runs at the **parent's** level, in its own domain:

* `jaato_isolated_t`, or `jaato_isolated_ro_t` under
  `isolated_read_only_workspace` (workspace list and read, no write);
* flat: `child_label == label`, and neither domain may exec anything, as
  the AppArmor body grants no exec (`ix`/`px`/`ux`) outside the interpreter;
* refused: `jaato_agent_config_t`, `jaato_prompts_t`, the user tier, ptys,
  `setexec`; allowed: the workspace, `jaato_authored_t`, claims (read-only
  in the `ro` domain), its own `jaato_tmp_t`, TCP and DNS.

`isolated_workspace_subpath` has no SELinux form (it would need the
subpath labelled at another level, which the parent's runner could then not
reach), so `_provision_isolated_selinux` refuses it by name, stage
`sub_profile`, before anything is provisioned.

An earlier draft of this section gave the sub-runner its own level for its
own workspace. There is no such workspace.

## 6. Labelling a workspace

Done by the daemon (unconfined, root or with `relabelfrom/relabelto` on the
jaato types), at provisioning, before the runner spawns.

1. `realpath` the workspace; refuse one under `/`, `/usr`, `/etc`, `/home`
   itself or `$HOME` itself (relabelling a whole home is never intended).
2. Create the authored directories that do not exist yet (empty), so they
   can carry `jaato_authored_t`. A runner creating one later would give it
   `jaato_workspace_t`; pre-creating them closes that.
3. Walk the tree with `lsetfilecon` (never following symlinks), setting
   `jaato_workspace_t` / `jaato_managed_ws_t` at the workspace level, and
   the authored and claims types on their subtrees. `.git` objects are
   labelled like any other file.
4. Record a stamp in the daemon registry (not in the workspace):
   `(level, policy_version, labelled_at, file_count)`.

Later sessions check the root's label and the stamp; a match skips the
walk. A mismatch (policy version changed, level re-allocated, someone ran
`restorecon -F`) re-runs it.

Measured in phase 0 on ext4: relabelling 101k entries took 0.50 s with a
Python `setxattr` walk and 0.64 s with `chcon -R`. It runs off the event
loop, once.

**Surviving `restorecon`.** `restorecon` without `-F` resets the type and
keeps the level; with `-F` it resets both. So the install step registers
one fcontext rule for the daemon's workspace root:

```
semanage fcontext -a -t jaato_managed_ws_t '/srv/jaato/workspaces(/.*)?'
restorecon -R /srv/jaato/workspaces
```

and the per-subtree authored types via `jaato.fc` patterns relative to it.
A plain `restorecon` then preserves isolation. A `restorecon -F` strips the
levels, which the stamp check detects at the next provisioning.

**Files created by the runner.** A new file gets the type of its parent
directory (default type inheritance) and the level of the creating process
(MCS default for files is the process's low level), and the SELinux user
of the creating process. All are right without
a `type_transition` rule, except in `/tmp`, where
`type_transition jaato_runner_t tmp_t:{file dir} jaato_tmp_t` applies.

**Renaming authored directories away.** The runner could try to rename
`.jaato/agents` and create a fresh one. Renaming a directory needs `rename`
(and `reparent` to move it) on the directory itself, and deleting needs
`rmdir`; `jaato_authored_t:dir` grants none of them, nor `add_name` /
`remove_name` inside it. Guarded by a test against the compiled policy
(`sesearch`), section 11.

**User checkouts (IPC, user-CWD).** Relabelling a user's own repository is
a visible change to their files. It stays opt-in, as AppArmor confinement
is for IPC today (`IPCClient(..., apparmor=True)` becomes `confine=True`,
old name kept). A checkout under a home directory needs an fcontext rule
the operator adds; the backend refuses to relabel under `$HOME` without
one, naming the command.

## 7. Entering the domain

### 7.1 Cold spawn: exec transition

`RunnerSpawner` already forks and `execvpe`s the runner. Between the two,
in the child, call `setexeccon(label)` (write `/proc/self/attr/exec`). The
`execve` then lands in `jaato_runner_t:level` atomically, single-threaded,
before any Python code of the runner runs. This is strictly simpler than
AppArmor's `aa_change_profile` on the runner's main thread, and it removes
the #1023 problem for cold spawns: there is no thread that predates the
transition.

`private_tmp` (#1381) keeps its order: the child unshares and binds before
the exec, because `jaato_runner_t` has no `mount`. The tmpfs mounted on
`/dev/shm` gets `context="<runner user>:object_r:jaato_tmp_t:<level>"` as a
mount option. The SELinux user is the runner's, not a fixed `system_u`:
a file takes its creator's user, and the policy's object-identity
constraint refused an `unconfined_u` runner creating a file in a
`system_u` mount at its own level (phase 0, 5.6). The value is quoted,
because a category list contains commas that `mount` would otherwise split
into options.

### 7.2 Pool slots: dynamic transition

A pool slot is forked from the template and never execs. It would have to
`setcon`, and SELinux refuses `setcon` in a multi-threaded process unless
the new domain is **bounded** by the old one (`typebounds`). A slot has
threads by the time it serves its first bootstrap.

Phase plan:

* **Phase 2b (shipped): SELinux-confined sessions do not use the pool.** The
  routing gate in `spawn_session_runner` already routes `cgroup_attach`
  sessions away from the pool; SELinux confinement joins it. Cost: the
  cold start (~7 s vs ~1 s warm). Unconfined sessions still use the pool.
* **Phase 4: bounded pool.** The template runs as
  `jaato_template_t:s0-s0:c0.c1023` with
  `typebounds jaato_template_t jaato_runner_t`, and each slot calls
  `setcon(jaato_runner_t:<level>)` on its main thread. Threads created
  before the transition keep the template label, so #1023's
  `recycle_worker_pools` + `verify_thread_confinement` apply unchanged
  (the scan reads `/proc/self/task/*/attr/current`, which SELinux fills
  with a context string). The slot key already contains the label via
  `confinement_id`, so #1033 and #1100 need no change. To be verified:
  whether `typebounds` constrains a Python runtime's template enough to be
  worth it, since the template must hold every permission any runner has.

### 7.3 Model-driven subprocesses

`make_child_transition_callback` becomes per backend. For SELinux the
`preexec_fn` writes `jaato_child_t:<level>` to `/proc/self/attr/exec`, and
the following `execve` transitions. Fail-closed as today: a failed write
raises in the child and the spawn fails. `cli`, `interactive_shell` and the
notebook kernel (#1323) take it through the existing
`set_apparmor_child_transition_callback` seam, renamed
`set_child_transition_callback` with the old name kept.

## 8. Feature map

| AppArmor feature | SELinux backend |
|---|---|
| per-session profile | per-workspace level + fixed domain |
| `//child` | `jaato_child_t` via exec transition |
| isolated sub-runner | `jaato_isolated_t` / `jaato_isolated_ro_t` at the parent's level, flat (§5.3) |
| `isolated_workspace_subpath` | **refused** by name under SELinux |
| complain mode (`JAATO_APPARMOR_COMPLAIN`) | per-domain permissive (`semanage permissive -a jaato_runner_t`), host-wide; the backend reads it and reports `selinux-permissive`, WARNING once, same as #1014 |
| per-thread verification (#1023) | same code; label comparison via `lsm_label` |
| mode check (#1014), `require_confinement` means enforce | same; enforcing and not permissive |
| private `/tmp` (#1381) | same namespace code; `jaato_tmp_t` labels |
| plugin-contributed rules (`get_apparmor_rules`) | **not translated.** Workspace-internal grants are already implied; exec grants in a managed workspace come from `jaato_managed_ws_t`; anything else is logged at WARNING as unsupported under SELinux. Plugins may add `get_selinux_booleans()` later if a real need appears |
| user / workspace / cache fragments | **not translated**, same WARNING. A fragment that needs a path outside the workspace needs that path labelled (`semanage fcontext -t jaato_shared_ro_t`), which is an operator act |
| exec scoping (`exec_scope: scoped`) | **unsupported**: exec is by type, not by binary path; the grant record reports `exec_scope: unscoped` and a scoped request is refused with a clear reason if the profile demands it |
| reference grants (`selectReferences`) | a reference inside the workspace needs nothing; one outside it is refused with a reason naming `jaato_shared_ro_t`, where AppArmor would add a fragment |
| grant record (#1326) | `{backend, domain, level, child_domain, labelled_roots, policy_version, unsupported: [...]}` |
| denial hints (#1348) | phase 1: none (verdict unknown). A later version can read the AVC from `audit` via the daemon; the runner cannot |
| `/proc/*/environ` of the runner itself, read in-process | **not denied.** `self:file read` covers `/proc/self`; AppArmor denies it by path. The application check `is_sensitive_proc_path` still covers file tools. Stated, not hidden |
| template version gate | policy module version gate |

The unsupported rows are why this backend must state its coverage in
`get_environment(aspect="runtime")` and in `explain oversight`, rather than
reusing the AppArmor wording.

## 9. Operator setup (target shape)

```bash
dnf install selinux-policy-devel policycoreutils-python-utils
make -C jaato-server/selinux -f /usr/share/selinux/devel/Makefile jaato.pp
semodule -i jaato-server/selinux/jaato.pp
semanage fcontext -a -t jaato_managed_ws_t '/srv/jaato/workspaces(/.*)?'
restorecon -R /srv/jaato/workspaces
# venv outside /home so jaato_runner_t can read site-packages:
semanage fcontext -a -t lib_t '/opt/jaato/venv(/.*)?'
restorecon -R /opt/jaato/venv
# the user tier a runner may use (jaato.fc labels these; a runner can
# never create them, since it has no add_name in ~/.jaato):
mkdir -p ~/.jaato/{agents,profiles,references,services,memories,prompts,skills}
restorecon -R ~/.jaato
JAATO_CONFINEMENT=selinux JAATO_REQUIRE_CONFINEMENT=1 \
  /opt/jaato/venv/bin/python -m jaato_server ...
```

The last step is the only per-host runtime requirement; the daemon needs
root (or the relabel permissions) for section 6 and nothing privileged per
session. That is an improvement on the AppArmor path, which runs
`sudo apparmor_parser -r` per boundary.

`jaato-doctor` reports `kernel confinement` by calling
`select_daemon_backend`, the function the daemon calls at startup, so it
names the backend a daemon started now would pick. It FAILs when that
daemon would refuse to start, and WARNs when no kernel backend is
available. Under SELinux it shows the host mode, whether `jaato_runner_t`
is permissive, and the interpreter's label. It also WARNs when either the
host or the runner domain is permissive, and when `~/.jaato` is not
`jaato_user_dir_t`, naming the `restorecon` to run. Module loading, its
version and the role/entrypoint checks are §10's readiness, so a failure
there is the reason shown on the "none" line. The check runs as the
doctor's own process, so a daemon started another way may hold another
context. There is no fcontext check for `workspace_root`: the daemon
labels the workspace itself (§6).

## 10. Availability check

`SELinuxBackend._check_availability`, first failing precondition wins and
becomes `unavailable_reason`, as today:

1. Linux, `/sys/fs/selinux` mounted, `libselinux.so.1` loadable (ctypes;
   no new Python dependency).
2. `security_getenforce()` returns 0 or 1 (disabled → unavailable).
3. The policy knows `jaato_runner_t` (`security_check_context` on
   `system_u:system_r:jaato_runner_t:s0`), and the module version is at
   least the one this build needs.
4. MCS is enabled (the policy is `targeted`/`mls` with categories).
5. The daemon may start a runner: its user and role are authorized for
   `jaato_runner_t` (`security_check_context` on
   `<own user>:<own role>:jaato_runner_t:s0`), it has `process transition`
   there, and `jaato_runner_t` has `file entrypoint` on the interpreter's
   label. The transition permission alone is not a check: on targeted,
   `unconfined_t` holds it for every domain, and phase 0 found the check
   passing with the module's own rule removed. Phase 2 adds `relabelto`
   on `jaato_workspace_t`.

Permissive host-wide (`getenforce` = 0) is **available but not enforced**:
sessions run with `sandbox_mode: selinux-permissive`, a WARNING, and
`--require-confinement` refuses, matching how complain mode is handled.

## 11. Testing

The AppArmor side is tested without a kernel (CI has no LSM) and says so.
The SELinux side can do better on one axis: **policy compilation works in a
container**, only loading needs a kernel.

| Layer | Where |
|---|---|
| label parsing, mode, sandbox_mode, slot key, allocation, relabel planning | unit tests, fabricated `attr/current` strings, like the AppArmor tests |
| the policy compiles; the authored types deny write/rename; `jaato_child_t` has no `setexec`/`dyntransition`; `jaato_runner_t` has no `setcurrent`; no domain has `sys_admin`/`mount` | the `selinux-policy` CI job (Fedora container): `make` the module, link it into targeted with `semodule -N -i`, and query the linked policy with the setools Python API (checks the rules exist, not that the kernel applies them) |
| the backend's contract matches AppArmor's for the shared features | one parametrised suite over both backends through the `ConfinementBackend` protocol |
| end to end on an enforcing kernel | a manual / scheduled job on a Rocky or Fedora VM (GitHub-hosted runners are Ubuntu; Testing Farm or a self-hosted runner). Not required for merge, like the AppArmor "not verified on an enforcing kernel" notes |

Each assertion declares the `jaato.te` edit that must fail it, and the same
job checks it does (the repository meta-guard cannot: it runs on Ubuntu).

## 12. Rollout

| Phase | Content | Behaviour change |
|---|---|---|
| 0 | **done** on Fedora 44 under WSL2 ([runbook](selinux-phase0-handoff.md), findings below): exec transition from `unconfined_service_t`, MCS on files and `/proc`, relabel cost, `/dev/shm` `context=` mount, threaded `setcon` | none |
| 1a | **shipped**: `server/confinement/` (the protocol, `select_backend`, the AppArmor adapter, the SELinux readiness checks of §10), `shared/lsm_label.py` (SELinux contexts, the `selinux` / `selinux-permissive` sandbox modes). No call site uses them yet | none |
| 1b | **shipped**: the WS pre-init hook, its post-init re-run and IPC provisioning go through `AppArmorBackend.provision(Boundary)`; envelope **v8** carries `confinement: {backend, label, child_label}`; the runner's self-confinement, `//child` callback and thread verification go through `server/runner/lsm_confine.py`, which refuses a backend it cannot enter | none (refactor) |
| 2a | **shipped**: the policy module (`jaato-server/selinux/jaato.{te,fc,if}`, `jaato_runner_t`, `jaato_child_t`, the five file types, marker `jaato_policy_v1_t`), and the `selinux-policy` CI job that links it into the targeted policy in a Fedora container and checks 39 properties with setools, each with its reversion. A kernel run is a [handoff](selinux-phase2a-handoff.md) (`jaato-server/selinux/tools/probe_policy.py`). No code loads the module | none |
| 2b | **shipped, verified on a kernel** (three runs, the last at 025dd212: probe 36/36 and live sessions 9/9 under a root and a uid-1000 runner, the pty path included): user-tier types, binds and authored-file transitions in the module (1.4.0); `SELinuxBackend.provision` (levels, labelling, tmpdir); daemon selection; IPC and WS provisioning; cold spawn by exec transition; the runner confirms its domain and moves children into `jaato_child_t`. Runbook: [handoff](selinux-phase2b-handoff.md); `jaato-doctor` reports the backend and the host facts | RHEL hosts get a kernel boundary; confined sessions skip the pool |
| 3 | **implemented; three kernel runs (the last at 573f1470: probe 61/61, live 8/8 per mode as root), the uid fix not yet re-run**: `jaato_isolated_t` / `jaato_isolated_ro_t`, `jaato_agent_config_t`, `jaato_prompts_t`, `jaato_runner_log_t`, module 1.6.0 (marker `jaato_policy_v3_t`, `REQUIRED_POLICY_VERSION = 3`), `SELinuxBackend.provision_isolated`, the daemon's isolated spawn through it. Runbook: [handoff](selinux-phase3-handoff.md) | isolated subagents confined on SELinux; a v1 module is refused |
| 4 | Bounded pool slots | confined sessions warm again |
| 5 | RPM packaging, AVC-based denial hints | operator convenience |

### What the phase 2a kernel run found

Run 2026-10-02 on Fedora 44 under WSL2 (`selinux-policy-targeted` 44.10,
enforcing, MCS on), module at 5c12bc86, with
`jaato-server/selinux/tools/probe_policy.py`. Of 25 probes, 22 passed
and 3 failed. Two of the failures were the probe's own fault (a buffered
write to `/proc/self/attr/*` raises only on close, after the verdict was
printed); the kernel refused correctly (`setcurrent`, `setexec`). One was
real:

* **A pty the runner opens is unusable.** `term_use_all_ptys` grants
  `ptynode`, but nothing relabels a new pty, so its slave end is
  `devpts_t`, outside `ptynode`, and `openpty` fails with
  `out of pty devices`. Static CI cannot see this: it checks allow rules,
  not the label a new object is born with. This affects every host, not
  only WSL, and probably `jaato_child_t` too (the probe now checks it).
* **And the grant itself is too wide.** `ptynode` contains 21 types,
  `user_devpts_t` and `sshd_devpts_t` among them: the ptys of login
  terminals, created at `s0`. The MCS constraint on `chr_file` is
  `h1 dom h2`, and a runner at `s0:c101,c102` dominates `s0`, so
  `term_use_all_ptys` lets a runner read, write and `ioctl` an
  administrator's terminal if it can name it. Found reading the linked
  policy after the run, not exercised on a kernel. The fix for the pty
  failure has to remove this grant, not add beside it.

Two AVCs beside passing probes, traced on an AppArmor host with an audit
hook and `strace -k` on the same import:

* `node_bind` on `tcp ::1` comes from **urllib3's IPv6 capability probe**
  (`urllib3/util/connection.py` `_has_ipv6`, at import). Denied, urllib3
  sets `HAS_IPV6 = False` and resolves IPv4 only.
* `search` on `/sys/fs/cgroup` comes from **cryptography's Rust core**
  (`std::thread::available_parallelism` reading `/proc/self/cgroup` and
  `cpu.max`, in `openssl::init`). Denied, Rust falls back to the affinity
  mask.

`dac_override` came from the probe itself: Python's `open(..., 'w')` on
`/proc/self/attr/*` adds `O_CREAT`, which asks for write on the read-only
directory. The rest were WSL host labelling (`/dev`, `/run`, `/mnt` never
labelled by WSL's systemd), not the module.

**Rerun at cf410bcf (module 1.1.0)**: 26 of 26 probes passed, as root and
with `--as-uid 1000`, with no pty, `::1` or cgroup denial and no
`dac_*` denial in either run. So the uid drop removes none the probe
causes; whether a real runner causes any is a 2b question.

### What the phase 2b kernel run found

Run 2026-10-02 at 33bfe22b on Fedora 44 / WSL2, enforcing.

* **Probe: 34 of 34, as root and as uid 1000**, including the new binds
  (port 80 refused by `name_bind` on `http_port_t`) and the user tier
  (persona writes refused on `jaato_user_config_t`, `*_auth.json` and home
  listing refused).
* **The live session was refused**: the daemon created the session tmpdir
  with a bare `makedirs`, so it was `user_tmp_t`, and
  `prepare_session_tmpdir` had no caller. Fixed in bf978cfe (provision
  labels it, a failure refuses the session). With the directory labelled
  by hand, every live check passed: runner in `jaato_runner_t:s0:c195,c418`,
  the command in `jaato_child_t` at the same level, `sandbox_mode: selinux`,
  the workspace, authored config and created files labelled at the level.
* **A root runner's child could walk into other homes.** Search on every
  `user_home_t` directory (1.2.0) let a child stat
  `/home/<user>/...`; only missing DAC capabilities stopped it at a 0700
  home. 1.3.0 gives `~/.jaato` its own type (`jaato_user_dir_t`) and drops
  search on `user_home_t` directories; `/root`'s subdirectories stay
  traversable by name (all `admin_home_t`), stated in `jaato.te`.
* **The runner listed `~/.jaato` at bootstrap**, most likely the #1215
  redactor's `*_auth.json` scan (the earliest AVC of the run, at step 1b).
  A confined runner no longer scans its home: it cannot read those files
  under either LSM. The next run's trace confirms or names another caller.
* **An `execute` on `ldconfig`** from a child the runner forked
  (`ctypes.util.find_library`, three runner-side callers). No evidence yet
  which; `live_session.py --trace-subprocess` records it.

### The second phase 2b kernel run (ec964406)

* **Both live sessions pass end to end, 9 of 9**: a root runner and one
  dropped to uid 1000 by `--runner-uid-policy workspace-owner`, with no
  workaround. The session tmpdir is `jaato_tmp_t` at the workspace level;
  the `~/.jaato` read and the `user_tmp_t` write are gone. Probe 34 of 34
  under both uids.
* **`ldconfig` was the notebook backend's `find_library("c")`** at import
  (named by `--trace-subprocess`). Refused, it fell back to running gcc
  and objdump at every runner start. It now takes `prctl` from the
  process's own symbols (`CDLL(None)`), as `private_tmp` already did; no
  subprocess at import on any LSM.
* **The prompt library's `~/.claude/skills`** is granted under AppArmor and
  was silently skipped under SELinux (17 enforced `search` denials for a
  user with a `~/.claude`). It now rides the #1465 user-tier snapshot under
  `@home/`, read through `user_tier.home_path`, rather than relabelling
  another tool's directory. (`~/.claude/commands`, also in the AppArmor
  grant, is not read from the home by any code.)

### The third phase 2b kernel run (025dd212)

* **Everything passed**: probe 36 of 36 under both uids (including the
  two authored-file refusals, each backed by its `create` AVC), the live
  session 9 of 9 under both uids, and the pty path (`interactive_shell`)
  9 of 9 under both, the child in `jaato_child_t` on a real `/dev/pts`.
  No unrequested jaato denial in any run.
* **`jaato-doctor` reported SELinux correctly and printed a false line
  above it**: `AppArmor confinement NOT available … falls back to
  directory sandboxing only`. `AppArmorManager.is_available()` logged that
  as a side effect of answering, and under `auto` the selection asks
  AppArmor first, so every SELinux host got it, the daemon's own startup
  included. The probe now records its reason at INFO; the WARNING comes
  from the code that settles on no kernel boundary: the daemon's selection
  line, and the WS server's startup when it was handed no SELinux backend
  (a standalone WS server runs no selection, so there it is still said).

### What the phase 3 kernel run found (64020ec9)

* **Probe 57/57** as root and as uid 1000, every refusal backed by an
  enforced AVC: the agent-config and prompts types, another workspace's
  level, the user tier, `reactors.json`, `execute_no_trans` on `bin_t`,
  `setexec`, and the read-only domain's writes.
* **The sub-runner entered `jaato_isolated_t` / `jaato_isolated_ro_t` at
  the parent's level** in all four live runs (rw, ro; root, uid 1000), and
  served RPC.
* **Every live spawn then failed at `stage=forwarding` with a bare
  `TimeoutError`, on a daemon defect, not SELinux.** The runner-RPC handler
  (`async def handle`) called the synchronous `_spawn_isolated_runner` on
  the daemon loop, which then waited 10 s on `rpc.start()`, a coroutine
  only that loop could run (`LOOP_STALL` in every daemon log). Unreachable
  before this branch, since the parent-id bug refused the spawn earlier.
  Fixed: the handler hands it to `asyncio.to_thread` (the #1355 rule), the
  #1355 guard lists `_spawn_isolated_runner`, and a test drives the real
  handler on a running loop.
* **The read-only sub-runner had no log**: three `append` denials on
  `.jaato/logs/runner-<id>__sub_*.log`, 0-byte logs. Fixed with
  `jaato_runner_log_t` (above).
* **Fixed in the tools and messages:** `live_session.py --isolated` no
  longer runs the child-file check, its read-only check needs the kernel's
  refusal rather than an absent file, and it checks the sub-runner's log;
  the spawn-failure message names what was actually rolled back (nothing,
  on an SELinux host without cgroups) instead of "sub-cgroup + sub-AppArmor
  profile"; `jaato-doctor` names the policy module version.

### The second phase 3 kernel run (a1d59347)

* **Probe 61/61** as root and as uid 1000; the four log checks pass, and
  truncating the log is refused as a `write` on the `jaato_runner_log_t`
  file. The doctor names policy module v3.
* **The spawn no longer deadlocks**, and the sub-runner's log is created,
  labelled and written by both domains (~6 KB).
* **Every live session then failed in the sub-runner's bootstrap**: the
  base-instructions loader asked `<ws>/.jaato/instructions` `is_dir()`,
  the boundary denies that directory down to `getattr`, and
  `Path.is_dir()` re-raises `EACCES`. The policy was right; the runner
  probed config its boundary denies. With that one check patched in the
  host's venv copy (diagnostic only), both domains passed 8/8 end to end,
  the read-only one with the kernel refusing its write.
* **GC was off in every isolated session** for the same reason
  (`~/.jaato/gc.json` `exists()`), and the isolated envelope dropped every
  number of a profile's `gc:` block but `type` (#1133 on this path).
* **Fixed (your choice: the daemon supplies, the runner never looks):**
  the isolated envelope carries `config_resolved_by_daemon` and, without
  a profile `gc:`, the `gc.json` the daemon found (`gc_file`). The runner
  builds its runtime with `read_config_tiers=False` (premium instructions
  only, no workspace or user tier) and installs GC from the envelope,
  never from disk. `gc:` crosses whole (`to_dict`).
* **Not fixed, stated:** plugins and provider `env.py` files still list
  `<ws>/.jaato` and `~/.jaato` through `resolve_config_search_path`, and
  probe them. Every such caller tolerates the refusal (no warning, no
  functional loss in the run's logs), but each probe is an AVC: 111 to
  472 per live run, most of them `search` on `~/.jaato`.

### The third phase 3 kernel run (573f1470)

* **Probe 61/61** at both uids. **As root, both live modes pass 8/8**:
  the sub-runner bootstraps, GC comes from the envelope, the read-write
  domain writes `iso-probe.txt` and the read-only one is refused by the
  kernel.
* **Under `--as-uid 1000` the sub-runner ran as root.** The parent runner
  dropped to the workspace owner (#1168), and the isolated spawn path
  applied no uid policy at all: its spawn line had no `runs_as`, and its
  write into the owner's workspace was refused for `dac_override`, which
  the isolated domains withhold.
* **Fixed:** the sub-runner inherits its parent's resolved `RunnerUser`
  (re-resolving the policy cannot work under `peer`: a sub-runner has no
  IPC peer). The daemon hands it the session tmpdir, the session storage
  directory and its log before the spawn, the spawner drops to it, and the
  envelope carries it with that user's user tier. The spawn line now names
  `runs_as` for every runner.

### What phase 3 decided

* **The parent's level, not a new one** (§5.3), because the sub-runner
  works in the parent's workspace.
* **Two domains, not a boolean**: read-only is a second domain with no
  write on the workspace types, as the AppArmor `ro` tightening is a second
  body.
* **The read-denied config gets its own types** rather than one type for all
  authored entries, because the isolated domain needs `references`,
  `templates` and the rest, and AppArmor denies only the persona entries.
  Relabelling is driven by the label stamp, which carries the policy
  version, so a v1 workspace is relabelled once.
* **Parity, stated where it is weak**: the isolated domain can write claims
  (AppArmor's isolated body can too).
* **The sub-runner's log has a type of its own** (`jaato_runner_log_t`,
  module 1.6.0 / policy v3). The daemon creates `.jaato/logs/runner-<id>.log`
  and labels it at the level before the spawn; both isolated domains may
  open and append to it, and nothing more (no write, truncate, unlink or
  rename). Without it the read-only domain was refused its inherited log
  fds at exec and its log handler, and ran with no log (kernel run below).
  A label failure refuses the spawn, as a session tmpdir does. Main runner
  logs keep the workspace type.
* **"Executes nothing" means its own entrypoint and nothing else.** The
  isolated domains keep `execute` + `map` on `bin_t`: after the exec
  transition the kernel maps the entry interpreter under the new domain,
  so without them the domain cannot start. They have no
  `execute_no_trans` and no `transition` of their own, and no `execute`
  on shells, workspace or tmp files. Targeted gives every `domain`
  `prelink_exec_t` execute (under `fips_mode`) and a transition to
  `abrt_helper_t` (inert without execute on its entry type); the module
  cannot remove those, and the runner and child carry them too.
* **Found on the way, fixed here:** the runner's subagent plugin read the
  parent id from `_session_id` / `session_id`, which `JaatoSession` does not
  have (it is `_daemon_session_id`), so every isolated spawn on the runner
  path was refused by the daemon (`parent_session_id must be a non-empty
  str`) on any LSM. Its routing test faked the same wrong attribute on a
  `MagicMock`. Found by driving `live_session.py --isolated` locally.

### What phase 2b decided

* **Parity with AppArmor, measured, where the two differ by construction.**
  Binds: any address, ports 1024 and up, both domains (AppArmor's
  `network inet stream` allows them; checked on an enforcing kernel). The
  user tier: exactly the `~/.jaato` subtrees the plugins grant under
  AppArmor, as `jaato_user_config_t` (read) and `jaato_user_data_t`
  (write, born at `s0` so the global memory tier stays one tier).
* **The handle is passed, not stashed.** The provisioned
  `ConfinementHandle` travels as an explicit `confinement=` argument
  through `spawn_session_runner`, `RunnerSpawner.spawn`,
  `build_session_envelope` and `dispatch_bootstrap_envelope`; an AppArmor
  boundary still rides `profile_name` alone, so every existing caller is
  unchanged. A stash read by each builder would fail open on the one path
  that forgot to read it (#735).
* **Selection happens once, at daemon start** (`_select_confinement_backend`),
  before the PID file. Only an SELinux choice is stored; AppArmor keeps its
  per-transport managers. An unknown `JAATO_CONFINEMENT`, or a required
  backend that is unavailable, exits with the reason.
* **SELinux sessions are cold spawned.** `_pool_may_serve` refuses them a
  slot; the child writes the runner's context to `/proc/self/attr/exec`
  after the privilege drop (#1168) and before `execve`, and the runner
  confirms it (step 1c and the cold-spawn entry point) rather than
  transitioning. Pool support is phase 4.
* **The daemon labels everything the runner cannot create**: the
  workspace tree once (stamped in `~/.jaato/selinux_levels.json`, outside
  the workspace), the private `<ws>/.tmp`, and the session tmpdir under
  `/tmp`, keyed on the boundary's id, which the envelope descriptor now
  carries because SELinux has no profile name to read it from.
* **Found on the way, fixed here:** `AppArmorBackend` looked its grant
  record up by the bare confinement id while the record is keyed by
  profile name, so `handle.grants` was always empty. **Found, fixed
  separately:** the runner reads `~/.jaato` files no profile grants, so a
  present `permissions.json` refused every confined session (#1465,
  PR #1466); and that file's policy is never applied (raised on #1466).
* **Not done in 2b**: the isolated sub-runner stays AppArmor-only and
  refuses (phase 3); denial hints and the diagnostics grant view are
  AppArmor-only.
* **An absent `.jaato/reactors.json` or `template_routing.yaml`** (AppArmor
  denies the path whether or not it exists; a label cannot sit on an absent
  file). Module 1.4.0: a file created under either name is born
  `jaato_authored_t`, which neither domain may create, so the creation is
  refused. Filename transitions match in any workspace directory; both
  names are jaato's own. Not closed: creating it under another name and
  renaming or hard-linking it into place, which keeps the source's type.
  AppArmor refuses that by path; SELinux could only by refusing `add_name`
  in all of `.jaato`, which holds runtime state the runner writes.

### What phase 2a decided

* **ptys get a type of jaato's own** (`jaato_devpts_t`, through
  `term_create_pty` for both domains), and `term_use_all_ptys` is gone.
  Granting `devpts_t` instead was two lines shorter and would have left
  every other domain's unrelabelled pty at `s0` reachable. CI now checks
  the `type_transition` itself, which allow rules alone could not show.
* **The two import AVCs are `dontaudit`, not grants**: urllib3 resolves
  IPv4 only inside a runner, and cryptography sizes its pool from the
  affinity mask. The bind is silenced for the runner only, because a child
  that binds (a test server, `npm run dev`) is refused by the same missing
  grant and that denial must stay visible. Whether a child may bind
  loopback ports is open for 2b.

* **The module stops at the two domains phase 2b needs.** `jaato_isolated_t`
  (phase 3) and `jaato_template_t` (phase 4) are not declared; declaring
  them before anything enters them would be policy nobody tests.
* **A presence check counts only unconditional rules.** Targeted grants
  `domain domain:fd use` under the boolean `domain_fd_use` (default on),
  so "the runner may use the daemon's fds" held with the module's own line
  removed, and the fd assertions were decorative until the reversion pass
  said so. The module grants fd use on the daemon's domains itself, and
  the checks ignore conditional rules, so a host that switches the boolean
  off keeps its runners. An absence check still counts every rule: a
  permission that any boolean can grant is a permission the domain may have.
* **The reversions run in the same job, not in the meta-guard.** The
  repository's reversion meta-guard runs on Ubuntu, which has no targeted
  policy. Each check carries the `jaato.te` edit that must break it, and
  `test_each_assertion_detects_its_reversion` rebuilds and re-queries.
  Linking costs ~9.5 s, so the job takes ~9 minutes.
* **Known gaps for 2b**, found writing the rules:
  * config under `~/.jaato` (user-tier profiles, agents) is `user_home_t`,
    which the runner cannot read by design. It needs a label, or the daemon
    hands the content over as it does for the rendered profile.
  * the session tmpdir under `/tmp` must be created and labelled
    `jaato_tmp_t` by the daemon; the module gives the runner no `add_name`
    on `tmp_t`, so a runner-created tmpdir is refused, not mislabelled.
  * two runners of one workspace share a level, so each can read the
    other's `/proc/<pid>`. Same boundary as AppArmor (#1033: one profile per
    workspace); stated, not new.

### What phase 1b decided

* **`Boundary.requested_fragments` is `Optional`, and `None` ≠ `()`.** The
  phase-1a adapter folded `[]` into `None`; no call site used it then, but
  moving IPC provisioning onto it would have given a stage that declared
  `apparmor_fragments: []` an unscoped `//child`. `fragment_field` keeps
  the distinction; `plugin_rule_fields` keeps #1326's per-plugin
  attribution (`Boundary.plugin_rule_owners`), which a tuple would have
  dropped.
* **`ConfinementHandle.complain`** carries #1014's rendered mode, so the IPC
  path records `apparmor-complain` from the handle rather than asking the
  manager a second question.
* **The descriptor is derived from `profile_name`** (`envelope_descriptor`),
  the one value every spawn path already carries, on both envelope
  builders (main and isolated sub-runner). It cannot disagree with
  `profile_name` for AppArmor, and the runner refuses one that does.
* **The runner checks the descriptor first**, at the top of step 1c, before
  the empty-`profile_name` no-op: an SELinux boundary carries no AppArmor
  profile name, and reading that as "the operator opted out" would serve it
  unconfined. Steps 2d, 4 and the #1023 verification run after that gate,
  so they call the dispatcher with the AppArmor backend.
* **Left for phase 2**, where SELinux needs them: the cold-spawn path in
  `runner/__main__.py` still confines from `JAATO_RUNNER_PROFILE` before any
  envelope exists (SELinux cold spawn enters its domain by exec transition
  instead), the step-1c idempotency readback still parses an AppArmor label,
  and the `sandbox_mode_is_kernel` rename of §3.3 is not done.

### What phase 1a decided

* **The SELinux backend reports `is_available() == False` on a ready host**,
  with the reason "the host is ready … but this build cannot provision
  SELinux boundaries yet". `host_readiness()` is the check itself. Saying
  "available" before provisioning exists would let selection pick a backend
  that confines nothing.
* **The policy version is read from a marker type**, `jaato_policy_v<N>_t`,
  which the module (phase 2) must declare. Asking whether a context is valid
  needs no privilege; `semodule -l` needs root.
* **An unknown `JAATO_CONFINEMENT` refuses to start** rather than falling back
  to `auto`.
* **The label's backend is named by the caller, never guessed** from the
  string, so an AppArmor profile name containing `:` cannot read as an
  SELinux context.
* **SELinux "confined" means one of jaato's domains.** Every task on an
  SELinux host has a domain; `unconfined_t` is not a jaato boundary.

### What phase 0 found

Run on Fedora 44 (`selinux-policy-targeted` 44.10, RHEL's upstream) under
WSL2, kernel 6.18.40.1-microsoft-standard-WSL2, with a throwaway module.

| Question | Answer |
|---|---|
| exec transition into `jaato_runner_t:s0:c1,c2` via `setexeccon` | works from `unconfined_t` and from `unconfined_service_t` (a `systemd-run` unit), including `s0` to a category pair |
| MCS on workspace files | cross-level read, write and directory search denied; same level works; a created file takes the creator's level |
| MCS on `/proc` | the whole `/proc/<pid>` directory is closed across levels (`search` on the directory), so `environ`, `status`, `cmdline`, `cwd` all fail at once |
| `restorecon` | plain keeps the level; `-F` resets level **and** user |
| relabel cost | 0.50–0.64 s per 101k entries |
| labelled `/dev/shm` tmpfs | works with the runner's own SELinux user; not with a fixed `system_u` (§7.1) |
| `setcon` in a threaded process | refused with `EPERM`; single-threaded succeeds (§7.2 holds) |
| readiness check 5 as first written | vacuous on targeted; rewritten (§10) |
| permissive domain | read as `selinux-permissive` by `lsm_label` |

Findings about WSL, which matter to anyone testing there and to AppArmor
users on WSL:

* **The stock WSL2 kernel runs SELinux, not AppArmor.** It boots with
  `lsm=capability,landlock,yama,safesetid,selinux,ima`; AppArmor is built in
  and skipped. So jaato's AppArmor backend is never available on a stock
  WSL2 host, whatever the distro, and Ubuntu under WSL has no MAC at all
  (SELinux with no policy allows everything). Enabling AppArmor needs an
  `lsm=` override in `.wslconfig`, which applies to every distro.
* **The policy is never loaded at boot.** systemd is not PID 1 in the
  kernel's sense and skips its SELinux setup. `securityfs` and `selinuxfs`
  must be mounted and `load_policy -i` run by hand after every boot,
  followed by a relabel, because files created before the load are
  `unlabeled_t`.
* **One kernel, one policy.** A policy loaded from one distro applies to
  every WSL distro, Docker Desktop included, until `wsl --shutdown`.
* **Enforcing breaks WSL itself.** Fedora's targeted policy confines
  `kernel_t`, which WSL's `/init` and session plumbing run in; the first
  enforcing load killed the session. Making `kernel_t` and
  `kernel_generic_helper_t` permissive was needed. WSL is a test bed for
  the policy, not a supported deployment.
* `sestatus` prints the policy name from `/etc/selinux/config` even when no
  policy is loaded; contexts reading `kernel` are the reliable sign.
  `/sys/fs/selinux` exists as an empty directory before `selinuxfs` is
  mounted, which is why check 1 looks for the `enforce` file.

## 13. Open questions

1. **Daemon domain.** Keeping the daemon in `unconfined_service_t` mirrors
   AppArmor. Confining the daemon itself (`jaato_daemon_t`) is possible and
   out of scope here.
2. **Relabel ownership on a non-root daemon.** A service user needs
   `relabelfrom/relabelto` and `process transition`; granting those to an
   `unconfined_t` service user is policy the operator must accept. Section 9
   assumes root, as the private `/tmp` does.
3. **Bounded pool.** Whether the template can be written as a bound
   (a superset domain) tight enough to be worth it, or whether confined
   sessions should simply always cold-spawn on SELinux.
4. **Shared read-only data.** Whether `jaato_shared_ro_t` (operator-labelled,
   readable by every runner at any level) is the right way to express
   "read `/srv/corpus`", given that it is global rather than per workspace.
