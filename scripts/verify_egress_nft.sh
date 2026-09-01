#!/usr/bin/env bash
# Real-host verification for the §5.11d-v2 cgroup-nft egress mechanism.
#
#   sudo bash scripts/verify_egress_nft.sh
#
# Proves a process in a per-session cgroup can reach the loopback egress proxy
# but NOT another service on IPv4 loopback, NOT anything on IPv6 loopback
# (#696), and NOT an external host — while a process outside the cgroup is
# unaffected.  PROVEN on Ubuntu kernel 6.8.0-85 / nftables 1.0.9.
# Self-contained + self-cleaning.  Must run as root (nft + cgroup writes).
#
# This is a TEST, not a demonstration: it installs the ruleset produced by the
# production renderer (server/egress_proxy/nft.py::render_ruleset) rather than
# a hand-written copy of it, asserts each probe against its expected outcome,
# and exits non-zero on any failure.  A script that mirrors the renderer passes
# identically on a fixed and an unfixed tree, which is no guard at all — the
# #696 regression (an unscoped `ip6 daddr ::1 accept`) has to reach the kernel
# from the real code path for the v6 probe to catch it.
set -u

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NFT_PY="$REPO_ROOT/jaato-server/server/egress_proxy/nft.py"
CG=/sys/fs/cgroup/jaato_egress_test
SESSION_ID=test                 # render_ruleset() -> table `jaato_egress_test`
TABLE=jaato_egress_test         # re-derived from the rendered ruleset below
PORTF=/tmp/jaato_egr_port
BPORTF=/tmp/jaato_egr_bport
RESF=/tmp/jaato_egr_result
LISTENER=""
BYSTANDER=""
FAILURES=0
SKIPS=0

log() { printf '%s\n' "$*"; }
cleanup() {
  [ -n "$LISTENER" ] && kill "$LISTENER" 2>/dev/null
  [ -n "$BYSTANDER" ] && kill "$BYSTANDER" 2>/dev/null
  nft delete table inet "$TABLE" 2>/dev/null
  if [ -d "$CG" ]; then
    # evacuate any stragglers, then remove
    if [ -f "$CG/cgroup.procs" ]; then
      while read -r pid; do echo "$pid" > /sys/fs/cgroup/cgroup.procs 2>/dev/null; done < "$CG/cgroup.procs"
    fi
    rmdir "$CG" 2>/dev/null
  fi
  rm -f "$PORTF" "$BPORTF" "$RESF" 2>/dev/null
}
trap cleanup EXIT

[ "$(id -u)" -eq 0 ] || { log "ERROR: must run as root (sudo bash $0)"; exit 2; }
[ -f "$NFT_PY" ] || { log "ERROR: cannot find $NFT_PY"; exit 2; }

log "=== 0. preconditions ==="
log "kernel: $(uname -r) ; nft: $(nft --version)"
log "renderer: $NFT_PY"

log "=== 1. create per-session test cgroup ==="
mkdir -p "$CG" || { log "FAIL: cannot mkdir cgroup"; exit 1; }
log "created $CG"

log "=== 2. loopback listener (egress-proxy stand-in), OUTSIDE the cgroup ==="
python3 - <<'PY' &
import socket, threading, time
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 0)); s.listen(4)
open("/tmp/jaato_egr_port", "w").write(str(s.getsockname()[1]))
def loop():
    while True:
        try: c, _ = s.accept(); c.close()
        except OSError: break
threading.Thread(target=loop, daemon=True).start(); time.sleep(60)
PY
LISTENER=$!
sleep 1
PORT=$(cat "$PORTF" 2>/dev/null)
[ -n "$PORT" ] || { log "FAIL: listener did not start"; exit 1; }
log "listener on 127.0.0.1:$PORT"

log "=== 2b. DUAL-STACK bystander service, OUTSIDE the cgroup ==="
log "        (stands in for ollama / the WS server / any other local daemon)"
python3 - <<'PY' &
import socket, threading, time
# Prefer a dual-stack bind so one listener stands in on BOTH ::1 and 127.0.0.1;
# fall back to IPv4-only on hosts built without IPv6 (the v6 probe is skipped).
try:
    s = socket.socket(socket.AF_INET6)
    s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("::", 0))
    v6 = True
except OSError:
    s = socket.socket(socket.AF_INET)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    v6 = False
s.listen(4)
open("/tmp/jaato_egr_bport", "w").write("%d %d" % (s.getsockname()[1], int(v6)))
def loop():
    while True:
        try: c, _ = s.accept(); c.close()
        except OSError: break
threading.Thread(target=loop, daemon=True).start(); time.sleep(60)
PY
BYSTANDER=$!
sleep 1
read -r BPORT HAS_V6 < "$BPORTF" 2>/dev/null
[ -n "${BPORT:-}" ] || { log "FAIL: bystander did not start"; exit 1; }
if [ "${HAS_V6:-0}" = "1" ]; then
  log "bystander on [::]:$BPORT (reachable as ::1:$BPORT and 127.0.0.1:$BPORT)"
else
  log "bystander on 127.0.0.1:$BPORT (host has no IPv6)"
fi

log "=== 3. render the ruleset with the PRODUCTION renderer ==="
# nft.py imports nothing but stdlib and has no package-relative imports, so it
# loads standalone by path — this script stays dependency-free while still
# exercising the code the daemon actually ships.
RULESET=$(python3 - "$NFT_PY" "$SESSION_ID" "$CG" "$PORT" <<'PY'
import importlib.util, sys
path, session_id, cgroup, port = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
spec = importlib.util.spec_from_file_location("jaato_egress_nft", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.stdout.write(mod.render_ruleset(session_id, cgroup, port))
PY
) || { log "FAIL: render_ruleset() raised — see traceback above"; exit 1; }

TABLE=$(printf '%s\n' "$RULESET" | sed -n 's/^table inet \([A-Za-z0-9_]\{1,\}\) {.*$/\1/p' | head -1)
[ -n "$TABLE" ] || { log "FAIL: could not read the table name out of the rendered ruleset"; exit 1; }
log "render_ruleset($SESSION_ID, $CG, $PORT) ->"
printf '%s\n' "$RULESET" | sed 's/^/    /'

log "=== 3b. load it ==="
if ! printf '%s\n' "$RULESET" | nft -f -; then
  log "FAIL: nft could not load the ruleset (socket cgroupv2 unsupported?)"
  exit 1
fi
log "loaded as table inet $TABLE:"
nft list table inet "$TABLE" | sed 's/^/    /'

# helper: run a python connect-probe, optionally joining the cgroup first.
# Writes machine-readable `key=STATUS` lines to $RESF and a human line to stdout.
probe() {  # $1 = "in" | "out"
  local where="$1"
  python3 - "$PORT" "$BPORT" "$where" "${HAS_V6:-0}" "$RESF" <<'PY'
import os, socket, sys
port = int(sys.argv[1]); bport = int(sys.argv[2]); where = sys.argv[3]
has_v6 = sys.argv[4] == "1"; resf = sys.argv[5]
if where == "in":
    open("/sys/fs/cgroup/jaato_egress_test/cgroup.procs", "w").write(str(os.getpid()))
out = []
def try_connect(key, label, addr):
    try:
        socket.create_connection(addr, 3).close(); status, detail = "CONNECTED", ""
    except PermissionError:              status, detail = "BLOCKED", " (PermissionError)"
    except ConnectionRefusedError:       status, detail = "BLOCKED", " (refused)"
    except OSError as e:                 status, detail = "BLOCKED", f" ({e.__class__.__name__}: {e})"
    print(f"    {label}: {status}{detail}")
    out.append(f"{key}={status}")
try_connect("proxy",    f"proxy      127.0.0.1:{port}",  ("127.0.0.1", port))
try_connect("other_v4", f"other v4   127.0.0.1:{bport}", ("127.0.0.1", bport))
if has_v6:
    try_connect("other_v6", f"other v6         ::1:{bport}", ("::1", bport))
else:
    print(f"    other v6         ::1:{bport}: SKIPPED (no IPv6 on this host)")
    out.append("other_v6=SKIPPED")
try_connect("external", "external    8.8.8.8:53",        ("8.8.8.8", 53))
open(resf, "w").write("\n".join(out) + "\n")
PY
}

expect() {  # $1 = key, $2 = expected status, $3 = what it proves
  local got
  got=$(sed -n "s/^$1=//p" "$RESF")
  if [ "${got:-}" = "SKIPPED" ]; then
    log "    SKIP  $1 — not exercised on this host ($3)"
    SKIPS=$((SKIPS + 1))
  elif [ "${got:-}" = "$2" ]; then
    log "    PASS  $1 = $got ($3)"
  else
    log "    FAIL  $1 — expected $2, got ${got:-<no result>} ($3)"
    FAILURES=$((FAILURES + 1))
  fi
}

# Reads the gate chain's reject counter.  Zero after a probe that should have
# been rejected means no packet ever reached the chain — i.e. `socket cgroupv2`
# did not match — which is an unsupported host, not a broken rule.
gate_rejects() {
  nft list table inet "$TABLE" 2>/dev/null \
    | sed -n 's/.*counter packets \([0-9]\{1,\}\) .*/\1/p' | head -1
}

log "=== 4. TEST A: process INSIDE the session cgroup ==="
log "    expect: proxy CONNECTED; other-v4, other-v6 (#696) and external BLOCKED"
probe in
if [ "$(gate_rejects)" = "0" ]; then
  log "    ERROR: the gate's reject counter is still 0 — not one packet from the"
  log "           probe reached the chain, so 'socket cgroupv2' never matched on"
  log "           this host.  Common inside containers, where /sys/fs/cgroup is a"
  log "           namespaced view and nft resolves the path against the host"
  log "           hierarchy.  The mechanism is UNVERIFIED here, not broken —"
  log "           re-run on a host with a real cgroup2 root."
  exit 2
fi
expect proxy    CONNECTED "the gate does not break the session's own egress"
expect other_v4 BLOCKED   "another loopback port is not reachable"
expect other_v6 BLOCKED   "#696 — IPv6 loopback is not wide open"
expect external BLOCKED   "the CONNECT allowlist is not bypassable"

log "=== 5. TEST B: process OUTSIDE the cgroup (scoping / host-safety proof) ==="
log "    expect: all CONNECTED (rule must not affect other processes)"
probe out
expect proxy    CONNECTED "the rule is cgroup-scoped"
expect other_v4 CONNECTED "the rule is cgroup-scoped"
expect other_v6 CONNECTED "the rule is cgroup-scoped"
OUTSIDE_EXTERNAL=$(sed -n 's/^external=//p' "$RESF")
if [ "${OUTSIDE_EXTERNAL:-}" = "CONNECTED" ]; then
  log "    PASS  external = CONNECTED (the rule is cgroup-scoped)"
else
  # Not a failure: a host with no internet blocks this outside the cgroup too.
  log "    NOTE  external = ${OUTSIDE_EXTERNAL:-<no result>} outside the cgroup —"
  log "          this host has no external egress, so TEST A's external leg"
  log "          proved nothing.  The loopback assertions above still hold."
fi

log "=== 6. verdict ==="
if [ "$SKIPS" -gt 0 ]; then
  log "    WARNING: $SKIPS assertion(s) SKIPPED — this host has no IPv6, so the"
  log "             #696 assertion this script exists for DID NOT RUN.  This run"
  log "             is not a regression proof; re-run on a dual-stack host."
fi
if [ "$FAILURES" -gt 0 ]; then
  log "    FAIL: $FAILURES assertion(s) failed (cleaning up table + cgroup)"
  exit 1
fi
log "    PASS: all assertions held (cleaning up table + cgroup)"
