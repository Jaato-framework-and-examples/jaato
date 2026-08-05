#!/usr/bin/env bash
# Real-host verification for the §5.11d-v2 cgroup-nft egress mechanism.
#
#   sudo bash scripts/verify_egress_nft.sh
#
# Proves a process in a per-session cgroup can reach the loopback egress proxy
# but NOT an external host, while a process outside the cgroup is unaffected.
# PROVEN on Ubuntu kernel 6.8.0-85 / nftables 1.0.9 (2026-07-04): Test A ->
# external BLOCKED (refused), Test B -> both CONNECTED. Self-cleaning.
# Enforcement test for the §5.11d-v2 cgroup-nft egress mechanism.
# Proves: a process in a per-session cgroup can reach the loopback egress proxy
# but NOT an external host, while a process OUTSIDE the cgroup is unaffected.
# Self-contained + self-cleaning. Must run as root (nft + cgroup writes).
set -u
CG=/sys/fs/cgroup/jaato_egress_test
TABLE=jaato_egress_test
PORTF=/tmp/jaato_egr_port
LISTENER=""

log() { printf '%s\n' "$*"; }
cleanup() {
  [ -n "$LISTENER" ] && kill "$LISTENER" 2>/dev/null
  nft delete table inet "$TABLE" 2>/dev/null
  if [ -d "$CG" ]; then
    # evacuate any stragglers, then remove
    if [ -f "$CG/cgroup.procs" ]; then
      while read -r pid; do echo "$pid" > /sys/fs/cgroup/cgroup.procs 2>/dev/null; done < "$CG/cgroup.procs"
    fi
    rmdir "$CG" 2>/dev/null
  fi
  rm -f "$PORTF" 2>/dev/null
}
trap cleanup EXIT

[ "$(id -u)" -eq 0 ] || { log "ERROR: must run as root (sudo bash $0)"; exit 2; }

log "=== 0. preconditions ==="
log "kernel: $(uname -r) ; nft: $(nft --version)"

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

log "=== 3. install cgroup-scoped egress ruleset ==="
# Gate ONLY our cgroup: allow ->127.0.0.1 (the proxy) + ::1, reject the rest.
# All other host traffic falls through to 'policy accept' (host unaffected).
nft -f - <<EOF
table inet $TABLE {
  chain out {
    type filter hook output priority 0; policy accept;
    socket cgroupv2 level 1 "jaato_egress_test" jump gate
  }
  chain gate {
    ip  daddr 127.0.0.1 accept
    ip6 daddr ::1       accept
    counter reject
  }
}
EOF
if [ $? -ne 0 ]; then
  log "FAIL: nft could not load the cgroupv2 ruleset (socket cgroupv2 unsupported?)"
  exit 1
fi
log "ruleset loaded:"
nft list table inet "$TABLE" | sed 's/^/    /'

# helper: run a python connect-probe, optionally joining the cgroup first
probe() {  # $1 = "in" | "out"
  local where="$1"
  python3 - "$PORT" "$where" <<'PY'
import os, socket, sys
port = int(sys.argv[1]); where = sys.argv[2]
if where == "in":
    open("/sys/fs/cgroup/jaato_egress_test/cgroup.procs", "w").write(str(os.getpid()))
def try_connect(label, addr):
    try:
        socket.create_connection(addr, 3).close(); print(f"    {label}: CONNECTED")
    except PermissionError: print(f"    {label}: BLOCKED (PermissionError)")
    except (ConnectionRefusedError,) as e: print(f"    {label}: BLOCKED (refused)")
    except OSError as e: print(f"    {label}: BLOCKED/err ({e.__class__.__name__}: {e})")
try_connect("loopback 127.0.0.1", ("127.0.0.1", port))
try_connect("external 8.8.8.8 ", ("8.8.8.8", 53))
PY
}

log "=== 4. TEST A: process INSIDE the session cgroup ==="
log "    expect: loopback CONNECTED, external BLOCKED"
probe in

log "=== 5. TEST B: process OUTSIDE the cgroup (scoping / host-safety proof) ==="
log "    expect: both CONNECTED (rule must not affect other processes)"
probe out

log "=== done (cleaning up table + cgroup) ==="
