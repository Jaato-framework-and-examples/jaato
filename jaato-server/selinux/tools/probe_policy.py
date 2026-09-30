#!/usr/bin/env python3
"""Probe the jaato SELinux module on an enforcing kernel (phase 2a).

Run as root on an SELinux host with the module loaded
(docs/design/selinux-phase2a-handoff.md).  Uses nothing from jaato: it
labels two scratch workspaces at two MCS levels and execs small probes
into ``jaato_runner_t`` and ``jaato_child_t`` the way the runtime will
(``setexeccon`` then ``execve``), then checks each property
``tests/test_policy_rules.py`` asserts statically, this time against the
kernel.  Prints one table and writes ``probe-results.json`` plus the AVC
denials the run produced.

    python3 probe_policy.py [--root /srv/jaato-2a] [--venv /opt/jaato/venv]

``--venv`` adds one check: the runner domain imports
``jaato_server.server.runner`` from that venv (it must be labelled
``lib_t``, design §9).

Every probe answers PASS / FAIL / SKIP with the observed value.  A FAIL is
a finding, not a script error; send the whole output back.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from typing import List, Optional, Tuple

L1 = "s0:c101,c102"
L2 = "s0:c103,c104"
RESULTS: List[dict] = []


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def own_user_role() -> Tuple[str, str]:
    ctx = open("/proc/self/attr/current").read().strip("\0\n ")
    user, role = ctx.split(":")[:2]
    return user, role


def context(domain: str, level: str) -> str:
    user, role = own_user_role()
    return f"{user}:{role}:{domain}:{level}"


def chcon(path: str, ftype: str, level: str, recursive: bool = True) -> None:
    user, _ = own_user_role()
    cmd = ["chcon"] + (["-R"] if recursive else []) + [
        "-u", user, "-r", "object_r", "-t", ftype, "-l", level, path]
    subprocess.run(cmd, check=True)


def run_in(domain: str, level: str, argv: List[str], *, pass_fds=(),
           timeout: float = 30.0) -> subprocess.CompletedProcess:
    """Exec *argv* in *domain*:*level*: setexeccon in the child, then exec."""
    label = context(domain, level)

    def preexec():
        with open("/proc/self/attr/exec", "w") as f:
            f.write(label)

    return subprocess.run(argv, preexec_fn=preexec, pass_fds=pass_fds,
                          capture_output=True, text=True, timeout=timeout)


def py_in(domain: str, level: str, code: str, **kw) -> subprocess.CompletedProcess:
    return run_in(domain, level, [sys.executable, "-I", "-c", code], **kw)


def record(name: str, ok: Optional[bool], observed: str) -> None:
    state = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append({"check": name, "result": state, "observed": observed[-400:]})
    print(f"{state:4}  {name}\n      {observed.strip()[-300:]}")


def expect_ok(name: str, p: subprocess.CompletedProcess, want: str = "OK") -> None:
    out = (p.stdout + p.stderr).strip()
    record(name, p.returncode == 0 and want in out, f"rc={p.returncode} {out}")


def expect_denied(name: str, p: subprocess.CompletedProcess) -> None:
    out = (p.stdout + p.stderr).strip()
    record(name, "DENIED" in out and "LEAK" not in out, f"rc={p.returncode} {out}")


# A probe prints OK on success, DENIED on EACCES/EPERM, LEAK when a thing
# that must fail succeeded.
TRY = '''
import os, sys
def attempt(fn, must_fail):
    try:
        fn()
    except PermissionError as e:
        print("DENIED", e); return
    except OSError as e:
        print("OSERROR", e); return
    print("LEAK" if must_fail else "OK")
'''


# ----------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------

def setup(root: str) -> dict:
    shutil.rmtree(root, ignore_errors=True)
    paths = {
        "wsA": f"{root}/wsA", "wsB": f"{root}/wsB", "wsU": f"{root}/wsU",
        "tmpA": f"/tmp/jaato-2a-{os.getpid()}",
    }
    for key in ("wsA", "wsB", "wsU"):
        os.makedirs(f"{paths[key]}/.jaato/agents", exist_ok=True)
        os.makedirs(f"{paths[key]}/.jaato/references-claims", exist_ok=True)
        os.makedirs(f"{paths[key]}/bin", exist_ok=True)
        open(f"{paths[key]}/file.txt", "w").write("workspace\n")
        open(f"{paths[key]}/.jaato/agents/a.md", "w").write("persona\n")
        shutil.copy("/usr/bin/true", f"{paths[key]}/bin/t")
    os.makedirs(paths["tmpA"], exist_ok=True)
    # wsA: a managed workspace at L1; wsB: another workspace at L2;
    # wsU: a user's own checkout (not executable) at L1.
    chcon(paths["wsA"], "jaato_managed_ws_t", L1)
    chcon(paths["wsB"], "jaato_managed_ws_t", L2)
    chcon(paths["wsU"], "jaato_workspace_t", L1)
    for key, lvl in (("wsA", L1), ("wsB", L2), ("wsU", L1)):
        chcon(f"{paths[key]}/.jaato/agents", "jaato_authored_t", lvl)
        chcon(f"{paths[key]}/.jaato/references-claims", "jaato_claims_t", lvl)
    chcon(paths["tmpA"], "jaato_tmp_t", L1)
    return paths


# ----------------------------------------------------------------------
# probes
# ----------------------------------------------------------------------

def probes(p: dict, venv: Optional[str]) -> None:
    R, C = "jaato_runner_t", "jaato_child_t"

    expect_ok("runner: enters jaato_runner_t at its level",
              py_in(R, L1, "print('OK', open('/proc/self/attr/current').read())"),
              want=f"{R}:{L1}")

    expect_ok("runner: reads and writes its workspace",
              py_in(R, L1, TRY + f"""
attempt(lambda: open('{p['wsA']}/file.txt').read(), False)
attempt(lambda: open('{p['wsA']}/new.txt','w').write('x'), False)
attempt(lambda: os.rename('{p['wsA']}/new.txt','{p['wsA']}/new2.txt'), False)
attempt(lambda: os.unlink('{p['wsA']}/new2.txt'), False)
attempt(lambda: os.mkdir('{p['wsA']}/d'), False)"""))

    expect_denied("runner: another workspace's files are refused by level",
                  py_in(R, L1, TRY + f"attempt(lambda: open('{p['wsB']}/file.txt').read(), True)"))

    expect_ok("runner: reads authored config",
              py_in(R, L1, TRY + f"attempt(lambda: open('{p['wsA']}/.jaato/agents/a.md').read(), False)"))
    for what, code in (
            ("writes an authored file", f"open('{p['wsA']}/.jaato/agents/a.md','a').write('x')"),
            ("creates an authored file", f"open('{p['wsA']}/.jaato/agents/b.md','w').write('x')"),
            ("renames the authored directory away", f"os.rename('{p['wsA']}/.jaato/agents','{p['wsA']}/.jaato/agents.old')"),
            ("removes an authored file", f"os.unlink('{p['wsA']}/.jaato/agents/a.md')")):
        expect_denied(f"runner: {what} is refused",
                      py_in(R, L1, TRY + f"attempt(lambda: {code}, True)"))

    expect_ok("runner: writes a reference claim",
              py_in(R, L1, TRY + f"attempt(lambda: open('{p['wsA']}/.jaato/references-claims/c.json','w').write('x'), False)"))
    expect_ok("runner: writes its session tmpdir",
              py_in(R, L1, TRY + f"attempt(lambda: open('{p['tmpA']}/t','w').write('x'), False)"))
    expect_denied("runner: creating a file in the host's /tmp is refused",
                  py_in(R, L1, TRY + f"attempt(lambda: open('/tmp/jaato-2a-leak-{os.getpid()}','w').write('x'), True)"))
    expect_denied("runner: /root is unreachable",
                  py_in(R, L1, TRY + "attempt(lambda: os.listdir('/root'), True)"))
    expect_denied("runner: cannot change its own domain",
                  py_in(R, L1, TRY + f"attempt(lambda: open('/proc/self/attr/current','w').write('{context('unconfined_t', 's0')}'), True)"))

    # The inherited socketpair: the daemon's RPC channel.
    for fd_use in ("on", "off"):
        if fd_use == "off":
            subprocess.run(["setsebool", "domain_fd_use", "0"], check=True)
        try:
            a, b = socket.socketpair()
            r = py_in(R, L1, f"import socket; s=socket.socket(fileno={b.fileno()}); s.sendall(b'OK-fd'); print('sent')",
                      pass_fds=(b.fileno(),))
            b.close()
            a.settimeout(5)
            try:
                got = a.recv(16).decode()
            except OSError as e:
                got = f"recv failed: {e}"
            a.close()
            record(f"runner: uses the daemon's socketpair (domain_fd_use {fd_use})",
                   got == "OK-fd", f"got={got!r} rc={r.returncode} {r.stdout}{r.stderr}")
        finally:
            if fd_use == "off":
                subprocess.run(["setsebool", "domain_fd_use", "1"], check=True)

    expect_ok("runner: opens a pty (interactive_shell)",
              py_in(R, L1, "import pty, os; m, s = pty.openpty(); os.write(s, b'x'); print('OK', os.read(m, 1))"))

    try:
        socket.create_connection(("1.1.1.1", 443), timeout=3).close()
        online = True
    except OSError:
        online = False
    if online:
        expect_ok("runner: resolves DNS and connects out over TCP",
                  py_in(R, L1, "import socket; socket.create_connection(('example.com', 443), timeout=10).close(); print('OK')"))
    else:
        record("runner: resolves DNS and connects out over TCP", None, "host is offline")

    # //child, entered the way cli will enter it: from inside the runner.
    child_ctx = context(C, L1)
    spawn = f"""
import subprocess
def pre():
    open('/proc/self/attr/exec','w').write('{child_ctx}')
p = subprocess.run(CMD, preexec_fn=pre, capture_output=True, text=True)
print(p.stdout + p.stderr)
"""
    expect_ok("child: the runner execs /bin/sh into jaato_child_t",
              py_in(R, L1, spawn.replace("CMD", "['/bin/sh','-c','cat /proc/self/attr/current; echo; echo OK']")),
              want=f"{C}:{L1}")
    expect_denied("child: cannot read the runner's /proc/<pid>/environ",
                  py_in(R, L1, spawn.replace("CMD", "['/bin/sh','-c','cat /proc/$PPID/environ >/dev/null 2>&1 && echo LEAK || echo DENIED']")))
    expect_denied("child: cannot set an exec context (cannot leave //child)",
                  py_in(R, L1, spawn.replace("CMD", f"['{sys.executable}','-I','-c',\"open('/proc/self/attr/exec','w').write('x')\"]")))
    expect_denied("child: cannot write a reference claim",
                  py_in(R, L1, spawn.replace("CMD", f"['/bin/sh','-c','echo x > {p['wsA']}/.jaato/references-claims/f.json && echo LEAK || echo DENIED']")))
    expect_ok("child: runs a binary a managed workspace holds",
              py_in(R, L1, spawn.replace("CMD", f"['/bin/sh','-c','{p['wsA']}/bin/t && echo OK']")))
    expect_denied("child: cannot run a binary from a user's own checkout",
                  py_in(R, L1, spawn.replace("CMD", f"['/bin/sh','-c','{p['wsU']}/bin/t && echo LEAK || echo DENIED']")))
    expect_ok("child: writes the workspace",
              py_in(R, L1, spawn.replace("CMD", f"['/bin/sh','-c','echo x > {p['wsA']}/child.txt && echo OK']")))

    if venv:
        expect_ok("runner: imports jaato_server.server.runner from the venv",
                  run_in(R, L1, [f"{venv}/bin/python", "-c",
                                 "import jaato_server.server.runner as m; print('OK', m.__file__)"]))
    else:
        record("runner: imports jaato_server.server.runner from the venv", None, "--venv not given")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/srv/jaato-2a")
    ap.add_argument("--venv")
    args = ap.parse_args()
    if os.geteuid() != 0:
        print("run as root (chcon, setsebool)", file=sys.stderr)
        return 2
    enforce = open("/sys/fs/selinux/enforce").read().strip()
    print(f"enforcing={enforce}  context={open('/proc/self/attr/current').read().strip(chr(0))}")
    start = time.strftime("%H:%M:%S")
    paths = setup(args.root)
    probes(paths, args.venv)
    time.sleep(1)
    avc = subprocess.run(["ausearch", "-m", "AVC,USER_AVC", "-ts", start, "-i"],
                         capture_output=True, text=True).stdout
    open("probe-avc.txt", "w").write(avc)
    json.dump({"enforcing": enforce, "results": RESULTS}, open("probe-results.json", "w"), indent=2)
    fails = [r for r in RESULTS if r["result"] == "FAIL"]
    print(f"\n{len(RESULTS) - len(fails)} of {len(RESULTS)} not failed; "
          f"{len(fails)} FAIL; AVC records: {avc.count('type=AVC')} (probe-avc.txt)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
