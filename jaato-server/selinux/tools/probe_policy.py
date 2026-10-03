#!/usr/bin/env python3
"""Probe the jaato SELinux module on an enforcing kernel (phases 2a to 4).

Run as root on an SELinux host with the module loaded
(docs/design/selinux-phase2a-handoff.md).  Uses nothing from jaato: it
labels two scratch workspaces at two MCS levels and execs small probes
into ``jaato_runner_t``, ``jaato_child_t`` and the two isolated domains
(phase 3), and enters the runner domain at fork the way a pool slot does
(phase 4, ``setcon`` in a forked child of a threaded process), the way the
runtime will
(``setexeccon`` then ``execve``), then checks each property
``tests/test_policy_rules.py`` asserts statically, this time against the
kernel.  Prints one table and writes ``probe-results.json`` plus the AVC
denials the run produced.

    python3 probe_policy.py [--root /srv/jaato-2a] [--venv /opt/jaato/venv]
                            [--as-uid 1000]

``--venv`` adds one check: the runner domain imports
``jaato_server.server.runner`` from that venv (it must be labelled
``lib_t``, design §9).  ``--as-uid`` runs every probe as that uid, as a
runner does under ``--runner-uid-policy`` (#1168), so the ``dac_override``
denials a root probe causes do not appear.

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
# --as-uid: the probes run as this uid, as runners do under #1168's
# --runner-uid-policy.  None = root, which adds dac_override AVCs a
# dropped runner would not produce.
AS_UID: Optional[int] = None


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
           cwd: Optional[str] = None,
           timeout: float = 30.0) -> subprocess.CompletedProcess:
    """Exec *argv* in *domain*:*level*: setexeccon in the child, then exec."""
    label = context(domain, level)

    def preexec():
        # Unbuffered: a buffered write raises at close, not at the call
        # (phase 2a run, finding 3a).
        fd = os.open("/proc/self/attr/exec", os.O_WRONLY)
        os.write(fd, label.encode())
        os.close(fd)
        if AS_UID is not None:
            os.setgroups([])
            os.setresgid(AS_UID, AS_UID, AS_UID)
            os.setresuid(AS_UID, AS_UID, AS_UID)

    # OLDPWD dropped: a child shell validates it at startup, and one
    # inherited from the operator's shell caused dac_* AVCs nothing in the
    # probe asked for (phase 2b runs).
    env = {k: v for k, v in os.environ.items() if k != "OLDPWD"}
    return subprocess.run(argv, preexec_fn=preexec, pass_fds=pass_fds, cwd=cwd,
                          env=env, capture_output=True, text=True, timeout=timeout)


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
def setattr_unbuffered(name, value):
    # os.write, never open().write(): a buffered write to /proc/self/attr/*
    # raises only when the file is finalised, after attempt() has already
    # printed its verdict (phase 2a run, finding 3a).
    fd = os.open("/proc/self/attr/" + name, os.O_WRONLY)
    try:
        os.write(fd, value.encode())
    finally:
        os.close(fd)
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

def probe_home() -> str:
    """The home a runner of this uid reads its user tier from (#1168)."""
    if AS_UID is None:
        return os.path.expanduser("~")
    import pwd
    return pwd.getpwuid(AS_UID).pw_dir


def _setup_user_tier() -> None:
    """Scratch user-tier files in the probe's home, labelled by jaato.fc."""
    home = os.path.join(probe_home(), ".jaato")
    for sub in ("agents", "memories"):
        os.makedirs(os.path.join(home, sub), exist_ok=True)
    open(os.path.join(home, "agents", "jaato-2a-probe.md"), "w").write("persona\n")
    open(os.path.join(home, "jaato-2a-probe_auth.json"), "w").write("{}\n")
    if AS_UID is not None:
        for dirpath, dirnames, filenames in os.walk(home):
            for name in [dirpath] + [os.path.join(dirpath, n) for n in dirnames + filenames]:
                os.chown(name, AS_UID, AS_UID)
    subprocess.run(["restorecon", "-R", home], check=True)


def setup(root: str) -> dict:
    shutil.rmtree(root, ignore_errors=True)
    paths = {
        "wsA": f"{root}/wsA", "wsB": f"{root}/wsB", "wsU": f"{root}/wsU",
        "tmpA": f"/tmp/jaato-2a-{os.getpid()}",
    }
    for key in ("wsA", "wsB", "wsU"):
        os.makedirs(f"{paths[key]}/.jaato/agents", exist_ok=True)
        os.makedirs(f"{paths[key]}/.jaato/references-claims", exist_ok=True)
        os.makedirs(f"{paths[key]}/.jaato/references", exist_ok=True)
        os.makedirs(f"{paths[key]}/.jaato/prompts", exist_ok=True)
        os.makedirs(f"{paths[key]}/bin", exist_ok=True)
        open(f"{paths[key]}/file.txt", "w").write("workspace\n")
        open(f"{paths[key]}/.jaato/agents/a.md", "w").write("persona\n")
        open(f"{paths[key]}/.jaato/references/r.json", "w").write("{}\n")
        open(f"{paths[key]}/.jaato/prompts/p.md", "w").write("prompt\n")
        os.makedirs(f"{paths[key]}/.jaato/logs", exist_ok=True)
        open(f"{paths[key]}/.jaato/logs/runner-iso.log", "w").write("")
        shutil.copy("/usr/bin/true", f"{paths[key]}/bin/t")
    os.makedirs(paths["tmpA"], exist_ok=True)
    _setup_user_tier()
    if AS_UID is not None:
        for key in ("wsA", "wsB", "wsU", "tmpA"):
            for dirpath, dirnames, filenames in os.walk(paths[key]):
                for name in [dirpath] + [os.path.join(dirpath, n) for n in dirnames + filenames]:
                    os.chown(name, AS_UID, AS_UID)
    # wsA: a managed workspace at L1; wsB: another workspace at L2;
    # wsU: a user's own checkout (not executable) at L1.
    chcon(paths["wsA"], "jaato_managed_ws_t", L1)
    chcon(paths["wsB"], "jaato_managed_ws_t", L2)
    chcon(paths["wsU"], "jaato_workspace_t", L1)
    for key, lvl in (("wsA", L1), ("wsB", L2), ("wsU", L1)):
        # v2: the persona layer is agent config, references stay authored.
        chcon(f"{paths[key]}/.jaato/agents", "jaato_agent_config_t", lvl)
        chcon(f"{paths[key]}/.jaato/references", "jaato_authored_t", lvl)
        chcon(f"{paths[key]}/.jaato/prompts", "jaato_prompts_t", lvl)
        chcon(f"{paths[key]}/.jaato/references-claims", "jaato_claims_t", lvl)
        # v3: an isolated sub-runner's log, as the daemon labels it.
        chcon(f"{paths[key]}/.jaato/logs/runner-iso.log", "jaato_runner_log_t", lvl,
              recursive=False)
    chcon(paths["tmpA"], "jaato_tmp_t", L1)
    return paths


# ----------------------------------------------------------------------
# probes
# ----------------------------------------------------------------------

def spawn_child_create(target: str) -> str:
    """Runner code that execs a child shell which tries to create *target*."""
    child_ctx = context("jaato_child_t", L1)
    return f"""
import os, subprocess
def pre():
    fd = os.open('/proc/self/attr/exec', os.O_WRONLY)
    os.write(fd, b'{child_ctx}')
    os.close(fd)
p = subprocess.run(['/bin/sh', '-c', 'echo x > {target} && echo LEAK || echo DENIED'],
                   preexec_fn=pre, capture_output=True, text=True)
print(p.stdout + p.stderr)
"""


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

    expect_ok("runner: reads agent config (.jaato/agents)",
              py_in(R, L1, TRY + f"attempt(lambda: open('{p['wsA']}/.jaato/agents/a.md').read(), False)"))
    expect_ok("runner: reads authored config (.jaato/references)",
              py_in(R, L1, TRY + f"attempt(lambda: open('{p['wsA']}/.jaato/references/r.json').read(), False)"))
    expect_ok("runner: writes the prompt library (.jaato/prompts)",
              py_in(R, L1, TRY + f"attempt(lambda: open('{p['wsA']}/.jaato/prompts/new.md','w').write('x'), False)"))
    for what, code in (
            ("writes an authored file", f"open('{p['wsA']}/.jaato/agents/a.md','a').write('x')"),
            ("creates an authored file", f"open('{p['wsA']}/.jaato/agents/b.md','w').write('x')"),
            ("renames the authored directory away", f"os.rename('{p['wsA']}/.jaato/agents','{p['wsA']}/.jaato/agents.old')"),
            ("removes an authored file", f"os.unlink('{p['wsA']}/.jaato/agents/a.md')")):
        expect_denied(f"runner: {what} is refused",
                      py_in(R, L1, TRY + f"attempt(lambda: {code}, True)"))

    expect_denied("runner: creating a missing .jaato/reactors.json is refused",
                  py_in(R, L1, TRY + f"attempt(lambda: open('{p['wsA']}/.jaato/reactors.json','w').write('{{}}'), True)"))
    expect_denied("child: creating a missing .jaato/template_routing.yaml is refused",
                  py_in(R, L1, spawn_child_create(p['wsA'] + "/.jaato/template_routing.yaml")))
    expect_ok("runner: writes a reference claim",
              py_in(R, L1, TRY + f"attempt(lambda: open('{p['wsA']}/.jaato/references-claims/c.json','w').write('x'), False)"))
    expect_ok("runner: writes its session tmpdir",
              py_in(R, L1, TRY + f"attempt(lambda: open('{p['tmpA']}/t','w').write('x'), False)"))
    expect_denied("runner: creating a file in the host's /tmp is refused",
                  py_in(R, L1, TRY + f"attempt(lambda: open('/tmp/jaato-2a-leak-{os.getpid()}','w').write('x'), True)"))
    expect_denied("runner: /root is unreachable",
                  py_in(R, L1, TRY + "attempt(lambda: os.listdir('/root'), True)"))
    expect_denied("runner: cannot change its own domain",
                  py_in(R, L1, TRY + f"attempt(lambda: setattr_unbuffered('current', '{context('unconfined_t', 's0')}'), True)"))

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

    # Binds (module 1.2.0): parity with AppArmor's `network inet stream`.
    expect_ok("runner: binds and listens on 127.0.0.1, an unreserved port",
              py_in(R, L1, "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); print('OK', s.getsockname())"))
    expect_ok("runner: binds the webhook's default port 9100 (hplip_port_t)",
              py_in(R, L1, "import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1',9100)); print('OK')"))
    expect_denied("runner: binding port 80 is refused",
                  py_in(R, L1, TRY + "import socket\nattempt(lambda: socket.socket().bind(('127.0.0.1', 80)), True)"))

    # The user tier, ~/.jaato (module 1.2.0), in the HOME this probe runs with.
    home = probe_home()
    expect_ok("runner: reads a user-tier persona (~/.jaato/agents)",
              py_in(R, L1, TRY + f"attempt(lambda: open('{home}/.jaato/agents/jaato-2a-probe.md').read(), False)"))
    expect_ok("runner: writes the user-tier memory store (~/.jaato/memories)",
              py_in(R, L1, TRY + f"attempt(lambda: open('{home}/.jaato/memories/jaato-2a-probe.txt','w').write('x'), False)"))
    expect_denied("runner: writing a user-tier persona is refused",
                  py_in(R, L1, TRY + f"attempt(lambda: open('{home}/.jaato/agents/jaato-2a-probe.md','a').write('x'), True)"))
    expect_denied("runner: a stored credential (~/.jaato/*_auth.json) is unreadable",
                  py_in(R, L1, TRY + f"attempt(lambda: open('{home}/.jaato/jaato-2a-probe_auth.json').read(), True)"))
    expect_denied("runner: the home directory cannot be listed",
                  py_in(R, L1, TRY + f"attempt(lambda: os.listdir('{home}'), True)"))

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
import os, subprocess
def pre():
    fd = os.open('/proc/self/attr/exec', os.O_WRONLY)
    os.write(fd, b'{child_ctx}')
    os.close(fd)
p = subprocess.run(CMD, preexec_fn=pre, capture_output=True, text=True)
print(p.stdout + p.stderr)
"""
    expect_ok("child: the runner execs /bin/sh into jaato_child_t",
              py_in(R, L1, spawn.replace("CMD", "['/bin/sh','-c','cat /proc/self/attr/current; echo; echo OK']")),
              want=f"{C}:{L1}")
    def child_py(code: str) -> str:
        # A Python probe run in the child, verdict from attempt(). Never a
        # shell `cmd >/dev/null && echo LEAK || echo DENIED`: that reports
        # DENIED when the redirect fails (phase 2a run, finding 3b).
        return spawn.replace("CMD", repr([sys.executable, "-I", "-c", TRY + code]))

    expect_denied("child: cannot read the runner's /proc/<pid>/environ",
                  py_in(R, L1, child_py(
                      "attempt(lambda: open('/proc/%d/environ' % os.getppid(), 'rb').read(1), True)")))
    expect_denied("child: cannot set an exec context (cannot leave //child)",
                  py_in(R, L1, child_py(
                      f"attempt(lambda: setattr_unbuffered('exec', '{context('unconfined_t', 's0')}'), True)")))
    expect_ok("child: opens a pty (interactive_shell's children)",
              py_in(R, L1, child_py("attempt(lambda: os.openpty(), False)")))
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
                  # From the workspace, as a runner starts: from the
                  # probe's own cwd (under /root) the import scanned it
                  # and logged admin_home_t denials (cf410bcf run).
                  run_in(R, L1, [f"{venv}/bin/python", "-c",
                                 "import jaato_server.server.runner as m; print('OK', m.__file__)"],
                         cwd=p["wsA"]))
    else:
        record("runner: imports jaato_server.server.runner from the venv", None, "--venv not given")


def isolated_probes(p: dict) -> None:
    """Phase 3: the isolated sub-runner domains, at the parent's level (L1).

    Translated from the AppArmor flat sub-profile: the parent's workspace
    (read-write, or read only), the shared authored config, the session
    tmpdir; never the persona config, the prompt library, the user tier,
    another workspace, or anything to exec.
    """
    I, RO = "jaato_isolated_t", "jaato_isolated_ro_t"
    expect_ok("isolated: enters jaato_isolated_t at the parent's level",
              py_in(I, L1, "print('OK', open('/proc/self/attr/current').read())"),
              want=f"{I}:{L1}")
    expect_ok("isolated: reads and writes its parent's workspace",
              py_in(I, L1, TRY + f"""
attempt(lambda: open('{p['wsA']}/file.txt').read(), False)
attempt(lambda: open('{p['wsA']}/iso.txt','w').write('x'), False)
attempt(lambda: os.unlink('{p['wsA']}/iso.txt'), False)"""))
    expect_ok("isolated: reads the shared authored config (.jaato/references)",
              py_in(I, L1, TRY + f"attempt(lambda: open('{p['wsA']}/.jaato/references/r.json').read(), False)"))
    for what, code in (
            ("a persona (.jaato/agents)", f"open('{p['wsA']}/.jaato/agents/a.md').read()"),
            ("the agent-config directory listing", f"os.listdir('{p['wsA']}/.jaato/agents')"),
            ("the prompt library (.jaato/prompts)", f"open('{p['wsA']}/.jaato/prompts/p.md').read()"),
            ("another workspace (level)", f"open('{p['wsB']}/file.txt').read()"),
            ("a user-tier persona (~/.jaato/agents)",
             f"open('{probe_home()}/.jaato/agents/jaato-2a-probe.md').read()")):
        expect_denied(f"isolated: reading {what} is refused",
                      py_in(I, L1, TRY + f"attempt(lambda: {code}, True)"))
    expect_denied("isolated: creating a missing .jaato/reactors.json is refused",
                  py_in(I, L1, TRY + f"attempt(lambda: open('{p['wsA']}/.jaato/reactors.json','w').write('{{}}'), True)"))
    expect_denied("isolated: executing /bin/true is refused (the flat profile execs nothing)",
                  py_in(I, L1, TRY + "import subprocess\nattempt(lambda: subprocess.run(['/bin/true']), True)"))
    expect_denied("isolated: cannot set an exec context",
                  py_in(I, L1, TRY + f"attempt(lambda: setattr_unbuffered('exec', '{context('unconfined_t', 's0')}'), True)"))
    expect_ok("isolated: writes a reference claim",
              py_in(I, L1, TRY + f"attempt(lambda: open('{p['wsA']}/.jaato/references-claims/i.json','w').write('x'), False)"))
    expect_ok("isolated: writes its session tmpdir",
              py_in(I, L1, TRY + f"attempt(lambda: open('{p['tmpA']}/i','w').write('x'), False)"))

    expect_ok("isolated read-only: enters jaato_isolated_ro_t",
              py_in(RO, L1, "print('OK', open('/proc/self/attr/current').read())"),
              want=f"{RO}:{L1}")
    expect_ok("isolated read-only: reads the workspace",
              py_in(RO, L1, TRY + f"attempt(lambda: open('{p['wsA']}/file.txt').read(), False)"))
    for what, code in (
            ("writing a workspace file", f"open('{p['wsA']}/file.txt','a').write('x')"),
            ("creating a workspace file", f"open('{p['wsA']}/ro.txt','w').write('x')"),
            ("writing a reference claim", f"open('{p['wsA']}/.jaato/references-claims/ro.json','w').write('x')")):
        expect_denied(f"isolated read-only: {what} is refused",
                      py_in(RO, L1, TRY + f"attempt(lambda: {code}, True)"))
    expect_ok("isolated read-only: writes its session tmpdir",
              py_in(RO, L1, TRY + f"attempt(lambda: open('{p['tmpA']}/ro','w').write('x'), False)"))
    log = f"{p['wsA']}/.jaato/logs/runner-iso.log"
    for dom, name in ((I, "isolated"), (RO, "isolated read-only")):
        expect_ok(f"{name}: appends to its log (jaato_runner_log_t)",
                  py_in(dom, L1, TRY + f"attempt(lambda: open('{log}','a').write('x'), False)"))
        expect_denied(f"{name}: truncating its log is refused",
                      py_in(dom, L1, TRY + f"attempt(lambda: open('{log}','w').write('x'), True)"))


# A threaded parent, as the pool template is, and a forked child that
# enters the runner domain while it has one thread (phase 4, design §7.2).
# CHILD is the child's body; the parent prints the child's output.
FORKED_SLOT = TRY + '''
import threading, time
threading.Thread(target=time.sleep, args=(30,), daemon=True).start()
CTX = {ctx!r}
AS_UID = {as_uid!r}
{parent}
r, w = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(r); os.dup2(w, 1); os.dup2(w, 2)
    try:
        if AS_UID is not None:
            os.setgroups([]); os.setresgid(AS_UID, AS_UID, AS_UID)
            os.setresuid(AS_UID, AS_UID, AS_UID)
        setattr_unbuffered("current", CTX)
{child}
    except BaseException as e:
        print("CHILD-ERROR", repr(e))
    finally:
        sys.stdout.flush(); os._exit(0)
os.close(w)
out = b""
while True:
    chunk = os.read(r, 4096)
    if not chunk:
        break
    out += chunk
os.waitpid(pid, 0)
print(out.decode(), end="")
'''


def forked_slot(child: str, parent: str = "") -> subprocess.CompletedProcess:
    """Run FORKED_SLOT unconfined, as the daemon's template runs."""
    body = "\n".join("        " + line for line in child.strip().splitlines())
    code = FORKED_SLOT.format(ctx=context("jaato_runner_t", L1), as_uid=AS_UID,
                              parent=parent, child=body)
    return subprocess.run([sys.executable, "-I", "-c", code],
                          capture_output=True, text=True, timeout=60)


def pool_probes(p: dict) -> None:
    """Phase 4: a pool slot enters jaato_runner_t at fork (design §7.2).

    The template runs in the daemon's domain and has threads; SELinux
    refuses setcon there, and allows it in the single-threaded child a
    fork produces, given the module's dyntransition rule.
    """
    target = context("jaato_runner_t", L1)
    expect_denied("pool: a threaded process cannot setcon into the runner",
                  forked_slot("pass", parent=f"attempt(lambda: setattr_unbuffered('current', CTX), True)"))
    expect_ok("pool: a forked child enters jaato_runner_t at the workspace level",
              forked_slot("print('OK', open('/proc/self/attr/current').read())"),
              want=target)
    expect_ok("pool: a thread the slot starts wears the same context",
              forked_slot('''
t = threading.Thread(target=time.sleep, args=(2,)); t.start()
labels = {open(f"/proc/self/task/{x}/attr/current").read().strip("\\0\\n ") for x in os.listdir("/proc/self/task")}
print("OK" if labels == {CTX} else f"LEAK {labels}")'''))
    expect_ok("pool: the slot reads its own workspace",
              forked_slot(f"attempt(lambda: open('{p['wsA']}/file.txt').read(), False)"))
    expect_denied("pool: the slot cannot read another workspace (level)",
                  forked_slot(f"attempt(lambda: open('{p['wsB']}/file.txt').read(), True)"))
    expect_denied("pool: the slot cannot change its domain back",
                  forked_slot(f"attempt(lambda: setattr_unbuffered('current', '{context('unconfined_t', 's0')}'), True)"))


def auditd_running() -> bool:
    return subprocess.run(["pidof", "auditd"], capture_output=True).returncode == 0


def kernel_avcs_since(marker: str) -> str:
    """AVC lines the kernel logged after *marker* (written to /dev/kmsg).

    The fallback when auditd is not running (WSL): the kernel then prints
    AVCs to its ring buffer instead (phase 2a run, finding 3c).
    """
    lines = subprocess.run(["dmesg"], capture_output=True, text=True).stdout.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if marker in lines[i]:
            lines = lines[i + 1:]
            break
    else:
        print(f"WARNING: marker {marker!r} not in dmesg (ring buffer wrapped?); "
              "AVCs below may predate this run", file=sys.stderr)
    return "\n".join(l for l in lines if "avc:" in l) + "\n"


def main() -> int:
    global AS_UID
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/srv/jaato-2a")
    ap.add_argument("--venv")
    ap.add_argument("--as-uid", type=int,
                    help="run every probe as this uid (not 0), as a runner "
                         "does under --runner-uid-policy (#1168)")
    args = ap.parse_args()
    if os.geteuid() != 0:
        print("run as root (chcon, setsebool)", file=sys.stderr)
        return 2
    if args.as_uid == 0:
        print("--as-uid 0 is the default; pass a non-root uid", file=sys.stderr)
        return 2
    AS_UID = args.as_uid
    enforce = open("/sys/fs/selinux/enforce").read().strip()
    print(f"enforcing={enforce}  context={open('/proc/self/attr/current').read().strip(chr(0))}"
          f"  as_uid={AS_UID if AS_UID is not None else 0}")
    start = time.strftime("%H:%M:%S")
    with_auditd = auditd_running()
    marker = f"jaato-2a-probe start pid={os.getpid()} {time.time():.0f}"
    ratelimit = None
    if not with_auditd:
        # Without auditd, the kernel rate-limits the AVCs it prints and
        # most of a run's denials are suppressed (phase 2a run, finding 3c).
        print("auditd is not running: collecting AVCs from dmesg, with "
              "kernel.printk_ratelimit=0 for the run", file=sys.stderr)
        ratelimit = open("/proc/sys/kernel/printk_ratelimit").read().strip()
        open("/proc/sys/kernel/printk_ratelimit", "w").write("0")
        open("/dev/kmsg", "w").write(marker + "\n")
    try:
        paths = setup(args.root)
        probes(paths, args.venv)
        isolated_probes(paths)
        pool_probes(paths)
        time.sleep(1)
    finally:
        if ratelimit is not None:
            open("/proc/sys/kernel/printk_ratelimit", "w").write(ratelimit)
    if with_auditd:
        avc = subprocess.run(["ausearch", "-m", "AVC,USER_AVC", "-ts", start, "-i"],
                             capture_output=True, text=True).stdout
        count = avc.count("type=AVC")
    else:
        avc = kernel_avcs_since(marker)
        count = avc.count("avc:")
    open("probe-avc.txt", "w").write(avc)
    json.dump({"enforcing": enforce, "as_uid": AS_UID, "avc_source":
               "ausearch" if with_auditd else "dmesg", "results": RESULTS},
              open("probe-results.json", "w"), indent=2)
    fails = [r for r in RESULTS if r["result"] == "FAIL"]
    print(f"\n{len(RESULTS) - len(fails)} of {len(RESULTS)} not failed; "
          f"{len(fails)} FAIL; AVC lines: {count} (probe-avc.txt, from "
          f"{'ausearch' if with_auditd else 'dmesg'})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
