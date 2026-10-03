# SELinux backend, phase 4: handoff for pool slots on an enforcing kernel

This is the instruction set for an agent (or a person) with root in the
Fedora 44 WSL2 distro that ran [phase 3](selinux-phase3-handoff.md).
Phase 4 lets SELinux-confined sessions use the pre-warm pool
([design §7.2](selinux-backend.md#72-pool-slots-dynamic-transition)):

- when a confined session finds no idle slot of its boundary, the daemon
  asks the template to fork one for it (`FORK_SLOT <json>`);
- the forked child, which has one thread, enters the private `/tmp`,
  drops to the runner user and writes `jaato_runner_t:<level>` to
  `/proc/self/attr/current` before it starts any thread;
- the slot returns to the pool after the session and serves the next
  session of the same boundary and uid;
- the template stays in the daemon's domain. The module grants
  `unconfined_t` and `unconfined_service_t` `dyntransition` into
  `jaato_runner_t`.

The module is now **1.7.0** (marker `jaato_policy_v4_t`), and the daemon
refuses an older one. A workspace labelled under an older module is
relabelled once.

**Third run (after b5dcc98f).** The second run's log fix refused every
pool-served session: the bootstrap flushed output that was still on the
daemon's log. Now the forked slot points its output at the session's log
before it confines itself, and the daemon creates `.jaato/logs` first. No
policy change. Expect every live check to pass in all three runs, both
sessions' `runner-<id>.log` to be non-empty with
`runner-session bootstrap: logging to …`, no `PermissionError` at
bootstrap, and no refused write on the daemon's log.

**Second run (after the slot-log fix).** The first run passed everything
and found that a pool slot wrote the daemon's log (refused under SELinux)
instead of its own. Each bootstrap now points the runner's fds 1 and 2 at
its session's `runner-<id>.log`. No policy change (still 1.7.0). Expect
the live tool's two new checks to pass (each session wrote its own runner
log; no runner write to the daemon's log was refused), and each
`runner-<id>.log` to contain `runner-session bootstrap: logging to …`.

CI checks the policy rules (Fedora container, setools) and the wiring with
fakes. This run answers what CI cannot: **does the kernel let a forked
child of the threaded template enter the runner domain, and does a real
session then run, and reuse its slot?** The output is one results file.

- Repository: `https://github.com/Jaato-framework-and-examples/jaato`
- Branch: `claude/selinux-phase4`
- Tools, under `jaato-server/selinux/tools/`:
  - `probe_policy.py`: the phase 3 probe plus six pool checks.
  - `live_session.py --sessions 2`: two sessions in turn in one private
    daemon; checks the first is served by a slot forked into its boundary
    and the second by the same slot.

## Rules for this run

The rules of the earlier handoffs apply unchanged:

- Report only. Do not push, comment or open issues.
- Never disable SELinux host-wide.
- Do not edit the module to make a check pass. A FAIL is a result.
- Work as root, on the Linux filesystem (not `/mnt/c`).
- `live_session.py` starts its own daemon with its own socket, PID file and
  log. **Never** run `jaato_server --stop`, `--restart` or `--status`
  without a private `--pid-file`, and never touch another daemon's socket.

Write the results to `/root/jaato-phase4/RESULTS.md` (template in §6) and
show the user its contents at the end.

## 1. Gate

Follow the 2a handoff §1. Proceed only when `getenforce` says `Enforcing`
and `id -Z` prints a real context.

## 2. Install the module and jaato

```bash
mkdir -p /root/jaato-phase4 && cd /root/jaato-phase4
cd /root/jaato && git fetch origin && git checkout claude/selinux-phase4 \
  && git pull --ff-only && git log --oneline -1
cd jaato-server/jaato_server/server/confinement/selinux_policy && make -f /usr/share/selinux/devel/Makefile jaato.pp \
  && semodule -i jaato.pp
semodule -l | grep jaato                 # jaato 1.7.0
seinfo -t | grep -c '^ *jaato_'          # 17 types
seinfo -t jaato_policy_v4_t              # present
sesearch -A -s unconfined_service_t -t jaato_runner_t -c process -p dyntransition
```

Reinstall the venv non-editable, as in 2b §2 (the code changed):

```bash
rm -rf /opt/jaato/venv && python3.12 -m venv /opt/jaato/venv
/opt/jaato/venv/bin/pip install -q -U pip
# [interactive] pulls in pexpect: without it interactive_shell is skipped and
# no live session exercises the pty path (jaato_devpts_t).
/opt/jaato/venv/bin/pip install -q /root/jaato/jaato-sdk '/root/jaato/jaato-server[interactive]'
semanage fcontext -a -t lib_t '/opt/jaato/venv(/.*)?' 2>/dev/null || true
restorecon -R /opt/jaato/venv
```

## 3. Readiness

As in the phase 3 handoff §3. Expected: `ready=True`, `is_available True`,
and the doctor's selinux row says `policy module v4`.

## 4. The policy probe, both uids

```bash
cd /root/jaato-phase4
P=/root/jaato/jaato-server/selinux/tools/probe_policy.py
python3 $P --root /srv/jaato-4 --venv /opt/jaato/venv | tee probe.txt
mkdir -p uid && cd uid && python3 $P --root /srv/jaato-4 --venv /opt/jaato/venv \
  --as-uid 1000 | tee probe.txt; cd ..
```

New since phase 3, all expected to PASS:

| Check | What a FAIL means |
|---|---|
| a threaded process cannot setcon into the runner | the premise of §7.2 changed: a threaded process may change domain |
| a forked child enters `jaato_runner_t` at the workspace level | the `dyntransition` rule, or the MCS constraint on it |
| a thread the slot starts wears the same context | a thread could start outside the domain |
| the slot reads its own workspace | the level reached by `setcon` is not the workspace's |
| the slot cannot read another workspace (level) | MCS does not hold after `setcon` |
| the slot cannot change its domain back | the runner domain can leave itself |

## 5. The live sessions

```bash
cd /root/jaato-phase4
L=/root/jaato/jaato-server/selinux/tools/live_session.py
/opt/jaato/venv/bin/python $L --root /srv/jaato-4 --sessions 2 | tee live.txt
cp /srv/jaato-4/live-session.json live.json; cp /srv/jaato-4/d.out live-d.out
cp -r /srv/jaato-4/ws/.jaato/logs live-logs
/opt/jaato/venv/bin/python $L --root /srv/jaato-4-uid --sessions 2 --as-uid 1000 | tee live-uid.txt
cp /srv/jaato-4-uid/live-session.json live-uid.json; cp /srv/jaato-4-uid/d.out live-uid-d.out
cp -r /srv/jaato-4-uid/ws/.jaato/logs live-uid-logs
# regression: the isolated sub-runner is still cold spawned
/opt/jaato/venv/bin/python $L --root /srv/jaato-4-iso --isolated rw | tee live-iso.txt
```

| Check | What a FAIL means |
|---|---|
| the first session is served by a slot forked into its boundary | no `forked slot pid=… into SELinux boundary` line: the pool is off, the template did not fork, or the session cold-spawned |
| the next session reuses that slot | the slot did not return to the pool (still loaded?), or its key did not match |
| the runner runs in `jaato_runner_t` at a workspace level, the command in `jaato_child_t` | the slot is not in the domain, or its children do not transition |
| (uid run) the runner is uid 1000 | the drop did not happen before `setcon` |
| the phase 2b and 3 checks | a regression |

A slot that could not enter its domain exits 124 and logs
`refusing to serve -- pool slot could not enter …`; quote that line, and
the `dyntransition` AVC if there is one. Also record, from `d.out`, the
time from `session.new` to the first turn for each session, and for a
cold-spawned session if you see one: the point of the phase is that a
pool-served session starts in about a second instead of about seven.

## 6. Results template (`/root/jaato-phase4/RESULTS.md`)

```markdown
# SELinux phase 4 results

- Date, host, `uname -r`, `rpm -q selinux-policy-targeted`:
- Branch commit:
- Module version and type count:

## Readiness and doctor
## Probe (root / --as-uid 1000): pass counts, every FAIL with its AVCs
## Live sessions (root, uid 1000; isolated regression): each check, the
   forked and served slot pids, the runner's context and uid, session
   start times
## AVCs naming a jaato type that no check asked for
## Surprises
## Anything that contradicts docs/design/selinux-backend.md
```

## 7. Cleanup

```bash
semodule -r jaato
semanage fcontext -d -t lib_t '/opt/jaato/venv(/.*)?'
rm -rf /srv/jaato-4* /opt/jaato
rm -f /root/.jaato/agents/jaato-2a-probe.md /root/.jaato/memories/jaato-2a-probe.txt \
      /root/.jaato/jaato-2a-probe_auth.json /root/.jaato/selinux_levels.json*
getsebool domain_fd_use                    # on
```

Leave `/root/jaato` and `/root/jaato-phase4` for the user. If you ran with
`--as-uid`, remove the same three probe files from that user's `~/.jaato`.
