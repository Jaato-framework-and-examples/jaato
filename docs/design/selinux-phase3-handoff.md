# SELinux backend, phase 3: handoff for the isolated sub-runner on an enforcing kernel

This is the instruction set for an agent (or a person) with root in the
Fedora 44 WSL2 distro that ran [phase 2b](selinux-phase2b-handoff.md).
Phase 3 confines the isolated sub-runner (`spawn_subagent` with
`agent_params.isolated: true`) under SELinux
([design §5.3](selinux-backend.md#53-isolated-sub-runners)):

- the sub-runner runs in `jaato_isolated_t`, or `jaato_isolated_ro_t`
  under `isolated_read_only_workspace`, at its **parent's** level;
- it is flat (its subprocesses stay in its domain) and may exec nothing;
- it cannot read the persona config (`jaato_agent_config_t`: agents,
  profiles, scripts, completion and spawn schemas, instructions,
  `reactors.json`) or the prompt library (`jaato_prompts_t`);
- `isolated_workspace_subpath` is refused by name.

The module is now **1.6.0** (marker `jaato_policy_v3_t`), and the daemon
refuses an older one. A workspace labelled under an older module is
relabelled once, because the label stamp carries the policy version.

**Third run (after a1d59347).** The second run passed the probe 61/61
and got past the spawn, then every sub-runner crashed in bootstrap
probing `.jaato/instructions`, which its boundary denies. Fixed: the
daemon now resolves the isolated session's config (instructions tier,
gc.json) and the sub-runner never reads it. No policy change: the module
is still 1.6.0, so reinstalling it is optional. Expect every live check
to pass, and the sub-runner's log to say
`GC installed from gc.json (via the daemon)` when the workspace or the
daemon's `~/.jaato` has a `gc.json`, or no GC line when neither has one.
Refused `search` / `getattr` AVCs on `~/.jaato`, `.jaato/agents` and
`.jaato/prompts` are still expected (non-fatal probes, a known
follow-up); a bootstrap `PermissionError` is not.

**Second run (after 64020ec9).** The first run's probe passed 57/57; its
live runs all failed at the spawn on a daemon deadlock (fixed: the spawn
now runs off the daemon loop), and the read-only sub-runner could not
write its log (fixed: `jaato_runner_log_t`). Expect the probe to grow by
four log checks and every live check to pass.

CI checks the policy rules (Fedora container, setools) and the wiring with
fakes. This run answers what CI cannot: **does a real isolated subagent end
up in the right domain at the right level, and does each domain allow and
refuse what the design says?** The output is one results file.

- Repository: `https://github.com/Jaato-framework-and-examples/jaato`
- Branch: `claude/selinux-phase3`
- Tools, under `jaato-server/selinux/tools/`:
  - `probe_policy.py`: the 2b probe plus the phase 3 checks (both isolated
    domains, run by exec transition from the probe).
  - `live_session.py --isolated rw|ro`: a private daemon, a confined parent
    session on the `echo` provider whose one tool call is `spawn_subagent`
    with `isolated: true`; the subagent's one tool call writes
    `iso-probe.txt` in the workspace.

## Rules for this run

The rules of the 2a and 2b handoffs apply unchanged:

- Report only. Do not push, comment or open issues.
- Never disable SELinux host-wide.
- Do not edit the module to make a check pass. A FAIL is a result.
- Work as root, on the Linux filesystem (not `/mnt/c`).
- `live_session.py` starts its own daemon with its own socket, PID file and
  log. **Never** run `jaato_server --stop`, `--restart` or `--status`
  without a private `--pid-file`, and never touch another daemon's socket.

Write the results to `/root/jaato-phase3/RESULTS.md` (template in §6) and
show the user its contents at the end.

## 1. Gate

Follow the 2a handoff §1. Proceed only when `getenforce` says `Enforcing`
and `id -Z` prints a real context.

## 2. Install the module and jaato

```bash
mkdir -p /root/jaato-phase3 && cd /root/jaato-phase3
cd /root/jaato && git fetch origin && git checkout claude/selinux-phase3 \
  && git pull --ff-only && git log --oneline -1
cd jaato-server/selinux && make -f /usr/share/selinux/devel/Makefile jaato.pp \
  && semodule -i jaato.pp
semodule -l | grep jaato                 # jaato 1.6.0
seinfo -t | grep -c '^ *jaato_'          # 17 types
seinfo -t jaato_policy_v3_t              # present
```

Reinstall the venv non-editable, as in 2b §2 (the code changed):

```bash
rm -rf /opt/jaato/venv && python3.12 -m venv /opt/jaato/venv
/opt/jaato/venv/bin/pip install -q -U pip
/opt/jaato/venv/bin/pip install -q /root/jaato/jaato-sdk /root/jaato/jaato-server
semanage fcontext -a -t lib_t '/opt/jaato/venv(/.*)?' 2>/dev/null || true
restorecon -R /opt/jaato/venv
```

The user tier is unchanged since 2b (`restorecon -R /root/.jaato`).

## 3. Readiness

```bash
/opt/jaato/venv/bin/python - <<'EOF'
from jaato_server.server.confinement.selinux import SELinuxBackend
b = SELinuxBackend()
print("host_readiness:", b.host_readiness())
print("is_available  :", b.is_available())
EOF
/opt/jaato/venv/bin/jaato-doctor 2>&1 | grep -i -A3 'confinement\|selinux'
```

Expected: `ready=True`, `is_available True`, and the doctor's selinux
row says `policy module v3`.

## 4. The policy probe, both uids

```bash
cd /root/jaato-phase3
P=/root/jaato/jaato-server/selinux/tools/probe_policy.py
python3 $P --root /srv/jaato-3 --venv /opt/jaato/venv | tee probe.txt
mkdir -p uid && cd uid && python3 $P --root /srv/jaato-3 --venv /opt/jaato/venv \
  --as-uid 1000 | tee probe.txt; cd ..
```

New since 2b, all expected to PASS:

| Check | What a FAIL means |
|---|---|
| runner reads `.jaato/references`, writes `.jaato/prompts` | the authored / prompts split took a grant from the runner |
| `jaato_isolated_t` entered at the parent's level | the transition or the MCS constraint |
| isolated reads and writes the workspace, reads `.jaato/references`, writes a claim and its tmpdir | a missing grant |
| isolated cannot read `.jaato/agents` (file or listing), `.jaato/prompts`, another workspace, a user-tier persona | the isolated boundary is open |
| isolated cannot create `.jaato/reactors.json`, exec `/bin/true`, or set an exec context | the flat, no-exec domain is not flat |
| `jaato_isolated_ro_t` reads the workspace and writes its tmpdir, and cannot write or create a workspace file or a claim | the read-only domain |
| both isolated domains append to a `jaato_runner_log_t` file, and truncating it is refused | the v3 log type |

## 5. The live session, both modes, both uids

```bash
cd /root/jaato-phase3
L=/root/jaato/jaato-server/selinux/tools/live_session.py
for mode in rw ro; do
  /opt/jaato/venv/bin/python $L --root /srv/jaato-3-$mode --isolated $mode | tee live-$mode.txt
  cp /srv/jaato-3-$mode/live-session.json live-$mode.json
  cp -r /srv/jaato-3-$mode/ws/.jaato/logs live-$mode-logs
  /opt/jaato/venv/bin/python $L --root /srv/jaato-3-$mode-uid --isolated $mode --as-uid 1000 \
    | tee live-$mode-uid.txt
  cp /srv/jaato-3-$mode-uid/live-session.json live-$mode-uid.json
  cp -r /srv/jaato-3-$mode-uid/ws/.jaato/logs live-$mode-uid-logs
done
```

| Check | What a FAIL means |
|---|---|
| `spawn_subagent returned` | the daemon refused the spawn; the JSON quotes its stage and error (`sub_profile` = provisioning) |
| the sub-runner ran in `jaato_isolated_t` / `jaato_isolated_ro_t` at the parent runner's level | the isolated handle did not reach the spawn, or the exec transition did not happen |
| `rw`: `iso-probe.txt` exists in the workspace at the level | the domain's write grant |
| `ro`: it does not, AND an enforced `jaato_isolated_ro_t` AVC refused a write on a workspace directory | the read-only domain is not read-only, or the sub-runner never tried |
| the sub-runner wrote its own log (`jaato_runner_log_t`, not empty) | the daemon did not label the log, or the domain cannot append to it |
| the parent runner and its labels as in 2b | a regression from the authored-config split |

In `ro` mode the subagent's write is expected to fail inside the subagent;
the parent turn still completes. Record the subagent's error text from the
runner logs either way.

A `LOOP_STALL` naming `_do_spawn_isolated_runner` in `d.out`, or a spawn
failing with a bare `TimeoutError`, means the venv predates the off-loop
fix.

Before this branch, every isolated spawn on the runner path was refused
with `parent_session_id must be a non-empty str` on any host; if that line
appears in `d.out`, the venv is not this branch's code.

## 6. Results template (`/root/jaato-phase3/RESULTS.md`)

```markdown
# SELinux phase 3 results

- Date, host, `uname -r`, `rpm -q selinux-policy-targeted`:
- Branch commit:
- Module version and type count:

## Readiness and doctor
## Probe (root / --as-uid 1000): pass counts, every FAIL with its AVCs
## Live isolated sessions (rw, ro; root, uid 1000): each check, the
   sub-runner's context next to the parent's, the subagent's result
## AVCs naming a jaato type that no check asked for
## Surprises
## Anything that contradicts docs/design/selinux-backend.md
```

## 7. Cleanup

```bash
semodule -r jaato
semanage fcontext -d -t lib_t '/opt/jaato/venv(/.*)?'
rm -rf /srv/jaato-3* /opt/jaato
rm -f /root/.jaato/agents/jaato-2a-probe.md /root/.jaato/memories/jaato-2a-probe.txt \
      /root/.jaato/jaato-2a-probe_auth.json /root/.jaato/selinux_levels.json*
getsebool domain_fd_use                    # on
```

Leave `/root/jaato` and `/root/jaato-phase3` for the user. If you ran with
`--as-uid`, remove the same three probe files from that user's `~/.jaato`.
