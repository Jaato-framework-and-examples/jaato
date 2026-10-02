# SELinux backend, phase 2b: handoff for a live session on an enforcing kernel

This is the instruction set for an agent (or a person) with root in the
Fedora 44 WSL2 distro that ran [phase 2a](selinux-phase2a-handoff.md).
Phase 2b wired the backend into the daemon. With `JAATO_CONFINEMENT=selinux`:

- the daemon labels a workspace at its own MCS level;
- it cold-spawns the runner straight into `jaato_runner_t` through an exec
  transition;
- it moves every command the model runs into `jaato_child_t`.

CI checks the policy rules and the wiring with fakes, but has no SELinux
kernel. This run answers the question CI cannot: **does a real jaato
session end up in the right domains, with the right labels, and does it
complete?** The output is one results file.

- Repository: `https://github.com/Jaato-framework-and-examples/jaato`
- Branch: `claude/awesome-lovelace-bbca88` (draft PR #1431)
- Two tools, both under `jaato-server/selinux/tools/`:
  - `probe_policy.py`, the 2a probe. It now also checks binds and the
    `~/.jaato` user tier.
  - `live_session.py`. It starts a private jaato daemon with SELinux
    selected, runs one confined session against the `echo` provider (no
    API key, no network), and checks:
    - the runner's context;
    - the context of the command the model runs;
    - the workspace labels;
    - the session record.

## Rules for this run

The rules of the 2a handoff apply unchanged:

- Report only. Do not push, comment or open issues.
- Never disable SELinux host-wide.
- Do not edit the module to make a check pass. A FAIL is a result.
- Work as root, on the Linux filesystem (not `/mnt/c`).

One addition: `live_session.py` starts its own daemon on
`/run/jaato-2b/d.sock` with its own PID file and log. **Never** run
`jaato_server --stop`, `--restart` or `--status` without a private
`--pid-file`, and never touch another daemon's socket.

Write the results to `/root/jaato-phase2b/RESULTS.md` (template in §6) and
show the user its contents at the end.

## 1. Gate

Follow the 2a handoff §1. Under WSL, after any `wsl --shutdown`, bring
SELinux up in exactly the order it gives. Proceed only when
`getenforce` says `Enforcing` and `id -Z` prints a real context.

## 2. Install the module and jaato

```bash
mkdir -p /root/jaato-phase2b && cd /root/jaato-phase2b
cd /root/jaato && git fetch origin && git checkout claude/awesome-lovelace-bbca88 \
  && git pull --ff-only && git log --oneline -1
cd jaato-server/selinux && make -f /usr/share/selinux/devel/Makefile jaato.pp \
  && semodule -i jaato.pp
seinfo -t | grep -c '^ *jaato_'          # 12 types (1.3.0)
```

**A venv the runner can read, outside `/root`** (labelled `lib_t`, design
§9). Install it non-editable: an editable install points back into
`/root/jaato`, which the runner cannot read.

```bash
rm -rf /opt/jaato/venv && python3.12 -m venv /opt/jaato/venv
/opt/jaato/venv/bin/pip install -q -U pip
/opt/jaato/venv/bin/pip install -q /root/jaato/jaato-sdk /root/jaato/jaato-server
semanage fcontext -a -t lib_t '/opt/jaato/venv(/.*)?' 2>/dev/null || true
restorecon -R /opt/jaato/venv
```

**The user tier.** `jaato.fc` labels these directories. A runner cannot
create them itself, because it has no `add_name` in `~/.jaato`.

```bash
mkdir -p /root/.jaato/{agents,profiles,references,services,memories,prompts,skills}
restorecon -R /root/.jaato
ls -Zd /root/.jaato /root/.jaato/agents /root/.jaato/memories
# jaato_user_dir_t / jaato_user_config_t / jaato_user_data_t
```

## 3. Readiness, from the venv

```bash
/opt/jaato/venv/bin/python - <<'EOF'
from jaato_server.server.confinement.selinux import SELinuxBackend
b = SELinuxBackend()
print("host_readiness:", b.host_readiness())
print("is_available  :", b.is_available())
EOF
```

Expected: `ready=True` and `is_available True`. Since 2b, a ready host is
available.

## 4. The policy probe, both uids

```bash
cd /root/jaato-phase2b
python3 /root/jaato/jaato-server/selinux/tools/probe_policy.py \
  --root /srv/jaato-2a --venv /opt/jaato/venv | tee probe.txt
mkdir -p uid && cd uid && python3 /root/jaato/jaato-server/selinux/tools/probe_policy.py \
  --root /srv/jaato-2a --venv /opt/jaato/venv --as-uid 1000 | tee probe.txt; cd ..
```

What is new since 2a, and expected to PASS:

- the runner binds `127.0.0.1` on an unreserved port, and on 9100;
- binding port 80 is refused;
- a user-tier persona is readable, but writing it is refused;
- the memory store is writable;
- a `*_auth.json` is unreadable;
- the home directory cannot be listed.

The `--as-uid` run uses that user's home (`/home/<user>/.jaato`). Make
sure the user exists and has a home before you start.

## 5. The live session

```bash
cd /root/jaato-phase2b
L=/root/jaato/jaato-server/selinux/tools/live_session.py
# 1. a root runner (the default policy)
/opt/jaato/venv/bin/python $L --root /srv/jaato-2b | tee live.txt
cp /srv/jaato-2b/live-session.json live-root.json; cp -r /srv/jaato-2b/ws/.jaato/logs live-root-logs
# 2. the runner dropped to uid 1000 (#1168), with a stack for every
#    subprocess the runner starts, to name what execs ldconfig
/opt/jaato/venv/bin/python $L --root /srv/jaato-2b --as-uid 1000 --trace-subprocess | tee live-uid.txt
cp /srv/jaato-2b/live-session.json live-uid.json; cp -r /srv/jaato-2b/ws/.jaato/logs live-uid-logs
grep -A14 JAATO-2B-SUBPROCESS live-uid-logs/runner-*.log | head -60
grep -A14 JAATO-2B-LISTDIR live-uid-logs/runner-*.log | head -30
```

Since bf978cfe the runner no longer scans `~/.jaato` for stored
credentials when it is confined, and since module 1.3.0 `~/.jaato` has a
type of its own and no other home subdirectory is searchable. Expected on
this run, and both are evidence for the change rather than assumptions:
no `read` AVC on `.jaato`, and no `JAATO-2B-LISTDIR` line. If either
appears, the stack names the code that lists it.

The tool now records a refused session as a FAIL with the daemon's reason
instead of crashing, reads the daemon's log from its stdout (`d.out`: a
foreground daemon ignores `--log-file`), sets `kernel.printk_ratelimit=0`
for the run, and drops `OLDPWD` from the daemon's environment.

Each run writes `/srv/jaato-2b/live-session.json`, which holds:

- the runner contexts it saw;
- the command's output;
- the workspace labels;
- the daemon's SELinux log lines;
- the AVCs naming a jaato type.

| Check | What a FAIL means |
|---|---|
| the runner runs in `jaato_runner_t` at a workspace level | the exec transition did not happen. Common causes: the venv's python is not `bin_t`, or the daemon's role is not authorized (readiness says which) |
| the model's command runs in `jaato_child_t` at the same level | the `//child` callback did not reach `cli` |
| the command wrote the workspace | workspace labelling, or the child's write grant |
| the turn completed without errors | anything above. The daemon log in the JSON says which |
| `sandbox_mode: selinux` | the IPC record path |
| the workspace, authored config and a created file carry the level / type | labelling (`selinux_labels`), or MCS inheritance |

If the daemon refuses to start, the first FAIL quotes the reason it
printed. Common reasons: `JAATO_CONFINEMENT=selinux` with the module
missing, or a readiness check failing.

## 6. Results template (`/root/jaato-phase2b/RESULTS.md`)

```markdown
# SELinux phase 2b results

- Date, host, `uname -r`, `rpm -q selinux-policy-targeted`:
- Branch commit:
- Module types (`seinfo -t | grep -c jaato_`):

## Readiness
## Probe (root / --as-uid 1000): pass counts, every FAIL with its AVCs
## Live session: each check, runner contexts, the command's output
## AVCs naming a jaato type that no check asked for
## Surprises
## Anything that contradicts docs/design/selinux-backend.md
```

## 7. Cleanup

```bash
semodule -r jaato
semanage fcontext -d -t lib_t '/opt/jaato/venv(/.*)?'
rm -rf /srv/jaato-2a /srv/jaato-2b /run/jaato-2b /tmp/jaato-2a-* /opt/jaato
rm -f /root/.jaato/agents/jaato-2a-probe.md /root/.jaato/memories/jaato-2a-probe.txt \
      /root/.jaato/jaato-2a-probe_auth.json /root/.jaato/selinux_levels.json*
getsebool domain_fd_use                    # on
```

Leave `/root/jaato` and `/root/jaato-phase2b` for the user. If you ran
with `--as-uid`, remove the same three probe files from that user's
`~/.jaato`.
