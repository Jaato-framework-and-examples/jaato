# SELinux backend, phase 2a: handoff for a run on an enforcing kernel

This file is an instruction set for an agent (or a person) with root in the
Fedora 44 WSL2 distro that ran [phase 0](selinux-phase0-handoff.md). Phase
2a shipped the policy module (`jaato-server/selinux/`) and nothing that
uses it: no jaato code path loads or enters it yet. CI compiles the module
and checks its rules against the linked policy (`jaato-server/selinux/tests/`),
but CI has no SELinux kernel. This run answers the question CI cannot:
**does the kernel, with this module loaded, allow what a runner needs and
refuse what it must not?** The output is one results file.

- Repository: `https://github.com/Jaato-framework-and-examples/jaato`
- Branch: `claude/awesome-lovelace-bbca88` (draft PR
  https://github.com/Jaato-framework-and-examples/jaato/pull/1431)
- The probe: `jaato-server/selinux/tools/probe_policy.py`. It imports
  nothing from jaato. It labels three scratch workspaces, execs small
  Python and shell probes into `jaato_runner_t` and `jaato_child_t` with
  `setexeccon` (the mechanism phase 2b will use, design §7.1 and §7.3), and
  prints one PASS / FAIL / SKIP line per property.

## Rules for this run

1. **Do not push, commit, comment on the PR, or open issues.** Report to
   the user.
2. **Do not disable SELinux** host-wide. `semanage permissive` on a jaato
   domain is not used by this run. The `kernel_t` / `kernel_generic_helper_t`
   permissive domains phase 0 needed to keep WSL alive stay as they are.
3. **Do not edit the module to make a probe pass.** A FAIL is the result.
   Record it, collect the AVCs, and move on. The one exception is §5,
   which asks for `audit2allow` output as *evidence*, never to load it.
4. Work as root. Keep test trees on the Linux root filesystem, never under
   `/mnt/c` (no labels there).

Write the results to `/root/jaato-phase2a/RESULTS.md` using the template in
§6, and show the user its contents at the end.

## 1. Gate

Same gate as phase 0 §1. **Under WSL, after any `wsl --shutdown`, bring
SELinux up in this order** (the 2026-10-02 run lost the distro doing it any
other way). `/etc/selinux/config` says `enforcing`, so `load_policy -i`
loads straight into enforcing, and with WSL's `/init` and session plumbing
confined in `kernel_t`, every exec fails, `sudo` included.

```bash
# 1. kernel_t / kernel_generic_helper_t permissive IN THE STORE (-N: no
#    reload yet). Phase 0's cleanup removed them; check before every load.
semanage permissive -l | grep -E 'kernel_t|kernel_generic_helper_t' || {
  semanage permissive -N -a kernel_t
  semanage permissive -N -a kernel_generic_helper_t; }
# 2. the filesystems the policy loader needs
mountpoint -q /sys/kernel/security || mount -t securityfs securityfs /sys/kernel/security
mountpoint -q /sys/fs/selinux     || mount -t selinuxfs  selinuxfs  /sys/fs/selinux
# 3. load WITHOUT -i: the mode stays permissive
load_policy
# 4. relabel, then the parts WSL's systemd never labels (they skewed the
#    first run: /dev/null and /dev/ptmx were device_t, resolv.conf tmpfs_t)
restorecon -RF / -e /mnt -e /proc -e /sys -e /dev -e /run
restorecon -R /dev /run/systemd/resolve; restorecon /mnt
[ -e /mnt/wsl/resolv.conf ] && chcon -t net_conf_t /mnt/wsl/resolv.conf
# 5. only now
setenforce 1
```

`auditd` does not run under WSL (PID 1 stays in `kernel_t`), so AVCs go to
`dmesg` only, rate-limited. The probe handles that itself (§4). Proceed
only when:

```bash
mkdir -p /root/jaato-phase2a && cd /root/jaato-phase2a
getenforce                         # Enforcing
sestatus | grep -E 'Loaded policy|MLS status|Current mode'
id -Z                              # a real context, not "kernel"
semodule -l | grep -E '^jaato'     # see §2: jaato_phase0 must go
```

## 2. Install the module

The throwaway `jaato_phase0` module declared the same types
(`jaato_runner_t`, `jaato_policy_v1_t`, ...). Two modules cannot both
declare a type, so remove it first.

```bash
semodule -r jaato_phase0 2>/dev/null; semodule -l | grep jaato   # nothing
dnf -y install git python3.12 selinux-policy-devel policycoreutils-python-utils \
  setools-console audit
cd /root && { [ -d jaato ] || git clone https://github.com/Jaato-framework-and-examples/jaato.git; }
cd jaato && git fetch origin && git checkout claude/awesome-lovelace-bbca88 \
  && git pull --ff-only && git log --oneline -3
cd jaato-server/selinux
make -f /usr/share/selinux/devel/Makefile jaato.pp
semodule -i jaato.pp
semodule -l | grep '^jaato'                       # jaato  1.0.0
seinfo -t | grep -c '^ *jaato_'                   # 8 types
```

Record the `make` output if it warns. If `semodule -i` fails, stop: every
later step depends on the module.

**A venv the runner can read.** `jaato_runner_t` cannot read `/root` or
`/home` (by type, which is the point), so the venv for the import probe
lives under `/opt` and is labelled `lib_t` (design §9). Install non-editable:
an editable install points back into `/root/jaato`, which the runner cannot
read.

```bash
python3.12 -m venv /opt/jaato/venv
/opt/jaato/venv/bin/pip install -q -U pip
/opt/jaato/venv/bin/pip install -q /root/jaato/jaato-sdk /root/jaato/jaato-server
semanage fcontext -a -t lib_t '/opt/jaato/venv(/.*)?'
restorecon -R /opt/jaato/venv
ls -Zd /opt/jaato/venv /opt/jaato/venv/bin/python
```

## 3. The branch's own checks

The policy tests should give the same answer here as in CI (78 passed).
**Never run them on this host directly**: each case links a module with
`semodule -N -i`, which writes the host's module store (it only skips the
reload). Run them in a container if Docker is available; otherwise skip
this step and say so.

```bash
cd /root/jaato
docker run --rm -v "$PWD":/w:ro -w /w -e JAATO_SELINUX_POLICY_TESTS=1 fedora:44 \
  bash -c 'dnf -y -q install selinux-policy-devel selinux-policy-targeted \
    setools-console policycoreutils make python3-pytest &&
    pytest -q -p no:cacheprovider --noconftest jaato-server/selinux/tests' \
  || echo "no docker: skipped"
```

Then the readiness check, which phase 1a wrote for this moment:

```bash
cd /root/jaato && python3.12 -m venv .venv 2>/dev/null
.venv/bin/pip install -q -e jaato-sdk/. -e jaato-server/.
.venv/bin/python - <<'EOF'
from jaato_server.server.confinement.selinux import SELinuxBackend
b = SELinuxBackend()
print("host_readiness     :", b.host_readiness())
print("unavailable_reason :", b.unavailable_reason)
EOF
```

Expected: `ready=True, reason=None, enforcing=True`, and the reason text
*"the host is ready ... cannot provision SELinux boundaries yet"* (phase 1a
decided `is_available()` stays `False` until 2b). Anything else is a
finding.

## 4. The probe

```bash
cd /root/jaato-phase2a
python3 /root/jaato/jaato-server/selinux/tools/probe_policy.py \
  --root /srv/jaato-2a --venv /opt/jaato/venv | tee probe.txt
```

Then run it again as a non-root uid, which is how a runner runs under
`--runner-uid-policy` (#1168). The `dac_override` denials of a root run
should disappear:

```bash
mkdir -p uid && cd uid && python3 /root/jaato/jaato-server/selinux/tools/probe_policy.py \
  --root /srv/jaato-2a --venv /opt/jaato/venv --as-uid 1000 | tee probe.txt; cd ..
```

Each run writes `probe.txt`, `probe-results.json` and `probe-avc.txt` (the
AVCs the run produced: from `ausearch` when `auditd` runs, otherwise from
`dmesg` after a marker the probe writes, with `kernel.printk_ratelimit`
set to 0 for the run and restored). It toggles the `domain_fd_use` boolean
off and back on for one probe; check it is back:

```bash
getsebool domain_fd_use            # on
```

What each line means, and why it is there:

| Check | Design | Why it can fail on a kernel and not in CI |
|---|---|---|
| enters `jaato_runner_t` at its level | §7.1 | `entrypoint` on the interpreter's real label; role/user authorization |
| reads and writes its workspace | §4.2 | a class or permission the interface macros miss (`rename`, `reparent`) |
| another workspace refused by level | §5.2 | MCS constraint applies to the jaato domains (`mcs_constrained`) |
| authored config readable, not writable/creatable/renamable/removable | §6 | |
| writes a reference claim | §4.2 | |
| writes its session tmpdir; host `/tmp` refused | §6 | |
| `/root` unreachable; cannot change its own domain | §4.4 | |
| uses the daemon's socketpair, with `domain_fd_use` on and off | §4.1 | the one inherited-fd case phase 0 found silent |
| opens a pty | §4.3 | `interactive_shell` |
| DNS + outbound TCP | §4.3 | SKIP when offline |
| runner execs `/bin/sh` into `jaato_child_t` | §7.3 | `setexec` + `transition` + `entrypoint` on `shell_exec_t` |
| child cannot read runner's `environ`, set an exec context, write claims | §4.3, §4.4 | |
| child opens a pty | §4.3 | the children `interactive_shell` starts own the pty's slave end |
| child runs a managed workspace's binary, not a user checkout's | §4.2 | |
| runner imports `jaato_server.server.runner` from `/opt/jaato/venv` | §9 | the venv's labels; `pyvenv.cfg` |

## 5. When a check fails

For each FAIL, record the probe line and the matching AVCs:

```bash
grep -A3 'type=AVC' probe-avc.txt | head -60
audit2allow -i probe-avc.txt -R > audit2allow.txt   # evidence only, never load it
```

`audit2allow` names the missing permission; whether granting it is right is
decided in phase 2b against the design, not here. A PASS with AVCs beside
it (a `dontaudit`-free denial that did not break the probe) is also worth
recording: phase 2b decides whether it becomes a `dontaudit` or a grant.

## 6. Results template (`/root/jaato-phase2a/RESULTS.md`)

```markdown
# SELinux phase 2a results

- Date, host: 
- `uname -r`, `rpm -q selinux-policy-targeted`: 
- Branch commit (`git log --oneline -1`): 
- Module (`semodule -l | grep ^jaato`): 

## Checks

| # | Check | Result | Observed |
|---|---|---|---|
| 3 | policy tests (container) | passed / skipped | |
| 3 | host_readiness | | |
| 4 | probe lines (paste probe.txt's table) | | |

## AVCs for each FAIL

## audit2allow (evidence only)

## Surprises

## Anything that contradicts docs/design/selinux-backend.md
```

## 7. Cleanup

```bash
semodule -r jaato
semanage fcontext -d -t lib_t '/opt/jaato/venv(/.*)?'
rm -rf /srv/jaato-2a /tmp/jaato-2a-* /opt/jaato
getsebool domain_fd_use            # on
```

Leave `/root/jaato` and `/root/jaato-phase2a` for the user.
