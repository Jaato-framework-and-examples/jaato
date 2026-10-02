# SELinux backend, phase 0: handoff for an e2e run on RHEL under WSL

This file is an instruction set for an agent (a Claude Cowork session)
with shell access to a RHEL instance running under WSL2. It covers
installing jaato's branch, running its tests, and running the phase-0
experiments from [the SELinux backend design](selinux-backend.md) §11.
Phase 0 answers questions the design assumes but nobody has checked on a
real kernel. **Nothing in this run changes jaato.** The output is one
results file.

- Repository: `https://github.com/Jaato-framework-and-examples/jaato`
- Branch: `claude/awesome-lovelace-bbca88` (draft PR
  https://github.com/Jaato-framework-and-examples/jaato/pull/1431)
- What the branch contains: the design doc, a label reader that
  understands SELinux contexts (`jaato_server/shared/lsm_label.py`), a
  backend seam (`jaato_server/server/confinement/`), and an SELinux
  backend that checks readiness only. `SELinuxBackend.is_available()` is
  always `False`, so it never confines anything yet.

## Rules for this run

1. **Do not push, commit, comment on the PR, or open issues.** Report to
   the user.
2. **Do not disable SELinux** (`setenforce 0`, `selinux=0`, editing
   `/etc/selinux/config` to `permissive`/`disabled`). If a step seems to
   need it, stop and report. Making one *domain* permissive with
   `semanage permissive` is allowed where a step says so, and must be
   undone.
3. Everything this run installs into the policy is the throwaway module
   `jaato_phase0`. Remove it at the end (`semodule -r jaato_phase0`), and
   delete `semanage` rules you added.
4. Work as root inside the distro (`wsl -d <distro> -u root`). Keep all
   test trees on the Linux root filesystem (ext4), **never under `/mnt/c`**:
   drvfs/9p mounts carry no SELinux labels.
5. Record every command's output that a result depends on. When a check
   fails, capture `ausearch -m AVC,SELINUX_ERR -ts recent` right after.
6. When the gate in §1 fails and its remedies do not fix it, **stop**.
   Report what you saw. Everything after §1 is meaningless without an
   active SELinux LSM.

Write the results to `/root/jaato-phase0/RESULTS.md` using the template in
§6, and show the user its contents at the end.

## 1. Gate: is SELinux actually running?

**Read this first.** WSL2 runs Microsoft's kernel, not RHEL's. That kernel
may have SELinux compiled in without enabling it at boot. And WSL's PID 1
is Microsoft's `/init`, not systemd, so the step that normally loads the
policy at boot may never run. A RHEL userland under WSL is therefore not
the same as a RHEL host, and phase 0 needs the real thing. Establish it
before anything else.

```bash
mkdir -p /root/jaato-phase0 && cd /root/jaato-phase0
cat /etc/redhat-release; uname -r
cat /proc/cmdline
cat /sys/kernel/security/lsm; echo
ls /sys/fs/selinux 2>&1 | head
getenforce; sestatus
id -Z; ps -eZ | head -5
cat /proc/1/comm
stat -fc %T /sys/fs/cgroup          # cgroup2fs = v2
```

The gate passes when **all** of these hold:

- `selinux` appears in `/sys/kernel/security/lsm`;
- `sestatus` reports `enabled`, policy `targeted`, **mode `enforcing`**,
  and `Policy MLS status: enabled` (targeted uses MCS through this);
- `id -Z` prints a real context (for example
  `unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023`), not `kernel`
  and not an error.

Remedies, in order. Record which ones you tried and what each changed.

1. **systemd.** Check that `/etc/wsl.conf` contains `[boot]` /
   `systemd=true`. If you add it, the user must run `wsl --shutdown` on
   Windows and restart the distro. This matters for experiment 2.1:
   services get `unconfined_service_t` only under systemd.
2. **Kernel command line.** On Windows, in `%UserProfile%\.wslconfig`:

   ```ini
   [wsl2]
   kernelCommandLine = security=selinux selinux=1 enforcing=0
   ```

   Then run `wsl --shutdown` and re-check. The user has to make this
   Windows-side change, so ask them. `enforcing=0` is only for the first
   boot, while files are relabelled; it is removed below.
3. **Policy not loaded** (the LSM is listed but `sestatus` says no policy
   is loaded): run `load_policy`, then relabel the root filesystem with
   `fixfiles -F onboot` or `restorecon -RF / -e /mnt` (slow). Remove
   `enforcing=0`, restart, and run `setenforce 1` if the mode is still
   permissive. If the policy has to be loaded by hand after every boot,
   note that in the results; it is a WSL property, not a jaato one.
4. **SELinux not compiled in** (`/sys/fs/selinux` never appears whatever
   the command line says): WSL needs a custom kernel built with
   `CONFIG_SECURITY_SELINUX=y`, set through `.wslconfig` `kernel=<path>`.
   **Do not attempt this unasked.** Stop and report. The recommended
   fallback is a Hyper-V (or other) VM running RHEL 9 or 10, where every
   step below applies unchanged.

## 2. Install

```bash
dnf -y install git python3.12 python3.12-devel gcc make krb5-devel \
  selinux-policy-devel policycoreutils-python-utils setools-console \
  audit util-linux
systemctl enable --now auditd 2>/dev/null || true   # for ausearch
```

(RHEL 9 needs 9.4 or later for `python3.12` in AppStream; on RHEL 10 it is
the system Python. `dnf` needs a registered subscription.) If `auditd`
cannot run under WSL, AVCs still reach `dmesg`: use
`dmesg | grep -i avc` wherever this document says `ausearch`.

Clone and check out the branch. The user supplies GitHub access, for
example `gh auth login` or an HTTPS token:

```bash
cd /root && git clone https://github.com/Jaato-framework-and-examples/jaato.git
cd jaato && git checkout claude/awesome-lovelace-bbca88 && git log --oneline -3
python3.12 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e jaato-sdk/. -e "jaato-server/.[all]" pytest pytest-asyncio \
  || .venv/bin/pip install -e jaato-sdk/. -e jaato-server/. pytest pytest-asyncio pexpect openai
```

Record which install line succeeded. If `[all]` failed, record the
package that broke it.

## 3. The branch's own tests

```bash
cd /root/jaato
.venv/bin/pytest -q \
  jaato-server/jaato_server/shared/tests/test_lsm_label_reads_selinux.py \
  jaato-server/jaato_server/server/tests/test_confinement_backend_seam.py
.venv/bin/python scripts/check.py --no-fetch
```

The two targeted files use fakes and must pass on any host: 13 and 22
tests. `check.py` runs the required `contract-guards` job plus the suite
legs the branch touches. Report failures verbatim. Where the cause is
clearly environmental (a missing optional SDK, WSL), say so, but keep the
output.

### 3.1 The readiness check, on this real kernel, before any policy

```bash
cat > /root/jaato-phase0/probe.py <<'EOF'
from jaato_server.shared.lsm_label import (
    active_lsm_backend, selinux_host_enforcing, parse_lsm_label,
    parse_selinux_context, selinux_domain_permissive)
from jaato_server.server.confinement import select_backend
from jaato_server.server.confinement.selinux import SELinuxBackend
raw = open("/proc/self/attr/current").read()
print("active_lsm_backend :", active_lsm_backend())
print("host enforcing     :", selinux_host_enforcing())
print("own context        :", parse_selinux_context(raw))
print("own label          :", parse_lsm_label(raw, "selinux"))
print("own domain permissive:", selinux_domain_permissive(raw.strip("\0\n ")))
b = SELinuxBackend()
print("host_readiness     :", b.host_readiness())
print("unavailable_reason :", b.unavailable_reason)
print("is_available       :", b.is_available())
c = select_backend(apparmor=lambda: None, selinux=SELinuxBackend,
                   environ={"JAATO_CONFINEMENT": "auto"})
print("select_backend     :", c.describe())
EOF
.venv/bin/python /root/jaato-phase0/probe.py
```

Expected before §4: `active_lsm_backend: selinux`; host enforcing `True`;
own label `identity=''` (an unconfined shell is not in a jaato domain);
readiness **not ready**, reason *"the jaato SELinux policy module is not
loaded (jaato_runner_t is unknown)"*; `is_available False`. Any other
reason is a finding. Record it exactly.

## 4. The throwaway policy module

This module is **not** jaato's policy. It is the smallest module that lets
the phase-0 questions be asked. The real module is phase 2.

```bash
mkdir -p /root/jaato-phase0/policy && cd /root/jaato-phase0/policy
cat > jaato_phase0.te <<'EOF'
policy_module(jaato_phase0, 1.0.0)

require {
    type unconfined_t;
    type unconfined_service_t;
    type bin_t;
    role unconfined_r;
    role system_r;
}

type jaato_runner_t;
type jaato_child_t;
type jaato_workspace_t;
type jaato_tmp_t;
type jaato_policy_v1_t;          # version marker the readiness check probes

domain_type(jaato_runner_t)
domain_type(jaato_child_t)
files_type(jaato_workspace_t)
files_type(jaato_tmp_t)
files_type(jaato_policy_v1_t)

role unconfined_r types { jaato_runner_t jaato_child_t };
role system_r types { jaato_runner_t jaato_child_t };

# The question under test: MCS isolation between two runners of one domain.
# In the targeted policy MCS constraints bind only types marked this way.
mcs_constrained(jaato_runner_t)
mcs_constrained(jaato_child_t)

# Exec transition into the runner (design §5.1: setexeccon then execve).
allow unconfined_t jaato_runner_t:process transition;
allow unconfined_service_t jaato_runner_t:process transition;
allow jaato_runner_t bin_t:file { entrypoint read open execute map getattr };
allow jaato_runner_t jaato_child_t:process transition;
allow jaato_child_t bin_t:file { entrypoint read open execute map getattr };

# Only for experiment 5.6 (setcon in a threaded process).
allow unconfined_t jaato_runner_t:process dyntransition;

# Enough to run coreutils.
corecmd_exec_bin(jaato_runner_t)
libs_use_ld_so(jaato_runner_t)
libs_use_shared_libs(jaato_runner_t)
miscfiles_read_localization(jaato_runner_t)
files_read_etc_files(jaato_runner_t)
kernel_read_system_state(jaato_runner_t)
domain_use_interactive_fds(jaato_runner_t)
userdom_use_inherited_user_terminals(jaato_runner_t)
allow jaato_runner_t self:process { fork sigchld signal getattr };
allow jaato_runner_t self:fifo_file rw_fifo_file_perms;

# Workspace and tmp: the type grants, MCS decides which level.
allow jaato_runner_t jaato_workspace_t:dir { list_dir_perms add_entry_dir_perms };
allow jaato_runner_t jaato_workspace_t:file { read_file_perms create_file_perms write_file_perms };
allow jaato_runner_t jaato_tmp_t:dir { list_dir_perms add_entry_dir_perms };
allow jaato_runner_t jaato_tmp_t:file { read_file_perms create_file_perms write_file_perms };

# Reading /proc of other runner processes (MCS is what must refuse it).
allow jaato_runner_t jaato_runner_t:dir list_dir_perms;
allow jaato_runner_t jaato_runner_t:file read_file_perms;
allow jaato_runner_t jaato_runner_t:lnk_file read_lnk_file_perms;
EOF
make -f /usr/share/selinux/devel/Makefile jaato_phase0.pp
semodule -i jaato_phase0.pp && semodule -l | grep jaato
```

If the build fails on an interface name (the names vary a little between
RHEL 9 and 10), look it up with `grep -rl <name> /usr/share/selinux/devel/`
and swap in the local equivalent. Record every substitution.

**Adjusting the module.** An experiment may hit a denial the module did
not anticipate. Run `ausearch -m AVC -ts recent | audit2allow -R` to see
what it suggests.

- You **may** add rules for infrastructure: libraries, locale,
  terminals/fds, `/proc/self`, signals to self.
- You **must not** add any rule that widens what the experiment measures:
  - no rule on `jaato_workspace_t`, `jaato_tmp_t` or `jaato_runner_t`
    beyond the ones above;
  - no `mcs*` attribute such as `mcsreadall`, `mcswriteall` or
    `mcsptraceall`;
  - nothing that makes a level check pass.

Record every rule you add, with the AVC it answered.

## 5. Experiments

Common setup:

```bash
P=/root/jaato-phase0; mkdir -p $P/wsA $P/wsB
echo secret-A > $P/wsA/f; echo secret-B > $P/wsB/f
chcon -R -t jaato_workspace_t -l s0:c1,c2 $P/wsA
chcon -R -t jaato_workspace_t -l s0:c3,c4 $P/wsB
ls -Zd $P/wsA $P/wsA/f $P/wsB $P/wsB/f
```

Each experiment records: the command, its output, pass/fail against the
expectation, and the AVCs.

### 5.1 Exec transition from a service context (design §5.1)

The daemon runs as a systemd service (`unconfined_service_t`) and enters
the runner with `setexeccon` + `execve`, the same thing libselinux's
`runcon` does.

```bash
cat > $P/enter.py <<'EOF'
import ctypes, os, sys
lib = ctypes.CDLL("libselinux.so.1", use_errno=True)
lib.setexeccon.argtypes = [ctypes.c_char_p]
if lib.setexeccon(sys.argv[1].encode()) != 0:
    e = ctypes.get_errno(); print("setexeccon failed", e, os.strerror(e)); sys.exit(2)
os.execv(sys.argv[2], sys.argv[2:])
EOF
# a) from the interactive root shell
/usr/bin/python3.12 $P/enter.py unconfined_u:unconfined_r:jaato_runner_t:s0:c1,c2 \
  /usr/bin/cat /proc/self/attr/current; echo
# b) from a transient systemd service (needs systemd, §1 remedy 1)
systemd-run --wait --pipe -q /usr/bin/id -Z
systemd-run --wait --pipe -q /usr/bin/python3.12 $P/enter.py \
  system_u:system_r:jaato_runner_t:s0:c1,c2 /usr/bin/cat /proc/self/attr/current; echo
```

Pass: (a) and (b) each print a `jaato_runner_t:s0:c1,c2` context, and
`id -Z` in (b) shows `unconfined_service_t`. If (b) cannot run because of
WSL, record that. Also test the move from `s0` to `s0:c1,c2`: whether a
source at `s0` may enter a higher category set is exactly the question.

### 5.2 MCS isolation on files

```bash
R='/usr/bin/python3.12 /root/jaato-phase0/enter.py unconfined_u:unconfined_r:jaato_runner_t'
$R:s0:c1,c2 /usr/bin/cat $P/wsA/f          # expect: secret-A
$R:s0:c1,c2 /usr/bin/cat $P/wsB/f          # expect: Permission denied
$R:s0:c3,c4 /usr/bin/cat $P/wsB/f          # expect: secret-B
$R:s0:c1,c2 /usr/bin/sh -c "echo x > $P/wsB/new"   # expect: denied
$R:s0:c1,c2 /usr/bin/sh -c "echo x > $P/wsA/new && ls -Z $P/wsA/new"
```

Pass: cross-level reads and writes are denied; same-level ones succeed;
a file created in `wsA` inherits `s0:c1,c2`. **If the cross-level read
succeeds, it is the most important finding of the run.** Check that
`mcs_constrained` took effect: `seinfo -a mcs_constrained_type -x | grep jaato`.

### 5.3 MCS isolation on `/proc` (design §6: reading another session's environ)

```bash
$R:s0:c1,c2 /usr/bin/sleep 300 & sleep 1
PID=$(pgrep -n sleep); ps -Z -p $PID
$R:s0:c3,c4 /usr/bin/cat /proc/$PID/environ | head -c 80; echo "rc=$?"
$R:s0:c3,c4 /usr/bin/cat /proc/$PID/status | head -3
$R:s0:c1,c2 /usr/bin/cat /proc/$PID/status | head -3
kill $PID
```

Pass: the `c3,c4` reads of the `c1,c2` process's `/proc` entries are
denied. The same-level read of `status` succeeds. Note which files differ.

### 5.4 Labels across `restorecon` (design §4.3)

```bash
semanage fcontext -a -t jaato_workspace_t "/root/jaato-phase0/ws(A|B)(/.*)?"
restorecon -Rv $P/wsA; ls -Z $P/wsA/f       # expect: level s0:c1,c2 kept
restorecon -RFv $P/wsA; ls -Z $P/wsA/f      # expect: level reset (design says -F strips it)
chcon -R -l s0:c1,c2 $P/wsA
semanage fcontext -d "/root/jaato-phase0/ws(A|B)(/.*)?"
```

Record both outcomes exactly. The design relies on plain `restorecon`
keeping the level.

### 5.5 Relabel cost on a large tree (design §4.3)

```bash
mkdir -p $P/big && cd $P/big
/usr/bin/python3.12 - <<'EOF'
import os
for d in range(1000):
    os.makedirs(f"n/{d}", exist_ok=True)
    for f in range(100):
        open(f"n/{d}/{f}.js", "w").write("x")
EOF
cd $P; find big -type f | wc -l
time chcon -R -t jaato_workspace_t -l s0:c5,c6 big
cat > $P/relabel.py <<'EOF'
import os, sys, time
ctx = sys.argv[2].encode() + b"\0"; t = time.monotonic(); n = 0
for root, dirs, files in os.walk(sys.argv[1]):
    for name in dirs + files:
        os.setxattr(os.path.join(root, name), "security.selinux", ctx,
                    follow_symlinks=False); n += 1
print(n, "entries", round(time.monotonic() - t, 2), "s")
EOF
/usr/bin/python3.12 $P/relabel.py big system_u:object_r:jaato_workspace_t:s0:c7,c8
rm -rf big
```

Record both timings (100,000 files plus 1,000 directories).

### 5.6 A private `/dev/shm` with a label (design §6, #1381)

```bash
unshare -m sh -c '
  mount --make-rprivate / &&
  mount -t tmpfs -o "nosuid,nodev,mode=1777,context=system_u:object_r:jaato_tmp_t:s0:c1,c2" tmpfs /dev/shm &&
  ls -Zd /dev/shm &&
  '"$R"':s0:c1,c2 /usr/bin/sh -c "echo a > /dev/shm/a && cat /dev/shm/a" ;
  '"$R"':s0:c3,c4 /usr/bin/cat /dev/shm/a ; echo "cross rc=$?"'
```

Pass: the mount shows the given context, the `c1,c2` runner can write and
read, and the `c3,c4` read is denied.

### 5.7 `setcon` in a threaded process (design §7, why pool slots are deferred)

```bash
cat > $P/setcon.py <<'EOF'
import ctypes, os, sys, threading, time
lib = ctypes.CDLL("libselinux.so.1", use_errno=True)
lib.setcon.argtypes = [ctypes.c_char_p]
if sys.argv[1] == "threaded":
    threading.Thread(target=time.sleep, args=(3,), daemon=True).start()
rc = lib.setcon(sys.argv[2].encode()); e = ctypes.get_errno()
os._exit(0 if rc == 0 else 10 + e)   # no I/O after setcon: the new domain may not write
EOF
C=unconfined_u:unconfined_r:jaato_runner_t:s0
/usr/bin/python3.12 $P/setcon.py single $C;   echo "single rc=$?"
/usr/bin/python3.12 $P/setcon.py threaded $C; echo "threaded rc=$?"
```

Expected: `single` exits 0; `threaded` exits `10+errno` (EPERM, so 11, or
EACCES, so 23) because `jaato_runner_t` is not bounded by `unconfined_t`.
Record the AVC or `SELINUX_ERR` line.

### 5.8 The readiness check with the module loaded

```bash
cd /root/jaato && .venv/bin/python /root/jaato-phase0/probe.py
```

Expected: `host_readiness` ready (`ready=True, reason=None,
enforcing=True`), and the reason text becomes *"the host is ready ...
cannot provision SELinux boundaries yet"*. If it is not ready, record the
reason. *"may not transition into jaato_runner_t"* would mean the probe's
own shell domain is not `unconfined_t`; record `id -Z`.

Then check that each readiness check fails with the right reason:

- Build a copy of the module without `type jaato_policy_v1_t;` and
  `files_type(jaato_policy_v1_t)`, install it in place of the original,
  and run the probe. Expect *"older than version 1"*. Reinstall the
  original afterwards.
- Remove the `allow unconfined_t jaato_runner_t:process transition;` line
  the same way. Expect *"may not transition into jaato_runner_t"*.
  Reinstall the original.

### 5.9 Permissive domain detection (design §3.3)

```bash
semanage permissive -a jaato_runner_t
/usr/bin/python3.12 $P/enter.py unconfined_u:unconfined_r:jaato_runner_t:s0:c1,c2 \
  /root/jaato/.venv/bin/python -c '
from jaato_server.shared.lsm_label import parse_lsm_label
print(parse_lsm_label(open("/proc/self/attr/current").read(), "selinux"))'
semanage permissive -d jaato_runner_t
```

Expected: `identity='jaato_runner_t:s0:c1,c2'`, `mode='permissive'`,
`enforced=False`. Run it again with the domain enforcing if the module
lets Python start. It probably will not, since the module grants only
enough for coreutils; if so, record the denial and move on.

### 5.10 Host facts

Record the output of these commands:

- `stat -fc %T /sys/fs/cgroup`
- `cat /sys/fs/cgroup/cgroup.controllers`
- `sestatus -v`
- `seinfo | head -20`
- `rpm -q selinux-policy-targeted libselinux kernel`

## 6. Results template (`/root/jaato-phase0/RESULTS.md`)

```markdown
# SELinux phase 0 results
- Date / operator:
- Host: RHEL <ver> under WSL2 | VM; kernel <uname -r>; cmdline <...>
- Gate (§1): pass/fail; remedies applied:
- Install (§2): which pip line worked; failures:
- Branch tests (§3): 13/13, 22/22, check.py: <summary>
- Readiness before policy (§3.1): <probe output>
- Module (§4): built/installed; interface substitutions; extra rules added (+ the AVC each answered)

| # | Experiment | Expected | Result | Notes / AVCs |
|---|---|---|---|---|
| 5.1a | exec transition from root shell | runner context | | |
| 5.1b | exec transition from systemd service | runner context | | |
| 5.2 | MCS on files | cross-level denied | | |
| 5.3 | MCS on /proc | cross-level denied | | |
| 5.4 | restorecon keeps level / -F strips | kept / stripped | | |
| 5.5 | relabel 101k entries | timings | | |
| 5.6 | labelled /dev/shm tmpfs | context applied, cross denied | | |
| 5.7 | setcon threaded | single ok, threaded refused | | |
| 5.8 | readiness with module / each failure reason | ready / right reasons | | |
| 5.9 | permissive domain read as permissive | mode=permissive | | |
| 5.10 | cgroup / policy versions | facts | | |

## Surprises
## Anything that contradicts docs/design/selinux-backend.md
```

## 7. Cleanup

```bash
semodule -r jaato_phase0
semanage permissive -l | grep jaato && semanage permissive -d jaato_runner_t
semanage fcontext -l -C | grep jaato     # should print nothing
getenforce                               # must still say Enforcing
```

Leave `/root/jaato-phase0/` in place: it holds the results and the module
sources. If the §1 remedies changed `.wslconfig` or `wsl.conf`, tell the
user, so they can decide whether to keep those changes.
