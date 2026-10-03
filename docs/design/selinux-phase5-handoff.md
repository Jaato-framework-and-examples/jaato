# SELinux backend, phase 5: handoff for the install command and the denial hint

This is the instruction set for an agent (or a person) with root in the
Fedora 44 WSL2 distro that ran [phase 4](selinux-phase4-handoff.md).
Phase 5 adds two things ([design §9 and the feature map](selinux-backend.md)):

- **`jaato-selinux install | uninstall | status`**. The module source now
  ships inside the jaato-server package. `install` builds it with the
  host's `selinux-policy-devel`, loads it, labels this venv `lib_t` and
  relabels `~/.jaato` (`--home DIR` for other users). It then runs the
  daemon's readiness check. This run installs the module **only** through
  it: no `make`, `semodule`, `semanage` or `restorecon` by hand.
- **SELinux denial hints in `cli`.** A command the policy refused carries a
  `denial_hint` naming SELinux, the child domain and the file's type. The
  runner asks the kernel (`security_compute_av`): module **1.8.0** (marker
  `jaato_policy_v5_t`) grants `jaato_runner_t` `security:compute_av`, and
  no other jaato domain.

CI checks the policy rules and the command's steps with fakes. This run
answers what CI cannot: does `install` produce a ready host, and does a
real refusal get a real hint? The output is one results file.

- Repository: `https://github.com/Jaato-framework-and-examples/jaato`
- Branch: `claude/selinux-phase5`
- Tools, under `jaato-server/selinux/tools/`:
  - `probe_policy.py`: the phase 4 probe plus two hint checks (the runner
    may ask the policy, a child may not).
  - `live_session.py --hint`: the model runs a program on the PATH that is
    labelled `var_t` for the run, which the child may not execute; the
    tool checks that the result carries an SELinux `denial_hint`.

## Rules for this run

The rules of the earlier handoffs apply unchanged:

- Report only. Do not push, comment or open issues.
- Never disable SELinux host-wide.
- Do not edit the module to make a check pass. A FAIL is a result.
- Work as root, on the Linux filesystem (not `/mnt/c`).
- `live_session.py` starts its own daemon with its own socket, PID file and
  log. **Never** run `jaato_server --stop`, `--restart` or `--status`
  without a private `--pid-file`, and never touch another daemon's socket.

Write the results to `/root/jaato-phase5/RESULTS.md` (template in §6) and
show the user its contents at the end.

## 1. Gate

Follow the 2a handoff §1. Proceed only when `getenforce` says `Enforcing`
and `id -Z` prints a real context. Make sure no jaato module is loaded
(`semodule -l | grep jaato` prints nothing; the last cleanup removed it).

## 2. Install jaato, then the module with the command

```bash
mkdir -p /root/jaato-phase5 && cd /root/jaato-phase5
cd /root/jaato && git fetch origin && git checkout claude/selinux-phase5 \
  && git pull --ff-only && git log --oneline -1
rm -rf /opt/jaato/venv && python3.12 -m venv /opt/jaato/venv
/opt/jaato/venv/bin/pip install -q -U pip
/opt/jaato/venv/bin/pip install -q /root/jaato/jaato-sdk '/root/jaato/jaato-server[interactive]'
cd /root/jaato-phase5
/opt/jaato/venv/bin/jaato-selinux status | tee status-before.txt
/opt/jaato/venv/bin/jaato-selinux install --home /root --home /home/apanoia | tee install.txt
echo "exit=$?"
/opt/jaato/venv/bin/jaato-selinux status | tee status-after.txt
semodule -l | grep jaato                 # jaato 1.8.0
seinfo -t jaato_policy_v5_t              # present
ls -Zd /opt/jaato/venv /opt/jaato/venv/bin/python
semanage fcontext -l -C | grep /opt/jaato/venv
```

| Check | What a FAIL means |
|---|---|
| `status` before install says `not ready: … not loaded …; run (as root): jaato-selinux install` | the readiness reason does not name the remedy |
| `install` exits 0 and its last line is `ready: policy module v5 …` | a step failed (it is named) or the result is not a ready host |
| the venv is `lib_t` and one local fcontext rule names it | the venv labelling step |
| a second `install` also exits 0 | re-running modifies the rule instead of failing on it |

Then run `uninstall`, check that the module and the rule are gone, and run
`install` again for the rest of the run.

## 3. Readiness

As in the phase 4 handoff §3. Expected: `ready=True`, and the doctor's
selinux row says `policy module v5`.

## 4. The policy probe, both uids

```bash
cd /root/jaato-phase5
P=/root/jaato/jaato-server/selinux/tools/probe_policy.py
python3 $P --root /srv/jaato-5 --venv /opt/jaato/venv | tee probe.txt
mkdir -p uid && cd uid && python3 $P --root /srv/jaato-5 --venv /opt/jaato/venv \
  --as-uid 1000 | tee probe.txt; cd ..
```

New, expected to PASS: `hint: the runner may ask the policy` (`OK refused`:
the answer arrives, and it is that a child may not execute a workspace
file) and `hint: a child may not ask the policy` (`DENIED`).

## 5. The live sessions

```bash
cd /root/jaato-phase5
L=/root/jaato/jaato-server/selinux/tools/live_session.py
/opt/jaato/venv/bin/python $L --root /srv/jaato-5 --hint | tee live-hint.txt
cp /srv/jaato-5/live-session.json live-hint.json; cp /srv/jaato-5/d.out live-hint-d.out
/opt/jaato/venv/bin/python $L --root /srv/jaato-5-uid --hint --as-uid 1000 | tee live-hint-uid.txt
cp /srv/jaato-5-uid/live-session.json live-hint-uid.json
# regressions
/opt/jaato/venv/bin/python $L --root /srv/jaato-5-pool --sessions 2 | tee live-pool.txt
/opt/jaato/venv/bin/python $L --root /srv/jaato-5-iso --isolated rw | tee live-iso.txt
```

| Check | What a FAIL means |
|---|---|
| the probe program was refused (it did not run) | the child may execute `var_t`, so there was nothing to explain |
| the refused command's result carries an SELinux `denial_hint` | the runner could not ask the policy (an AVC on `security_t` `compute_av`), could not read the file's label, or the cli output was not recognised; quote the tool result from `live-hint.json` |
| the phase 2b to 4 checks in the other runs | a regression |

Quote the `denial_hint` text in the results. `/usr/local/bin/jaato-hint-probe`
is created and removed by the tool; check it is gone.

## 6. Results template (`/root/jaato-phase5/RESULTS.md`)

```markdown
# SELinux phase 5 results

- Date, host, `uname -r`, `rpm -q selinux-policy-targeted`:
- Branch commit:

## jaato-selinux: status before, install (twice), uninstall, install
## Readiness and doctor
## Probe (root / --as-uid 1000): pass counts, every FAIL with its AVCs
## Live sessions (hint root / uid; pool, isolated regressions): each
   check; the denial_hint as received
## AVCs naming a jaato type that no check asked for
## Surprises
## Anything that contradicts docs/design/selinux-backend.md
```

## 7. Cleanup

```bash
/opt/jaato/venv/bin/jaato-selinux uninstall
rm -rf /srv/jaato-5* /opt/jaato
rm -f /root/.jaato/agents/jaato-2a-probe.md /root/.jaato/memories/jaato-2a-probe.txt \
      /root/.jaato/jaato-2a-probe_auth.json /root/.jaato/selinux_levels.json*
ls /usr/local/bin/jaato-hint-probe 2>/dev/null   # absent
getsebool domain_fd_use                    # on
```

Leave `/root/jaato` and `/root/jaato-phase5` for the user. If you ran with
`--as-uid`, remove the same three probe files from that user's `~/.jaato`.
