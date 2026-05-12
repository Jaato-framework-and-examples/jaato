# Phase 5 §5.10e — sub-runner subprocess spawn: skip §5.10c install

**Parent plan:** `per_session_confined_runner_phase5_plan.md` §5.10
(Theme C, memory backlog `project_backlog_apparmor_child_subprofile`).
**Predecessor:** §5.10a/b/c/d (PRs #64, #65, #69).
**v15 author sign-off:** received — Approach A confirmed; the
asymmetry between case 2 (skip + INFO) and case 3 (audible-failure)
is intentional.
**Status:** Shipped (this commit).

---

## 1. Finding — three-level nesting is NOT the right shape

The §5.10 audit doc (§6.4 "Sub-runner sub-profile case (§5.10e
preview)") posited that the §4.3.4 isolated-subagent sub-profile
"has the same escape vector — it also grants
`change_profile -> unconfined` + `/proc/self/attr/current w`".  **That
premise is wrong** — verified empirically by re-reading the v15
template at `apparmor.py:1135-1270`.

The sub-profile body, **as of v15** (predates §5.10), explicitly
DROPS those rules:

```
# ---- DROP: change_profile transitions (sub-runner stays in this
# profile for its lifetime; no further self-transitions).
```

And the proc rule (lines 1257-1258) carries only `r,`:

```
owner /proc/*/attr/current      r,
owner /proc/*/task/*/attr/current r,
```

vs. the base profile's `rw,` at line 484.

**The sub-profile already closes the escape vector for the
sub-runner case** — there's no writable attr/current to defend
against, and no `change_profile -> unconfined` capability.
Audit 6 + §4.3.4's design intent ("sub-runner stays in this
profile for its lifetime") gave the sub-profile a "no escape
primitive" posture from day one.

## 2. Latent break introduced by §5.10c

`server/runner/session.py:bootstrap_session` reads
`JAATO_RUNNER_PROFILE` and installs the §5.10c transition callback.
`RunnerSpawner._build_env` sets `JAATO_RUNNER_PROFILE` to the
profile name for every runner — including sub-runners, where the
name is `jaato-ws-{parent}//{subagent}` per Audit 6.

The §5.10c install path under a sub-runner:

1. `make_child_transition_callback("jaato-ws-parent//subagent")` →
   constructs a callable.  No I/O.  Install succeeds.
2. Sub-runner's cli/interactive_shell spawns a subprocess.
3. preexec_fn fires in the forked child: tries to write
   `changeprofile jaato-ws-parent//subagent//child` to
   `/proc/self/attr/current`.
4. **Sub-profile denies the write** (only `r,` capability).
   `os.open(..., O_WRONLY)` raises `PermissionError(13)`.
5. Popen sees a spawn failure → the cli/interactive_shell tool
   returns an error to the supervisor.

**Net effect of §5.10c on isolated subagents:** every cli or
interactive_shell tool call from a sub-runner fails with a
preexec_fn error.  Latent because the §5.10c real-host
verification (audit §7.4) targeted the main-session case, not the
isolated-subagent case.

§5.10e closes this regression.

## 3. Approach options

### Approach A — skip install for sub-runners (recommended)

In `bootstrap_session`, detect sub-runner mode (profile name
contains `//`, the Audit 6 nesting separator) and skip the
§5.10c install.  Subprocesses spawned by the sub-runner inherit
the sub-profile, which already lacks the escape primitive — no
additional confinement to apply.  Log INFO explaining the skip.

**Pros:**
- Smallest possible code change (~15 LoC + branch in
  bootstrap_session).
- No template change needed — v15's "no escape primitive" posture
  is already correct.
- Zero security regression — the sub-runner subprocess inherits
  the strict sub-profile by construction.
- Restores §4.3.4's spawn semantics for isolated subagents.

**Cons:**
- Two install paths to reason about (main: install; sub-runner:
  skip).  Documented inline + via INFO log so operators see the
  decision in their session bootstrap output.

### Approach B — three-level //child sub-sub-profile

Per the original §5.10 audit's §6.4 preview: add
`profile child { ... }` inline inside the sub-profile, grant the
sub-profile `/proc/self/attr/current w` (re-introducing the write
capability), and have the sub-runner's preexec_fn transition into
`//subagent//child`.

**Cons (load-bearing rejection):**
- Re-introduces write capability to attr/current in the
  sub-profile.  This is the EXACT escape primitive §5.10 closes
  for the main session.  Re-introducing it for sub-runners
  weakens the sub-profile's posture below what §4.3.4 / v15
  established — net security loss.
- The `//child` sub-sub-profile would mirror the sub-profile's
  body minus the (re-added) write rules.  But the sub-profile
  ALREADY lacks the write rules.  Net: the new `//child`
  sub-sub-profile is byte-equivalent to the existing sub-profile,
  which makes the transition a no-op functionally but adds two
  cgroup-style writes per spawn.
- Three-level transition strings (`X//Y//Z`) introduce parser
  complexity for marginal/negative benefit.

**Reject Approach B.**

### Approach C — defer to Phase 6

§5.10e isn't strictly required — Approach A is the right scope.
But the regression introduced by §5.10c (sub-runner spawn fails)
IS a latent bug that needs closure before Phase 5 ships.
Deferring would leave isolated subagents broken on
AppArmor-enabled hosts.  Reject.

## 4. Recommended fix — Approach A

### 4.1 Detection: profile name contains `//`

Audit 6's sub-profile naming convention is
`jaato-ws-{parent}//{subagent}`.  The `//` is the kernel-visible
nesting separator.  Any AppArmor profile name containing `//` is
a sub-profile by definition.

`bootstrap_session` reads `JAATO_RUNNER_PROFILE`; the presence of
`//` is the discriminator.

### 4.2 Skip semantics

When `//` is present:

- Skip the `make_child_transition_callback` construction.
- Skip the executor's `set_apparmor_child_transition_callback`.
- INFO log: "JAATO_RUNNER_PROFILE is a sub-profile ({name});
  skipping AppArmor //child transition install (sub-profile
  already drops the escape primitive per §4.3.4 v15 template)."

When `//` is absent (main session case): existing §5.10c logic
applies unchanged.  No regression to §5.10c's main-session
contract.

### 4.3 Why this is correct (not a degradation)

The §5.10c contract: "subprocesses spawned by model-controlled
plugins cannot escape the per-session profile via changeprofile".

For main sessions, the per-session profile carries the escape
primitive (base profile has `/proc/self/attr/current rw,`), so we
need the //child transition to drop it.

For sub-runners, the sub-profile NEVER carried the escape
primitive (template v15 designed it that way per §4.3.4).
Subprocesses spawned by the sub-runner inherit the sub-profile.
The contract holds by construction — no transition needed.

The audible-failure contract from the §5.10c peer review
("operator opts into kernel confinement; install MUST succeed or
bootstrap MUST fail") applies to the main-session case where the
install is the ONLY thing closing the escape vector.  In the
sub-runner case, the install isn't needed AT ALL — skipping it
isn't a degradation, it's the correct response to a context
that's already secure.

## 5. Scope

### In-scope

- Branch in `server/runner/session.py:bootstrap_session`: when
  `JAATO_RUNNER_PROFILE` contains `//`, skip the install and INFO-
  log the rationale.
- Regression tests pinning:
  - Sub-runner skip path: install not called, no error, INFO log
    fires with the rationale.
  - Main-session unchanged: install still called when profile
    lacks `//`.
  - Sub-profile template still drops the escape rules (smoke
    test against template-edit regression).

### Out of scope

- **Sub-sub-profile `//child` block in the sub-profile template.**
  Approach B, rejected (would re-introduce the write capability
  for no net security gain).
- **`make_child_transition_callback` API change.**  The factory
  already supports three-level nesting (verified by the §5.10b
  `test_session_profile_name_used_verbatim` test); we just don't
  invoke it for sub-runners.

## 6. Test plan

Regression pins extend
`server/runner/tests/test_runner_session_apparmor_child_install.py`
(the §5.10c install test file):

1. `test_install_skipped_when_runner_profile_is_subprofile` —
   `JAATO_RUNNER_PROFILE=jaato-ws-parent//subagent` → executor
   setter NOT called, INFO log explains the skip with the
   profile name + audit reference.
2. `test_install_runs_when_runner_profile_is_main_session` —
   `JAATO_RUNNER_PROFILE=jaato-ws-main_session` → install runs
   as before (pins §5.10c happy path stays intact).
3. `test_subprofile_skip_distinguished_from_disable_confine` —
   sub-profile path is INFO-level skip; `JAATO_RUNNER_PROFILE`
   empty is also INFO-level skip; both paths reach successful
   session bootstrap.

Plus a sub-profile template smoke test:

4. `test_sub_profile_template_drops_escape_rules` (in
   `shared/tests/test_apparmor_sub_profile.py` if it exists,
   else inline) — pins that the rendered sub-profile body
   does NOT contain uncommented
   `change_profile -> unconfined` or
   writable `/proc/*/attr/current` rules.  Catches accidental
   re-introduction by future template edits.

## 7. Real-host verification

The skip path is observable but not security-critical (no
escape primitive to test).  Recommended verification:

1. From a confined main session (post-§5.10c verified), spawn an
   isolated subagent via `agent_params.isolated=true`.
2. From the supervisor, ask the subagent to run a cli tool
   (e.g., `ls /etc`).
3. **Pre-§5.10e**: the cli tool fails with PermissionError
   (preexec_fn EACCES) — the sub-runner's spawn is broken.
4. **Post-§5.10e**: the cli tool succeeds.  Subprocess inherits
   the sub-profile.

Bonus exploit check (to confirm no regression of the §5.10c
guarantee):

5. From the subagent, run a cli with the exploit string
   `python3 -c 'open("/proc/self/attr/current","w").write("changeprofile unconfined")'`.
6. Subprocess inherits the sub-profile (no `w` capability) →
   PermissionError at open() time.  Same EACCES as §5.10c's
   main-session repro, just from inheritance rather than
   transition.

## 8. Phase 6 carryover

None.  §5.10e closes the §5.10 sub-track for Phase 5.

The original "three-level //child sub-sub-profile" idea is
parked permanently — it would require re-introducing the
write capability to the sub-profile, which contradicts the
§4.3.4 v15 design intent.  Any future hardening that needs
finer-grained sub-runner subprocess confinement should
start from "what additional rules does the sub-profile need
to deny" rather than "add a sub-sub-profile transition".

---

End of audit.
