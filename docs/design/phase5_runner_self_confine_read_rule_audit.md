# Phase 5 §X — Runner self-confine read-rule fix (template v15)

**Status:** Audit done, fix ready.
**Surfaced:** 2026-05-12 real-host verification of §4.3 isolated-subagent
spawn on Ubuntu 24.04 + AppArmor 4.0.1.

## 1. Problem

The IPC apparmor opt-in path (and the WS-default path) silently fails
to bring up a confined runner.  Daemon log shows:

```
[INFO] server.apparmor: Loaded AppArmor profile jaato-ws-<id>
[INFO] server.runner_spawn: runner spawned ... confined=True
[WARNING] server.runner_spawn: runner session.bootstrap failed: runner RPC closed
```

Runner subprocess crashes at `bootstrap.py:read_current_profile`:

```python
PermissionError: [Errno 13] Permission denied: '/proc/self/attr/current'
  File "server/runner/bootstrap.py", line 188, in confine_to_profile
    actual = read_current_profile(proc_attr_path)
  File "server/runner/bootstrap.py", line 119, in read_current_profile
    with open(proc_attr_path, "r") as f:
```

`confine_to_profile` does:

1. `aa_change_profile(profile_name)` via libapparmor → runner now in
   the profile.  Write side is allowed by the profile's existing
   `/proc/self/attr/current w,` rule.
2. **`read_current_profile(proc_attr_path)`** → verify kernel reports
   the new profile.  The READ is denied because the profile has
   `w,` only, no `r,`.

## 2. Root cause + scope

Two profile contexts are affected:

| Profile | Source | Read needed | Fix |
|---------|--------|-------------|-----|
| Base profile `jaato-ws-<id>` | `apparmor.py:460-461` (PROFILE_TEMPLATE) | YES — runner subprocess self-confines | change `w,` → `rw,` |
| tool_hat sub-profile | `apparmor.py:1874-1875` (`_build_tool_hat_subprofile`) | DEFENSIVE — daemon's `apparmor_confine` only writes, doesn't read.  Bring read in for symmetry + future-proofing | change `w,` → `rw,` |
| Isolated sub-profile `<parent>//<subagent>` | `apparmor.py:_render_sub_profile` | YES — sub-runner subprocess also calls `confine_to_profile` | add `/proc/*/attr/current r,` + `/proc/*/task/*/attr/current r,` next to existing `/proc/*/` rules.  **`r,` only, NOT `rw,` — intentional, see Phase 5 §5.10e (`phase5_5_10e_sub_runner_skip_audit.md`).  Sub-profile staying write-less is what lets §5.10e's bootstrap_session skip the §5.10c install for sub-runners: subprocesses inherit the no-escape-primitive posture by construction, no //child transition needed.** |
| //child sub-profile (Phase 5 §5.10a) | `apparmor.py:_build_child_subprofile` | NO — subprocesses entering //child via preexec_fn don't call `confine_to_profile`; they only get changeprofile'd from the parent context | no change |

NOT a §5.10 regression — broken since `bootstrap.py:read_current_profile`
landed (commit `4a8ec141`, 2026-05-07).  CI doesn't catch it because
the runner-side self-confine path is mocked in unit tests; the real-
host gate (audit §7.4) was only just exercised today.

## 3. Why no `r,` rule was added originally

Hypothesis: `confine_to_profile`'s verify-read step was added late in
Phase 2 (server/runner/ package skeleton, 2026-05-07) but the template
work pre-dated it.  Each landed without exercising the integration on
a real host, so the dependency on the read rule was missed.

## 4. Fix

Three template patches in `apparmor.py`:

```diff
@@ base profile body @@
-  /proc/self/attr/current      w,
-  /proc/self/task/*/attr/current w,
+  /proc/self/attr/current      rw,
+  /proc/self/task/*/attr/current rw,

@@ _build_tool_hat_subprofile @@
-    /proc/self/attr/current      w,
-    /proc/self/task/*/attr/current w,
+    /proc/self/attr/current      rw,
+    /proc/self/task/*/attr/current rw,

@@ _render_sub_profile (isolated subagent) @@
   /proc/*/fd/*              r,
+  /proc/*/attr/current      r,
+  /proc/*/task/*/attr/current r,
```

Bump `_TEMPLATE_VERSION` 14 → 15 to invalidate `apparmor_parser`'s
binary cache.

## 5. Test plan

Three regression pins extending `shared/tests/test_apparmor.py`:

1. `test_template_v15_base_profile_allows_attr_current_read` —
   `_render_profile()` output contains `/proc/self/attr/current` with
   either `rw,` or `r` permission, NOT pure `w,`.
2. `test_template_v15_tool_hat_allows_attr_current_read` — same
   assertion against the tool_hat sub-profile body.
3. `test_template_v15_isolated_sub_profile_allows_attr_current_read` —
   `_render_sub_profile()` output includes a rule matching
   `/proc/*/attr/current` with read permission.

Real-host integration test (gated; runs only when apparmor + cgroup
v2 are available + sudo apparmor_parser is passwordless):

4. `test_real_host_runner_self_confines_under_v15_profile` — provision
   a session via `_provision_profile_impl`, exec a Python subprocess
   under that profile, verify `cat /proc/self/attr/current` returns
   the expected profile name.  Existing `TestRealKernel` test class
   pattern.

## 6. Phase 5 §X status

Filed as a Phase 5 ad-hoc fix (not part of the original Theme structure).
Mark as shipped once the patch lands; unblocks §3.11 isolated-subagent
real-host verification + any future apparmor-opt-in IPC integration
testing.
