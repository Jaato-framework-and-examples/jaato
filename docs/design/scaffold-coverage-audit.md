# Scaffold coverage audit: the last 50 merged PRs

Audited 2026-10-04 against `main` at `cef2ec14` (PR #1516). The question:
did any of the last 50 merged pull requests change the framework in a way
that should also have updated a `jaato-scaffold` surface, and not do so?

## Method

- **The 50 PRs** are the first-parent merge commits on `main`, newest
  first, from #1516 back to #1405 (`git log --first-parent --merges`).
  The `gh` CLI was not usable in this environment, so the list comes from
  the merge history rather than the pulls API. Every PR in the window was
  merged with a merge commit, so the two give the same set.
- **Per PR**, the merge diff (tests and docs excluded) was scanned for the
  signals the audit cares about: `subagent/config.py`, `get_config_schema`
  and `config.get(...)` reads, `os.environ` / `get_session_env` reads,
  `PROVIDER_*` contracts, new plugin packages, `TRAIT_*` constants,
  `jaato_sdk/events.py`, `apparmor.py`, and paths under `.jaato/`. Each
  hit was then checked against the surface it needed by reading the code
  on `main` and running the scaffold.
- **Run against a scratch workspace**: `validate` on a profile using every
  knob the window added (`references.allow_inline_content`,
  `witness_proposals`, `refresh_catalog`, `max_transitive_references`,
  `template.allow_inline_template`, `memory.require_curation`,
  `runtime_limits.seccomp` / `seccomp_allow`, `todo.initial_plan_name`);
  `explain env`, `runtime`, `paths`, `transports`, `pool`, `runner-user`,
  `plugin permission`, `plugin references`.
- **Snapshot**: `python -m jaato_server.shared.scaffold.authoring_contracts
  --write` leaves `authoring_snapshot.json` unchanged. Current.
- **Guards on `main`**: `scripts/check.py` (contract-guards) passes, and
  the scaffold, SDK scaffold and explain/validate tests pass (578). Nothing
  was red before this audit.

## Confirmed gaps

| PR | Change | Surface it needed | Status |
|---|---|---|---|
| #1502 Reuse a loaded AppArmor profile | `JAATO_APPARMOR_PROFILE_GRACE_SECONDS` | `explain env` description (`# env:` comment on the read line); `env_scope` was updated | **missing → fixed here** |
| #1454 Daemon lock profiling | `JAATO_LOCK_HOLD_WARN_MS` | same | **missing → fixed here** |
| #1476 Resize the runner pool live (protocol 1.35) | new `explain pool` topic | the integration skill's topic list | **missing → fixed here** |
| #1463 `explain runner-user` | new topic | the integration skill's topic list | **missing → fixed here** |
| #1516 Posture-aware pool capacity (#1507) | the floor counts VIRGIN slots, not unreserved ones | skill `references/cascade.md` still said "a floor on unreserved idle slots" | **stale → fixed here** |
| #1466 User tier on the envelope (#1465) | runners read `~/.jaato` files from a snapshot shipped at spawn | `explain paths` still described `~/.jaato` as read live | **missing → fixed here** |
| #1490 permissions.json decides a session (#1474) | `~/.jaato/permissions.json` is policy layer 2 | `explain plugin permission`, `validate` and `gitignore` were updated; `explain paths` did not mention it | **minor → fixed here** (same lines as #1466) |
| #1431, #1484, #1488 SELinux backend, phases 2–4 | a second confinement backend, `JAATO_CONFINEMENT`, MCS-labelled workspaces, `jaato_runner_t` / `jaato_child_t` | no `explain` topic or skill text names it; `validate`'s AppArmor-fragment checks do not say fragments are inert under SELinux | **missing → issue #1531** |

Two defects in the guards themselves turned up while fixing these:

- `test_every_topic_the_skill_lists_is_one_the_cli_has` matched topics with
  `[a-z]+`, so a hyphenated topic would be read as its prefix
  (`runner-user` as `runner`). It now matches `[a-z-]+`.
- Nothing required the reverse direction, which is how `explain pool` and
  `explain runner-user` shipped without reaching the skill.
  `test_every_topic_the_cli_has_is_one_the_skill_lists` now does. The five
  older topics the skill also omitted (`commands`, `event`,
  `integrations`, `releases`, `gh`) are added.

The `# env:` gap was wider than the window: 15 older `JAATO_*` vars and
`LEDGER_PATH` also rendered with no description (`JAATO_CHROME_AI_*`,
`JAATO_WEBMCP_*`, `JAATO_CREDENTIAL_LOCK_TIMEOUT`,
`JAATO_OAUTH_REFRESH_MARGIN`, `JAATO_LEDGER_INTEGRITY`,
`JAATO_PROFILE_SET`, `JAATO_APPARMOR_COMPLAIN`). All are described now,
and `test_every_jaato_env_var_is_described.py` fails a new framework var
that has none.

## Updated in the PR that made the change

| PR | Change | Surface | Status |
|---|---|---|---|
| #1505 seccomp child filter (#1503) | `runtime_limits.seccomp` / `seccomp_allow` | `explain runtime` (families, enforcer, inheritance); `validate` `seccomp_unknown_family`, `seccomp_disabled` (in `HIGH_RISK_ESCALATED_CODES`) | updated |
| #1490 permissions.json layers (#1474) | policy layers, `permission_file_allow` | `explain plugin permission`, `validate`, `gitignore` `AUTHORED` (`confined=True`), `env_scope` | updated |
| #1461 Runner uid drop (#1168) | `--runner-uid-policy`, `JAATO_RUNNER_UID_POLICY` | `env_scope`; topic in #1463 | updated |
| #1454 Lock profiling | `JAATO_LOCK_HOLD_WARN_MS` | `env_scope` (description: see gaps) | updated |
| #1502 AppArmor profile reuse | `JAATO_APPARMOR_PROFILE_GRACE_SECONDS` | `env_scope` (description: see gaps) | updated |
| #1431 SELinux phase 1–2 | `JAATO_RUNNER_SELINUX_LABEL` | `env_scope` (internal) | updated |
| #1483 Defer the embedder (#1482) | offline-first model load | `env_scope` | updated |
| #1449 Pages from catalog templates | `template.allow_inline_template`, `references.allow_inline_content` | declared in `get_config_schema`; `validate` `template_only_gate_leaks` | updated |
| #1436, #1420 wikiLLM references | `proposeReference`, claims dir, `witness_proposals`, `require_curation`, typed links | knobs declared; `gitignore` `CONFINED_STATE` (`references-claims/`); `validate` `reference_link_*` | updated |
| #1473 `see-also` link (#1472) | a fourth link relation | `explain plugin references` link table (derived from `links.REL_DOCS`) | updated |
| #1489 Revision claims (#1437) | `revises`, `revisions[]` | tool schema rendered live by `explain plugin references`; `validate_reference_file` accepts the keys | updated |
| #1497 Per-app workspace root (#1496) | `--ws-app-credentials` entry shape | `explain transports` ("per-application workspaces") | updated |
| #1476 Pool resize | `--pool-size`, `pool.resize` | `explain pool` (topic list: see gaps) | updated |
| #1516 Posture-aware pool | virgin floor, new counters | `explain pool` | updated |
| #1451 Facade `session_timeout` (#1450) | new facade kwarg | `explain clients`, `new cascade` template | updated |
| #1456 Candidate command pins (#1455) | install commands | `scaffold/releases.py` | updated |
| #1405, #1423, #1424, #1425, #1426, #1428 | `jaato-scaffold` itself (#1267 tiers, #1306) | — | the surface |
| #1427, #1448, #1453, #1480 release bumps | version | `authoring_snapshot.json` regenerated; current on `main` | updated |

## Not applicable

No scaffold-relevant change (runtime fixes, wire-level behaviour, protocol
additions whose fields `explain events` renders live, or web-coder
application code): #1518, #1517, #1515, #1512, #1500, #1498, #1493,
#1487, #1481, #1479, #1477, #1459, #1447, #1446, #1439, #1435, #1432,
#1421, #1419, #1417, #1416.

## What the audit could not check

- Whether a protocol verb deserved prose beyond `explain events`, which
  renders every event from the SDK's own models and so cannot be stale.
- The web coder (`jaato-web-coder-*`) and its out-of-tree toolchain plugin
  are application code and outside `jaato-scaffold`.

---
_Generated by [Claude Code](https://claude.ai/code)_
