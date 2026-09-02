"""AppArmor Manager for Jaato Server.

Manages per-session AppArmor profiles for workspace confinement.  When
available, provides kernel-enforced filesystem isolation so that CLI and
interactive-shell commands executed by one session cannot access files
belonging to another session.

When AppArmor is not available (non-Linux, not installed, or insufficient
privileges), all methods are no-ops and ``is_available()`` returns False.
Callers should check availability and fall back to directory-level sandboxing
(which is the existing default behaviour).

Profile naming convention: ``jaato-ws-{session_id}``

Thread-level confinement
------------------------

``apparmor_confine(profile_name)`` is a context manager that transitions
the *current OS thread* into the given AppArmor profile by writing to
``/proc/self/attr/current``, and restores the previous profile on exit.
This is used by the tool executor to confine in-process file I/O
(``readFile``, ``glob_files``, ``file_edit``) under the same AppArmor
profile that subprocesses get via ``aa-exec``.

``make_confine_context(profile_name)`` returns a zero-argument callable
that produces the context manager.  This callable is passed through
the ``server → shared`` boundary so that ``ToolExecutor`` (in ``shared/``)
can confine tools without importing ``server.apparmor``.
"""

import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import platform
import re
import shutil
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple

from shared.session_id import validate_session_id

logger = logging.getLogger(__name__)


class AppArmorManager:
    """Manages AppArmor profiles for per-session workspace confinement.

    Lifecycle (called by the WebSocket server or workspace provisioner):

    1. ``provision_profile(session_id, workspace_path)`` — renders a
       profile from the template, writes it to the profile directory,
       and loads it via ``apparmor_parser -r``.
    2. Tool execution is confined via the thread-level
       ``apparmor_confine()`` context manager (see below), which the
       ``ToolExecutor`` wraps around every tool call.  Subprocesses
       inherit the parent thread's confinement via fork+exec.
    3. ``teardown_profile(session_id)`` — unloads and removes the profile.

    All methods are safe to call even when AppArmor is unavailable; they
    degrade to no-ops.
    """

    # Bump this whenever the PROFILE_TEMPLATE changes in a way that
    # requires confined sessions to pick up new rules.  The value is
    # embedded as a comment in every rendered profile, which changes the
    # content hash and forces ``apparmor_parser`` to recompile against
    # its cache at ``~/.jaato/apparmor-cache`` instead of reusing a stale
    # entry.  Operators don't need to clear the cache manually; the next
    # session will load the new rules automatically.
    #
    # History:
    #   1 — initial template
    #   2 — memories/ folder (raw queue + curated.jsonl), retain legacy
    #       memories.jsonl for migration reads.
    #   3 — narrow write rule for /proc/self/{{task/*/,}}attr/current so
    #       apparmor_confine().__exit__ can actually restore unconfined.
    #       Without this, the kernel denied the file-write that executes
    #       the change_profile transition, leaving thread-pool workers
    #       stuck in the session's enforce-mode profile across sessions —
    #       subsequent operations (verify_auth reading ~/.jaato/*.json,
    #       reads from external sandbox-added paths) all hit EACCES.
    #   4 — read access to ~/.jaato/services/ so user-tier services
    #       (github, phoenix, etc.) are reachable from confined WS
    #       sessions.  SchemaStore gained tiered lookup in the same
    #       change; the profile needed to follow so the reads succeed.
    #   5 — ``include if exists`` directive at the end of the profile
    #       pointing at a per-session ``.refs.d/`` fragment directory.
    #       Allows ``add_reference_fragment()`` to grant readonly access
    #       to selected reference paths (selectReferences) without
    #       editing the base profile.  Empty/missing dir is a no-op
    #       thanks to ``if exists``, so existing sessions keep working.
    #   6 — ``{config_root_rules}`` placeholder grants read access to a
    #       client-supplied ``config_root`` override (from
    #       ``ClientConfigRequest.config_root``) so opt-in IPC AppArmor
    #       confinement (``IPCClient(apparmor=True)``) can serve
    #       framework config from somewhere other than the workspace.
    #   7 — Two additions:
    #       (a) Runtime-injected ``{env_file_rule}`` placeholder so
    #           reactor-spawned sessions can read the client-supplied
    #           env_file (``ClientConfigRequest.env_file``) from inside
    #           the parent's confined thread.
    #       (b) ``include if exists "~/.jaato/apparmor-fragments/*.rules"``
    #           directive at end of profile.  Daemon extensions drop
    #           their own fragment files there to grant access to
    #           extension-owned state (e.g. premium reactor's
    #           ``handoff_gates.json``) without polluting the public
    #           template with extension-specific paths.  Empty/missing
    #           dir is a no-op thanks to ``if exists``.
    #   8 — Two fixes for confined IPC sessions surfaced in early
    #       runs:
    #       (a) ``~/.jaato/references/`` granted read so the
    #           references plugin can ``initialize()`` (it scans the
    #           directory at construction time).  Without this the
    #           plugin fails to expose, agents lose preselected
    #           reference content, and start hallucinating "no
    #           services configured".
    #       (b) Extension-fragments include resolved to an absolute
    #           path at render time (``{fragments_dir}`` placeholder)
    #           instead of ``@{{HOME}}/...``.  ``apparmor_parser``
    #           running under sudo evaluates ``@{{HOME}}`` against
    #           the parser's environment, not the daemon user's
    #           home, so the include silently misses extension
    #           fragments.
    #   9 — Extension fragments are now INLINED at render time
    #       instead of resolved via ``include if exists`` directive.
    #       Even with absolute paths, ``apparmor_parser`` was finding
    #       fragment files via the include glob but not inlining
    #       their content into the parsed profile (silent miss with
    #       no error).  Reading the fragments at render time and
    #       embedding them directly below the header sidesteps the
    #       parser quirk and produces a self-contained profile
    #       that's also easier to debug from kernel denial logs.
    #  10 — Comment block introducing the inline-fragments section
    #       cleaned up to remove backtick-wrapped strings and
    #       brace-wrapped placeholder names.  ``str.format()`` was
    #       substituting placeholder names mentioned inside the
    #       comment, which produced a corrupted profile (backticks
    #       and stray fragment content escaping into the comment
    #       body) and tripped the AppArmor lexer with
    #       ``unexpected character: '`'`` errors at load time.
    #  11 — ``.jaato/`` carve-out from workspace rw rule.  Three
    #       ``audit deny`` rules block w/l/k under
    #       ``{workspace_path}/.jaato/**`` while the surrounding
    #       ``{workspace_path}/** rwkl`` keeps the rest of the
    #       workspace fully writable.  Closes the framework write-
    #       policy gap where confined sessions and reactors could
    #       write to ``.jaato`` (daemon-tier state) just like any
    #       other workspace path.  Server 0.6.52+; coordinates with
    #       0.6.49 (profile-load before configure), 0.6.50 (prefetch
    #       runs confined), 0.6.51 (diagnostic logging on abort).
    #  12 — Framework-subtree carve-outs added to the .jaato deny.
    #       Pre-v12, the broad deny blocked the daemon's OWN writes
    #       to .jaato/sessions/ (session journals) and .jaato/logs/
    #       (per-session log handlers) when triggered from a confined
    #       thread (e.g. log call inside a tool callback).  v12 adds
    #       more-specific allow rules for sessions/, logs/, cache/,
    #       vision/, services/_discovered/, waypoints.json, and
    #       *_auth.json so framework infrastructure writes succeed
    #       regardless of which thread initiated them.  Session-content
    #       writes to other paths under .jaato (e.g. arbitrary
    #       ``.jaato/state/...``) still hit the deny.  AppArmor's
    #       more-specific-wins rule does the carve-out resolution.
    #  13 — Two structural changes after v12 was empirically shown
    #       broken (cascade v17 / 7:3 reproducer, 2026-05-06):
    #
    #       (a) Broad ``audit deny .jaato/** w,l,k`` REPLACED with
    #           narrow per-subpath denies on user-authored config
    #           only (agents/, profiles/, prompts/, scripts/,
    #           services/<name>/, reactors.json, completion_schemas/,
    #           spawn_schemas/, instructions/, references/).  AppArmor
    #           empirically does NOT let more-specific allow override
    #           less-specific deny — the v12 carve-outs failed to
    #           lift the deny on ``.jaato/sessions/`` etc., causing
    #           confined-reactor session-journal saves to EACCES.
    #           Tenant-runtime subpaths (sessions/, logs/, cache/,
    #           vision/, services/_discovered/, memory/, todos/,
    #           waypoints.json, *_auth.json) are no longer under any
    #           deny — they fall through to the broad workspace rwkl
    #           and writes succeed regardless of confinement context.
    #
    #       (b) NEW ``profile tool_hat`` sub-profile added inside the
    #           per-session profile.  Tool execution
    #           (``ToolExecutor.execute``) enters the sub-profile via
    #           ``change_profile -> jaato-ws-{session_id}//tool_hat``
    #           and exits back to base on tool finish.  The sub-profile
    #           ADDS read-denies on user-authored config subpaths so
    #           LLM-driven file-read tools (cat via cli, file_edit's
    #           readFile, filesystem_query) cannot inspect agents/,
    #           profiles/, prompts/, schemas/, instructions/,
    #           reactors.json, scripts/.  Information-isolation
    #           between agents in a cascade.  Reactors and session-
    #           init stay in base (need reads to load agent persona,
    #           profile JSON, schemas).
    #
    #       Empirically verified: scripts/verify_apparmor_hat.sh,
    #       19/19 assertions pass on AppArmor 4.0.1.
    #
    #       AppArmor sub-profile semantics: hat/sub-profile rules
    #       REPLACE base profile rules (no inheritance).  The
    #       sub-profile must redeclare every allow + deny it wants
    #       to honor.  This template duplicates the base body inside
    #       ``profile tool_hat { ... }`` and appends the
    #       hat-specific read-denies.
    # Phase 5 §5.10 (template v14): add ``profile child`` sub-profile
    # that mirrors ``tool_hat``'s body but drops the three escape-
    # vector rules (``change_profile -> unconfined``,
    # ``/proc/self/attr/current w``, and the task variant).
    # Subprocess-spawning plugins transition into the sub-profile via
    # preexec_fn between fork() and exec(), so a model-controlled
    # subprocess can't break confinement by writing changeprofile to
    # attr/current.  See
    # docs/design/phase5_5_10_apparmor_child_subprofile_audit.md.
    # Phase 5 ad-hoc (template v15): add ``r`` permission to
    # ``/proc/self/attr/current`` (parent + tool_hat) and
    # ``/proc/*/attr/current`` (isolated sub-profile) so the runner's
    # ``confine_to_profile.read_current_profile`` verify-after-write
    # step in ``server/runner/bootstrap.py:188`` doesn't EACCES.
    # Surfaced 2026-05-12 by real-host §4.3 isolated-subagent
    # verification.  Not a §5.10 regression; broken since
    # bootstrap.py landed (commit 4a8ec141, 2026-05-07) but masked by
    # mock-only CI.  See
    # docs/design/phase5_runner_self_confine_read_rule_audit.md.
    # v16 (2026-05-12): grant ``mr`` (read + mmap-exec) on venv
    # C-extension shared objects (``*.so`` + ``*.so.*``).  The
    # ``#include <abstractions/python>`` line above covers
    # ``/usr/lib/python*/**`` and ``/usr/local/lib/python*/**``
    # with mmap-exec, but daemon venvs at other paths (``/tmp/...``,
    # ``~/.venvs/...``, etc.) fall outside those globs and have no
    # PROT_EXEC permission on their ``.so`` files.  Symptom:
    # ``import anthropic`` / ``import numpy`` / any other module
    # with a C-extension dependency fails inside the runner with
    # ``failed to map segment from shared object`` (kernel-level
    # mmap denial wrapped as a misleading "package not installed"
    # ToolError).  Narrow grant — only ELF shared objects get
    # mmap-exec; .py source and data files in the venv stay r-only.
    # v19 (2026-05-15): ``flags=(...)`` is now a placeholder so the
    # diagnostic env ``JAATO_APPARMOR_COMPLAIN`` can append ``complain``
    # mode — denials get logged to dmesg without enforcement.  Used to
    # harvest the missing-rule set in one cascade run, then ship
    # targeted grants from the kernel-audit data instead of guessing.
    # Default (env unset) behaviour is byte-identical to v18.
    # v20 (2026-05-16): plugin-contribution hook (Phase 0).  Plugins
    # implementing the optional ``ToolPlugin.get_apparmor_rules``
    # classmethod can contribute their own rule lines instead of having
    # them hardcoded in this template.  Daemon-side resolution walks
    # ``profile.plugins`` at session-spawn time, unions the rules, and
    # splices into the ``{plugin_contributed_rules}`` placeholder in
    # base + tool_hat + child profiles.  Default (no plugin overrides)
    # emits a "(none for this session)" comment line — byte-equivalent
    # rule semantics to v19.  See
    # ``docs/design/plugin-apparmor-contribution.md``.
    # v21 (2026-05-16): ML model caches (HF + torch) migrated to
    # ReferencesPlugin.get_apparmor_rules.  Sessions WITHOUT references
    # in ``profile.plugins`` no longer carry these grants (least-
    # privilege).  WS-server-side path also wired to feed plugin_rules
    # through; previously WS bypassed the resolver.
    # v22 (2026-05-16): ``JAATO_APPARMOR_COMPLAIN=1`` now propagates the
    # ``complain`` flag to the ``tool_hat`` + ``//child`` sub-profiles,
    # not just the base profile.  Sub-profiles do NOT inherit the base
    # flag in AppArmor; pre-v22 the env knob loosened the base but left
    # cli_based_tool subprocesses (which transition into ``//child`` via
    # ``change_profile``) running ENFORCE.  That asymmetry produced
    # silent denials surfacing as ENOENT (path-traversal block at the
    # kernel) rather than the expected complain-mode AVC log entries.
    # The flag knob now does what its name implies — entire profile
    # chain is complain when set.
    # v28 (2026-05-20): Phase 3 cascade-sharing — added
    # ``change_profile -> jaato-ws-*,`` rule to the main runner scope
    # so a reused pool slot can ``aa_change_profile`` to the next
    # session's profile during ``session.bootstrap`` step 1c re-entry.
    # See docs/design/runner-cascade-sharing.md §4.4.  The ``//child``
    # sub-profile does NOT inherit this rule; only the runner main
    # thread can trigger the transition (the LLM-driven scope can't
    # cross profile boundaries).
    _TEMPLATE_VERSION = 28

    # AppArmor profile template.  Placeholders are filled per-session by
    # ``_render_profile()``.
    PROFILE_TEMPLATE = '''\
# jaato-apparmor-template-version: {template_version}
#include <tunables/global>

profile jaato-ws-{session_id} flags=({profile_flags}) {{
  #include <abstractions/base>
  #include <abstractions/nameservice>
  #include <abstractions/python>

  # ---- workspace: read-write with narrow integrity write-denies ----
  # Sessions and session-scoped reactors can read/write/link anywhere
  # in the workspace, INCLUDING tenant-runtime ``.jaato/`` subpaths
  # (sessions/, logs/, cache/, vision/, services/_discovered/, memory/,
  # todos/, waypoints.json, *_auth.json) — those are tenant-owned
  # state and need to be writable from confined threads (reactor-
  # spawned session journaling, per-session log handlers, etc.).
  #
  # Server 0.6.55+ (template v13): broad ``.jaato/** w`` deny replaced
  # with narrow per-subpath denies on user-authored config ONLY
  # (agents, profiles, prompts, scripts, schemas, services definitions,
  # reactors.json, instructions, references).  Pre-0.6.55 the broad
  # deny was empirically shown to win over more-specific allows on
  # AppArmor 4.0 — the carve-out approach didn't work as designed.
  # Narrow per-subpath denies avoid the deny-vs-allow specificity
  # conflict entirely: tenant-runtime subpaths are not under any deny,
  # only the explicitly-named user-authored config subpaths are.
  #
  # ``audit deny`` makes denials visible in ``dmesg | grep apparmor``
  # so policy violations are diagnosable.  Each rule combines
  # ``w,l,k`` permissions because:
  # - ``w`` (write): primary integrity protection
  # - ``l`` (link/hardlink): a confined writer could otherwise hardlink
  #   the file into a writable path and write through the alias,
  #   bypassing the write deny
  # - ``k`` (file lock): otherwise a confined process could grab a
  #   lock that blocks daemon-side reads of the same file
  {workspace_path}/   rw,
  {workspace_path}/** rwkl,
  audit deny {workspace_path}/.jaato/agents/**             wlk,
  audit deny {workspace_path}/.jaato/profiles/**           wlk,
  # .jaato/prompts/ intentionally OMITTED from the write-denies: prompt_library's
  # savePrompt/deletePrompt run in the confined runner and need write/unlink
  # (unlink is granted by 'w' — classic AppArmor has no standalone 'd' mode).
  # Accepted posture tradeoff: a confined agent may author/rewrite prompts (whose
  # content is later executed). agents/profiles/scripts/schemas stay protected.
  audit deny {workspace_path}/.jaato/scripts/**            wlk,
  audit deny {workspace_path}/.jaato/services/*/           wlk,
  audit deny {workspace_path}/.jaato/reactors.json         wlk,
  audit deny {workspace_path}/.jaato/completion_schemas/** wlk,
  audit deny {workspace_path}/.jaato/spawn_schemas/**      wlk,
  audit deny {workspace_path}/.jaato/instructions/**       wlk,
  audit deny {workspace_path}/.jaato/references/**         wlk,
  # Workspace-tier AppArmor fragments — read by _render_profile on
  # the NEXT session spawn.  A confined runner that could write a
  # fragment here would be authoring its own future-session rules
  # (privilege escalation: confined now, broader-confined-or-
  # unconfined next).  Same wlk pattern as the other user-authored
  # config subpaths.
  audit deny {workspace_path}/.jaato/apparmor-fragments/** wlk,

  # ---- shared read-only resources ----
  {venv_path}/           r,
  {venv_path}/**         r,
  # v16: mmap-exec on venv C-extension shared objects.  Without
  # ``m`` on ``*.so`` / ``*.so.*`` the kernel rejects PROT_EXEC
  # mmap when Python imports a C extension whose ``.so`` lives
  # outside ``abstractions/python``'s /usr/lib/python and
  # /usr/local/lib/python coverage.  Narrow grant — only ELF
  # shared objects, not .py.
  {venv_path}/**/*.so    mr,
  {venv_path}/**/*.so.*  mr,
  {venv_path}/bin/*      ix,

  # ---- jaato source tree (read-only, for editable installs) ----
  # Required so plugin discovery and module imports work when jaato
  # is installed via `pip install -e`. Python loads modules from the
  # source tree, not from the venv site-packages, in editable mode.
  {source_root}/         r,
  {source_root}/**       r,

  # ---- premium package (read-only, optional) ----
  # Profile discovery, instructions, and other premium content must be
  # readable so discover_profiles() can scan all three tiers.
  {premium_rules}

  # ---- client-supplied config_root (read-only, optional) ----
  # When the client connects with ``config_root`` set on
  # ``ClientConfigRequest`` (decoupling read-only framework config from
  # the agent's workspace), the profiles, agent .md files, prompts,
  # references, completion_schemas, instructions, scripts, and services
  # all live under that root rather than ``<workspace>/.jaato/``.  We
  # grant read access to it so the daemon's discovery sites — and
  # plugins that surface those files to the agent (references,
  # prompt_library, etc.) — can keep working under confinement.
  {config_root_rules}

{user_global_block}

  # ---- client-supplied env_file (read-only, optional) ----
  # When the client passes ``env_file`` on ``ClientConfigRequest``,
  # the daemon's session-creation path reads it to populate provider
  # config (PROJECT_ID, JAATO_PROVIDER, etc.).  Reactor-spawned
  # subagents go through the same path during ``create_session``,
  # which fires from inside the parent's confined thread — without
  # this grant, every reactor-spawned session fails with
  # ``Permission denied`` on the env file.
  {env_file_rule}

  # ---- global memories: migrated to memory plugin (template v23) ----
  # Previously this block hardcoded the memories paths in the
  # framework template.  MemoryPlugin.get_apparmor_rules now contributes
  # those rules, spliced into the plugin-contributed section below.
  # Sessions whose profile.plugins lacks memory no longer carry these
  # grants -- least privilege.

  # ---- Claude Code interop + user prompts: migrated to prompt_library plugin (template v23) ----
  # Previously this block hardcoded the user-tier prompts/skills paths
  # and Claude Code interop paths.  PromptLibraryPlugin.get_apparmor_rules
  # now contributes those rules.  Sessions whose profile.plugins lacks
  # prompt_library no longer carry the grants.

  # ---- ML model caches: migrated to references plugin (template v21) ----
  # Previously this block hardcoded ~/.cache/huggingface + ~/.cache/torch
  # in the framework template.  ReferencesPlugin.get_apparmor_rules
  # now contributes those rules, spliced into the plugin-contributed
  # section below.  Sessions whose profile.plugins lacks references
  # no longer carry these grants -- least privilege.

  # ---- temp files scoped to session ----
  # Allow both file-prefix style (/tmp/jaato-<id>-foo) and subfolder
  # style (/tmp/jaato-<id>/foo) so plugins can use either layout.
  /tmp/jaato-{session_id}-** rw,
  /tmp/jaato-{session_id}/   rw,
  /tmp/jaato-{session_id}/** rw,

  # Note: sibling workspaces are implicitly denied by AppArmor's
  # default-deny policy.  An explicit deny on the sessions root would
  # override the workspace allow rule above (deny wins over allow at
  # the same specificity without priority annotations), blocking the
  # agent from reading its own workspace.

  # ---- basic system access ----
  /usr/bin/**          ix,
  /usr/local/bin/**    ix,
  /bin/**              ix,
  /usr/lib/**          rm,
  /lib/**              rm,
  /etc/ld.so.cache     r,
  /etc/passwd          r,
  /etc/nsswitch.conf   r,
  /proc/self/**        r,
  /dev/null            rw,
  /dev/urandom         r,
  /dev/pts/*           rw,

  # ---- network: outbound only ----
  network inet  stream,
  network inet6 stream,
  network inet  dgram,
  network inet6 dgram,
  deny network raw,

  # ---- deny dangerous capabilities ----
  deny ptrace,
  deny mount,
  deny capability sys_admin,
  deny capability net_admin,
  deny capability sys_ptrace,

  # ---- profile transitions ----
  # Required for the framework's apparmor_confine.__exit__ to restore
  # unconfined when a tool finishes.  Two rules are both needed:
  #
  #   1. ``change_profile -> unconfined`` — authorizes the semantic
  #      transition (AppArmor capability check).
  #   2. Write access to /proc/self/{{task/*/,}}attr/current — authorizes
  #      the file-write that EXECUTES the transition (AppArmor file check).
  #
  # Without both, apparmor_confine().__exit__ silently fails to restore
  # and the thread stays trapped in this profile.  Thread-pool workers
  # are reused across sessions, so a trapped worker picked up for
  # verify_auth in the next session gets EACCES on ~/.jaato/*_auth.json,
  # and one picked up for an external-sandbox-added path gets EACCES
  # reading that path — both observed in session 20260421_202113 before
  # this rule landed.
  #
  # !!! KNOWN ESCAPE VECTOR !!!
  # These rules ARE exploitable from any code running under this profile,
  # not just the framework's __exit__.  Verified empirically against
  # template version 10 (jaato-ws-20260501_123847, 2026-05-01):
  #
  #     sudo aa-exec -p jaato-ws-... -- python3 -c '
  #         open("/proc/self/attr/current","w").write("changeprofile unconfined")
  #         os.listdir("/etc/apt/apt.conf.d")  # 19 entries — UNCONFINED'
  #
  # write_ok=True, opendir succeeded.  The kernel gates by writer's
  # profile capabilities, not by who-in-userspace called write().  Two
  # surfaces inside jaato that can hit this:
  #
  #   - Subprocesses inherit the profile via fork+exec.  cli /
  #     interactive_shell tool calls can write the same string from
  #     model-controlled scripts.
  #   - In-process tools running on the worker thread (file_edit,
  #     readFile, glob_files, ...) are themselves in this profile.
  #     If their path validation lets the agent target
  #     /proc/self/attr/current, the write succeeds and the worker
  #     thread itself becomes unconfined for every subsequent call.
  #
  # Mitigations today (defense in depth, none kernel-enforced via this
  # profile):
  #
  #   - sandbox_utils.check_path_with_jaato_containment denies
  #     /proc/**/attr/** explicitly (even when workspace_root is unset),
  #     so file_edit / readFile / glob_files reject the in-process attack.
  #   - cli / interactive_shell subprocesses still inherit the rules.
  #     This is a known gap.
  #
  # Tracked fix: child subprofile (jaato-ws-S//child) for subprocesses
  # that drops both rules; parent keeps them for the framework's __exit__.
  # Helper-thread "do unconfine from outside the profile" doesn't work
  # because proc_pid_attr_write enforces ``current != task → -EACCES`` —
  # only the task itself can write its own attr/current.
  # See ``project_backlog_apparmor_child_subprofile`` memory.
  #
  # Phase 5 template v15: rule form switched from ``/proc/self/...`` to
  # ``owner /proc/*/...``.  Empirically, AppArmor resolves the
  # /proc/self/ symlink to /proc/<pid>/ BEFORE matching against rules,
  # so ``/proc/self/attr/current r,`` never matches a read.  The
  # write rule worked accidentally on v14 because procfs's
  # proc_pid_attr_write takes a different code path that bypassed the
  # full LSM file-mediation check.  The ``owner`` qualifier covers
  # both read and write against the resolved path while preserving
  # the process-is-owner constraint.
  #
  # The ``r`` permission is required by the runner subprocess's
  # ``confine_to_profile.read_current_profile`` verify-after-write
  # step (server/runner/bootstrap.py:188).
  #
  # Phase 5 §5.10 (template v17, 2026-05-15): add explicit
  # ``change_profile -> jaato-ws-{session_id}//child,``.  Pre-v17,
  # subprocess transitions into the inline-declared ``//child``
  # sub-profile were silently denied by the kernel.  The legacy
  # assumption (still papered into
  # ``make_child_transition_callback``'s pre-v17 docstring) was
  # "parent profile implicitly authorizes transitions to inline
  # sub-profiles".  Empirically false on AppArmor 4.0.1: kernel
  # ``dmesg`` shows
  # ``apparmor="DENIED" operation="change_profile" target="...//child"``
  # for every cli_based_tool subprocess.  The ``//child`` sub-
  # profile was declared but the transition was never authorized.
  #
  # Why this stayed latent: until per-stack apparmor_fragments
  # (Piece 1 + walker) landed, confined cascades ran with an
  # over-permissive workspace-tier fragment that masked the
  # denial (subprocesses likely fell through to the parent
  # profile or unconfined depending on the path).  Strict
  # per-stage scoping exposes it because the //child path is
  # actually taken now.
  #
  # tool_hat transitions do NOT need the symmetric rule because
  # ``apparmor_confine`` defensively writes ``changeprofile
  # unconfined`` first (see :func:`apparmor_confine` rationale),
  # bouncing through unconfined before entering the sub-profile.
  # The cli ``preexec_fn`` path can't do that bounce — the
  # security boundary requires the forked child to inherit the
  # parent profile and transition directly to ``//child``, so
  # the rule must be granted explicitly.
  change_profile -> unconfined,
  change_profile -> jaato-ws-{session_id}//child,
  # Phase 3 cascade-sharing (server 0.6.146+): permit transitions
  # between any two ``jaato-ws-*`` profiles so a reused pool slot can
  # re-confine to the next session's profile via ``aa_change_profile``
  # during ``session.bootstrap`` step 1c re-entry.  The transition
  # space is closed — every ``jaato-ws-*`` profile is framework-
  # composed against a per-session template; there's no operator-
  # untrusted profile matching the glob.  The ``//child`` sub-profile
  # does NOT inherit this rule: the LLM-driven scope cannot trigger
  # ``change_profile`` (no kernel-level capability + sandbox_utils
  # path validation denies /proc/**/attr/** writes).  See
  # docs/design/runner-cascade-sharing.md §4.4.
  change_profile -> jaato-ws-*,
  owner /proc/*/attr/current      rw,
  owner /proc/*/task/*/attr/current rw,

  # ---- per-session reference fragments ----
  # ``add_reference_fragment(session_id, ref_id, path)`` writes one
  # bare-rule file per selected reference into the .refs.d/ dir below
  # and reloads this profile.  ``if exists`` keeps things working when
  # the dir is absent (no references selected yet) and absorbs the
  # race between profile load and first selection.  Removed at session
  # teardown via _remove_all_reference_fragments().
  include if exists "{refs_include_glob}"

  # ---- extension-supplied fragments (host-wide, optional) ----
  # Daemon extensions (e.g. premium's reactor framework) may need
  # additional grants for state files they own (reactor handoff
  # gates, future cluster-state files, etc.) without leaking their
  # specific paths into this public template.  Each extension drops
  # a "*.rules" file at "~/.jaato/apparmor-fragments/" containing
  # bare rules; the daemon's AppArmorManager._render_profile reads
  # every matching file at session-creation time and INLINES the
  # contents directly below this header.
  #
  # Inlining (instead of "include if exists") because apparmor_parser
  # does not reliably expand globbed include paths in confinement
  # context — fragments would land on disk but silently miss the
  # profile, leaving confined sessions with Permission denied on
  # extension-owned state.  Inline expansion at render time
  # eliminates that whole class of failure.  Empty / missing dir is
  # a no-op (the placeholder renders as a single comment line).
  #
  # Note: this comment block deliberately avoids backtick-wrapped
  # strings and brace-wrapped tokens.  str.format() applied to this
  # template substitutes any brace-pair like {{NAME}} (or {{NAME}})
  # even inside comments, so embedding placeholder names would either
  # corrupt the rendered profile (when the name resolves) or raise
  # KeyError (when it does not).
{extension_fragments_inline}

{plugin_contributed_rules}

  # ---- tool_hat sub-profile (server 0.6.55+, template v13) ----
  # Tool execution context.  ``ToolExecutor.execute`` enters this
  # sub-profile via ``change_profile -> jaato-ws-{session_id}//tool_hat``
  # before invoking a plugin, and exits back to the parent profile
  # on tool finish.  AppArmor sub-profiles do NOT inherit base rules
  # (verified empirically against AppArmor 4.0.1) — every allow and
  # deny the sub-profile honors must be redeclared.
  #
  # The sub-profile mirrors the base body exactly + ADDS read-denies
  # on user-authored config so LLM-driven file reads can't inspect
  # other agents' personas, profile JSON, prompts, schemas, scripts,
  # instructions, or reactors.json.  Information-isolation between
  # agents in a cascade.
  #
  # Reactors and session-init stay in BASE (not in tool_hat) because
  # they need read access to the same paths to load agent personas,
  # validate completion payloads, etc.  ``apparmor_confine`` for
  # tool execution uses the sub-profile name; for everything else
  # (prefetch, reactor dispatch) uses the base profile name.
{tool_hat_subprofile}

  # ---- //child sub-profile (Phase 5 §5.10, template v14) ----
  # Subprocess execution context.  Subprocesses spawned by the cli /
  # interactive_shell plugins enter this sub-profile via
  # ``change_profile -> jaato-ws-{session_id}//child`` between fork()
  # and exec() (preexec_fn writes to /proc/self/attr/current).  The
  # body mirrors ``tool_hat``'s rule set verbatim and additionally
  # DROPS the three escape-vector lines that the base + tool_hat
  # profiles must keep for the framework's apparmor_confine.__exit__
  # path:
  #
  #     change_profile -> unconfined,        ← removed in //child
  #     /proc/self/attr/current      w,      ← removed in //child
  #     /proc/self/task/*/attr/current w,    ← removed in //child
  #
  # A process in //child cannot write changeprofile to attr/current
  # (kernel rejects with EACCES) and cannot transition anywhere.
  # The dangerous primitive is moved from "any subprocess can use
  # it" to "only the framework's parent-tier code can use it" —
  # closes the verified escape at apparmor.py:413-449.
  #
  # See docs/design/phase5_5_10_apparmor_child_subprofile_audit.md.
{child_subprofile}
}}
'''

    def __init__(
        self,
        workspace_root: str,
        venv_path: Optional[str] = None,
        profile_dir: str = "/etc/apparmor.d/jaato",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """Initialize the AppArmor manager.

        Args:
            workspace_root: Root directory containing ``sessions/``.
                The deny rule for sibling workspaces uses
                ``{workspace_root}/sessions/``.
            venv_path: Path to the Python venv (read-only access for the
                confined process).  Defaults to ``sys.prefix``.
            profile_dir: Directory to write profile files.  Defaults to
                ``/etc/apparmor.d/jaato``.
            loop: The daemon's main asyncio loop, captured at startup.
                Used by ``_run_unconfined`` to dispatch ``/etc/apparmor.d/``
                file writes from confined worker threads onto the
                always-unconfined main loop.  When ``None``, mutations
                run in the calling thread (suitable for tests or
                non-confined deployments).
        """
        import sys

        self._workspace_root = Path(workspace_root).expanduser().resolve()
        self._sessions_root = self._workspace_root / "sessions"
        self._venv_path = Path(venv_path or sys.prefix).resolve()
        self._profile_dir = Path(profile_dir)

        # User-local cache directory for apparmor_parser, avoiding the
        # system-level /var/cache/apparmor which requires root access.
        self._cache_dir = Path("~/.jaato/apparmor-cache").expanduser().resolve()

        # Detect the jaato source root for editable installs.  When
        # jaato is installed via ``pip install -e``, Python loads modules
        # from the source tree (not the venv site-packages), so the
        # source directory must be readable by the confined thread for
        # plugin discovery, model_provider initialization, etc. to work.
        # apparmor.py lives at jaato-server/server/apparmor.py, so the
        # repo root is two levels up from this file.
        self._source_root = Path(__file__).resolve().parents[2]

        # Detect premium package root (if installed).  Premium content
        # (profiles, instructions, etc.) must be readable by confined
        # sessions for profile discovery and instruction assembly.
        self._premium_root: Optional[Path] = None
        try:
            from shared.jaato_runtime import _get_premium_content_path
            premium_profiles = _get_premium_content_path("profiles")
            if premium_profiles:
                # Content paths are like <pkg>/profiles — parent is the package root
                self._premium_root = Path(premium_profiles).resolve().parent
        except ImportError:
            pass

        self._available: Optional[bool] = None  # Lazy-checked
        # Human-readable reason confinement is unavailable, set by
        # ``_check_availability`` on the first failing precondition.
        # ``None`` until the check runs or when confinement is available.
        # Surfaced via :attr:`unavailable_reason` so operators (and the
        # require-confinement startup gate) can report *why* the kernel
        # boundary is absent rather than just *that* it is.
        self._unavailable_reason: Optional[str] = None

        # Per-session locks for fragment-write + parser-reload races.
        # Lazily allocated by _session_lock(); cleaned up on teardown.
        # The guard lock protects the dict itself.
        self._session_locks: Dict[str, threading.Lock] = {}
        self._session_locks_guard = threading.Lock()

        # Daemon's main asyncio loop, captured at startup by the IPC /
        # WS apparmor hook.  Mutations to /etc/apparmor.d/jaato/ that
        # originate from confined worker threads (e.g. the references
        # plugin's selectReferences tool) get dispatched here so the
        # actual file write happens on the always-unconfined main loop
        # thread.  Without this, confined workers EACCES on the file
        # write and the only recovery is a daemon restart.
        self._loop = loop

    # ------------------------------------------------------------------
    # Cross-thread dispatch: confined-worker → unconfined main loop
    # ------------------------------------------------------------------

    def _run_unconfined(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Run *fn* on the daemon's main asyncio loop thread.

        The daemon's main loop is the only thread we can reliably know
        is unconfined — it's the thread that started the process before
        any per-session AppArmor profile was loaded, and we never
        confine it.  Worker pool threads (IPC executor, session
        ``_model_thread``, etc.) can become confined either deliberately
        (the tool executor's ``apparmor_confine`` context) or by
        accident (a confine-restore failure leaving the thread stuck in
        a per-session profile).  Reading
        ``/proc/self/task/<tid>/attr/current`` to detect "am I confined"
        is fragile — the kernel may report a profile name even after
        the kernel-side profile has been unloaded, the proc semantics
        vary across hosts, and a thread that's silently been confined
        without our knowledge will lie.

        So we don't ask "am I confined" — we ask "am I the loop thread".
        If yes, run *fn* in place (avoids scheduling-on-self deadlock).
        If no, dispatch *fn* to the loop unconditionally and block
        until it returns.  This guarantees that every code path which
        touches ``/etc/apparmor.d/jaato/`` runs on the loop thread,
        which is the only thread we trust to be unconfined.

        When ``self._loop`` is None (unit tests, non-AppArmor hosts),
        *fn* runs in the calling thread.

        Timeout: 60 s.  ``apparmor_parser -r`` can take 10–30 s on
        slow hosts; 60 s leaves headroom without masking a stuck loop.
        """
        if self._loop is None:
            return fn(*args, **kwargs)

        # If we're already running on the loop thread, call directly.
        # ``get_running_loop`` raises RuntimeError off-loop, which is
        # the signal we need to dispatch.
        try:
            if asyncio.get_running_loop() is self._loop:
                return fn(*args, **kwargs)
        except RuntimeError:
            pass

        async def _coro():
            return fn(*args, **kwargs)

        future = asyncio.run_coroutine_threadsafe(_coro(), self._loop)
        try:
            return future.result(timeout=60)
        except concurrent.futures.TimeoutError:
            logger.error(
                "AppArmor mutation timed out after 60s on main loop "
                "(fn=%s)", getattr(fn, "__name__", repr(fn)),
            )
            raise

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check whether AppArmor can be managed by this process.

        Returns ``True`` only when all of the following hold:

        - Running on Linux
        - ``apparmor_parser`` is on ``PATH``
        - ``aa-exec`` is on ``PATH``
        - ``/sys/kernel/security/apparmor`` exists (kernel module loaded)
        - The profile directory is writable (or can be created)
        """
        if self._available is not None:
            return self._available

        self._available = self._check_availability()
        if self._available:
            logger.info("AppArmor confinement available")
        else:
            # WARNING, not INFO: when the kernel boundary is absent the
            # only remaining workspace isolation is the bypassable
            # directory-sandbox heuristic. Operators need this in the log
            # at a visible level, with the specific failing precondition,
            # so the degradation is a deliberate, observable choice rather
            # than a silent one. Use the require-confinement startup gate
            # (--apparmor / JAATO_REQUIRE_APPARMOR) to fail closed instead.
            logger.warning(
                "AppArmor confinement NOT available (%s) — workspace "
                "isolation falls back to directory sandboxing only, which "
                "is a heuristic and not a kernel-enforced boundary. To "
                "require confinement and refuse to start unconfined, pass "
                "--apparmor (WS) or set JAATO_REQUIRE_APPARMOR=1.",
                self._unavailable_reason or "reason unknown",
            )
        return self._available

    def _check_availability(self) -> bool:
        """Perform the actual availability check (called once).

        On the first failing precondition, records a human-readable
        reason in ``self._unavailable_reason`` (surfaced via
        :attr:`unavailable_reason`) and returns ``False``. On success,
        clears the reason and returns ``True``.
        """
        def _unavailable(reason: str) -> bool:
            self._unavailable_reason = reason
            logger.debug("AppArmor unavailable: %s", reason)
            return False

        if platform.system() != "Linux":
            return _unavailable("not running on Linux")

        if not shutil.which("apparmor_parser"):
            return _unavailable("apparmor_parser not found on PATH")

        if not shutil.which("aa-exec"):
            return _unavailable("aa-exec not found on PATH")

        if not Path("/sys/kernel/security/apparmor").exists():
            return _unavailable(
                "kernel module not loaded "
                "(/sys/kernel/security/apparmor missing)"
            )

        # Check profile directory
        try:
            self._profile_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return _unavailable(
                f"cannot create profile dir {self._profile_dir}"
            )

        if not os.access(self._profile_dir, os.W_OK):
            return _unavailable(
                f"profile dir not writable: {self._profile_dir}"
            )

        # Verify sudo access to apparmor_parser (required for loading profiles)
        try:
            result = subprocess.run(
                ["sudo", "-n", "apparmor_parser", "--version"],
                capture_output=True, timeout=5,
            )
            if result.returncode != 0:
                return _unavailable(
                    "passwordless sudo for apparmor_parser unavailable "
                    "(no sudoers rule?)"
                )
        except (subprocess.TimeoutExpired, OSError):
            return _unavailable("sudo apparmor_parser check failed")

        # Create user-local cache directory for apparmor_parser
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return _unavailable(
                f"cannot create cache dir {self._cache_dir}"
            )

        self._unavailable_reason = None
        return True

    @property
    def unavailable_reason(self) -> Optional[str]:
        """Why confinement is unavailable, or ``None`` if available/unchecked.

        Populated by :meth:`_check_availability` (run on first
        :meth:`is_available` call). ``None`` before the check runs or when
        confinement is available.
        """
        return self._unavailable_reason

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def provision_profile(
        self,
        session_id: str,
        workspace_path: str,
        config_root: Optional[str] = None,
        env_file: Optional[str] = None,
        requested_fragments: Optional[List[str]] = None,
        plugin_rules: Optional[List[str]] = None,
    ) -> bool:
        """Create and load an AppArmor profile for a session.

        Dispatches to ``_provision_profile_impl`` via ``_run_unconfined``
        so the file write + ``apparmor_parser -r`` reload run on the
        unconfined main loop when called from a confined worker
        (e.g. a reactor-spawned child session whose creation fires
        from a confined parent thread).
        """
        return self._run_unconfined(
            self._provision_profile_impl,
            session_id, workspace_path, config_root, env_file,
            requested_fragments, plugin_rules,
        )

    def _provision_profile_impl(
        self,
        session_id: str,
        workspace_path: str,
        config_root: Optional[str] = None,
        env_file: Optional[str] = None,
        requested_fragments: Optional[List[str]] = None,
        plugin_rules: Optional[List[str]] = None,
    ) -> bool:
        """Inner body of :meth:`provision_profile`.

        Writes the rendered profile to
        ``{profile_dir}/jaato-ws-{session_id}`` and loads it with
        ``apparmor_parser -r``.

        Args:
            session_id: Session identifier (used in profile name).
            workspace_path: Absolute path to the session's workspace
                directory.
            config_root: Optional path to the client's
                ``config_root`` override.  When set, the profile grants
                read access to ``{config_root}/`` and ``{config_root}/**``
                so the daemon's discovery sites can read profiles,
                agent .md files, prompts, etc. from that root under
                confinement.  When ``None``, the agent reads framework
                config exclusively from the user-tier ``~/.jaato/``
                rules already in the template.
            env_file: Optional absolute path to the client-supplied
                env_file (from ``ClientConfigRequest.env_file``).  When
                set, the profile grants read access to that exact file
                so reactor-spawned sessions can populate provider
                config (PROJECT_ID, JAATO_PROVIDER, …) during
                ``create_session`` from inside the parent's confined
                thread.  Without this, a confined parent that triggers
                a reactor-spawned child fails with ``Permission
                denied`` on the env file.
            requested_fragments: Piece 1 (2026-05-14) — per-profile
                fragment scoping.  When ``None`` (default), composes
                EVERY ``*.rules`` file found under the user-tier,
                workspace-tier, and walker-cache fragment directories
                (back-compat — pre-Piece-1 behaviour).  When a list,
                composes ONLY fragments whose basename (without
                ``.rules``) matches an entry.  ``[]`` is distinct from
                ``None`` — explicit empty list composes NO fragments,
                yielding a maximally locked-down profile.  Unknown
                fragment names log WARNING but don't abort.  See
                ``project_backlog_per_profile_apparmor_fragments``
                for the cascade least-privilege motivation.

        Returns:
            True on success, False on failure (logged, not raised).
        """
        if not self.is_available():
            return False

        profile_name = self.get_profile_name(session_id)
        profile_path = self._profile_dir / profile_name
        profile_content = self._render_profile(
            session_id, workspace_path, config_root, env_file,
            requested_fragments=requested_fragments,
            plugin_rules=plugin_rules,
        )

        try:
            profile_path.write_text(profile_content)
        except OSError:
            logger.exception("Failed to write AppArmor profile %s", profile_path)
            return False

        try:
            result = subprocess.run(
                ["sudo", "apparmor_parser", "-r", "--cache-loc", str(self._cache_dir), str(profile_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error(
                    "apparmor_parser failed for %s: %s",
                    profile_name,
                    result.stderr.strip(),
                )
                # Clean up the written file
                profile_path.unlink(missing_ok=True)
                return False
        except subprocess.TimeoutExpired:
            logger.error("apparmor_parser timed out for %s", profile_name)
            profile_path.unlink(missing_ok=True)
            return False
        except OSError:
            logger.exception("Failed to run apparmor_parser for %s", profile_name)
            profile_path.unlink(missing_ok=True)
            return False

        logger.info("Loaded AppArmor profile %s", profile_name)
        return True

    # ──────────────────────────────────────────────────────────────────
    # Phase 4 §4.3.4 — sub-AppArmor profile for isolated subagents
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def _validate_subagent_id(cls, subagent_id: Any) -> Optional[str]:
        """Validate ``subagent_id`` for use in a sub-profile name +
        filename + ``change_profile`` target.

        Phase 4 §4.3.4 anchored this on AppArmorManager.  §4.3.5
        hoisted the body to ``server.subagent_id.validate_subagent_id``
        so the sub-cgroup validator can share the same implementation.
        The classmethod stays for backwards compat with existing
        callers + tests.

        Returns:
            ``None`` if valid; an error message string if invalid.
        """
        from server.subagent_id import validate_subagent_id
        return validate_subagent_id(subagent_id)

    @classmethod
    def get_sub_profile_name(
        cls, parent_session_id: str, subagent_id: str,
    ) -> str:
        """Return the kernel-visible sub-profile name.

        Template: ``jaato-ws-{parent}//{subagent_id}``.  The ``//``
        is a naming convention (standalone-with-prefix-name per
        Audit 6); the kernel treats this as a separate profile, not
        a true AppArmor hat.

        Caller is responsible for validating ``subagent_id`` first via
        :meth:`_validate_subagent_id` — this method does no checks
        because it's also used by callers that have already validated
        (e.g., test fixtures, log message builders).
        """
        return f"jaato-ws-{parent_session_id}//{subagent_id}"

    @classmethod
    def get_sub_profile_filename(
        cls, parent_session_id: str, subagent_id: str,
    ) -> str:
        """Return the on-disk filename for a sub-profile.

        Template: ``jaato-ws-{parent}__sub_{subagent_id}``.  Filenames
        can't contain ``/`` (path separator), so the kernel name's
        ``//`` is replaced with ``__sub_`` — matching the
        ``isolated_session_id`` template
        ``SessionManager._spawn_isolated_runner`` generates in §4.3.3.
        Callers can derive the filename from the isolated session id
        without re-encoding the convention.

        Caller is responsible for validating ``subagent_id`` first.
        """
        return f"jaato-ws-{parent_session_id}__sub_{subagent_id}"

    def provision_sub_profile(
        self,
        parent_session_id: str,
        subagent_id: str,
        workspace_path: str,
        tightenings: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Create and load an AppArmor sub-profile for an isolated
        subagent (Phase 4 §4.3.4).

        Standalone profile with a ``//``-prefixed name (per Audit 6's
        structural decision).  Mirrors :meth:`provision_profile`'s
        pattern: dispatches the file write + ``apparmor_parser -r``
        reload through :meth:`_run_unconfined`.

        Args:
            parent_session_id: Parent session id.  Used in the sub-
                profile's name + filename.  Trusted (originates from
                the handler-bound id, not runner-supplied for this
                method).
            subagent_id: Subagent id.  Sanitized strictly per
                :meth:`_validate_subagent_id` — invalid ids return
                ``(False, error_message)`` without touching disk.
            workspace_path: Workspace path (inherited from parent
                per §4.3 invariant).
            tightenings: Optional dict of validated supervisor-
                declared sub-profile tightenings (Phase 5 §5.9).
                Validated upstream by
                ``server/runner_rpc_handlers/sub_profile_tightenings_schema.py``;
                any value present here is trusted to have passed
                the schema check.  ``None`` / empty dict means
                "no tightenings — render the default sub-profile
                body" (byte-identical to the pre-§5.9 output).

        Returns:
            ``(True, sub_profile_name)`` on success — sub-profile
            loaded into kernel + file written.  The returned name is
            the kernel-visible identifier the sub-runner will
            ``change_profile`` to.

            ``(False, error_message)`` on validation / kernel / IO
            failure.  Error message is human-readable and suitable
            for forwarding back to the runner-side caller via the
            stage envelope.
        """
        err = self._validate_subagent_id(subagent_id)
        if err is not None:
            return False, f"subagent_id rejected: {err}"

        return self._run_unconfined(
            self._provision_sub_profile_impl,
            parent_session_id, subagent_id, workspace_path, tightenings,
        )

    def _provision_sub_profile_impl(
        self,
        parent_session_id: str,
        subagent_id: str,
        workspace_path: str,
        tightenings: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Inner body of :meth:`provision_sub_profile`.

        Writes the rendered sub-profile body to
        ``{profile_dir}/jaato-ws-{parent}__sub_{subagent}`` and loads
        it with ``apparmor_parser -r``.  Cleans up the file on parser
        failure (mirrors ``_provision_profile_impl``).
        """
        if not self.is_available():
            return False, "AppArmor unavailable on this host"

        sub_profile_name = self.get_sub_profile_name(
            parent_session_id, subagent_id,
        )
        sub_profile_filename = self.get_sub_profile_filename(
            parent_session_id, subagent_id,
        )
        sub_profile_path = self._profile_dir / sub_profile_filename
        sub_profile_content = self._render_sub_profile(
            parent_session_id=parent_session_id,
            subagent_id=subagent_id,
            workspace_path=workspace_path,
            tightenings=tightenings,
        )

        try:
            sub_profile_path.write_text(sub_profile_content)
        except OSError as exc:
            logger.exception(
                "Failed to write sub-profile %s", sub_profile_path,
            )
            return False, f"sub-profile file write failed: {exc}"

        try:
            result = subprocess.run(
                ["sudo", "apparmor_parser", "-r",
                 "--cache-loc", str(self._cache_dir),
                 str(sub_profile_path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                logger.error(
                    "apparmor_parser failed for sub-profile %s: %s",
                    sub_profile_name, result.stderr.strip(),
                )
                sub_profile_path.unlink(missing_ok=True)
                return False, (
                    f"apparmor_parser exit={result.returncode}: "
                    f"{result.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            logger.error(
                "apparmor_parser timed out for sub-profile %s",
                sub_profile_name,
            )
            sub_profile_path.unlink(missing_ok=True)
            return False, "apparmor_parser timed out (30s)"
        except OSError as exc:
            logger.exception(
                "Failed to run apparmor_parser for sub-profile %s",
                sub_profile_name,
            )
            sub_profile_path.unlink(missing_ok=True)
            return False, f"apparmor_parser invocation failed: {exc}"

        logger.info("Loaded AppArmor sub-profile %s", sub_profile_name)
        return True, sub_profile_name

    def teardown_sub_profile(
        self,
        parent_session_id: str,
        subagent_id: str,
    ) -> bool:
        """Unload and remove an isolated-subagent sub-profile
        (Phase 4 §4.3.4).

        Symmetric to :meth:`teardown_profile`.  Worker-sweep is NOT
        invoked here — §4.3.6 wires the sub-runner lifecycle, and at
        that point the sub-runner's own executor will be torn down
        separately.  Today's daemon worker pool is not confined
        under the sub-profile (the sub-profile only confines the
        sub-runner subprocess), so the sweep is unnecessary.

        Args:
            parent_session_id: Parent session id.
            subagent_id: Subagent id.  Validated; invalid ids return
                False (we still try to unlink any leftover file with
                a sanitized name to be defensive about cleanup).

        Returns:
            True on success or "already gone"; False on failure
            (logged, not raised).
        """
        err = self._validate_subagent_id(subagent_id)
        if err is not None:
            logger.warning(
                "teardown_sub_profile: invalid subagent_id %r — %s",
                subagent_id, err,
            )
            return False
        return self._run_unconfined(
            self._teardown_sub_profile_impl,
            parent_session_id, subagent_id,
        )

    def _teardown_sub_profile_impl(
        self,
        parent_session_id: str,
        subagent_id: str,
    ) -> bool:
        """Inner body of :meth:`teardown_sub_profile`."""
        if not self.is_available():
            return False

        sub_profile_name = self.get_sub_profile_name(
            parent_session_id, subagent_id,
        )
        sub_profile_filename = self.get_sub_profile_filename(
            parent_session_id, subagent_id,
        )
        sub_profile_path = self._profile_dir / sub_profile_filename

        if not sub_profile_path.exists():
            return True  # Already gone

        try:
            subprocess.run(
                ["sudo", "apparmor_parser", "-R",
                 "--cache-loc", str(self._cache_dir),
                 str(sub_profile_path)],
                capture_output=True, text=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError):
            logger.exception(
                "Failed to unload sub-profile %s", sub_profile_name,
            )
            # Continue to try deleting the file.

        try:
            sub_profile_path.unlink(missing_ok=True)
        except OSError:
            logger.exception(
                "Failed to delete sub-profile file %s", sub_profile_path,
            )
            return False
        return True

    def _render_sub_profile(
        self,
        parent_session_id: str,
        subagent_id: str,
        workspace_path: str,
        tightenings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render the sub-profile body for an isolated subagent.

        Conservative-tightening rule set per Audit 6:

        - Workspace allows mirror parent base (§4.3 invariant).
        - Integrity-deny rules mirror parent base.
        - Tool-hat-style read-denies mirror the parent's tool_hat
          (information-isolation between agents).
        - DROP fragment-admit (no add_reference_fragment from the
          sub-runner; Phase 5+ behind explicit opt-in).
        - DROP tool_hat sub-sub-profile (flat profile; isolated
          subagent doesn't need finer-grained tool scoping yet).

        The profile NAME is ``jaato-ws-{parent}//{subagent}``
        (Audit 6's standalone-with-prefix-name decision).

        Phase 5 §5.9: ``tightenings`` is an optional dict of
        supervisor-declared additive tightenings (validated
        upstream by
        ``server/runner_rpc_handlers/sub_profile_tightenings_schema.py``).
        Supported keys:

        - ``isolated_workspace_subpath: str`` — narrows the
          workspace allow rule from ``{workspace}/`` to
          ``{workspace}/{subpath}/``.  Subagent cannot reach
          siblings or parent directories of the subpath.
        - ``isolated_read_only_workspace: bool`` — when True,
          downgrades workspace ``rwkl`` to ``r``.  Subagent
          cannot write/lock/link anywhere under the workspace
          tree.

        Tightenings compose: subpath + read-only yields read-only
        access scoped to the subpath.  Integrity-deny rules and
        DROP comments are NOT affected by tightenings (they're
        baseline invariants that tightenings can only strengthen,
        never erode).
        """
        sub_profile_name = self.get_sub_profile_name(
            parent_session_id, subagent_id,
        )

        if self._premium_root:
            premium_rules = (
                f"{self._premium_root}/         r,\n"
                f"  {self._premium_root}/**       r,"
            )
        else:
            premium_rules = "# (no premium package installed)"

        # ── Phase 5 §5.9: supervisor-declared tightenings ────────
        # Both keys default to "no tightening" so untightened sub-
        # profiles render byte-identical to the pre-§5.9 body
        # (snapshot-test invariant).
        tightenings = tightenings or {}
        subpath = tightenings.get("isolated_workspace_subpath")
        read_only = tightenings.get("isolated_read_only_workspace", False)

        if subpath:
            ws_root = f"{workspace_path}/{subpath}"
        else:
            ws_root = workspace_path

        if read_only:
            # Read-only sub-profile: drop write, lock, link,
            # delete capabilities on the workspace tree.  Dir
            # perm drops the trailing ``w`` so even creating new
            # files under the workspace is denied.
            ws_dir_perm = "r,"
            ws_file_perm = "r,"
        else:
            ws_dir_perm = "rw,"
            ws_file_perm = "rwkl,"

        return f"""# AppArmor profile for isolated subagent
# Generated by jaato Phase 4 §4.3.4 (sub-AppArmor profile).
# Parent session: {parent_session_id}
# Subagent id:    {subagent_id}
#
# Standalone profile with a //-prefixed name per Audit 6.
# Loaded as a separate kernel-level profile; the sub-runner
# self-confines to this profile via change_profile.

profile "{sub_profile_name}" flags=(attach_disconnected) {{
  #include <abstractions/base>
  #include <abstractions/nameservice>
  #include <abstractions/python>

  # ---- workspace (inherited from parent per §4.3 invariant) ----
  {ws_root}/   {ws_dir_perm}
  {ws_root}/** {ws_file_perm}

  # ---- integrity-protected write-denies (mirror parent base) ----
  audit deny {workspace_path}/.jaato/agents/**             wlk,
  audit deny {workspace_path}/.jaato/profiles/**           wlk,
  # .jaato/prompts/ intentionally OMITTED from the write-denies: prompt_library's
  # savePrompt/deletePrompt run in the confined runner and need write/unlink
  # (unlink is granted by 'w' — classic AppArmor has no standalone 'd' mode).
  # Accepted posture tradeoff: a confined agent may author/rewrite prompts (whose
  # content is later executed). agents/profiles/scripts/schemas stay protected.
  audit deny {workspace_path}/.jaato/scripts/**            wlk,
  audit deny {workspace_path}/.jaato/services/*/           wlk,
  audit deny {workspace_path}/.jaato/reactors.json         wlk,
  audit deny {workspace_path}/.jaato/completion_schemas/** wlk,
  audit deny {workspace_path}/.jaato/spawn_schemas/**      wlk,
  audit deny {workspace_path}/.jaato/instructions/**       wlk,
  audit deny {workspace_path}/.jaato/references/**         wlk,
  # Workspace-tier AppArmor fragments — privilege-escalation guard
  # (mirrors base; isolated sub-runner could otherwise plant rules
  # for the next session's profile).
  audit deny {workspace_path}/.jaato/apparmor-fragments/** wlk,

  # ---- read-denies on user-authored config ----
  # Information-isolation between agents in a cascade — same as
  # the parent's tool_hat, applied at the sub-profile level here.
  audit deny {workspace_path}/.jaato/agents/**             r,
  audit deny {workspace_path}/.jaato/profiles/**           r,
  audit deny {workspace_path}/.jaato/prompts/**            r,
  audit deny {workspace_path}/.jaato/scripts/**            r,
  audit deny {workspace_path}/.jaato/completion_schemas/** r,
  audit deny {workspace_path}/.jaato/spawn_schemas/**      r,
  audit deny {workspace_path}/.jaato/instructions/**       r,
  audit deny {workspace_path}/.jaato/reactors.json         r,

  # ---- shared read-only resources ----
  {self._venv_path}/           r,
  {self._venv_path}/**         r,
  # v16: mmap-exec on venv C-extension shared objects (mirrors base).
  {self._venv_path}/**/*.so    mr,
  {self._venv_path}/**/*.so.*  mr,
  {self._venv_path}/bin/*      ix,
  {self._source_root}/         r,
  {self._source_root}/**       r,
  {premium_rules}

  # ---- user-global jaato config (read-only) ----
  # ~/.jaato/agents/ migrated to subagent + prompt_library plugins (template v26).
  # ~/.jaato/profiles/ migrated to subagent plugin (template v26).
  # ~/.jaato/prompts/ migrated to prompt_library plugin (template v23; this
  # isolated-subprofile site closed in template v26 — Phase 2/3 missed it).
  # ~/.jaato/skills/ migrated to prompt_library plugin (template v23; this
  # isolated-subprofile site closed in template v26 — Phase 2/3 missed it).
  @{{HOME}}/.jaato/keybindings.json r,
  @{{HOME}}/.jaato/theme.json       r,
  # ~/.jaato/gc.json migrated to gc plugin subsystem (template v25).
  # ~/.jaato/memories/ migrated to memory plugin (template v23; this
  # isolated-subprofile site closed in template v26 — Phase 2/3 missed it).

  # ---- temp files ----
  /tmp/                     r,
  /tmp/jaato-*              rw,
  /tmp/jaato-*/**           rwkl,

  # ---- self-introspection ----
  /proc/*/                  r,
  /proc/*/status            r,
  /proc/*/stat              r,
  /proc/*/cmdline           r,
  /proc/*/comm              r,
  /proc/*/fd/               r,
  /proc/*/fd/*              r,
  # Read-only access to attr/current — required by the sub-runner's
  # confine_to_profile.read_current_profile verify-after-write step
  # (server/runner/bootstrap.py:188).  ``owner`` qualifier matches the
  # kernel's resolved /proc/<pid>/ path AND verifies the process is
  # the file's owner (always true for the process's own attr/current).
  # No write permission: by design the sub-runner stays in this
  # profile for its lifetime (no further self-transitions per the
  # DROP block below), so the write capability the parent + tool_hat
  # profiles keep is intentionally absent.  Phase 5 template v15.
  owner /proc/*/attr/current      r,
  owner /proc/*/task/*/attr/current r,

  # ---- DROP: external-reference admit (no add_reference_fragment
  # from sub-runner — Phase 5+ behind opt-in).
  # ---- DROP: tool_hat sub-sub-profile (flat profile).
  # ---- DROP: change_profile transitions (sub-runner stays in this
  # profile for its lifetime; no further self-transitions).

  # ---- signal denies (mirror parent base) ----
  signal (receive) peer=unconfined,
  signal (send,receive) peer="{sub_profile_name}",
}}
"""

    def teardown_profile(self, session_id: str) -> bool:
        """Unload and remove an AppArmor profile.

        Dispatches to ``_teardown_profile_impl`` via ``_run_unconfined``
        so unloading + file deletion run on the unconfined main loop
        when called from a confined session-end path.
        """
        return self._run_unconfined(self._teardown_profile_impl, session_id)

    def _teardown_profile_impl(self, session_id: str) -> bool:
        """Inner body of :meth:`teardown_profile`.

        Runs ``apparmor_parser -R`` to unload the profile, then deletes
        the profile file.

        Server 0.6.47+: BEFORE running ``apparmor_parser -R``, sweeps
        every registered ``ApparmorSafeThreadPoolExecutor`` to dispatch
        a defensive ``changeprofile unconfined`` to every worker thread.
        This ensures no worker is left flagged-as-confined to a profile
        that's about to disappear (which would surface as a phantom-
        profile evaluation in the kernel — opaque EACCES on the
        worker's next operation).

        Args:
            session_id: Session whose profile should be removed.

        Returns:
            True on success, False on failure (logged, not raised).
        """
        if not self.is_available():
            return False

        profile_name = self.get_profile_name(session_id)
        profile_path = self._profile_dir / profile_name

        if not profile_path.exists():
            return True  # Already gone

        # Pre-removal sweep: reset every thread-pool worker so no
        # worker is left holding the profile-name flag once
        # apparmor_parser -R removes the profile.  Belt-and-braces
        # for any worker whose tool-exit restoration failed.
        try:
            from shared.safe_pool import sweep_registered_executors
            n_reset = sweep_registered_executors()
            if n_reset > 0:
                logger.debug(
                    "AppArmor: pre-teardown sweep dispatched %d worker resets "
                    "before removing profile %s",
                    n_reset, profile_name,
                )
        except Exception as exc:
            # Sweep is defense-in-depth; failure here must not block
            # the actual teardown.
            logger.warning(
                "AppArmor: pre-teardown sweep failed for %s: %s — "
                "continuing with apparmor_parser -R",
                profile_name, exc,
            )

        try:
            subprocess.run(
                ["sudo", "apparmor_parser", "-R", "--cache-loc", str(self._cache_dir), str(profile_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError):
            logger.exception("Failed to unload AppArmor profile %s", profile_name)
            # Continue to try deleting the file

        try:
            profile_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to delete AppArmor profile file %s", profile_path)
            return False

        # Drop any per-session reference fragments and the lock state.
        # Both are session-scoped — keeping them around after the
        # profile is gone would leak across session_id reuse.
        self._remove_all_reference_fragments(session_id)
        with self._session_locks_guard:
            self._session_locks.pop(session_id, None)

        logger.info("Removed AppArmor profile %s", profile_name)
        return True

    # ------------------------------------------------------------------
    # Reference fragments (per-selectReferences readonly grants)
    # ------------------------------------------------------------------
    #
    # Each ``selectReferences`` call writes one bare-rule file into
    # ``{profile_dir}/jaato-ws-{session_id}.refs.d/`` and reloads the
    # base profile via ``sudo apparmor_parser -r``.  The base profile's
    # ``include if exists`` directive (added in template v5) splices the
    # fragment in.  Unselect deletes the fragment and reloads.  Session
    # teardown removes the directory.
    #
    # Layering note: AppArmor enforcement is the kernel side of the
    # authorization story.  The application layer (sandbox_manager
    # ``add_path_programmatic``) is *also* required so that in-process
    # readers (``readFile``, glob, etc.) check the path against the
    # session's allow list before even attempting an open.  Both layers
    # are managed independently — the references plugin calls each in
    # turn.

    # AppArmor rule globs interpret these characters specially.  We
    # reject any reference path containing them at fragment-write time
    # so the fragment we emit means exactly what the path says.
    _APPARMOR_GLOB_CHARS = frozenset("[]{}*?\\")

    def _validate_path_for_fragment(self, path: str) -> Optional[str]:
        """Return an error message if *path* cannot be encoded as an
        AppArmor rule, otherwise ``None``.

        Rejected: relative paths (AppArmor needs absolute), paths
        containing AppArmor glob metacharacters, paths containing
        newlines (would break the fragment file format).
        """
        if not path:
            return "path is empty"
        if not path.startswith("/"):
            return f"path must be absolute: {path!r}"
        if "\n" in path or "\r" in path:
            return f"path contains newline/CR: {path!r}"
        bad = sorted(set(path) & self._APPARMOR_GLOB_CHARS)
        if bad:
            return (
                f"path contains AppArmor glob metacharacter(s) "
                f"{bad!r}: {path!r}"
            )
        return None

    @staticmethod
    def _safe_fragment_filename(ref_id: str) -> str:
        """Sanitize ``ref_id`` for use as a fragment filename.

        AppArmor's ``include`` glob picks up everything in the dir
        regardless of name, so the only constraint is filesystem
        validity and avoiding collisions.  We collapse anything outside
        ``[A-Za-z0-9._-]`` to ``_`` and refuse the empty string.
        """
        cleaned = re.sub(r'[^A-Za-z0-9._-]', '_', ref_id)
        return cleaned or "ref"

    def _fragment_content(self, path: str) -> str:
        """Build the AppArmor rule body for granting readonly access to *path*.

        Files get a single ``r,`` rule.  Directories get two — one for
        the directory listing itself and one (``**``) for all
        descendants — matching the workspace allow rule pattern at the
        top of the base template.  The path is wrapped in double quotes
        so spaces and other shell-special characters in the path don't
        confuse the parser.
        """
        p = Path(path)
        is_dir = False
        try:
            is_dir = p.is_dir()
        except OSError:
            # Path may not exist yet (selection happened first); treat
            # as a file by default — single ``r,`` rule is harmless if
            # later turned into a dir, but we'll re-fragment then.
            is_dir = False

        if is_dir:
            return (
                f"# auto-generated jaato reference fragment\n"
                f'  "{path}/"   r,\n'
                f'  "{path}/**" r,\n'
            )
        return (
            f"# auto-generated jaato reference fragment\n"
            f'  "{path}" r,\n'
        )

    def _session_lock(self, session_id: str) -> threading.Lock:
        """Return (lazily allocating) the per-session fragment-write lock.

        Reload races would otherwise collapse a parallel
        select+unselect into "one of the two changes wins".
        """
        with self._session_locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            return lock

    def _reload_profile(self, profile_path: Path) -> tuple[bool, str]:
        """Run ``apparmor_parser -r`` on *profile_path*.

        Returns ``(ok, stderr)`` so the caller can decide rollback
        and surface a clear error.  Atomic from the kernel's view —
        on failure the previously-loaded policy stays active.
        """
        try:
            result = subprocess.run(
                ["sudo", "apparmor_parser", "-r",
                 "--cache-loc", str(self._cache_dir),
                 str(profile_path)],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            return False, "apparmor_parser timed out"
        except OSError as exc:
            return False, f"apparmor_parser failed to run: {exc}"

        if result.returncode != 0:
            return False, result.stderr.strip() or "apparmor_parser non-zero exit"
        return True, ""

    def add_reference_fragment(
        self, session_id: str, ref_id: str, path: str,
    ) -> bool:
        """Grant readonly access to *path* for the session's profile.

        Dispatches to ``_add_reference_fragment_impl`` via
        ``_run_unconfined`` so the fragment file write and the
        ``apparmor_parser -r`` reload run on the unconfined main loop
        when called from a confined worker thread (the typical path —
        ``selectReferences`` runs from inside a tool call).
        """
        return self._run_unconfined(
            self._add_reference_fragment_impl, session_id, ref_id, path,
        )

    def _add_reference_fragment_impl(
        self, session_id: str, ref_id: str, path: str,
    ) -> bool:
        """Inner body of :meth:`add_reference_fragment`.

        Writes ``{refs_dir}/{safe_ref_id}`` and reloads the base
        profile.  Failure (validation, file write, parser reload) is
        logged and returned as ``False`` — the caller (references
        plugin) propagates this so the model sees the selection didn't
        actually grant access.

        Returns ``True`` on success or when AppArmor is unavailable
        (no kernel layer to mutate, so nothing to fail).
        """
        if not self.is_available():
            return True  # No kernel layer; succeed at this layer.

        err = self._validate_path_for_fragment(path)
        if err:
            logger.error(
                "AppArmor fragment rejected (session=%s ref=%s): %s",
                session_id, ref_id, err,
            )
            return False

        refs_dir = self._refs_dir(session_id)
        profile_path = self._profile_dir / self.get_profile_name(session_id)
        if not profile_path.exists():
            logger.error(
                "AppArmor base profile missing for session %s; cannot "
                "add reference fragment", session_id,
            )
            return False

        safe_id = self._safe_fragment_filename(ref_id)
        fragment_path = refs_dir / safe_id
        fragment_body = self._fragment_content(path)

        with self._session_lock(session_id):
            try:
                refs_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = fragment_path.with_suffix(".tmp")
                tmp_path.write_text(fragment_body)
                tmp_path.replace(fragment_path)  # atomic on same fs
            except OSError:
                logger.exception(
                    "Failed to write AppArmor fragment %s", fragment_path,
                )
                return False

            ok, err = self._reload_profile(profile_path)
            if not ok:
                logger.error(
                    "apparmor_parser reload failed adding fragment %s: %s",
                    fragment_path, err,
                )
                # Roll back the bad fragment so the next reload doesn't
                # keep failing on the same bad rule.
                fragment_path.unlink(missing_ok=True)
                return False

        logger.info(
            "AppArmor fragment added (session=%s ref=%s path=%s)",
            session_id, ref_id, path,
        )
        return True

    def remove_reference_fragment(
        self, session_id: str, ref_id: str,
    ) -> bool:
        """Revoke readonly access previously granted to *ref_id*.

        Dispatches to ``_remove_reference_fragment_impl`` via
        ``_run_unconfined`` so the fragment file delete and the
        ``apparmor_parser -r`` reload run on the unconfined main loop
        when called from a confined worker thread.
        """
        return self._run_unconfined(
            self._remove_reference_fragment_impl, session_id, ref_id,
        )

    def _remove_reference_fragment_impl(
        self, session_id: str, ref_id: str,
    ) -> bool:
        """Inner body of :meth:`remove_reference_fragment`.

        Idempotent: missing fragment is treated as success.  Always
        succeeds at the kernel layer — even if reload fails the
        fragment is gone, so the next successful reload (e.g. from a
        subsequent add) drops the rule.
        """
        if not self.is_available():
            return True

        safe_id = self._safe_fragment_filename(ref_id)
        refs_dir = self._refs_dir(session_id)
        profile_path = self._profile_dir / self.get_profile_name(session_id)
        fragment_path = refs_dir / safe_id

        with self._session_lock(session_id):
            if not fragment_path.exists():
                return True

            try:
                fragment_path.unlink()
            except OSError:
                logger.exception(
                    "Failed to remove AppArmor fragment %s", fragment_path,
                )
                return False

            if profile_path.exists():
                ok, err = self._reload_profile(profile_path)
                if not ok:
                    logger.warning(
                        "apparmor_parser reload failed removing fragment "
                        "%s: %s (fragment file gone, rule still in kernel "
                        "until next reload)", fragment_path, err,
                    )
                    # Still return True — the fragment file is gone, so
                    # the next add (or session teardown) reload will
                    # drop the rule.  Returning False would mislead the
                    # caller into thinking the unselection failed.

        logger.info(
            "AppArmor fragment removed (session=%s ref=%s)",
            session_id, ref_id,
        )
        return True

    def _remove_all_reference_fragments(self, session_id: str) -> None:
        """Delete the entire per-session refs dir.

        Called from ``teardown_profile`` after the base profile has
        been unloaded — at that point no further reload is needed
        because the include directive points at a profile that no
        longer exists in the kernel.
        """
        refs_dir = self._refs_dir(session_id)
        if not refs_dir.exists():
            return
        try:
            shutil.rmtree(refs_dir)
            logger.debug("Removed AppArmor refs dir: %s", refs_dir)
        except OSError:
            logger.exception(
                "Failed to remove AppArmor refs dir %s", refs_dir,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_profile_name(self, session_id: str) -> str:
        """Return the AppArmor profile name for a session.

        Fail-closed on an unsafe id: the name is interpolated into the profile
        grammar and the on-disk profile filename, so a traversal / injection id
        must never reach it.
        """
        validate_session_id(session_id)
        return f"jaato-ws-{session_id}"

    @staticmethod
    def _format_plugin_contributed_rules(
        plugin_rules: Optional[List[str]],
        indent: str = "  ",
    ) -> str:
        """Format plugin-contributed rules as a template-spliceable section.

        Returns a single string with each rule on its own line, indented to
        ``indent`` (defaults to base profile's 2-space indentation).  Empty
        / None input renders a "(none for this session)" comment so the
        section header is always visible in rendered profiles — operators
        can grep for ``plugin-contributed`` and find it in every profile,
        whether populated or not.

        Sub-profile builders (tool_hat, child) call this with ``indent="    "``
        to nest correctly inside their 4-space-indented block.
        """
        if not plugin_rules:
            return f"{indent}# ---- plugin-contributed rules (none for this session) ----"
        lines = [f"{indent}# ---- plugin-contributed rules ----"]
        for rule in plugin_rules:
            lines.append(f"{indent}{rule}")
        return "\n".join(lines)

    def _render_profile(
        self,
        session_id: str,
        workspace_path: str,
        config_root: Optional[str] = None,
        env_file: Optional[str] = None,
        requested_fragments: Optional[List[str]] = None,
        plugin_rules: Optional[List[str]] = None,
    ) -> str:
        """Render the profile template with session-specific values.

        The template uses Python ``str.format()`` placeholders.
        Sibling workspaces are implicitly denied by AppArmor's default-
        deny policy — only the session's own ``workspace_path`` is in
        the allow list.

        Piece 1 (2026-05-14): ``requested_fragments`` scopes which
        AppArmor fragments compose into the rendered profile.
        ``None`` (default) keeps the pre-Piece-1 behaviour — every
        fragment under the search path is composed.  A list (possibly
        empty) restricts composition to fragments whose basename
        matches an entry.
        """
        if self._premium_root:
            premium_rules = (
                f"{self._premium_root}/         r,\n"
                f"  {self._premium_root}/**       r,"
            )
        else:
            premium_rules = "# (no premium package installed)"

        if config_root:
            cr = str(Path(config_root).expanduser().resolve())
            config_root_rules = (
                f"{cr}/         r,\n"
                f"  {cr}/**       r,"
            )
        else:
            config_root_rules = "# (no client-supplied config_root override)"

        if env_file:
            ef = str(Path(env_file).expanduser().resolve())
            env_file_rule = f"{ef} r,"
        else:
            env_file_rule = "# (no client-supplied env_file)"

        # Inline extension-supplied fragments at render time.  Walks
        # three tiers of search paths and concatenates the contents
        # below the header in PROFILE_TEMPLATE.  Each fragment's
        # content is wrapped with a ``# === <tier>/<filename> ===``
        # comment so kernel denials can be traced back to the
        # originating extension when debugging
        # ``dmesg | grep apparmor`` output.
        #
        # **Search tiers (in walk order):**
        #
        # 1. ``user`` — ``~/.jaato/apparmor-fragments/`` — applies
        #    to every workspace under this user account.
        # 2. ``workspace`` — ``<workspace>/.jaato/apparmor-fragments/``
        #    — hand-authored, lives with the repo.  Only applies to
        #    sessions on that workspace.
        # 3. ``cache`` (Piece 1, 2026-05-14) —
        #    ``<workspace>/.jaato/.cache/apparmor-fragments/`` —
        #    walker-generated.  Sister workspaces (e.g.
        #    kb-enablement-2.0) write stack-specific rule files here
        #    at preflight time from a higher-level config (e.g.
        #    ``.jaato/stacks/<active>/host_tools.yaml``).  Cache
        #    wins on basename collision so walker output can shadow
        #    hand-authored placeholders without renaming.  This keeps
        #    the basename-namespace flat — the
        #    ``profile.apparmor_fragments: [name]`` field pairs
        #    naturally with one logical name per fragment.
        #
        # **Per-profile scoping (Piece 1, 2026-05-14):**
        # ``requested_fragments`` filters the composed set:
        # - ``None`` (default) — compose every fragment found in any
        #   of the three search paths (back-compat for non-cascade
        #   workspaces).
        # - ``[]`` — compose nothing.  Maximally locked-down stage.
        # - non-empty list — compose only fragments whose basename
        #   (without ``.rules``) matches an entry in the list.
        #   Unknown names log WARNING but don't abort.
        # Order does not affect AppArmor semantics (allow rules
        # union, deny rules union, deny wins at equal specificity).
        SEARCH_TIERS = [
            ("user", Path("~/.jaato/apparmor-fragments").expanduser().resolve()),
            ("workspace", (Path(workspace_path) / ".jaato" / "apparmor-fragments").resolve()),
            ("cache", (Path(workspace_path) / ".jaato" / ".cache" / "apparmor-fragments").resolve()),
        ]

        # Phase 1: discover every available fragment, keyed by
        # basename (without ``.rules``).  Later tiers overwrite
        # earlier ones — cache wins over workspace wins over user.
        # The tier label is retained for the inlined comment.
        discovered: Dict[str, Tuple[str, Path]] = {}
        for tier_label, fragments_dir in SEARCH_TIERS:
            if not fragments_dir.is_dir():
                continue
            for path in sorted(fragments_dir.glob("*.rules")):
                discovered[path.stem] = (tier_label, path)

        # Phase 2: decide which fragments to include.  ``None`` =
        # all discovered, in tier-iteration insertion order
        # (user → workspace → cache, alphabetical within tier).
        # Otherwise filter by basename in the operator's declaration
        # order.
        if requested_fragments is None:
            selected_names = list(discovered.keys())
        else:
            selected_names = []
            requested_set = set(requested_fragments)
            available_set = set(discovered.keys())
            unknown = requested_set - available_set
            if unknown:
                logger.warning(
                    "AppArmor profile %s requests fragments %s but the "
                    "following are not in the search path "
                    "(%s/.jaato/apparmor-fragments + .cache; "
                    "~/.jaato/apparmor-fragments): %s.  Continuing "
                    "without them; check the operator authored the "
                    "right filenames.",
                    session_id, sorted(requested_set),
                    workspace_path, sorted(unknown),
                )
            # Preserve the operator's declaration order; AppArmor
            # semantics are order-insensitive but matching the
            # declaration order keeps rendered profiles readable.
            for name in requested_fragments:
                if name in available_set:
                    selected_names.append(name)
        logger.info(
            "AppArmor profile %s composing %d fragments: %s",
            session_id, len(selected_names), selected_names,
        )

        # Phase 3: inline the selected fragments' bodies.
        chunks: List[str] = []
        for name in selected_names:
            tier_label, path = discovered[name]
            try:
                body = path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "Skipping unreadable AppArmor fragment %s: %s",
                    path, exc,
                )
                continue
            # Indent each line by two spaces to match the
            # surrounding profile body's indentation.  Strip
            # trailing whitespace per-line, preserve blanks.
            indented = "\n".join(
                f"  {line.rstrip()}" if line.strip() else ""
                for line in body.splitlines()
            )
            chunks.append(
                f"  # === {tier_label}/{path.name} ===\n{indented}"
            )
        extension_fragments_inline = (
            "\n\n".join(chunks) if chunks else "  # (no extension fragments)"
        )

        # Phase 0 (template v20, 2026-05-16): plugin-contribution hook.
        # Splice plugin-contributed rules into base + sub-profiles.  When
        # ``plugin_rules`` is None/empty (no plugin overrides), both
        # variants render a "(none for this session)" comment line —
        # rule semantics unchanged from v19.
        plugin_rules_base_inline = self._format_plugin_contributed_rules(
            plugin_rules, indent="  ",
        )
        plugin_rules_subprofile_inline = self._format_plugin_contributed_rules(
            plugin_rules, indent="    ",
        )

        # Build the tool_hat sub-profile body (server 0.6.55+).
        # AppArmor sub-profiles do NOT inherit base rules — every allow
        # and deny must be redeclared.  We mirror the base body
        # verbatim and append the hat-specific read-denies on
        complain = os.environ.get("JAATO_APPARMOR_COMPLAIN", "").lower() in ("1", "true", "yes")  # env: generate AppArmor profiles in complain (log-only) mode; confinement debugging aid
        profile_flags = "attach_disconnected, complain" if complain else "attach_disconnected"
        # v22 (2026-05-16): propagate complain to sub-profiles.  AppArmor
        # sub-profiles do NOT inherit the parent's flag set, so
        # JAATO_APPARMOR_COMPLAIN=1 has to be emitted on each sub-profile
        # header explicitly.  When the env is unset, the clause is empty
        # — preserving the v21 ``profile tool_hat {`` shape byte-for-byte
        # (no ``attach_disconnected`` on sub-profiles per AppArmor doc:
        # the flag is base-profile-only).
        subprofile_flag_clause = " flags=(complain)" if complain else ""

        # user-authored config.  Indentation is 4 spaces (vs 2 for
        # base body) so the rendered profile is visually nested.
        tool_hat_subprofile = self._build_tool_hat_subprofile(
            workspace_path=workspace_path,
            premium_rules=premium_rules,
            config_root_rules=config_root_rules,
            env_file_rule=env_file_rule,
            refs_include_glob=f"{self._refs_dir(session_id)}/*",
            extension_fragments_inline=extension_fragments_inline,
            plugin_contributed_rules=plugin_rules_subprofile_inline,
            session_id=session_id,
            subprofile_flag_clause=subprofile_flag_clause,
        )

        # Phase 5 §5.10: build the //child sub-profile body.  Mirrors
        # tool_hat's rule set but drops the three escape-vector lines
        # (change_profile -> unconfined + /proc/self/attr/current w +
        # task variant).  Subprocess-spawning plugins transition
        # into //child via preexec_fn so model-controlled subprocesses
        # cannot escape via the kernel-level transition primitive.
        child_subprofile = self._build_child_subprofile(
            workspace_path=workspace_path,
            premium_rules=premium_rules,
            config_root_rules=config_root_rules,
            env_file_rule=env_file_rule,
            refs_include_glob=f"{self._refs_dir(session_id)}/*",
            extension_fragments_inline=extension_fragments_inline,
            plugin_contributed_rules=plugin_rules_subprofile_inline,
            session_id=session_id,
            subprofile_flag_clause=subprofile_flag_clause,
        )

        return self.PROFILE_TEMPLATE.format(
            template_version=self._TEMPLATE_VERSION,
            session_id=session_id,
            workspace_path=workspace_path,
            venv_path=str(self._venv_path),
            source_root=str(self._source_root),
            premium_rules=premium_rules,
            config_root_rules=config_root_rules,
            env_file_rule=env_file_rule,
            refs_include_glob=f"{self._refs_dir(session_id)}/*",
            extension_fragments_inline=extension_fragments_inline,
            plugin_contributed_rules=plugin_rules_base_inline,
            tool_hat_subprofile=tool_hat_subprofile,
            child_subprofile=child_subprofile,
            profile_flags=profile_flags,
            user_global_block=self._render_user_global_block(2),
        )

    def _build_tool_hat_subprofile(
        self,
        workspace_path: str,
        premium_rules: str,
        config_root_rules: str,
        env_file_rule: str,
        refs_include_glob: str,
        extension_fragments_inline: str,
        plugin_contributed_rules: str,
        session_id: str,
        subprofile_flag_clause: str = "",
    ) -> str:
        """Build the ``profile tool_hat { ... }`` sub-profile body
        (server 0.6.55+).

        The sub-profile mirrors the base profile's rule set verbatim
        (sub-profiles don't inherit) and ADDS read-denies on user-
        authored config subpaths.  ``ToolExecutor.execute`` enters the
        sub-profile via ``change_profile -> jaato-ws-{session_id}//tool_hat``
        before invoking each plugin.
        """
        # Re-indent extension_fragments_inline from 2-space to 4-space
        # so it nests correctly inside the sub-profile block.
        nested_extension_fragments = "\n".join(
            f"  {line}" if line.strip() else line
            for line in extension_fragments_inline.splitlines()
        )
        return f"""  profile tool_hat{subprofile_flag_clause} {{
    #include <abstractions/base>
    #include <abstractions/nameservice>
    #include <abstractions/python>

    # ---- workspace + integrity write-denies (mirrors base) ----
    {workspace_path}/   rw,
    {workspace_path}/** rwkl,
    audit deny {workspace_path}/.jaato/agents/**             wlk,
    audit deny {workspace_path}/.jaato/profiles/**           wlk,
    # .jaato/prompts/ intentionally OMITTED from the write-denies (see base): the
    # confined runner needs write/unlink for savePrompt/deletePrompt. Accepted
    # posture tradeoff. agents/profiles/scripts/schemas stay protected.
    audit deny {workspace_path}/.jaato/scripts/**            wlk,
    audit deny {workspace_path}/.jaato/services/*/           wlk,
    audit deny {workspace_path}/.jaato/reactors.json         wlk,
    audit deny {workspace_path}/.jaato/completion_schemas/** wlk,
    audit deny {workspace_path}/.jaato/spawn_schemas/**      wlk,
    audit deny {workspace_path}/.jaato/instructions/**       wlk,
    audit deny {workspace_path}/.jaato/references/**         wlk,
    # Workspace-tier AppArmor fragments — privilege-escalation guard
    # (mirrors base; a tool execution under tool_hat could otherwise
    # plant rules for the next session's profile).
    audit deny {workspace_path}/.jaato/apparmor-fragments/** wlk,

    # ---- tool_hat-specific read-denies on user-authored config ----
    # The whole point of the sub-profile: tool execution can't read
    # other agents' personas, profile JSON, prompts, schemas,
    # instructions, scripts, or reactors.json.  Information-isolation
    # between agents in a cascade.  Framework loading (which needs
    # these reads) happens in the BASE profile, not in the hat.
    audit deny {workspace_path}/.jaato/agents/**             r,
    audit deny {workspace_path}/.jaato/profiles/**           r,
    audit deny {workspace_path}/.jaato/prompts/**            r,
    audit deny {workspace_path}/.jaato/scripts/**            r,
    audit deny {workspace_path}/.jaato/completion_schemas/** r,
    audit deny {workspace_path}/.jaato/spawn_schemas/**      r,
    audit deny {workspace_path}/.jaato/instructions/**       r,
    audit deny {workspace_path}/.jaato/reactors.json         r,

    # ---- shared read-only resources (mirrors base) ----
    {self._venv_path}/           r,
    {self._venv_path}/**         r,
    # v16: mmap-exec on venv C-extension shared objects (mirrors base).
    {self._venv_path}/**/*.so    mr,
    {self._venv_path}/**/*.so.*  mr,
    {self._venv_path}/bin/*      ix,
    {self._source_root}/         r,
    {self._source_root}/**       r,
    {premium_rules}
    {config_root_rules}

{self._render_user_global_block(4)}
    {env_file_rule}

    # ---- temp files (mirrors base) ----
    /tmp/jaato-{session_id}-** rw,
    /tmp/jaato-{session_id}/   rw,
    /tmp/jaato-{session_id}/** rw,

    # ---- basic system access (mirrors base) ----
    /usr/bin/**          ix,
    /usr/local/bin/**    ix,
    /bin/**              ix,
    /usr/lib/**          rm,
    /lib/**              rm,
    /etc/ld.so.cache     r,
    /etc/passwd          r,
    /etc/nsswitch.conf   r,
    /proc/self/**        r,
    /dev/null            rw,
    /dev/urandom         r,
    /dev/pts/*           rw,

    # ---- network (mirrors base) ----
    network inet  stream,
    network inet6 stream,
    network inet  dgram,
    network inet6 dgram,
    deny network raw,

    # ---- denied capabilities (mirrors base) ----
    deny ptrace,
    deny mount,
    deny capability sys_admin,
    deny capability net_admin,
    deny capability sys_ptrace,

    # ---- profile transitions (mirrors base, needed to exit hat) ----
    # ``owner /proc/*/...`` form matches the kernel's resolved path
    # for both read and write (see v15 note in the base profile body).
    change_profile -> unconfined,
    owner /proc/*/attr/current      rw,
    owner /proc/*/task/*/attr/current rw,

    # ---- per-session reference fragments (mirrors base) ----
    include if exists "{refs_include_glob}"

    # ---- extension fragments (mirrors base, re-indented) ----
{nested_extension_fragments}

{plugin_contributed_rules}
  }}"""

    def _build_child_subprofile(
        self,
        workspace_path: str,
        premium_rules: str,
        config_root_rules: str,
        env_file_rule: str,
        refs_include_glob: str,
        extension_fragments_inline: str,
        plugin_contributed_rules: str,
        session_id: str,
        subprofile_flag_clause: str = "",
    ) -> str:
        """Build the ``profile child { ... }`` sub-profile body
        (Phase 5 §5.10, template v14).

        Subprocesses spawned by the cli / interactive_shell plugins
        transition into this sub-profile via
        ``change_profile -> jaato-ws-{session_id}//child`` between
        fork() and exec() (preexec_fn writes ``changeprofile`` to
        ``/proc/self/attr/current``).

        The body mirrors :meth:`_build_tool_hat_subprofile`'s rule
        set verbatim and **drops** the three escape-vector lines that
        the base + tool_hat profiles must keep for the framework's
        ``apparmor_confine.__exit__`` path:

            change_profile -> unconfined,
            /proc/self/attr/current      w,
            /proc/self/task/*/attr/current w,

        A subprocess in ``//child`` cannot write ``changeprofile`` to
        ``attr/current`` (kernel rejects with EACCES) and cannot
        transition anywhere — the dangerous primitive is moved from
        "any subprocess can use it" to "only the framework's
        parent-tier code can use it".

        See
        ``docs/design/phase5_5_10_apparmor_child_subprofile_audit.md``.
        """
        # Re-indent extension_fragments_inline from 2-space to 4-space
        # so it nests correctly inside the sub-profile block.  Same
        # transform applied by _build_tool_hat_subprofile.
        nested_extension_fragments = "\n".join(
            f"  {line}" if line.strip() else line
            for line in extension_fragments_inline.splitlines()
        )
        return f"""  profile child{subprofile_flag_clause} {{
    #include <abstractions/base>
    #include <abstractions/nameservice>
    #include <abstractions/python>

    # ---- workspace + integrity write-denies (mirrors tool_hat) ----
    {workspace_path}/   rw,
    {workspace_path}/** rwkl,
    audit deny {workspace_path}/.jaato/agents/**             wlk,
    audit deny {workspace_path}/.jaato/profiles/**           wlk,
    # .jaato/prompts/ intentionally OMITTED from the write-denies (see base): the
    # confined runner needs write/unlink for savePrompt/deletePrompt. Accepted
    # posture tradeoff. agents/profiles/scripts/schemas stay protected.
    audit deny {workspace_path}/.jaato/scripts/**            wlk,
    audit deny {workspace_path}/.jaato/services/*/           wlk,
    audit deny {workspace_path}/.jaato/reactors.json         wlk,
    audit deny {workspace_path}/.jaato/completion_schemas/** wlk,
    audit deny {workspace_path}/.jaato/spawn_schemas/**      wlk,
    audit deny {workspace_path}/.jaato/instructions/**       wlk,
    audit deny {workspace_path}/.jaato/references/**         wlk,
    # Workspace-tier AppArmor fragments — privilege-escalation guard
    # (mirrors base; a //child subprocess could otherwise plant
    # rules for the next session's profile).
    audit deny {workspace_path}/.jaato/apparmor-fragments/** wlk,

    # ---- tool_hat-style read-denies (mirrors tool_hat) ----
    # Same information-isolation as the in-process tool_hat: a
    # subprocess can't read other agents' personas, profile JSON,
    # prompts, schemas, instructions, scripts, or reactors.json.
    audit deny {workspace_path}/.jaato/agents/**             r,
    audit deny {workspace_path}/.jaato/profiles/**           r,
    audit deny {workspace_path}/.jaato/prompts/**            r,
    audit deny {workspace_path}/.jaato/scripts/**            r,
    audit deny {workspace_path}/.jaato/completion_schemas/** r,
    audit deny {workspace_path}/.jaato/spawn_schemas/**      r,
    audit deny {workspace_path}/.jaato/instructions/**       r,
    audit deny {workspace_path}/.jaato/reactors.json         r,

    # ---- shared read-only resources (mirrors tool_hat) ----
    {self._venv_path}/           r,
    {self._venv_path}/**         r,
    # v16: mmap-exec on venv C-extension shared objects (mirrors tool_hat).
    {self._venv_path}/**/*.so    mr,
    {self._venv_path}/**/*.so.*  mr,
    {self._venv_path}/bin/*      ix,
    {self._source_root}/         r,
    {self._source_root}/**       r,
    {premium_rules}
    {config_root_rules}

{self._render_user_global_block(4)}
    {env_file_rule}

    # ---- temp files (mirrors tool_hat) ----
    /tmp/jaato-{session_id}-** rw,
    /tmp/jaato-{session_id}/   rw,
    /tmp/jaato-{session_id}/** rw,

    # ---- basic system access (mirrors tool_hat, MINUS broad ix) ----
    # Template v18 (2026-05-15): the broad ``/usr/bin/** ix``,
    # ``/usr/local/bin/** ix``, ``/bin/** ix`` rules are
    # INTENTIONALLY OMITTED from //child.  Rationale:
    #
    # The //child sub-profile is the agent-controlled exec
    # context.  Per-profile ``apparmor_fragments`` (Piece 1, PR
    # #107) authorize stage-specific binaries
    # (e.g. host_validator's fragment declares
    # ``/usr/bin/java ix, /usr/bin/mvn ix``).  Pre-v18, the broad
    # ``/usr/bin/** ix`` rule shadowed those declarations — every
    # cascade stage could exec ANY ``/usr/bin/*`` binary
    # regardless of what its fragment listed.  Peer's v83 caught
    # this empirically: a host_validator agent improvised
    # ``curl`` when ``mvn dependency:get`` raised
    # ``MojoExecutionException``; ``curl`` ran because the broad
    # rule allowed it even though the fragment only listed
    # java/mvn/sh/coreutils.  The per-profile scoping promise
    # ("codegen/transform/build_descriptor cannot exec java
    # because their fragment does not list it") was empty.
    #
    # In v18, ``apparmor_fragments`` becomes the SOLE source of
    # exec authority in //child:
    # - A fragment-less stage (or
    #   ``apparmor_fragments: []``) gets NO exec authority — the
    #   maximally locked-down cascade stage.
    # - A stage with ``apparmor_fragments: [host_validator]``
    #   gets only the binaries the host_validator.rules fragment
    #   declares.
    # - Shell helpers (sh, cat, grep, ...) that probes need must
    #   appear in the fragment explicitly OR in a separately-
    #   included ``shell_essentials`` fragment if operators
    #   compose one later.
    #
    # The base profile + tool_hat sub-profile KEEP the broad
    # ``ix`` grants — framework code (Python stdlib subprocess
    # machinery, runner-tier plugin subprocesses, references
    # plugin's SentenceTransformer load) executes there, not in
    # //child.  Narrowing those would risk breaking framework
    # internals invisibly; the agent-controlled-exec threat
    # model belongs uniquely to //child.
    #
    # Library mapping (``rm``) is preserved unchanged — execs
    # need shared-library mmap rights to load the binary they're
    # about to run.  The deny is on ``ix`` (the exec capability),
    # not on the libraries.
    /usr/lib/**          rm,
    /lib/**              rm,
    /etc/ld.so.cache     r,
    /etc/passwd          r,
    /etc/nsswitch.conf   r,
    /proc/self/**        r,
    /dev/null            rw,
    /dev/urandom         r,
    /dev/pts/*           rw,

    # ---- network (mirrors tool_hat) ----
    network inet  stream,
    network inet6 stream,
    network inet  dgram,
    network inet6 dgram,
    deny network raw,

    # ---- denied capabilities (mirrors tool_hat) ----
    deny ptrace,
    deny mount,
    deny capability sys_admin,
    deny capability net_admin,
    deny capability sys_ptrace,

    # ---- DROP: profile transitions ----
    # Phase 5 §5.10: the three rules that base + tool_hat carry are
    # intentionally absent here:
    #
    #     change_profile -> unconfined,
    #     /proc/self/attr/current      w,
    #     /proc/self/task/*/attr/current w,
    #
    # A subprocess in //child cannot escape to unconfined, and the
    # kernel rejects writes to /proc/self/attr/current with EACCES.
    # No transition out of //child is possible for the subprocess's
    # lifetime.

    # ---- per-session reference fragments (mirrors tool_hat) ----
    include if exists "{refs_include_glob}"

    # ---- extension fragments (mirrors tool_hat, re-indented) ----
{nested_extension_fragments}

{plugin_contributed_rules}
  }}"""

    def _refs_dir(self, session_id: str) -> Path:
        """Per-session AppArmor reference-fragment directory.

        Lives next to the base profile file so AppArmor's ``include``
        glob picks up every fragment file written by
        ``add_reference_fragment()``.
        """
        return self._profile_dir / f"{self.get_profile_name(session_id)}.refs.d"

    @staticmethod
    def _render_user_global_block(indent_spaces: int = 2) -> str:
        """Render the user-global config block shared across the
        base profile and its two nested sub-profiles (tool_hat, child).

        Phase 5 (template v27, 2026-05-16): consolidates the residual
        user-global grants and migration-marker comments into a single
        helper so future plugin migrations edit ONE location instead of
        three.  Closes the smell behind why Phases 1-4 kept hitting
        copy-paste patterns across multiple renderers.

        **Scope: base + tool_hat + child only.**  The fourth renderer
        (``_render_sub_profile``, isolated subagent) historically
        carries a narrower active block (no ``themes/``) per its
        minimal-grants security posture, and is NOT covered by this
        helper.  See ``_render_sub_profile``'s own user-global block.

        Active grants (client-side, not plugin-attributable):
        - ``~/.jaato/themes/`` (and ``**``) — TUI theme definitions
        - ``~/.jaato/keybindings.json`` — TUI keybinding config
        - ``~/.jaato/theme.json`` — TUI theme override

        Migrated paths (now contributed by plugin classmethods via
        :func:`resolve_plugin_apparmor_rules`):
        - ``~/.jaato/agents/`` — subagent + prompt_library (v26)
        - ``~/.jaato/profiles/`` — subagent (v26)
        - ``~/.jaato/prompts/``, ``~/.jaato/skills/`` — prompt_library (v23)
        - ``~/.jaato/services/`` — service_connector (v24)
        - ``~/.jaato/references/`` — references plugin catalog (v24)
        - ``~/.jaato/memories/``, ``memories.jsonl`` — memory plugin (v23)
        - ``~/.jaato/gc.json`` — gc subsystem helper (v25)
        - ``~/.claude/skills/``, ``~/.claude/commands/`` —
          prompt_library (v23)
        - ``~/.cache/huggingface/``, ``~/.cache/torch/`` — references
          plugin (v21)

        Args:
            indent_spaces: Leading whitespace per line.  ``2`` for the
                base profile (uses 2-space indent inside the
                ``profile { ... }`` block).  ``4`` for
                ``_build_tool_hat_subprofile`` and
                ``_build_child_subprofile`` (which nest one level
                deeper inside ``profile foo { ... }``).

        Returns:
            The block as a single string with embedded newlines, ready
            to splice into a template via ``.format()`` or f-string.
            Does NOT include a leading or trailing newline.
        """
        indent = " " * indent_spaces
        lines = [
            "# ---- user-global jaato config (read-only) ----",
            "# All plugin-attributable paths migrated to plugins' "
            "get_apparmor_rules (Phases 1-4, templates v21-v26).",
            "# Remaining grants are client-side reads not owned by any plugin.",
            "@{HOME}/.jaato/themes/         r,",
            "@{HOME}/.jaato/themes/**       r,",
            "@{HOME}/.jaato/keybindings.json r,",
            "@{HOME}/.jaato/theme.json       r,",
        ]
        return "\n".join(indent + line for line in lines)


def resolve_plugin_apparmor_rules(
    server: Any,
    profile: Optional[Any],
    session_id: str,
    workspace_path: str,
    config_root: Optional[str],
) -> Optional[List[str]]:
    """Union the AppArmor rules contributed by every plugin the runner loads.

    Phase 0/1 of the plugin-apparmor-contribution refactor
    (template v20+, 2026-05-16).  See
    ``docs/design/plugin-apparmor-contribution.md``.

    Module-level helper so both the IPC path
    (``SessionManager._provision_ipc_apparmor_and_spawn_runner``) and
    the WS path (``websocket.py:_apparmor_pre_init_hook``) can call it
    without duplicating the resolution logic.

    Iterates **every discovered plugin** (``registry.all_plugins()``),
    not just ``profile.plugins``, and calls each plugin's optional
    ``get_apparmor_rules`` classmethod with session context.  The full-set
    walk is deliberate: the confined runner initializes ALL discovered
    runner-tier plugins regardless of ``profile.plugins`` (the load-gate was
    disabled in #114), so the grant set must match the *loaded* set, not the
    *declared* set — otherwise auto-loaded plugins init without their
    host-path rules and fail with ``[Errno 13]``.  The walk self-filters:
    only runner-tier filesystem plugins define ``get_apparmor_rules`` (see
    the body comment).  ``plugin_configs`` is still applied per plugin.

    Returns ``None`` when the server has no registry or no plugin
    contributes anything — the renderer treats ``None`` and ``[]``
    identically (renders "(none for this session)" marker).

    Failures inside an individual plugin's ``get_apparmor_rules`` are
    logged but do not abort — the framework baseline still renders.
    A confined runner missing one plugin's rules beats a session that
    fails to start.
    """
    if profile is None:
        return None
    rules: List[str] = []

    registry = getattr(server, "registry", None)
    plugin_configs = getattr(profile, "plugin_configs", None) or {}
    if registry is not None:
        # Grant for the plugins the RUNNER actually loads — NOT the declared
        # ``profile.plugins``.  ``expose_all`` (runner/session.py) initializes
        # ALL discovered runner-tier plugins regardless of ``profile.plugins``:
        # the PR-112 load-gate that would have scoped init to the declared set
        # was disabled in #114 (it broke cascades).  So an auto-loaded plugin
        # (memory / references / subagent — none of which must be declared)
        # initializes and touches ``~/.jaato/{memories,references,profiles,
        # agents}``; if the apparmor profile only grants ``profile.plugins``,
        # those inits hit ``[Errno 13] Permission denied`` and the plugin is
        # disabled for the whole session.  This regressed 2026-05-16 when the
        # grants migrated out of the baseline template keyed on
        # ``profile.plugins`` membership, while the loader had — since #114
        # (2026-05-15) — already been loading them unconditionally.  The two
        # subsystems disagreed on the plugin set; this realigns the grant set
        # to the load set.
        #
        # Iterating the full discovered set SELF-FILTERS to the runner-tier
        # filesystem plugins: only they define ``get_apparmor_rules``.
        # Daemon-tier plugins run daemon-side, never in the confined runner,
        # and contribute nothing (verified: every ``get_apparmor_rules``
        # plugin is ``PLUGIN_TIER="runner"``; ``test_plugin_tier_partition``
        # enforces the daemon/runner split).  ``plugin_configs`` is still
        # honoured per-plugin so a declared plugin's profile config reaches
        # its rule contributor.
        for plugin_name, plugin in registry.all_plugins().items():
            get_rules = getattr(plugin, "get_apparmor_rules", None)
            if get_rules is None or not callable(get_rules):
                continue
            try:
                contributed = get_rules(
                    workspace_path=workspace_path,
                    session_id=session_id,
                    config_root=config_root,
                    plugin_config=plugin_configs.get(plugin_name, {}),
                )
            except Exception:  # noqa: BLE001 — boundary surface
                logger.exception(
                    "plugin %s get_apparmor_rules failed for session %s — "
                    "skipping its contribution",
                    plugin_name, session_id,
                )
                continue
            if contributed:
                rules.extend(contributed)

    # Phase 3b (template v25, 2026-05-16) — gc rules.
    # gc plugins live in ``profile.gc`` (a single GCProfileConfig), not in
    # ``profile.plugins``, so the per-plugin walk above doesn't see them.
    # The gc.json rule is shared across all 4 gc strategies — surfaced as a
    # module-level helper rather than a per-plugin classmethod (no concrete
    # base class on the GCPlugin Protocol).
    if getattr(profile, "gc", None) is not None:
        try:
            from shared.plugins.gc import get_gc_apparmor_rules
            rules.extend(get_gc_apparmor_rules())
        except Exception:  # noqa: BLE001 — boundary surface
            logger.exception(
                "gc apparmor rules failed for session %s — skipping",
                session_id,
            )

    # Dedup identical rule lines while preserving first-seen order.  Distinct
    # plugins can contribute the same literal grant (e.g. cli + notebook +
    # interactive_shell all contribute the pip/distro OS-id reads); a rule line
    # is idempotent, so collapsing keeps rendered profiles readable without any
    # change in AppArmor semantics.
    if rules:
        seen = set()
        deduped = []
        for r in rules:
            if r not in seen:
                seen.add(r)
                deduped.append(r)
        rules = deduped

    return rules if rules else None


# ------------------------------------------------------------------
# Thread-level AppArmor confinement
# ------------------------------------------------------------------

def _get_thread_attr_path() -> str:
    """Return the path to the current thread's AppArmor attr file."""
    tid = threading.get_native_id()
    return f"/proc/self/task/{tid}/attr/current"


@contextmanager
def apparmor_confine(profile_name: str):
    """Context manager that confines the current thread to an AppArmor profile.

    Defensively writes ``unconfined`` first to the thread's ``attr/current``
    to recover from any prior session's stuck-confinement state, then
    writes *profile_name* to enter the requested profile.  On exit, writes
    ``unconfined`` to restore the thread.

    **Cross-session state poisoning (the bug this defensive reset closes).**
    ``run_in_executor`` reuses worker threads across asyncio tasks.  If a
    prior session's exit path failed to restore unconfined (kernel
    transient, profile-rule gap, etc.), the worker stays in the prior
    profile.  When a NEW session is dispatched to the same worker:
    1. Entry writes ``changeprofile <new>`` — the prior profile only has
       ``change_profile -> unconfined`` (not ``-> <new>``), so the kernel
       denies the write.
    2. The exception-catching code below sets ``confined=False`` and
       proceeds — but the thread is still stuck in the prior profile.
    3. The yielded work runs under the prior profile's rules, hitting
       EACCES on paths the new session's workspace expects.
    The defensive ``changeprofile unconfined`` on entry breaks the cycle:
    the prior profile grants the unconfine transition, restoration is
    deterministic, and the new confinement enters from a known-clean
    state every time.

    If the entry write fails after the defensive reset (AppArmor not
    available, profile not loaded, insufficient privileges), logs a
    warning and proceeds without confinement — never raises.

    Args:
        profile_name: The AppArmor profile to confine to
            (e.g. ``"jaato-ws-20260405_123456"``).
    """
    attr_path = _get_thread_attr_path()
    confined = False

    # Defensive reset.  Recovers from a prior session's failed
    # exit-restoration (see docstring).  Safe whether the thread is
    # currently unconfined (no-op write) or stuck in a prior session's
    # profile (rule grants change_profile -> unconfined).  Any failure
    # here is unactionable — a failed reset will surface as a failed
    # entry below, which IS handled.  Logged at debug for forensics.
    try:
        with open(attr_path, "w") as f:
            f.write("changeprofile unconfined")
    except (OSError, PermissionError) as e:
        logger.debug(
            "AppArmor: defensive pre-confine reset to unconfined failed "
            "(thread %d, target %s): %s — likely already unconfined or "
            "kernel state will surface in entry below",
            threading.get_native_id(), profile_name, e,
        )

    try:
        with open(attr_path, "w") as f:
            f.write(f"changeprofile {profile_name}")
        confined = True
    except (OSError, PermissionError) as e:
        logger.debug("AppArmor: thread confinement failed for %s: %s", profile_name, e)

    try:
        yield
    finally:
        if confined:
            try:
                with open(attr_path, "w") as f:
                    f.write("changeprofile unconfined")
            except (OSError, PermissionError) as e:
                # Cannot restore — thread stays confined until it exits.
                # The defensive reset on the next confine entry will
                # recover the state, but flag it loudly so the cause
                # (transient kernel issue, missing profile rule, etc.)
                # gets investigated rather than silently absorbed.
                logger.error(
                    "AppArmor: could not restore unconfined for thread %d "
                    "(profile %s): %s — thread is stuck in this profile "
                    "until it exits OR the next apparmor_confine() call "
                    "on this thread (which defensively resets first). "
                    "Investigate AppArmor kernel state if this persists.",
                    threading.get_native_id(), profile_name, e,
                )


def make_confine_context(profile_name: str) -> Callable[[], ContextManager]:
    """Create a callable that returns an AppArmor confinement context manager.

    The returned callable takes no arguments and produces a context
    manager suitable for ``with confine_ctx(): ...``.  This is passed
    through the ``server → shared`` boundary so that ``ToolExecutor``
    can confine tools without importing ``server.apparmor``.

    Used for BASE-profile entry (prefetch / dynamic-instructions
    expansion, reactor dispatch, session lifecycle work that needs
    full read access to ``.jaato/agents/``, ``.jaato/profiles/``,
    ``.jaato/scripts/``, etc.).  For tool execution (LLM-driven
    dispatch through ToolExecutor) use ``make_tool_confine_context``
    instead — that enters the sub-profile that ADDS read-denies on
    user-authored config.

    Args:
        profile_name: The AppArmor profile name to confine to.

    Returns:
        A zero-argument callable that returns a context manager.
    """
    def _confine():
        return apparmor_confine(profile_name)
    return _confine


def make_tool_confine_context(
    session_profile_name: str,
) -> Callable[[], ContextManager]:
    """Create a callable that enters the per-session ``tool_hat``
    sub-profile (server 0.6.55+, template v13+).

    Tool execution context.  ``ToolExecutor.execute`` calls this
    factory once per tool call to wrap the plugin's executor in
    ``apparmor_confine("jaato-ws-X//tool_hat")``.  The sub-profile
    REPLACES the parent's rules (AppArmor sub-profiles don't
    inherit) and adds read-denies on user-authored config —
    information-isolation between agents in a cascade.

    On exit, the existing apparmor_confine machinery writes
    ``changeprofile unconfined`` to the thread's attr/current,
    same as for base-profile exit.

    Args:
        session_profile_name: The session's BASE profile name (e.g.
            ``"jaato-ws-20260506_135835"``).  The sub-profile name
            ``"//tool_hat"`` is appended automatically.

    Returns:
        A zero-argument callable that returns a context manager.
    """
    return make_confine_context(f"{session_profile_name}//tool_hat")


def make_child_transition_callback(
    session_profile_name: str,
) -> Callable[[], None]:
    """Return a ``preexec_fn``-style callback that transitions the
    forked child into the per-session ``//child`` sub-profile
    (Phase 5 §5.10, template v14+).

    Designed for ``subprocess.Popen(preexec_fn=...)``: runs in the
    forked child between ``fork()`` and ``exec()``.  The callback
    writes ``changeprofile {session_profile_name}//child`` to
    ``/proc/self/attr/current``.

    **Template v17 (2026-05-15) fix.**  Pre-v17 this docstring
    claimed "no explicit ``change_profile -> //child`` rule is
    needed in the parent profile" — empirically false on AppArmor
    4.0.1.  The kernel's ``change_profile`` mediation runs against
    the parent profile's rules even when the target is an inline
    sub-profile, and the transition is DENIED unless explicitly
    authorized.  Template v17 adds the
    ``change_profile -> jaato-ws-{session_id}//child,`` grant
    alongside ``-> unconfined``.  See the corresponding
    PROFILE_TEMPLATE block at ``apparmor.py:511-540`` for the
    full rationale + why this stayed latent until per-stack
    fragment scoping (Piece 1) exposed it.

    After the ``exec()``, the new program is in ``//child``, which
    mirrors ``tool_hat``'s body but drops the three escape-vector
    rules.  A subprocess in ``//child`` cannot write
    ``changeprofile`` to ``/proc/self/attr/current`` (kernel
    rejects with EACCES) and cannot transition anywhere — closing
    the verified escape vector at ``apparmor.py:413-449``.

    **Fail-closed semantics.**  If the write fails (e.g.,
    AppArmor unavailable on the host, or the source profile
    lacks the necessary permissions), the callback raises an
    ``OSError`` from the forked child.  ``subprocess.Popen``
    surfaces this as a spawn failure — the subprocess never
    starts.  This is intentional: a failed transition would mean
    the child inherits the dangerous parent profile, which is the
    exact escape we're trying to prevent.  Spawn failure is the
    correct fail-closed posture.

    Used alongside :meth:`CgroupsManager.make_attach_callback` —
    plugins compose both callables in their ``preexec_fn``,
    apparmor transition first (so the new profile applies during
    the cgroup write), then cgroup attach, then exec.

    Args:
        session_profile_name: The session's BASE profile name
            (e.g. ``"jaato-ws-20260506_135835"``).  The sub-profile
            name ``"//child"`` is appended automatically.

    Returns:
        A zero-argument callable suitable for
        ``Popen(preexec_fn=...)``.
    """
    target = f"{session_profile_name}//child"
    payload = f"changeprofile {target}".encode("ascii")

    def _transition() -> None:
        # Open + write in one syscall pair; can't use Path.write_text
        # because /proc/self/attr/current rejects the truncate that
        # write_text implies, and signal-handler-style territory
        # (preexec_fn) shouldn't allocate Python objects unnecessarily.
        fd = os.open(
            "/proc/self/attr/current", os.O_WRONLY,
        )
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)

    return _transition


# ------------------------------------------------------------------
# Thread-pool safe-submit wrapper (server 0.6.47+)
# ------------------------------------------------------------------
#
# ``apparmor_confine()`` writes ``changeprofile unconfined`` defensively
# on entry and again on exit, but those writes only fire on tasks that
# go through ``apparmor_confine`` itself (tool execution path).  A
# significant amount of work runs on the SAME thread-pool workers
# WITHOUT going through ``apparmor_confine``: enrichment plugins,
# event-bus broadcasts, workspace-monitor callbacks, log I/O,
# ``loop.run_in_executor`` calls in IPC handlers, etc.
#
# If a tool's exit-restoration write to ``/proc/self/task/<tid>/attr/current``
# fails (kernel transient, profile removed mid-tool, signal interruption),
# the worker is stuck in the session profile.  Subsequent non-tool work
# on that worker hits EACCES.  Eventually a new tool DOES run on it and
# the defensive entry-reset recovers — but the non-tool work in between
# fails.
#
# The fix wraps ``ThreadPoolExecutor.submit()`` so EVERY task starts
# with a defensive ``changeprofile unconfined``, regardless of whether
# the task is a confined tool or unconfined housekeeping.  Cost: one
# 27-byte syscall per submit.  Coverage: complete for the wrapped
# executor.
#
# Long-lived shared pools that should use this subclass:
#   - ``ToolExecutor._auto_background_pool``  (auto-background tools)
#   - ``SubagentPlugin._executor``            (subagent submission)
#   - ``BackgroundCapable._bg_executor``      (per-plugin background)
#   - ``asyncio`` default executor            (run_in_executor calls)
#
# Ephemeral ``with ThreadPoolExecutor(...)`` pools (parallel tool
# execution, registry shutdown, web_search timeouts) close their
# workers at scope exit so cross-session leak isn't a concern, but
# wrapping them is still cheap insurance — workers spawned inside an
# apparmor_confine context inherit the parent's profile, so the first
# task on each worker can hit a stuck state otherwise.
#
# Pools are auto-registered on construction in
# ``_REGISTERED_EXECUTORS`` (a WeakSet so dropped pools don't leak),
# and ``sweep_registered_executors()`` dispatches a defensive reset to
# every worker in every registered pool.  ``AppArmorManager.teardown_profile()``
# calls the sweep before invoking ``apparmor_parser -R`` so no worker
# is left flagged-as-confined to a profile that's about to disappear
# (which would surface as a phantom-profile evaluation in the kernel).


def _thread_unconfine_safe() -> None:
    """Best-effort defensive reset of the calling thread's AppArmor
    profile to unconfined (server 0.6.47+).

    Writes ``changeprofile unconfined`` to ``/proc/self/task/<tid>/attr/current``.
    Silent on failure: if the thread is already unconfined the write is
    a no-op; if the thread is genuinely stuck and the kernel rejects
    the write, the next operation will surface the issue with its own
    EACCES.  Never raises — this helper is only ever called as a
    cheap pre-emptive cleanup before unrelated work runs.
    """
    try:
        with open(_get_thread_attr_path(), "w") as f:
            f.write("changeprofile unconfined")
    except (OSError, PermissionError):
        pass  # not on Linux, AppArmor unavailable, or already unconfined.


# Phase 2 (confined runner): the daemon never confines its own
# threads to a per-session AppArmor profile, so the pre-task
# defensive-reset hook is no longer needed.  The
# ``_thread_unconfine_safe`` helper above is kept for any
# third-party reuse and ``SafeThreadPoolExecutor`` is unchanged
# from the perspective of its remaining callers (auto-background
# pool, subagent pool) — they get a plain ThreadPoolExecutor
# subclass with an empty pre-task hook list.
#
# ``AppArmorManager._teardown_profile_impl`` still calls
# ``sweep_registered_executors()`` defensively; with no hooks
# registered, the sweep dispatches no-op tasks — wasted work,
# not a correctness issue.  Phase 6 cleanup deletes the sweep
# call along with the rest of the thread-confinement machinery.


# ------------------------------------------------------------------
# Reference authorizer (selectReferences → fragment management)
# ------------------------------------------------------------------

class ReferenceAuthorizer:
    """Per-session handle the references plugin uses to mutate the
    session's AppArmor profile when a reference is selected/unselected.

    Mirrors the ``make_confine_context()`` injection pattern: the
    server (which owns the ``AppArmorManager``) constructs an instance
    bound to one session and hands it across the ``server → shared``
    boundary via ``JaatoServer.set_reference_authorizer()``.  The
    references plugin queries it from the session and calls
    ``authorize`` / ``deauthorize`` alongside the existing
    ``sandbox_manager`` calls.

    When AppArmor is unavailable, ``JaatoWSServer.get_reference_authorizer``
    returns ``None`` and the plugin skips the fragment path entirely —
    the in-process ``sandbox_manager`` allowlist is the only layer.
    """

    def __init__(self, apparmor: "AppArmorManager", session_id: str) -> None:
        self._apparmor = apparmor
        self._session_id = session_id

    def authorize(self, ref_id: str, path: str) -> bool:
        """Grant kernel-level readonly access to *path* for this session."""
        return self._apparmor.add_reference_fragment(
            self._session_id, ref_id, path,
        )

    def deauthorize(self, ref_id: str) -> bool:
        """Revoke a previously-authorized reference."""
        return self._apparmor.remove_reference_fragment(
            self._session_id, ref_id,
        )
