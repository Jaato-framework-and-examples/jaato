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
import logging
import os
import platform
import re
import shutil
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Optional

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
    _TEMPLATE_VERSION = 10

    # AppArmor profile template.  Placeholders are filled per-session by
    # ``_render_profile()``.
    PROFILE_TEMPLATE = '''\
# jaato-apparmor-template-version: {template_version}
#include <tunables/global>

profile jaato-ws-{session_id} flags=(attach_disconnected) {{
  #include <abstractions/base>
  #include <abstractions/nameservice>
  #include <abstractions/python>

  # ---- workspace: read-write ----
  {workspace_path}/   rw,
  {workspace_path}/** rwkl,

  # ---- shared read-only resources ----
  {venv_path}/           r,
  {venv_path}/**         r,
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

  # ---- user-global jaato config (read-only) ----
  # Allow agent/profile/prompt/theme definitions from ~/.jaato/.
  # NOT allowed: credentials, *_auth.json, sibling workspaces.
  @{{HOME}}/.jaato/agents/         r,
  @{{HOME}}/.jaato/agents/**       r,
  @{{HOME}}/.jaato/profiles/       r,
  @{{HOME}}/.jaato/profiles/**     r,
  @{{HOME}}/.jaato/prompts/        r,
  @{{HOME}}/.jaato/prompts/**      r,
  @{{HOME}}/.jaato/skills/         r,
  @{{HOME}}/.jaato/skills/**       r,
  @{{HOME}}/.jaato/themes/         r,
  @{{HOME}}/.jaato/themes/**       r,
  @{{HOME}}/.jaato/services/       r,
  @{{HOME}}/.jaato/services/**     r,
  @{{HOME}}/.jaato/references/     r,
  @{{HOME}}/.jaato/references/**   r,
  @{{HOME}}/.jaato/keybindings.json r,
  @{{HOME}}/.jaato/theme.json       r,
  @{{HOME}}/.jaato/gc.json          r,

  # ---- client-supplied env_file (read-only, optional) ----
  # When the client passes ``env_file`` on ``ClientConfigRequest``,
  # the daemon's session-creation path reads it to populate provider
  # config (PROJECT_ID, JAATO_PROVIDER, etc.).  Reactor-spawned
  # subagents go through the same path during ``create_session``,
  # which fires from inside the parent's confined thread — without
  # this grant, every reactor-spawned session fails with
  # ``Permission denied`` on the env file.
  {env_file_rule}

  # ---- global memories (read-write) ----
  # All sessions can propose and read cross-session memories.
  # The maturity lifecycle (raw → validated) is the quality gate,
  # not filesystem permissions.
  #
  # Layout (current):
  #   memories/raw/{{id}}.json    — pending queue, one file per memory
  #   memories/curated.jsonl      — curator-managed knowledge base
  # Layout (legacy, retained for migration):
  #   memories.jsonl              — pre-split single-file store
  #
  # The first two rules cover folder creation (mkdir on the parent),
  # directory enumeration, and atomic tempfile + rename writes inside
  # the raw/ subdirectory and against curated.jsonl.  The third rule
  # keeps pre-split data readable until it's migrated away.
  @{{HOME}}/.jaato/memories/       rw,
  @{{HOME}}/.jaato/memories/**     rw,
  @{{HOME}}/.jaato/memories.jsonl  rw,

  # ---- Claude Code interop (read-only) ----
  # The prompt_library plugin reads ~/.claude/skills and ~/.claude/commands
  # so jaato can use Claude Code skill/command definitions interchangeably.
  @{{HOME}}/.claude/skills/        r,
  @{{HOME}}/.claude/skills/**      r,
  @{{HOME}}/.claude/commands/      r,
  @{{HOME}}/.claude/commands/**    r,

  # ---- ML model caches (read-write) ----
  # The embedding provider (sentence-transformers, HuggingFace transformers,
  # ONNX runtime) loads models from these caches. Read-write because the
  # libraries write lockfiles and metadata even for cached models.
  @{{HOME}}/.cache/huggingface/    rw,
  @{{HOME}}/.cache/huggingface/**  rwk,
  @{{HOME}}/.cache/torch/          rw,
  @{{HOME}}/.cache/torch/**        rwk,

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
  change_profile -> unconfined,
  /proc/self/attr/current      w,
  /proc/self/task/*/attr/current w,

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
            logger.info("AppArmor confinement not available — falling back to directory sandboxing")
        return self._available

    def _check_availability(self) -> bool:
        """Perform the actual availability check (called once)."""
        if platform.system() != "Linux":
            logger.debug("AppArmor: not Linux")
            return False

        if not shutil.which("apparmor_parser"):
            logger.debug("AppArmor: apparmor_parser not found")
            return False

        if not shutil.which("aa-exec"):
            logger.debug("AppArmor: aa-exec not found")
            return False

        if not Path("/sys/kernel/security/apparmor").exists():
            logger.debug("AppArmor: kernel module not loaded")
            return False

        # Check profile directory
        try:
            self._profile_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.debug("AppArmor: cannot create profile dir %s", self._profile_dir)
            return False

        if not os.access(self._profile_dir, os.W_OK):
            logger.debug("AppArmor: profile dir not writable: %s", self._profile_dir)
            return False

        # Verify sudo access to apparmor_parser (required for loading profiles)
        try:
            result = subprocess.run(
                ["sudo", "-n", "apparmor_parser", "--version"],
                capture_output=True, timeout=5,
            )
            if result.returncode != 0:
                logger.debug("AppArmor: sudo apparmor_parser not available (no sudoers rule?)")
                return False
        except (subprocess.TimeoutExpired, OSError):
            logger.debug("AppArmor: sudo apparmor_parser check failed")
            return False

        # Create user-local cache directory for apparmor_parser
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.debug("AppArmor: cannot create cache dir %s", self._cache_dir)
            return False

        return True

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def provision_profile(
        self,
        session_id: str,
        workspace_path: str,
        config_root: Optional[str] = None,
        env_file: Optional[str] = None,
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
        )

    def _provision_profile_impl(
        self,
        session_id: str,
        workspace_path: str,
        config_root: Optional[str] = None,
        env_file: Optional[str] = None,
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

        Returns:
            True on success, False on failure (logged, not raised).
        """
        if not self.is_available():
            return False

        profile_name = self.get_profile_name(session_id)
        profile_path = self._profile_dir / profile_name
        profile_content = self._render_profile(
            session_id, workspace_path, config_root, env_file,
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
        """Return the AppArmor profile name for a session."""
        return f"jaato-ws-{session_id}"

    def _render_profile(
        self,
        session_id: str,
        workspace_path: str,
        config_root: Optional[str] = None,
        env_file: Optional[str] = None,
    ) -> str:
        """Render the profile template with session-specific values.

        The template uses Python ``str.format()`` placeholders.
        Sibling workspaces are implicitly denied by AppArmor's default-
        deny policy — only the session's own ``workspace_path`` is in
        the allow list.
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

        # Inline extension-supplied fragments at render time.  Reads
        # every ``*.rules`` file under ``~/.jaato/apparmor-fragments/``
        # and concatenates the contents below the header in
        # PROFILE_TEMPLATE.  Each fragment's content is wrapped with a
        # ``# === <filename> ===`` comment so kernel denials can be
        # traced back to the originating extension when debugging
        # ``dmesg | grep apparmor`` output.
        fragments_dir = (
            Path("~/.jaato/apparmor-fragments").expanduser().resolve()
        )
        extension_fragments_inline = (
            "  # (no extension fragments)"
        )
        if fragments_dir.is_dir():
            chunks: List[str] = []
            for path in sorted(fragments_dir.glob("*.rules")):
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
                chunks.append(f"  # === {path.name} ===\n{indented}")
            if chunks:
                extension_fragments_inline = "\n\n".join(chunks)

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
        )

    def _refs_dir(self, session_id: str) -> Path:
        """Per-session AppArmor reference-fragment directory.

        Lives next to the base profile file so AppArmor's ``include``
        glob picks up every fragment file written by
        ``add_reference_fragment()``.
        """
        return self._profile_dir / f"{self.get_profile_name(session_id)}.refs.d"


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

    Args:
        profile_name: The AppArmor profile name to confine to.

    Returns:
        A zero-argument callable that returns a context manager.
    """
    def _confine():
        return apparmor_confine(profile_name)
    return _confine


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
