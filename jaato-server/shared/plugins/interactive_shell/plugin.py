"""Interactive shell plugin for driving user-interactive commands.

Provides tools that let the model spawn persistent PTY sessions and
interact with programs that require back-and-forth input: REPLs, password
prompts, wizards, debuggers, SSH sessions, etc.

Design philosophy: the model reads whatever the process outputs and makes
its own decisions about what to type next. No expect patterns required —
the intelligence is in the model, not the tool.

Containment (#722, #503)
========================
This plugin spawns a real PTY and is strictly more capable than ``cli``,
which has always refused paths outside the session workspace.  Until #722
it refused nothing, so ``shell_spawn("cat /etc/hostname")`` read a file
``cli`` had just declined.  Three layers answer that now, and they are
deliberately not equal:

1. **The spawn command** is a shell command by construction, so it is
   checked exactly as ``cli`` checks one — same analyzer, same workspace
   rules (:mod:`shared.plugins.command_containment`) — and **fails closed**
   on a command the analyzer cannot model.
2. **Text sent to a live session** is checked too, but **fails open** on a
   parse failure: that text may be Python, SQL or a password, and refusing
   everything the shell grammar cannot parse would refuse most legitimate
   input.  It catches the direct attempt (``cat /etc/shadow``) and claims
   nothing more.
3. **Kernel confinement** is the only real boundary for a live PTY, whose
   working directory drifts under ``cd`` and which can name paths through
   channels no string check sees.  When the runner installed an AppArmor
   child-profile transition the spawned child enters it; when it did not,
   the plugin says so at WARNING once per session rather than running
   unconfined in silence, and ``require_confinement: true`` makes that
   posture fail closed instead.
"""

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional

from jaato_sdk.plugins.base import UserCommand
from jaato_sdk.plugins.model_provider.types import (
    WithMetadata,
    ToolSchema,
    DISCOVERABILITY_DEFERRED,
)
from .session import ShellSession, _BACKEND, _BACKEND_ERROR, IS_MSYS2
from .ansi import strip_ansi
from shared.ai_tool_runner import get_current_tool_output_callback
from shared.plugins.runner_forwarding import RunnerForwardingMixin
from shared.secret_scrub import DEFAULT_SECRET_ENV_PATTERNS, resolve_scrub_patterns
from shared.command_analysis import UnanalyzableCommand
from ..command_containment import first_denied_path
from ..workspace_venv import (
    resolve_venv_path, ensure_workspace_venv, pip_apparmor_rules,
)

logger = logging.getLogger(__name__)


# Maximum concurrent interactive sessions
DEFAULT_MAX_SESSIONS = 8

# Session reaper interval (seconds)
REAPER_INTERVAL = 30.0

# Max idle time before a session is reaped (seconds)
DEFAULT_MAX_IDLE = 300  # 5 minutes


class InteractiveShellPlugin(RunnerForwardingMixin):
    """Plugin that provides interactive shell session management.

    Allows the model to spawn long-lived sessions and drive any
    interactive command by reading output and sending input.

    Platform backends (selected at import time by ``session.py``):

    - **pexpect** (Unix / macOS, or MSYS Python with ``pty``): full PTY.
    - **popen_spawn** (MSYS2 with MINGW Python): ``subprocess.Popen`` pipes.
      No real PTY — child ``isatty()`` is ``False``, terminal dimensions
      are ignored, password-prompt echo control is unavailable — but
      timeout/idle detection works reliably.
    - **wexpect** (native Windows): Windows console APIs + named pipes.

    The active backend is exposed as ``session._BACKEND`` and logged in
    the plugin's trace output at ``initialize()`` and ``spawn()`` time.

    Containment: every spawn command, and every text sent to a live
    session, is checked against the session workspace before it reaches
    the PTY (see the module docstring for the three layers and why the
    two string checks fail in opposite directions).  ``workspace_root``
    is both the check's boundary and the spawn ``cwd``; with no workspace
    root configured there is no sandbox and nothing is refused, exactly
    as in ``cli``.

    Configuration:
        max_sessions: Maximum concurrent sessions (default: 8).
        max_lifetime: Max session lifetime in seconds (default: 600).
        max_idle: Max idle time before reaping in seconds (default: 300).
        idle_timeout: Seconds of silence for output settling (default: 0.5).
        workspace_root: Working directory for spawned processes, and the
            containment boundary their commands are checked against.
        require_confinement: Refuse to spawn at all when no AppArmor
            child-profile transition is installed (default: False).
    """

    def __init__(self):
        self._sessions: Dict[str, ShellSession] = {}
        self._session_counter = 0
        self._lock = threading.Lock()

        # Configuration (set during initialize)
        self._max_sessions = DEFAULT_MAX_SESSIONS
        self._max_idle = DEFAULT_MAX_IDLE
        self._max_lifetime = 600
        self._idle_timeout = 0.5
        self._workspace_root: Optional[str] = None
        # Workspace-scoped venv path for spawned sessions (None/empty = off).
        # See shared/plugins/workspace_venv.py.
        self._workspace_venv: Optional[str] = None
        # Secrets-broker scrub (#10 / #503 / #863): env-var name globs
        # stripped from the inherited environment of every spawned PTY
        # session, so a model-driven REPL cannot ``echo $GITHUB_TOKEN``.
        # ON by default — the framework set applies until ``initialize``
        # resolves the operator's ``scrub_secret_env`` knob.
        self._scrub_secret_env: List[str] = list(DEFAULT_SECRET_ENV_PATTERNS)
        # Confinement posture (#722).  A PTY child is a separate process,
        # the same risk class as a ``cli`` subprocess, so the default is
        # ``cli``'s: run, with the string containment below, and ANNOUNCE
        # that the kernel boundary is absent rather than inheriting the
        # old answer ("unconfined, silently").  ``require_confinement``
        # takes the notebook plugin's posture instead and fails closed.
        # ``_unconfined_announced`` keeps the WARNING to once per
        # initialize() — it is a property of the deployment, not of the
        # spawn, and one line per spawn would be noise nobody reads.
        self._require_confinement = False
        self._unconfined_announced = False
        # Plugin registry, for the operator's authorized / denied external
        # paths.  Wired by set_plugin_registry(); None means "no grants,
        # no denials", not an error.
        self._plugin_registry = None
        self._agent_name: Optional[str] = None
        self._initialized = False
        self._tool_output_callback: Optional[Callable[[str], None]] = None

        # Per-session runtime limits installed by the executor via
        # set_runtime_limits().  ``_cgroup_attach`` becomes preexec_fn
        # at spawn time so the new PTY child joins the session's cgroup
        # before exec().  ``_runtime_limits`` is currently informational
        # — interactive sessions are inherently long-lived, so the
        # ``tool_timeout_seconds`` cap doesn't apply per-call; the
        # max_lifetime config remains the relevant ceiling.  Output
        # caps from RuntimeLimits are not enforced here either: each
        # ``shell_input``/``shell_read`` already trims to a fixed
        # buffer at the protocol layer (see read_until_idle in session.py).
        self._cgroup_attach: Optional[Callable[[], None]] = None
        self._runtime_limits = None
        # Phase 5 §5.10d: AppArmor child-profile transition callback
        # installed via set_apparmor_child_transition_callback().  When
        # set, the plugin's ShellSession spawn preexec_fn writes
        # ``changeprofile <profile>//child`` to
        # /proc/self/attr/current between fork() and exec() so the new
        # PTY child enters the per-session ``//child`` sub-profile
        # (which drops the escape-vector rules).  None until the
        # executor calls — same contract as _cgroup_attach.  See
        # docs/design/phase5_5_10_apparmor_child_subprofile_audit.md.
        self._apparmor_child_transition: Optional[Callable[[], None]] = None
        # Reaper thread
        self._reaper_thread: Optional[threading.Thread] = None
        self._reaper_stop = threading.Event()

    @property
    def name(self) -> str:
        return "interactive_shell"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    agent_prefix = f"@{self._agent_name}" if self._agent_name else ""
                    f.write(f"[{ts}] [InteractiveShell{agent_prefix}] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the interactive shell plugin.

        Args:
            config: Optional dict with:
                - max_sessions: Max concurrent sessions (default: 8)
                - max_lifetime: Session lifetime ceiling in seconds (default: 600)
                - max_idle: Max idle seconds before reaping (default: 300)
                - idle_timeout: Output settling time in seconds (default: 0.5)
                - workspace_root: Working directory for spawned processes,
                  and the boundary their commands are contained to (#722).
                  Absent/empty = no sandboxing, as in ``cli``.
                - require_confinement: Refuse every spawn when the runner
                  installed no AppArmor child-profile transition (default
                  False — announce at WARNING and run, which is ``cli``'s
                  posture for the same class of subprocess)
                - agent_name: Agent context for trace logging
                - scrub_secret_env: env-var name globs stripped from every
                  spawned session's inherited environment ('default' /
                  absent = the framework set; 'none' = off, announced at
                  WARNING; a list may carry 'default' and '!EXEMPT'
                  entries).  See ``shared.secret_scrub``.
        """
        if config:
            self._agent_name = config.get('agent_name')
            if 'max_sessions' in config:
                self._max_sessions = config['max_sessions']
            if 'max_lifetime' in config:
                self._max_lifetime = config['max_lifetime']
            if 'max_idle' in config:
                self._max_idle = config['max_idle']
            if 'idle_timeout' in config:
                self._idle_timeout = config['idle_timeout']
            if 'workspace_root' in config:
                workspace = config['workspace_root']
                if workspace:
                    self._workspace_root = os.path.realpath(
                        os.path.abspath(workspace)
                    )
            if 'workspace_venv' in config:
                self._workspace_venv = config['workspace_venv']
            if 'require_confinement' in config:
                self._require_confinement = bool(config['require_confinement'])

        # Secrets-broker scrub (#503's second gap, closed by #863): absent
        # means the framework set; ``none`` is the announced opt-out; a
        # malformed value fails closed.
        self._scrub_secret_env = list(resolve_scrub_patterns(
            (config or {}).get('scrub_secret_env'), surface=self.name,
        ))

        self._initialized = True
        self._unconfined_announced = False
        self._start_reaper()
        self._trace(
            f"initialize: max_sessions={self._max_sessions}, "
            f"max_lifetime={self._max_lifetime}, "
            f"max_idle={self._max_idle}, "
            f"workspace_root={self._workspace_root}, "
            f"require_confinement={self._require_confinement}, "
            f"backend={_BACKEND or 'NONE'}, "
            f"msys2={IS_MSYS2}"
        )
        if _BACKEND is None and _BACKEND_ERROR:
            self._trace(f"initialize: WARNING - no backend: {_BACKEND_ERROR}")

    def shutdown(self) -> None:
        """Shutdown the plugin, closing all sessions."""
        self._trace("shutdown: closing all sessions")
        self._stop_reaper()

        with self._lock:
            for session_id in list(self._sessions.keys()):
                try:
                    self._sessions[session_id].close()
                except Exception:
                    pass
            self._sessions.clear()

        self._initialized = False

    def reset_for_next_session(self) -> None:
        """Cascade-sharing reset (Phase 1b, server 0.6.143+).

        Per Daniel's litmus test: PTY sessions are typically per-agent,
        per-session.  The next session's agent has no awareness of the
        prior session's spawns and cannot meaningfully address them.
        Carrying them across the boundary leaves orphan PTYs accumulating
        — bounded by the reaper's max_lifetime/max_idle, but still
        operator-confusing.

        Per-session state CLEARED:
        - ``_sessions``: PTY shell session map.  CLOSE each session
          + clear the dict.  Same logic as ``shutdown()`` per-session
          cleanup but without stopping the reaper thread (reaper
          persists across cascade sessions).
        - ``_session_counter``: per-session counter reset to 0.
        - ``_agent_name``: re-set by next session's ``initialize()``.
        - ``_tool_output_callback``: re-wired by next session.

        Survives the reset:
        - ``_max_sessions``, ``_max_idle``, ``_max_lifetime``,
          ``_idle_timeout``: workspace-tier config.
        - ``_workspace_root``: constant within cascade.  It is also the
          containment boundary (#722), and the next session's
          ``initialize()`` re-states it.
        - ``_require_confinement``, ``_unconfined_announced``: the
          confinement posture is workspace-tier config like the caps
          above; the announcement latch is re-armed by the next
          ``initialize()``, so each session says it once.
        - ``_initialized``: stays True (reaper still running).
        - Reaper thread + control flags: keep alive between sessions.
        - ``_cgroup_attach``, ``_runtime_limits``,
          ``_apparmor_child_transition``: lifecycle hooks (re-wired
          on next session).

        """
        self._trace("reset_for_next_session: closing per-session PTYs, keeping reaper")
        with self._lock:
            for session_id in list(self._sessions.keys()):
                try:
                    self._sessions[session_id].close()
                except Exception:
                    pass
            self._sessions.clear()
        self._session_counter = 0
        self._agent_name = None
        self._tool_output_callback = None

    def get_config_schema(self) -> dict:
        """Return JSON Schema for this plugin's configuration."""
        return {
            "type": "object",
            "properties": {
                "max_sessions": {
                    "type": "integer",
                    "default": 8,
                    "description": "Maximum concurrent shell sessions",
                },
                "max_lifetime": {
                    "type": "integer",
                    "default": 600,
                    "description": "Session lifetime ceiling in seconds",
                },
                "max_idle": {
                    "type": "integer",
                    "default": 300,
                    "description": "Max idle seconds before session reaping",
                },
                "idle_timeout": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Output settling time in seconds",
                },
                "scrub_secret_env": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "default": "default",
                    "description": (
                        "Env-var name globs (case-insensitive fnmatch) stripped "
                        "from the inherited environment of every spawned "
                        "session, so a model-driven shell/REPL cannot read raw "
                        "credentials the runner holds. 'default' (also when "
                        "absent) = the framework set; 'none' = off (announced "
                        "at WARNING); a list may carry 'default' and '!NAME' "
                        "exemption entries. Overrides the profile-level "
                        "scrub_secret_env for this surface."
                    ),
                },
                "require_confinement": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Refuse every shell_spawn when the runner installed "
                        "no AppArmor child-profile transition, i.e. when the "
                        "spawned PTY would run without a kernel-enforced "
                        "boundary. Default false: the plugin spawns and "
                        "announces the absence at WARNING (a PTY child is a "
                        "separate process, the same risk class as a cli "
                        "subprocess, which does not fail closed either). Set "
                        "true on hosts where the workspace boundary must be "
                        "kernel-enforced or not offered at all."
                    ),
                },
                "workspace_venv": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Path to a workspace-scoped venv to activate for "
                        "spawned sessions (empty = off). Relative paths "
                        "resolve against the workspace root. Created if "
                        "absent with --system-site-packages. Recommended: "
                        ".jaato/tool-venv"
                    ),
                },
            },
        }

    @classmethod
    def get_apparmor_rules(
        cls,
        *,
        workspace_path: str,
        session_id: str,
        config_root: Optional[str],
        plugin_config: Dict[str, Any],
    ) -> List[str]:
        """Contribute pip's AppArmor rules to the profile.

        Interactive shells can run ``pip``: the distro/UA OS-id reads (crashes
        without them under confinement) plus, when a ``workspace_venv`` is set,
        an ``ix`` grant on the venv bin so a bare ``pip`` / console script runs.
        Scoped to sessions that load ``interactive_shell`` — least-privilege.
        See ``pip_apparmor_rules``.
        """
        return pip_apparmor_rules(plugin_config.get("workspace_venv"), workspace_path)

    def set_workspace_path(self, path: Optional[str]) -> None:
        """Update the workspace root path.

        Args:
            path: The new workspace root path, or None.
        """
        if path:
            self._workspace_root = os.path.realpath(os.path.abspath(path))
        else:
            self._workspace_root = None
        self._trace(f"set_workspace_path: {self._workspace_root}")

    def set_plugin_registry(self, registry) -> None:
        """Receive the plugin registry, for external-path authorization.

        Called by ``PluginRegistry.expose_tool`` during plugin wiring (see
        "Plugin Auto-Wiring" in the root ``CLAUDE.md``).  The registry
        carries the operator's ``sandbox add`` grants and explicit denials,
        which
        :func:`~shared.plugins.command_containment.path_within_workspace`
        consults — so a path the operator authorised outside the workspace
        is reachable from a shell exactly as it is from ``cli``, and a
        denied one is refused in both.  ``None`` until this is called,
        which the containment check reads as "no grants, no denials"
        rather than as an error.

        Args:
            registry: The ``PluginRegistry`` instance.
        """
        self._plugin_registry = registry

    def set_runtime_limits(self, attach_callback, limits) -> None:
        """Receive per-session cgroup attach + app-layer caps from the executor.

        Forwarded by ``ToolExecutor.set_runtime_limits``.  At spawn
        time, ``attach_callback`` becomes ``preexec_fn`` on the
        backend's spawn call so the forked PTY child joins the
        session's cgroup before ``exec``.  Existing sessions are not
        re-attached — operators tuning limits mid-session must close
        and respawn (rare for interactive PTY workflows).

        ``limits`` is stored but not actively enforced here:
        ``tool_timeout_seconds`` doesn't map cleanly to a long-lived
        PTY (the existing ``max_lifetime`` config covers that), and
        per-read output caps are already enforced by
        ``read_until_idle``'s buffer.  Stored for future use and so
        operators can introspect what the plugin received.
        """
        self._cgroup_attach = attach_callback
        self._runtime_limits = limits
        self._trace(
            f"set_runtime_limits: attach={attach_callback is not None} "
            f"limits={limits!r}"
        )

    def set_apparmor_child_transition_callback(
        self,
        callback: Optional[Callable[[], None]],
    ) -> None:
        """Install the AppArmor child-profile transition callback
        (Phase 5 §5.10d).

        Mirrors the cli plugin's §5.10c wiring exactly.  Forwarded by
        ``ToolExecutor.set_apparmor_child_transition_callback`` at
        runner-side bootstrap.  When set, the plugin's
        ``ShellSession`` spawn composes this callback with the
        cgroup-attach callback: AppArmor transition FIRST, then
        cgroup attach, then exec.  Order matters — the new ``//child``
        profile applies during the cgroup write.

        Closes the verified escape at ``apparmor.py:413-449`` for the
        interactive_shell surface: a PTY child running
        ``python3 -c 'open("/proc/self/attr/current","w").write("changeprofile unconfined")'``
        from a shell tool cannot escape the per-session profile.

        Argument may be ``None`` when the runner isn't AppArmor-
        confined (JAATO_RUNNER_DISABLE_CONFINE=1 or daemon-side
        legacy paths) — spawn falls back to cgroup-only preexec_fn.
        """
        self._apparmor_child_transition = callback
        self._trace(
            f"set_apparmor_child_transition_callback: "
            f"transition={callback is not None}"
        )

    def _build_subprocess_preexec_fn(
        self,
    ) -> Optional[Callable[[], None]]:
        """Phase 5 §5.10d: compose the apparmor + cgroup preexec_fn.

        Mirrors :meth:`CLIToolPlugin._build_subprocess_preexec_fn` —
        same four-case ladder (none / apparmor-only / cgroup-only /
        both), same apparmor-first ordering, same fail-closed
        semantics (an exception in preexec_fn propagates as a spawn
        failure; ShellSession surfaces the failure to the caller
        instead of returning a half-started session that's lost
        confinement).
        """
        apparmor_cb = self._apparmor_child_transition
        cgroup_cb = self._cgroup_attach
        if apparmor_cb is None and cgroup_cb is None:
            return None
        if apparmor_cb is None:
            return cgroup_cb
        if cgroup_cb is None:
            return apparmor_cb

        def _composite() -> None:
            apparmor_cb()
            cgroup_cb()

        return _composite

    # --- Path containment (#722, #503) ---

    def _containment_refusal(
        self,
        command: str,
        tool: str,
        on_parse_error: str,
    ) -> Optional[Dict[str, Any]]:
        """Refuse *command* when it names a path outside the workspace.

        The plugin-side half of #722.  ``cli`` has always checked the paths
        in a command against the session workspace; this is the same check,
        on the same analyzer, applied at the two points where a string this
        plugin controls reaches a shell (``shell_spawn``'s command and
        ``shell_input``'s text).

        Args:
            command: The string about to reach the PTY.
            tool: Tool name, used only to prefix the refusal so the model
                can tell which call was refused.
            on_parse_error: ``"deny"`` or ``"allow"`` — what an unparseable
                string means.  ``shell_spawn`` passes ``"deny"`` (its
                command IS a shell command, so a string the analyzer cannot
                model is refused, as in ``cli``); ``shell_input`` passes
                ``"allow"`` (its text is whatever the running program
                reads — Python, SQL, a password — and a shell-grammar
                failure there is the normal case, not an evasion).

        Returns:
            ``None`` when the command may run; otherwise a ready-to-return
            executor result whose ``error`` names the offending path and
            the boundary, because a refusal the model cannot distinguish
            from a broken environment costs it a turn guessing (see the
            ``/dev/null`` lesson in :mod:`shared.plugins.sandbox_utils`).
        """
        if not self._workspace_root or not command:
            return None

        try:
            denied = first_denied_path(
                command,
                self._workspace_root,
                self._plugin_registry,
                on_parse_error=on_parse_error,
            )
        except UnanalyzableCommand as exc:
            self._trace(f"containment: {tool} unparseable, refused ({exc})")
            return {
                'error': (
                    f'{tool}: refused — this command cannot be parsed as a '
                    f'shell command ({exc}), so its paths cannot be checked '
                    f'against the workspace. Rewrite it with balanced '
                    f'quotes and complete redirections.'
                ),
            }

        if denied is None:
            return None

        path, mode = denied
        self._trace(
            f"containment: {tool} blocked path={path!r} mode={mode} "
            f"workspace={self._workspace_root}"
        )
        return {
            'error': (
                f'{tool}: refused — {path!r} is outside the session '
                f'workspace ({self._workspace_root}), which this session '
                f'may not {mode}. Use a path inside the workspace.'
            ),
        }

    def _confinement_refusal(self) -> Optional[Dict[str, Any]]:
        """Apply the confinement posture to one spawn.

        A live PTY cannot be contained by reading strings — its working
        directory drifts under ``cd`` and a program inside it can name
        paths through channels no analyzer sees — so the kernel boundary
        is the real one, and whether it is present is worth stating
        (#722).  The plugin's evidence for it is
        ``_apparmor_child_transition``: the runner installs that callback
        only when it is itself confined, and it is what puts the forked
        child into the per-session ``//child`` profile.

        Returns:
            ``None`` when the spawn may proceed (confined, or unconfined
            and permitted).  Otherwise an executor result refusing the
            spawn, which happens only under ``require_confinement``.

        Side effects:
            Announces the unconfined posture once per ``initialize()``, at
            WARNING — the same treatment ``scrub_secret_env: none`` and
            ``--ws-unsafe-no-auth`` get, so running without the kernel
            boundary is never silent.
        """
        if self._apparmor_child_transition is not None:
            return None

        if self._require_confinement:
            self._trace("spawn: refused — require_confinement and no AppArmor transition")
            return {
                'error': (
                    'shell_spawn: refused — this deployment requires '
                    'kernel-enforced confinement for interactive shells '
                    '(plugin_configs.interactive_shell.require_confinement) '
                    'and no AppArmor child profile is active for this '
                    'session.'
                ),
            }

        if not self._unconfined_announced:
            self._unconfined_announced = True
            logger.warning(
                "interactive_shell: spawning PTY sessions WITHOUT kernel "
                "confinement (no AppArmor child-profile transition is "
                "installed). Commands and typed input are checked against "
                "the workspace (%s), but a live PTY can reach paths that "
                "check cannot see. Set "
                "plugin_configs.interactive_shell.require_confinement: true "
                "to refuse spawning instead.",
                self._workspace_root or "no workspace root configured",
            )
        return None

    # --- Tool schemas ---

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for all interactive shell tools."""
        return [
            ToolSchema(
                name='shell_spawn',
                description=(
                    'Start a new interactive process and return its session_id. '
                    'Use shell_input(session_id=...) for ALL subsequent '
                    'interactions — calling shell_spawn again creates a '
                    'SEPARATE process, it does NOT send input to this one. '
                    'Use for programs that require interactive input: '
                    'REPLs (python, node, psql), password prompts (ssh, sudo), '
                    'wizards (npm init), debuggers (gdb, pdb).'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "The command to run interactively. Examples: "
                                "'python3', 'ssh user@host', 'psql -U admin mydb', "
                                "'npm init', 'gdb ./myprogram'"
                            ),
                        },
                        "session_name": {
                            "type": "string",
                            "description": (
                                "Optional human-readable name for this session. "
                                "Auto-generated if omitted (e.g., 'session_0')."
                            ),
                        },
                        "rows": {
                            "type": "integer",
                            "description": "PTY height in rows (default: 24).",
                        },
                        "cols": {
                            "type": "integer",
                            "description": "PTY width in columns (default: 80).",
                        },
                    },
                    "required": ["command"],
                },
                category="system",
                discoverability=DISCOVERABILITY_DEFERRED,
            ),
            ToolSchema(
                name='shell_input',
                description=(
                    'Send text input to a running interactive session (by '
                    'session_id from shell_spawn) and return whatever the '
                    'program outputs next. This is the ONLY tool for '
                    'interacting with an already-running session — do not '
                    'use shell_spawn to send input.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID from shell_spawn.",
                        },
                        "input": {
                            "type": "string",
                            "description": (
                                "Text to type into the session. Include \\n at "
                                "the end to press Enter. For example: "
                                "'yes\\n', 'SELECT * FROM users;\\n', "
                                "'my-project-name\\n'"
                            ),
                        },
                    },
                    "required": ["session_id", "input"],
                },
                category="system",
                discoverability=DISCOVERABILITY_DEFERRED,
            ),
            ToolSchema(
                name='shell_read',
                description=(
                    'Read pending output from a session without sending any '
                    'input. Useful for checking on long-running operations or '
                    'reading output that arrived since the last interaction.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID from shell_spawn.",
                        },
                        "timeout": {
                            "type": "number",
                            "description": (
                                "Seconds to wait for output (default: 2). "
                                "Use longer values for slow operations."
                            ),
                        },
                    },
                    "required": ["session_id"],
                },
                category="system",
                discoverability=DISCOVERABILITY_DEFERRED,
            ),
            ToolSchema(
                name='shell_control',
                description=(
                    'Send a control key to an interactive session. '
                    'Use for interrupting (Ctrl+C), sending EOF (Ctrl+D), '
                    'suspending (Ctrl+Z), or clearing screen (Ctrl+L).'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID from shell_spawn.",
                        },
                        "key": {
                            "type": "string",
                            "description": (
                                "Control key to send. Options: "
                                "'c-c' (Ctrl+C, interrupt), "
                                "'c-d' (Ctrl+D, EOF/exit), "
                                "'c-z' (Ctrl+Z, suspend), "
                                "'c-\\\\' (Ctrl+\\\\, quit), "
                                "'c-l' (Ctrl+L, clear)"
                            ),
                            "enum": ["c-c", "c-d", "c-z", "c-\\", "c-l"],
                        },
                    },
                    "required": ["session_id", "key"],
                },
                category="system",
                discoverability=DISCOVERABILITY_DEFERRED,
            ),
            ToolSchema(
                name='shell_close',
                description=(
                    'Close an interactive session. Sends EOF, then SIGTERM, '
                    'then SIGKILL if needed. Returns exit status and any '
                    'final output the program produced while shutting down.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID from shell_spawn.",
                        },
                    },
                    "required": ["session_id"],
                },
                category="system",
                discoverability=DISCOVERABILITY_DEFERRED,
            ),
            ToolSchema(
                name='shell_list',
                description=(
                    'List all active interactive sessions with their status, '
                    'command, and age. Useful for keeping track of what is '
                    'running, especially in long conversations.'
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                },
                category="system",
                discoverability=DISCOVERABILITY_DEFERRED,
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor mapping for all tools.

        Phase 3 §3.5 wave 2: forwards via runner-RPC when a runner
        is attached so spawned PTY subprocesses inherit the runner's
        AppArmor profile (kernel-confined per-session multitenancy
        becomes real for shell sessions).  Falls through to in-
        process for sessions without a runner.

        Cgroup attach + runtime-limits plumbing migration (per plan
        §3.5 / peer-review M2) lands in a follow-on commit; for now
        the runner-side path uses the default cgroup attach the
        runner's ``set_runtime_limits`` already wires.
        """
        return self.wrap_executors_for_runner_forwarding({
            'shell_spawn': self._exec_spawn,
            'shell_input': self._exec_input,
            'shell_read': self._exec_read,
            'shell_control': self._exec_control,
            'shell_close': self._exec_close,
            'shell_list': self._exec_list,
        })

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for interactive shell tools."""
        return """INTERACTIVE SHELL SESSIONS (interactive_shell plugin):

STATE MODEL:
Sessions are STATEFUL and PERSISTENT. shell_spawn creates one process
that remains alive across multiple tool calls. You interact with it
using the session_id returned by shell_spawn. The process remembers
everything — previous input, output, state, working directory, variables.

WORKFLOW:
1. shell_spawn(command=...) → session_id + initial output
2. Read the output. Understand what the program is asking.
3. shell_input(session_id=..., input="response\\n") → next output
4. Repeat 2-3 until done.
5. shell_close(session_id=...)

TOOL ROLES:
- shell_spawn: Start a NEW process (once per command). Returns session_id.
- shell_input: Send input to an EXISTING session. Use for ALL interactions after spawn.
- shell_read: Check for new output without sending input.
- shell_control: Send Ctrl+C, Ctrl+D, etc.
- shell_close: Terminate a session.
- shell_list: See running sessions.

WHEN TO USE vs cli_based_tool:
These tools spawn a real PTY session — use them ONLY when the command
requires interactive back-and-forth input during execution. Do NOT use
shell_spawn for commands that simply run and produce output:
- Non-interactive commands (ls, grep, git status, make) → use cli_based_tool
- Commands with -y/--yes flags or piped stdin → use cli_based_tool
- Shell scripts that run to completion without prompts → use cli_based_tool
- Build commands (npm run build, cargo build, pytest) → use cli_based_tool
Use shell_spawn ONLY when the program will ask questions, prompt for
passwords, or present a REPL that requires you to type responses:
- Password prompts (ssh, sudo) → shell_spawn
- REPLs (python, node, psql) → shell_spawn
- Wizards (npm init, interactive installers) → shell_spawn
- Debuggers (gdb, pdb) → shell_spawn
If unsure → try cli_based_tool first; use shell_spawn only if it hangs
waiting for input that cli_based_tool cannot provide

COMMON MISTAKES:
- Calling shell_spawn to send input to a running session. shell_spawn
  starts a NEW process — use shell_input instead.
- Spawning multiple sessions for the same command. Spawn once, then use
  shell_input with the session_id for all subsequent interactions.
- Losing track of the session_id. Note the session_id from shell_spawn
  and reuse it consistently.

WORKSPACE BOUNDARY:
Sessions are confined to the session workspace, exactly as cli_based_tool
is. A spawn command or typed input naming a path outside it is REFUSED
with an error naming that path — the command does not run and nothing is
sent to the session. Re-issuing it unchanged will be refused again; use a
path inside the workspace, or ask the user to authorize the outside path.

ENVIRONMENT VARIABLES:
Spawned processes inherit session environment variables. When a command
needs credentials, use shell variable references — do NOT attempt to
echo, print, or retrieve their values. The user manages which env vars
are set. If a command fails due to a missing variable, report it.

IMPORTANT NOTES:
- Always end text input with \\n — that's pressing Enter
- For password prompts, the terminal won't echo what you type — that's normal
- If a process seems stuck, use shell_read() to check for new output
- Use shell_control(key="c-c") to interrupt, shell_control(key="c-d") for EOF
- Sessions are automatically cleaned up after inactivity or timeout
- Always shell_close() sessions when done to free resources"""

    def get_auto_approved_tools(self) -> List[str]:
        """Interactive shell tools require permission — return empty list."""
        # shell_list and shell_read are read-only but we still require
        # permission because they access active session state.
        return ['shell_list']

    def get_user_commands(self) -> List[UserCommand]:
        """No user-facing commands — all tools are model-invoked."""
        return []

    # --- Executors ---

    def _allocate_session_id(self, session_name: Optional[str]) -> str:
        """Pick an unused id for a new session.

        **Caller must hold ``self._lock``** — this reads ``_sessions`` and
        writes ``_session_counter``, and the id it returns is only unique
        for as long as that lock is held.  ``_exec_spawn`` is the single
        caller and registers the session under the returned id inside the
        same critical section.

        Args:
            session_name: The caller's preferred name, or ``None`` for an
                auto-generated ``session_<n>``.  A name already in use
                gains the lowest free ``_<suffix>``, so a model that
                re-uses a name gets a second session rather than an error
                (and the returned id tells it which it got).

        Returns:
            An id not currently present in ``self._sessions``.
        """
        if not session_name:
            session_id = f"session_{self._session_counter}"
            self._session_counter += 1
            return session_id

        if session_name not in self._sessions:
            return session_name

        suffix = 1
        while f"{session_name}_{suffix}" in self._sessions:
            suffix += 1
        return f"{session_name}_{suffix}"

    def _exec_spawn(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell_spawn: start a new interactive session.

        Creates a ``ShellSession`` using the platform-appropriate backend
        (see ``session.py`` module docstring).  On MSYS2 with the
        ``popen_spawn`` backend the child process has no PTY — ``isatty()``
        returns ``False`` and terminal dimensions are not forwarded.  Most
        REPLs and debuggers still work; password prompts may not hide input.

        Two gates run before anything is spawned (#722), in this order:

        1. :meth:`_confinement_refusal` — the deployment's posture on
           running a PTY with no kernel boundary.  First, because under
           ``require_confinement`` no command is acceptable, and because
           its WARNING should be emitted before a refusal that looks like
           an ordinary path error.
        2. :meth:`_containment_refusal` with ``on_parse_error="deny"`` —
           the same workspace check ``cli`` applies to a command, failing
           closed on a command that cannot be parsed.

        The session's working directory is the workspace root and is not
        model-controllable; ``ShellSession`` re-asserts that (#503).
        """
        command = args.get('command')
        if not command:
            return {'error': 'shell_spawn: command is required'}

        refusal = self._confinement_refusal()
        if refusal is not None:
            return refusal

        refusal = self._containment_refusal(
            command, 'shell_spawn', on_parse_error="deny"
        )
        if refusal is not None:
            return refusal

        session_name = args.get('session_name')
        rows = args.get('rows', 24)
        cols = args.get('cols', 80)

        with self._lock:
            # Check session limit
            alive_count = sum(
                1 for s in self._sessions.values() if s.is_alive
            )
            if alive_count >= self._max_sessions:
                return {
                    'error': (
                        f'Maximum concurrent sessions ({self._max_sessions}) '
                        f'reached. Close an existing session first.'
                    ),
                    'active_sessions': [
                        {'session_id': sid, 'command': s.command}
                        for sid, s in self._sessions.items()
                        if s.is_alive
                    ],
                }

            session_id = self._allocate_session_id(session_name)

        # AppArmor confinement (if any) is inherited from the parent
        # thread via fork+exec — see ToolExecutor.set_apparmor_context.
        # Phase 5 §5.10d — preexec_fn composes two callbacks between
        # fork() and exec():
        #   1. AppArmor child-profile transition (writes
        #      ``changeprofile <session>//child`` to
        #      /proc/self/attr/current).  The forked PTY child enters
        #      the ``//child`` sub-profile, dropping the escape-vector
        #      rules — a model-controlled shell that writes to
        #      attr/current gets EACCES.
        #   2. Cgroup attach (writes the forked child's PID to
        #      cgroup.procs), so memory.max / pids.max / cpu.weight
        #      apply from the first instruction of the new program.
        # Either callback may be None; the composite handles all
        # four (none, apparmor-only, cgroup-only, both).
        spawn_command = command

        self._trace(f"spawn: id={session_id}, cmd={command[:80]}, backend={_BACKEND}")

        # Resolve + create-if-absent the workspace venv (if configured) so the
        # spawned session runs with it activated (pip persists, imports resolve).
        venv_path = resolve_venv_path(self._workspace_venv, self._workspace_root)
        if venv_path:
            ensure_workspace_venv(venv_path)

        try:
            session = ShellSession(
                command=spawn_command,
                session_id=session_id,
                rows=rows,
                cols=cols,
                idle_timeout=self._idle_timeout,
                max_lifetime=self._max_lifetime,
                cwd=self._workspace_root,
                preexec_fn=self._build_subprocess_preexec_fn(),
                workspace_venv=venv_path,
                scrub_env=self._scrub_secret_env or None,
                # Same value as cwd, passed as the boundary rather than
                # assumed from it (#503): ShellSession verifies one
                # against the other, so the invariant survives a future
                # caller that computes cwd some other way.
                workspace_root=self._workspace_root,
            )

            # Read initial output (program banner, first prompt, etc.)
            initial_output = session.read_initial_output()

            with self._lock:
                self._sessions[session_id] = session

            self._stream_output(initial_output)

            result = {
                'session_id': session_id,
                'output': strip_ansi(initial_output),
                'is_alive': session.is_alive,
            }

            if not session.is_alive:
                result['note'] = (
                    'Process exited immediately. Check the command and output.'
                )

            # _telemetry: Convention-based telemetry
            result['_telemetry'] = {
                'jaato.shell.operation': 'spawn',
                'jaato.shell.session_id': session_id,
                'jaato.shell.command': command[:200],
                'jaato.shell.is_alive': session.is_alive,
            }

            self._trace(
                f"spawn: id={session_id} output_len={len(initial_output)} "
                f"alive={session.is_alive}"
            )
            if session.is_alive:
                return WithMetadata(result, {"continuation_id": session_id,
                                            "show_output": True})
            return result

        except Exception as exc:
            self._trace(f"spawn: FAILED: {exc}")
            return {'error': f'shell_spawn: {exc}'}

    def _exec_input(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell_input: send text and return response.

        The typed text is run through the same workspace containment as a
        spawn command, with one deliberate difference: it fails **open**
        on a parse failure (``on_parse_error="allow"``).  What reaches a
        live session is whatever the running program reads — Python, SQL,
        a password, a menu choice — so a string the shell grammar cannot
        model is the normal case here, where for a spawn command it is an
        evasion.  The check therefore catches the direct attempt
        (``cat /etc/shadow``) and claims nothing beyond it; the boundary
        that holds for a live PTY is kernel confinement (see the module
        docstring).

        Ordering: containment is checked **after** the session lookup, so
        a text aimed at a session that does not exist still reports the
        missing session — the fact the caller needs — rather than a path
        refusal about a session it never had.
        """
        session_id = args.get('session_id')
        text = args.get('input', '')

        if not session_id:
            return {'error': 'shell_input: session_id is required'}

        session = self._get_session(session_id)
        if session is None:
            return self._session_not_found(session_id)

        refusal = self._containment_refusal(
            text, 'shell_input', on_parse_error="allow"
        )
        if refusal is not None:
            refusal['is_alive'] = session.is_alive
            return refusal

        if not session.is_alive:
            return {
                'error': f'Session {session_id!r} has exited.',
                'is_alive': False,
            }

        self._trace(
            f"input: id={session_id} "
            f"text={text[:50]!r}{'...' if len(text) > 50 else ''}"
        )

        try:
            output = session.send_input(text)
            self._stream_output(output)
            result = {
                'output': strip_ansi(output),
                'is_alive': session.is_alive,
                # _telemetry: Convention-based telemetry
                '_telemetry': {
                    'jaato.shell.operation': 'input',
                    'jaato.shell.session_id': session_id,
                    'jaato.shell.input_len': len(text),
                    'jaato.shell.output_len': len(output),
                    'jaato.shell.is_alive': session.is_alive,
                },
            }
            if session.is_alive:
                return WithMetadata(result, {"continuation_id": session_id,
                                            "show_output": False})
            return result
        except Exception as exc:
            self._trace(f"input: id={session_id} FAILED: {exc}")
            return {'error': f'shell_input: {exc}', 'is_alive': session.is_alive}

    def _exec_read(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell_read: read pending output without sending."""
        session_id = args.get('session_id')
        timeout = args.get('timeout', 2.0)

        if not session_id:
            return {'error': 'shell_read: session_id is required'}

        session = self._get_session(session_id)
        if session is None:
            return self._session_not_found(session_id)

        self._trace(f"read: id={session_id} timeout={timeout}")

        try:
            output = session.read_output(timeout=timeout)
            self._stream_output(output)
            result = {
                'output': strip_ansi(output),
                'is_alive': session.is_alive,
            }
            if session.is_alive:
                return WithMetadata(result, {"continuation_id": session_id,
                                            "show_output": False,
                                            "show_popup": False})
            return result
        except Exception as exc:
            return {'error': f'shell_read: {exc}', 'is_alive': session.is_alive}

    def _exec_control(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell_control: send a control key."""
        session_id = args.get('session_id')
        key = args.get('key')

        if not session_id:
            return {'error': 'shell_control: session_id is required'}
        if not key:
            return {'error': 'shell_control: key is required'}

        session = self._get_session(session_id)
        if session is None:
            return self._session_not_found(session_id)

        if not session.is_alive:
            return {
                'error': f'Session {session_id!r} has exited.',
                'is_alive': False,
            }

        self._trace(f"control: id={session_id} key={key}")

        try:
            output = session.send_control(key)
            self._stream_output(output)
            result = {
                'output': strip_ansi(output),
                'is_alive': session.is_alive,
                # _telemetry: Convention-based telemetry
                '_telemetry': {
                    'jaato.shell.operation': 'control',
                    'jaato.shell.session_id': session_id,
                    'jaato.shell.key': key,
                    'jaato.shell.is_alive': session.is_alive,
                },
            }
            if session.is_alive:
                return WithMetadata(result, {"continuation_id": session_id,
                                            "show_output": False})
            return result
        except ValueError as exc:
            return {'error': str(exc)}
        except Exception as exc:
            return {'error': f'shell_control: {exc}', 'is_alive': session.is_alive}

    def _exec_close(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell_close: terminate a session."""
        session_id = args.get('session_id')

        if not session_id:
            return {'error': 'shell_close: session_id is required'}

        session = self._get_session(session_id)
        if session is None:
            return self._session_not_found(session_id)

        self._trace(f"close: id={session_id}")

        try:
            result = session.close()

            with self._lock:
                self._sessions.pop(session_id, None)

            self._trace(
                f"close: id={session_id} exit_status={result.get('exit_status')}"
            )
            # _telemetry: Convention-based telemetry
            result['_telemetry'] = {
                'jaato.shell.operation': 'close',
                'jaato.shell.session_id': session_id,
                'jaato.shell.exit_status': result.get('exit_status'),
            }
            return WithMetadata(result, {"show_output": True})

        except Exception as exc:
            # Clean up even on error
            with self._lock:
                self._sessions.pop(session_id, None)
            return {'error': f'shell_close: {exc}'}

    def _exec_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell_list: list active sessions."""
        with self._lock:
            sessions = []
            for sid, session in self._sessions.items():
                sessions.append({
                    'session_id': sid,
                    'command': session.command,
                    'is_alive': session.is_alive,
                    'age_seconds': round(session.age_seconds, 1),
                    'idle_seconds': round(session.idle_seconds, 1),
                })

        return {'sessions': sessions, 'count': len(sessions)}

    # --- Helpers ---

    def set_tool_output_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set the callback for streaming output during tool execution.

        Called by ToolExecutor before/after each tool call. Also supports
        thread-local fallback for parallel execution.
        """
        self._tool_output_callback = callback

    def _stream_output(self, text: str) -> None:
        """Forward raw output text to the tool output callback if available.

        Splits on \\n only (not \\r) to preserve carriage returns and ANSI
        escape sequences within lines for proper terminal emulation by pyte.
        """
        callback = get_current_tool_output_callback() or self._tool_output_callback
        if callback and text:
            parts = text.split('\n')
            # Remove trailing empty from trailing \n (avoids extra blank line)
            if parts and not parts[-1]:
                parts.pop()
            for chunk in parts:
                callback(chunk)

    def _get_session(self, session_id: str) -> Optional[ShellSession]:
        """Look up a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def _session_not_found(self, session_id: str) -> Dict[str, Any]:
        """Return a helpful error for missing sessions."""
        with self._lock:
            available = [
                sid for sid, s in self._sessions.items() if s.is_alive
            ]
        result: Dict[str, Any] = {
            'error': f'No session with id {session_id!r}.',
        }
        if available:
            result['available_sessions'] = available
            result['hint'] = 'Use shell_list to see all sessions.'
        else:
            result['hint'] = 'No active sessions. Use shell_spawn to start one.'
        return result

    # --- Reaper thread ---

    def _start_reaper(self) -> None:
        """Start the background reaper thread."""
        if self._reaper_thread is not None:
            return

        self._reaper_stop.clear()
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop,
            daemon=True,
            name="interactive-shell-reaper",
        )
        self._reaper_thread.start()
        self._trace("reaper: started")

    def _stop_reaper(self) -> None:
        """Stop the background reaper thread."""
        self._reaper_stop.set()
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=5.0)
            self._reaper_thread = None
        self._trace("reaper: stopped")

    def _reaper_loop(self) -> None:
        """Periodically check for expired or dead sessions."""
        while not self._reaper_stop.wait(timeout=REAPER_INTERVAL):
            self._reap_sessions()

    def _reap_sessions(self) -> None:
        """Close sessions that are expired, idle, or dead."""
        to_reap = []

        with self._lock:
            for sid, session in self._sessions.items():
                reason = None
                if not session.is_alive:
                    reason = "process exited"
                elif session.is_expired:
                    reason = f"lifetime exceeded ({self._max_lifetime}s)"
                elif session.idle_seconds > self._max_idle:
                    reason = f"idle too long ({self._max_idle}s)"

                if reason:
                    to_reap.append((sid, session, reason))

        for sid, session, reason in to_reap:
            self._trace(f"reaper: closing {sid} ({reason})")
            try:
                session.close()
            except Exception:
                pass
            with self._lock:
                self._sessions.pop(sid, None)


def create_plugin() -> InteractiveShellPlugin:
    """Factory function to create the interactive shell plugin instance."""
    return InteractiveShellPlugin()
