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

import logging
import os
import platform
import shutil
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, ContextManager, Optional

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
    _TEMPLATE_VERSION = 2

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
  @{{HOME}}/.jaato/keybindings.json r,
  @{{HOME}}/.jaato/theme.json       r,
  @{{HOME}}/.jaato/gc.json          r,

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
  # Allow the framework to restore the unconfined state on tool exit.
  # Without this, apparmor_confine().__exit__ silently fails to restore,
  # and the thread stays trapped in this profile across tool calls —
  # breaking framework-level tools (like spawn_subagent) that rely on
  # opting out of confinement via TRAIT_FRAMEWORK_LEVEL.
  #
  # Note: this does NOT let agent code escape confinement.  Writes to
  # /proc/self/attr/current are gated by the change_profile capability,
  # which only the framework's apparmor_confine context manager invokes.
  # Agent tool code can't write to /proc/self/attr/current because file
  # writes there are not in the workspace allow list.
  change_profile -> unconfined,
}}
'''

    def __init__(
        self,
        workspace_root: str,
        venv_path: Optional[str] = None,
        profile_dir: str = "/etc/apparmor.d/jaato",
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
    ) -> bool:
        """Create and load an AppArmor profile for a session.

        Writes the rendered profile to
        ``{profile_dir}/jaato-ws-{session_id}`` and loads it with
        ``apparmor_parser -r``.

        Args:
            session_id: Session identifier (used in profile name).
            workspace_path: Absolute path to the session's workspace
                directory.

        Returns:
            True on success, False on failure (logged, not raised).
        """
        if not self.is_available():
            return False

        profile_name = self.get_profile_name(session_id)
        profile_path = self._profile_dir / profile_name
        profile_content = self._render_profile(session_id, workspace_path)

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

        logger.info("Removed AppArmor profile %s", profile_name)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_profile_name(self, session_id: str) -> str:
        """Return the AppArmor profile name for a session."""
        return f"jaato-ws-{session_id}"

    def _render_profile(self, session_id: str, workspace_path: str) -> str:
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

        return self.PROFILE_TEMPLATE.format(
            template_version=self._TEMPLATE_VERSION,
            session_id=session_id,
            workspace_path=workspace_path,
            venv_path=str(self._venv_path),
            source_root=str(self._source_root),
            premium_rules=premium_rules,
        )


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

    Writes *profile_name* to ``/proc/self/task/<tid>/attr/current`` on
    entry, which transitions the calling OS thread into that profile.
    On exit, writes ``unconfined`` to restore the thread.

    If the write fails (AppArmor not available, profile not loaded,
    insufficient privileges), logs a warning and proceeds without
    confinement — never raises.

    Args:
        profile_name: The AppArmor profile to confine to
            (e.g. ``"jaato-ws-20260405_123456"``).
    """
    attr_path = _get_thread_attr_path()
    confined = False
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
            except (OSError, PermissionError):
                # Cannot restore — thread stays confined until it exits.
                # This is safe: the profile allows the session's workspace,
                # and the thread will be reused for the same session.
                logger.warning(
                    "AppArmor: could not restore unconfined for thread %d",
                    threading.get_native_id(),
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
