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
from typing import Callable, ContextManager, List, Optional

logger = logging.getLogger(__name__)


class AppArmorManager:
    """Manages AppArmor profiles for per-session workspace confinement.

    Lifecycle (called by the WebSocket server or workspace provisioner):

    1. ``provision_profile(session_id, workspace_path)`` — renders a
       profile from the template, writes it to the profile directory,
       and loads it via ``apparmor_parser -r``.
    2. ``wrap_command(session_id, command)`` — prepends ``aa-exec`` to a
       command so it runs under the session's profile.
    3. ``teardown_profile(session_id)`` — unloads and removes the profile.

    All methods are safe to call even when AppArmor is unavailable; they
    degrade to no-ops.
    """

    # AppArmor profile template.  Placeholders are filled per-session by
    # ``_render_profile()``.
    PROFILE_TEMPLATE = '''\
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

  # ---- temp files scoped to session ----
  /tmp/jaato-{session_id}-** rw,

  # ---- deny sibling workspaces ----
  deny {sessions_root}/ rw,
  deny {sessions_root}/** rw,

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
    # Command wrapping
    # ------------------------------------------------------------------

    def wrap_command(self, session_id: str, command: List[str]) -> List[str]:
        """Wrap a command to run under the session's AppArmor profile.

        Args:
            session_id: Session whose profile to use.
            command: The command as a list of arguments.

        Returns:
            ``["aa-exec", "-p", profile_name, "--"] + command``
            if AppArmor is available, otherwise the original command
            unchanged.
        """
        if not self.is_available():
            return command

        profile_name = self.get_profile_name(session_id)
        return ["aa-exec", "-p", profile_name, "--"] + command

    def wrap_shell_command(self, session_id: str, command: str) -> str:
        """Wrap a shell command string to run under AppArmor.

        For use with ``shell=True`` subprocess calls or pexpect, where
        the command is a single string rather than an argv list.

        Args:
            session_id: Session whose profile to use.
            command: The shell command string.

        Returns:
            ``aa-exec -p <profile> -- /bin/sh -c '<command>'``
            if AppArmor is available, otherwise the original string.
        """
        if not self.is_available():
            return command

        profile_name = self.get_profile_name(session_id)
        # Escape single quotes in the command for safe shell embedding
        escaped = command.replace("'", "'\"'\"'")
        return f"aa-exec -p {profile_name} -- /bin/sh -c '{escaped}'"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_profile_name(self, session_id: str) -> str:
        """Return the AppArmor profile name for a session."""
        return f"jaato-ws-{session_id}"

    def _render_profile(self, session_id: str, workspace_path: str) -> str:
        """Render the profile template with session-specific values.

        The template uses Python ``str.format()`` placeholders.  The
        workspace path's deny rule uses the ``sessions_root`` to block
        access to sibling workspaces while allowing the specific session's
        own workspace via the more-specific allow rule.
        """
        return self.PROFILE_TEMPLATE.format(
            session_id=session_id,
            workspace_path=workspace_path,
            venv_path=str(self._venv_path),
            sessions_root=str(self._sessions_root),
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
