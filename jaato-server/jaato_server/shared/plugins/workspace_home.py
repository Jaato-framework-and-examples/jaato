"""Workspace-scoped HOME for model-driven tool subprocesses (#1225).

A runner's subprocesses -- ``cli`` commands, ``interactive_shell`` PTY
children, the notebook kernel -- otherwise inherit the daemon's ``HOME``
(``/root`` on a root daemon), so every tool that writes to ``~`` shares one
home across every tenant and workspace on the daemon: ``~/.gitconfig`` (and
``gh auth setup-git`` rewrites it for everyone), ``~/.config/gh``,
``~/.npmrc``, pip and npm caches, shell history, ``~/.ssh/known_hosts``.
Under AppArmor most of ``/root`` is probably denied too, so the same tools
fail in confusing ways.

A ``workspace_home`` path, shaped exactly like ``workspace_venv``
(:mod:`shared.plugins.workspace_home`'s sibling
:mod:`shared.plugins.workspace_venv`), gives each workspace its own home
INSIDE the workspace (``<ws>/.home/`` by default): a relative path resolved
against the session workspace root, absolute allowed, a relative path with
no workspace root refused (no silent fallback to cwd).

Only the environment handed to the SUBPROCESS is changed -- ``HOME`` plus
the four XDG base-directory variables.  The runner's own ``HOME`` is
untouched, so jaato's ``~/.jaato`` lookups and the ``${HOME}`` profile
expansion variable keep their meaning (§8).

**Secrets never live here.**  It persists on disk and the model-driven file
tools can read it; tokens ride the session env, never the workspace home.

Where each piece runs
----------------------
- The **daemon** decides the effective home (``effective_workspace_home``),
  mirrors it across the surfaces that honour it
  (``inject_workspace_home`` -- the shape ``inject_scrub_secret_env`` uses),
  and creates the directory + its ``*`` gitignore BEFORE the runner spawns
  (``ensure_workspace_home_dir``), as it does the session tmpdir (#1171):
  a confined runner must not be relied on to create it.
- The **plugins** (``cli`` / ``interactive_shell`` / ``notebook``) resolve
  the raw value against their workspace root (``resolve_home_path``) and
  point the subprocess env at it (``apply_home_to_env``).
"""

import logging
import os
from typing import List, MutableMapping, Optional

from .jaato_tools_path import append_path_entry

logger = logging.getLogger(__name__)


# The framework default home, used for workspaces the daemon manages under
# ``workspace_root``.  A relative path so it resolves inside the workspace.
DEFAULT_WORKSPACE_HOME = ".home"

# The surfaces that redirect ``HOME``.  The author-facing knob lives under
# ``plugin_configs.cli.workspace_home`` (§8); the value is mirrored into the
# other two namespaces so each plugin reads it from its own config, the same
# per-surface shape ``scrub_secret_env`` has.
HOME_SURFACES = ("cli", "interactive_shell", "notebook")

_CONFIG_KEY = "workspace_home"


def resolve_home_path(
    raw: Optional[str], workspace_root: Optional[str],
) -> Optional[str]:
    """Resolve a configured ``workspace_home`` value to an absolute path.

    Mirrors :func:`shared.plugins.workspace_venv.resolve_venv_path`.

    Args:
        raw: The raw ``workspace_home`` config value.  Empty / None / blank
            means the feature is off for this session.
        workspace_root: The session workspace root, used to resolve a
            relative home path.

    Returns:
        The absolute home path, or ``None`` when the feature is off.

    Raises:
        ValueError: If a relative path is given but ``workspace_root`` is
            unset (no silent fallback to cwd -- the target would be
            non-deterministic).
    """
    if not raw or not raw.strip():
        return None
    path = os.path.expanduser(raw.strip())
    if os.path.isabs(path):
        return os.path.realpath(path)
    if not workspace_root:
        raise ValueError(
            f"workspace_home={raw!r} is relative but no workspace_root is "
            "set; provide an absolute path or run within a session workspace")
    return os.path.realpath(os.path.join(workspace_root, path))


def apply_home_to_env(
    env: MutableMapping[str, str], home_path: str,
) -> None:
    """Point ``HOME`` and the XDG base dirs at ``home_path`` in ``env``.

    The XDG variables are set to subdirectories under the home, matching the
    XDG base-directory spec's own defaults relative to ``HOME``.  Setting
    them explicitly is not redundant: a daemon whose environment exports
    ``XDG_CONFIG_HOME=/root/.config`` would otherwise have that value
    inherited by the subprocess and win over the redirected ``HOME``.

    ``<home>/.local/bin`` is also APPENDED to ``PATH`` (#1273), so a program
    a user-level installer put there resolves by name on the next command.

    Mutates ``env`` in place.  Nothing on disk is created here -- the tools
    make their own XDG subdirectories on demand, under the ``<home>`` the
    daemon created.
    """
    env["HOME"] = home_path
    env["XDG_CONFIG_HOME"] = os.path.join(home_path, ".config")
    env["XDG_CACHE_HOME"] = os.path.join(home_path, ".cache")
    env["XDG_DATA_HOME"] = os.path.join(home_path, ".local", "share")
    env["XDG_STATE_HOME"] = os.path.join(home_path, ".local", "state")
    # #1273: ``uv tool install``, ``pipx`` and ``pip install --user`` put the
    # programs they install in ``~/.local/bin``.  With HOME redirected that is
    # ``<home>/.local/bin``, which was on nobody's PATH, so a tool installed
    # by one command was "not found" by the next.  Appended, so it can never
    # shadow a host binary; per workspace, so it does not leak across them.
    append_path_entry(env, local_bin_dir(home_path))


def local_bin_dir(home_path: str) -> str:
    """``<home>/.local/bin``: where user-level installers put programs."""
    return os.path.join(home_path, ".local", "bin")


def home_exec_apparmor_rules(
    workspace_home_raw: Optional[str], workspace_path: Optional[str],
) -> List[str]:
    """AppArmor ``ix`` grants for programs installed under the workspace home.

    The workspace is ``rwkl`` with no exec, so a program ``uv tool install``
    or ``pipx`` put under the home could be written and not run.  Two
    grants, because exec is mediated on the RESOLVED path: ``.local/bin/*``
    for files installed there directly (``pip install --user`` scripts), and
    ``.local/share/**/bin/*`` for the targets the ``uv tool`` and ``pipx``
    symlinks in ``.local/bin`` resolve to.  The same trust class as the
    workspace venv's ``bin/*`` grant: a location the model can already write
    executables into.  Empty when the home is off or unresolvable.
    """
    try:
        home_path = resolve_home_path(workspace_home_raw, workspace_path)
    except ValueError:
        return []
    if not home_path:
        return []
    return [
        f"{local_bin_dir(home_path)}/* ix,",
        f"{os.path.join(home_path, '.local', 'share')}/**/bin/* ix,",
    ]


def is_daemon_managed(
    workspace_path: Optional[str], managed_workspace_root: Optional[str],
) -> bool:
    """Public name for :func:`_is_daemon_managed` (shared with ``workspace_venv``)."""
    return _is_daemon_managed(workspace_path, managed_workspace_root)


def _is_daemon_managed(
    workspace_path: Optional[str], managed_workspace_root: Optional[str],
) -> bool:
    """True iff ``workspace_path`` is a workspace the daemon manages.

    A workspace under the WS server's ``workspace_root`` is one the daemon
    provisions; a session pointed at a user's own checkout (IPC / user-CWD)
    passes ``managed_workspace_root=None`` and is never managed -- so it does
    not grow a ``.home/`` unless the profile explicitly asks for one.
    """
    if not workspace_path or not managed_workspace_root:
        return False
    try:
        ws = os.path.realpath(workspace_path)
        root = os.path.realpath(managed_workspace_root)
    except OSError:
        return False
    return ws == root or ws.startswith(root + os.sep)


def effective_workspace_home(
    cli_config: Optional[dict],
    workspace_path: Optional[str],
    managed_workspace_root: Optional[str],
) -> Optional[str]:
    """The raw ``workspace_home`` value for this session, or ``None`` (off).

    - An explicit ``plugin_configs.cli.workspace_home`` is honoured verbatim
      (a blank value opts out, exactly as ``workspace_venv`` does), and the
      managed default is NOT applied on top of it.
    - Otherwise the framework default (``.home``) applies iff the workspace
      is one the daemon manages under ``managed_workspace_root`` (§8:
      "on for workspaces the daemon manages under workspace_root, opt-in
      elsewhere").
    """
    if cli_config is not None and _CONFIG_KEY in cli_config:
        raw = cli_config.get(_CONFIG_KEY)
        return str(raw) if raw and str(raw).strip() else None
    if _is_daemon_managed(workspace_path, managed_workspace_root):
        return DEFAULT_WORKSPACE_HOME
    return None


def inject_workspace_home(
    plugin_configs: dict,
    workspace_path: Optional[str],
    managed_workspace_root: Optional[str],
) -> Optional[str]:
    """Fold the effective ``workspace_home`` into the surfaces that honour it.

    The author-facing knob is ``plugin_configs.cli.workspace_home`` (§8);
    ``interactive_shell`` and the notebook kernel read the value from their
    own namespaces, so it is mirrored there -- the shape
    ``inject_scrub_secret_env`` uses across ``cli`` / ``interactive_shell`` /
    ``mcp``.  An explicit per-surface value is never overwritten.

    Args:
        plugin_configs: The envelope's plugin-config dict, mutated in place.
        workspace_path: The session workspace.
        managed_workspace_root: The daemon's provisioning root, or ``None``
            for a session pointed at a user's own checkout.

    Returns:
        The raw value applied, or ``None`` when the feature is off for this
        session (for a caller that also needs to create the directory).
    """
    if not isinstance(plugin_configs, dict):
        return None
    cli_cfg = plugin_configs.get("cli")
    cli_cfg = cli_cfg if isinstance(cli_cfg, dict) else None
    raw = effective_workspace_home(
        cli_cfg, workspace_path, managed_workspace_root,
    )
    if not raw:
        return None
    for surface in HOME_SURFACES:
        section = dict(plugin_configs.get(surface) or {})
        section.setdefault(_CONFIG_KEY, raw)
        plugin_configs[surface] = section
    return raw


def ensure_workspace_home(home_path: str) -> None:
    """Create the workspace home and a ``*`` gitignore, best-effort.

    The daemon creates it before the runner spawns, as it does the session
    tmpdir (#1171): the confined runner is not relied on to make it.  The
    nested ``.gitignore`` (``*``) keeps the whole home out of git without
    editing the user's own ``.gitignore``.

    Best-effort and audible: a failure logs at WARNING and returns rather
    than taking down a session that would otherwise run.
    """
    try:
        os.makedirs(home_path, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "workspace_home: could not create home dir %s (%s: %s); "
            "model-driven subprocesses will fall back to the daemon's HOME",
            home_path, type(exc).__name__, exc,
        )
        return
    gitignore = os.path.join(home_path, ".gitignore")
    try:
        if not os.path.exists(gitignore):
            with open(gitignore, "w", encoding="utf-8") as handle:
                handle.write("*\n")
    except OSError as exc:
        logger.warning(
            "workspace_home: could not write %s (%s: %s); the home may be "
            "committed to git unless the workspace .gitignore covers it",
            gitignore, type(exc).__name__, exc,
        )


def ensure_workspace_home_dir(
    profile: object,
    workspace_path: Optional[str],
    managed_workspace_root: Optional[str],
) -> Optional[str]:
    """Resolve and create this session's workspace home, daemon-side.

    Reads the explicit ``plugin_configs.cli.workspace_home`` off *profile*
    (``None`` for a profile-less session) and applies the managed default
    otherwise, then creates ``<ws>/<home>/`` + its ``*`` gitignore.  The
    directory is the one the plugins later point the subprocess env at.

    Returns the absolute home path created, or ``None`` when the feature is
    off / the value is relative with no workspace (the plugin surfaces the
    latter as a config error when it runs).
    """
    plugin_configs = getattr(profile, "plugin_configs", None) or {}
    cli_cfg = plugin_configs.get("cli") if isinstance(plugin_configs, dict) else None
    cli_cfg = cli_cfg if isinstance(cli_cfg, dict) else None
    raw = effective_workspace_home(
        cli_cfg, workspace_path, managed_workspace_root,
    )
    if not raw:
        return None
    try:
        home_path = resolve_home_path(raw, workspace_path)
    except ValueError:
        # Relative with no workspace root -- the plugin raises the same
        # ValueError when it resolves, which is where it belongs.
        return None
    if home_path:
        ensure_workspace_home(home_path)
    return home_path
