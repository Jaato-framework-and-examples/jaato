"""Workspace-scoped virtualenv provisioning for tool subprocesses.

When a tool plugin (``cli`` / ``interactive_shell`` / ``notebook``) is
configured with a ``workspace_venv`` path, its subprocesses run with that
venv **activated** so the model's in-tool ``pip install X`` persists to the
venv and a later ``import X`` resolves.  The venv lives INSIDE the workspace
(the confined runner has rw there; the runner's own base environment stays
read-only).

Why activation and not a bare ``PYTHONPATH`` prepend
----------------------------------------------------
A bare ``PYTHONPATH`` prepend under the *base* interpreter makes ``import X``
resolve once ``X`` is present, but a bare ``pip install X`` under that same
base interpreter targets the base environment (read-only under confinement →
denied) or ``--user`` — NOT the venv.  For the install-then-import cycle to
work end-to-end the subprocess must run with the venv activated: the venv
``bin`` dir ahead on ``PATH`` (so ``pip`` / ``python`` resolve to the venv)
plus ``VIRTUAL_ENV``, and for the notebook kernel the venv interpreter itself.
The site-packages dir is *also* prepended to ``PYTHONPATH`` — this is the
symmetric half of the in-process host-tool import contract (the client
prepends the SAME site-packages to its own ``sys.path`` so a host tool imports
the dep the runner installed).

What the tool-venv can import from the runner (#1322)
----------------------------------------------------
Exactly one thing: ``pip``.  The venv is created ``--without-pip`` (venv
creation runs ``ensurepip``, which the confined runner could not), so the
``bin/pip`` shim runs the RUNNER's pip.  ``_write_pip_bridge`` makes that
possible with a ``.pth`` naming one directory inside the venv that holds one
symlink, ``pip`` -> the runner's pip package.  pip vendors its dependencies,
so nothing else comes with it.

It used to list the runner's whole site-packages plus jaato's source roots,
for the notebook kernel's sake.  That was not harmless on ``cli`` /
``interactive_shell``, which run the MODEL's commands:

- a session developing jaato imported the daemon's installed ``jaato_server``
  instead of the checkout it was editing, and ``pip install -e <checkout>``
  lost too (an editable install's finder is consulted after the path search);
- pip in the tool-venv saw the daemon's packages as installed.

The notebook kernel is the one process that needs jaato: it IS jaato's own
module (``kernel_main``) run under the tool-venv interpreter.  It now gets
:func:`kernel_import_dirs` at its own launch, in argv rather than
``PYTHONPATH``, so a cell's ``!python -m pytest`` starts as clean as a
``cli`` command does.  See ``SubprocessKernelBackend._kernel_argv``.

Why those dirs are plain paths and not ``site.addsitedir``: ``addsitedir``
would run every editable ``.pth`` finder in the base dir, dragging unrelated
src dirs (jaato_premium, a client's own package) in with it.

``--system-site-packages`` is the other way in, and it is only safe when the
runner is itself a venv (see :func:`_system_site_packages_are_the_runners`).
A tool-venv made from a runner VENV resolves its base to the interpreter that
venv was made from, so the system site holds OS packages and not jaato.  A
runner installed straight into a base interpreter -- a container's system
Python, a CI image -- has jaato in that interpreter's own site-packages, so
``--system-site-packages`` would hand the tool-venv the daemon's whole
install, the #1322 defect by another door.  In that case the flag is not
passed, and an existing venv's ``pyvenv.cfg`` is switched off on the next
``ensure``.

Contract
--------
- Empty / unset path = feature OFF.  Outside daemon-managed workspaces there
  is no implicit default venv — the path is the explicit agreement between
  the tool subprocess and the client's in-process import path.
- **Managed default (#1274).**  A workspace the daemon manages under the WS
  server's ``workspace_root`` gets :data:`DEFAULT_WORKSPACE_VENV`, the rule
  #1225 applies to ``workspace_home``.  Those workspaces are profile-less
  (a bare ``.env``), so no ``plugin_configs`` channel reaches them, and
  without a venv the model's ``pip install`` runs the HOST's pip: as root
  against the system Python on a root daemon, or into PEP 668's refusal.
  An explicit ``plugin_configs.cli.workspace_venv`` (``""`` opts out) wins,
  and an explicit per-surface value is never overwritten.  Folded daemon-side
  by :func:`inject_workspace_venv`, both into the envelope and into the
  AppArmor rule resolution, so the ``ix`` grant follows the venv.
- Create-if-absent is idempotent, but the pip bridge is refreshed on every
  ``ensure`` (so a venv created elsewhere, or by a release that bridged the
  whole runner, is brought to spec).
- Relative paths resolve against the session workspace root; a relative path
  with no workspace root is a configuration error (raised, not defaulted).
"""

import glob
import importlib
import importlib.util
import logging
import os
import site
import subprocess
import sys
from typing import List, MutableMapping, Optional

logger = logging.getLogger(__name__)


# The ``.pth`` that makes the runner's pip importable in the tool-venv.  The
# name predates #1322, when it bridged the runner's whole site-packages; it is
# kept so the next ``ensure`` overwrites that file in an existing venv rather
# than leaving it beside a new one.
_BRIDGE_PTH = "_jaato_runner_bridge.pth"

# Directory inside the venv holding the single ``pip`` symlink the ``.pth``
# names.  Not under site-packages, so pip never treats it as an installed
# distribution.
_PIP_BRIDGE_DIR = "jaato-pip"

# The venv daemon-managed workspaces get when nothing says otherwise (#1274).
# Under ``.jaato/`` so the scaffolded ``.gitignore`` block (``.jaato/*`` plus
# re-included authored dirs) already keeps it out of git; the workspace
# monitor keeps it out of the Files panel.
DEFAULT_WORKSPACE_VENV = ".jaato/tool-venv"

# The surfaces that run model-driven Python and honour ``workspace_venv``.
VENV_SURFACES = ("cli", "interactive_shell", "notebook")

_VENV_CONFIG_KEY = "workspace_venv"


# AppArmor rules that let a confined tool run ``pip`` at all: pip builds its
# HTTP User-Agent via the ``distro`` module, which reads the OS-identification
# files below.  Without these reads EVERY ``pip install`` in a confined runner
# crashes constructing the UA header (PermissionError on /etc/debian_version)
# before any network I/O.  Contributed via ``get_apparmor_rules`` by the tools
# that can run pip (cli / interactive_shell / notebook), so the grant is scoped
# to sessions that load one of them — least-privilege vs the core template.
# The files are world-readable OS metadata (distro name / version / codename).
PIP_APPARMOR_RULES: List[str] = [
    "/etc/os-release      r,",
    "/usr/lib/os-release  r,",
    "/etc/lsb-release     r,",
    "/etc/debian_version  r,",
    "/etc/*-release       r,",
]


def resolve_venv_path(raw: Optional[str], workspace_root: Optional[str]) -> Optional[str]:
    """Resolve a configured ``workspace_venv`` value to an absolute path.

    Args:
        raw: The raw ``workspace_venv`` config value.  Empty / None / blank
            means the feature is off.
        workspace_root: The session workspace root, used to resolve a relative
            venv path.

    Returns:
        The absolute venv path, or ``None`` when the feature is off.

    Raises:
        ValueError: If a relative path is given but ``workspace_root`` is unset
            (no silent fallback to cwd — the target would be non-deterministic).
    """
    if not raw or not raw.strip():
        return None
    path = os.path.expanduser(raw.strip())
    if os.path.isabs(path):
        return os.path.realpath(path)
    if not workspace_root:
        raise ValueError(
            f"workspace_venv={raw!r} is relative but no workspace_root is set; "
            "provide an absolute path or run within a session workspace")
    return os.path.realpath(os.path.join(workspace_root, path))


def _bin_dir(venv_path: str) -> str:
    """The venv's executable directory (``bin`` on POSIX, ``Scripts`` on nt)."""
    return os.path.join(venv_path, "Scripts" if os.name == "nt" else "bin")


def venv_python(venv_path: str) -> str:
    """Absolute path to the venv's Python interpreter."""
    exe = "python.exe" if os.name == "nt" else "python"
    return os.path.join(_bin_dir(venv_path), exe)


def venv_site_packages(venv_path: str) -> Optional[str]:
    """The venv's ``site-packages`` directory, or ``None`` if not found.

    POSIX venvs place it at ``<venv>/lib/pythonX.Y/site-packages``; Windows at
    ``<venv>/Lib/site-packages``.  Resolved by glob so the exact ``X.Y`` need
    not be known by the caller.
    """
    if os.name == "nt":
        candidate = os.path.join(venv_path, "Lib", "site-packages")
        return candidate if os.path.isdir(candidate) else None
    matches = sorted(glob.glob(os.path.join(venv_path, "lib", "python*", "site-packages")))
    return matches[0] if matches else None


def runner_site_dirs() -> List[str]:
    """The current (runner) interpreter's site-package directories.

    These are the dirs where jaato itself is installed — either as a normal
    package (wheel install) or via a PEP 660 editable ``.pth`` that registers
    an import finder.  Handed to the notebook kernel's launch (see
    :func:`kernel_import_dirs`) and to nothing else in the tool-venv.
    """
    dirs: List[str] = list(site.getsitepackages())
    user = site.getusersitepackages()
    if user:
        dirs.append(user)
    seen = set()
    out: List[str] = []
    for d in dirs:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def jaato_source_dirs() -> List[str]:
    """Source roots of the jaato packages the notebook kernel imports.

    The subprocess kernel (``jaato_server.shared.plugins.notebook.kernel_main``)
    imports ``jaato_server`` (jaato-server) and ``jaato_sdk`` (its
    ``tool_stubs`` needs ``ToolSchema``, which pulls in ``pydantic``).  For an
    **editable** install these are the external src dirs (e.g.
    ``.../jaato-server``, ``.../jaato-sdk``); for a **wheel** install they
    resolve into the base site-packages (deduped by the caller).  Deliberately
    NOT the whole base site-packages — that would surface UNRELATED editable
    installs (jaato_premium, a client's own editable package).

    ``os.path.dirname(mod.__path__[0])`` is the directory that CONTAINS the
    package, which is what a bridged venv needs on ``sys.path`` to
    ``import jaato_server`` / ``import jaato_sdk``.
    """
    dirs: List[str] = []
    for mod_name in ("jaato_server", "jaato_sdk"):
        mod = importlib.import_module(mod_name)
        dirs.append(os.path.dirname(mod.__path__[0]))
    return dirs


def kernel_import_dirs() -> List[str]:
    """Dirs the notebook kernel appends to ``sys.path`` (order-preserving, existing).

    The runner's site-packages (``pydantic`` and the other deps the kernel's
    ``tool_stubs`` import) plus the jaato source roots.  Used ONLY for the
    kernel's launch, never for the tool-venv as a whole (#1322).
    """
    seen = set()
    out: List[str] = []
    for d in runner_site_dirs() + jaato_source_dirs():
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def runner_pip_dir() -> Optional[str]:
    """The runner's ``pip`` package directory, or ``None`` if it has none.

    Located without importing pip.  A runner venv built without pip (``uv``
    creates them that way) has none, and then neither does the tool-venv.
    """
    spec = importlib.util.find_spec("pip")
    locations = list(spec.submodule_search_locations or []) if spec else []
    return locations[0] if locations and os.path.isdir(locations[0]) else None


def _write_pip_bridge(venv_path: str) -> None:
    """(Re)write the ``.pth`` that makes the runner's pip importable, and only it.

    ``<venv>/jaato-pip/pip`` is a symlink to the runner's pip package and the
    ``.pth`` names ``<venv>/jaato-pip``.  A ``.pth`` line that is not an
    ``import`` is appended to ``sys.path`` verbatim, after the venv's own
    site-packages, so a pip the model installs into the venv takes priority.
    A symlink rather than a copy keeps it tracking the runner's pip.

    Refreshed on every ``ensure``.  That is also what migrates a venv written
    before #1322, whose ``.pth`` under the same name listed the runner's whole
    site-packages.
    """
    site_dir = venv_site_packages(venv_path)
    if site_dir is None:
        raise RuntimeError(
            f"workspace venv at {venv_path} has no site-packages directory; "
            "cannot bridge the runner's pip")
    bridge_dir = os.path.join(venv_path, _PIP_BRIDGE_DIR)
    os.makedirs(bridge_dir, exist_ok=True)
    link = os.path.join(bridge_dir, "pip")
    pip_dir = runner_pip_dir()
    lines: List[str] = []
    try:
        if pip_dir is None:
            if os.path.lexists(link):
                os.unlink(link)
            logger.warning(
                "workspace_venv: the runner has no pip, so the tool-venv at "
                "%s has none either; `pip` there will fail", venv_path)
        else:
            if not (os.path.islink(link) and os.readlink(link) == pip_dir):
                tmp = f"{link}.tmp-{os.getpid()}"
                if os.path.lexists(tmp):
                    os.unlink(tmp)
                os.symlink(pip_dir, tmp)
                os.replace(tmp, link)
            lines.append(bridge_dir)
    except OSError as exc:
        logger.warning(
            "workspace_venv: could not link the runner's pip into %s (%s); "
            "`pip` there will fail", venv_path, exc)
        lines = []
    with open(os.path.join(site_dir, _BRIDGE_PTH), "w", encoding="utf-8") as f:
        f.write("".join(f"{ln}\n" for ln in lines))


def _system_site_packages_are_the_runners(base_python: Optional[str] = None) -> bool:
    """Whether a venv made from *base_python* would see the runner's install.

    True when the interpreter is a BASE interpreter (not a venv): its own
    site-packages are what ``--system-site-packages`` exposes, and that is
    where the runner's jaato and dependencies live.  False when it is a venv:
    a venv made from it resolves its system site to the interpreter IT was
    made from, which does not hold the runner's packages (#1322).

    *base_python* ``None`` (or this process's own interpreter) is answered
    in-process; any other interpreter is asked.
    """
    if base_python in (None, sys.executable):
        return sys.prefix == sys.base_prefix
    try:
        out = subprocess.run(
            [base_python, "-c", "import sys; print(sys.prefix == sys.base_prefix)"],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return True   # unknown: the safe answer is "do not include"
    return out.stdout.strip() == "True"


def _exclude_system_site_packages(venv_path: str) -> None:
    """Switch an existing venv's ``include-system-site-packages`` off.

    For a venv created before #1322 by a runner whose system site holds its
    own install.  Only ever switches it OFF: a venv someone made without the
    system site is not given one.
    """
    cfg = os.path.join(venv_path, "pyvenv.cfg")
    try:
        with open(cfg, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    changed = False
    for i, line in enumerate(lines):
        key, sep, value = line.partition("=")
        if (sep and key.strip() == "include-system-site-packages"
                and value.strip().lower() == "true"):
            lines[i] = "include-system-site-packages = false\n"
            changed = True
    if changed:
        try:
            with open(cfg, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except OSError as exc:
            logger.warning(
                "workspace_venv: could not stop %s seeing the runner's own "
                "packages (%s)", venv_path, exc)


def _venv_pip(venv_path: str) -> str:
    """Absolute path to the venv's ``pip`` console script."""
    exe = "pip.exe" if os.name == "nt" else "pip"
    return os.path.join(_bin_dir(venv_path), exe)


def _ensure_venv_pip(venv_path: str) -> None:
    """Write a ``bin/pip`` (+ ``pip3``) shim if absent, so a bare ``pip`` /
    notebook ``!pip`` installs into the tool-venv.

    A venv with no ``<venv>/bin/pip`` script (created ``--without-pip``, incl.
    by us — see ``ensure_workspace_venv``) makes a bare ``pip`` resolve to the
    SYSTEM pip on ``PATH``, which the confined runner denies.

    NOT ``ensurepip``: (1) the pip bridge makes ``import pip`` succeed,
    so ensurepip no-ops ("already satisfied") and never writes ``bin/pip``;
    (2) ensurepip extracts its wheel to ``/tmp`` (denied under confinement) and
    would pin an OLD bundled pip.  Instead the shim is a 2-line ``sh`` wrapper
    that execs ``<venv>/bin/python -m pip "$@"`` — running the BRIDGED pip (the
    runner's, see ``_write_pip_bridge``) and installing into the tool-venv
    (``sys.prefix``=venv),
    the exact path proven to work under confinement.  The interpreter path is an
    ``exec`` argument (no shebang-length limit).

    Executing the shim ALSO needs an AppArmor ``ix`` grant on the venv bin —
    ``{workspace}/** rwkl`` has no exec bit, and ``bin/python`` runs only
    because it symlinks to the exec-allowed base python.  See
    ``workspace_venv_bin_exec_rule`` (contributed by the tool plugins'
    ``get_apparmor_rules``).

    POSIX only (the confinement target); on Windows this is a no-op and callers
    use ``python -m pip``.  Non-fatal on failure.
    """
    if os.name == "nt" or os.path.exists(_venv_pip(venv_path)):
        return
    bin_dir = _bin_dir(venv_path)
    pip_path = _venv_pip(venv_path)
    try:
        with open(pip_path, "w", encoding="utf-8") as f:
            f.write(f'#!/bin/sh\nexec "{venv_python(venv_path)}" -m pip "$@"\n')
        os.chmod(pip_path, 0o755)
        pip3 = os.path.join(bin_dir, "pip3")
        if not os.path.exists(pip3):
            os.symlink("pip", pip3)   # relative, within bin/
    except OSError as exc:
        logger.warning(
            "workspace_venv: could not write pip shim in %s (%s); bare "
            "`pip`/`!pip` unavailable — use `python -m pip`", venv_path, exc)


def workspace_venv_bin_exec_rule(
    raw: Optional[str], workspace_root: Optional[str],
) -> Optional[str]:
    """AppArmor rule granting exec (``ix``) on the tool-venv's ``bin/`` scripts.

    The broad ``{workspace}/** rwkl`` grant has NO exec bit, so a real
    ``bin/pip`` (or any installed package's console script) in the tool-venv
    can't be exec'd — ``bin/python`` works only because it symlinks to the
    exec-allowed base python.  ``ix`` (inherit-exec) keeps the script in the
    SAME confined profile — no escalation beyond the arbitrary-code exec the
    venv python already permits.  Returns ``None`` when no venv is configured.
    """
    try:
        venv_path = resolve_venv_path(raw, workspace_root)
    except ValueError:
        return None
    if not venv_path:
        return None
    return f"{os.path.join(venv_path, 'bin')}/* ix,"


def pip_apparmor_rules(
    workspace_venv_raw: Optional[str], workspace_path: Optional[str],
) -> List[str]:
    """AppArmor rules for a pip-capable tool (cli / notebook / interactive_shell).

    The distro/UA reads pip needs (``PIP_APPARMOR_RULES``) plus, when a
    ``workspace_venv`` is configured, an ``ix`` grant on the venv bin so a bare
    ``pip`` / notebook ``!pip`` / installed console script can be exec'd.
    Contributed from each tool's ``get_apparmor_rules``.
    """
    rules = list(PIP_APPARMOR_RULES)
    exec_rule = workspace_venv_bin_exec_rule(workspace_venv_raw, workspace_path)
    if exec_rule:
        rules.append(exec_rule)
    return rules


def ensure_workspace_venv(venv_path: str, base_python: Optional[str] = None) -> str:
    """Create the workspace venv if absent + materialize pip.

    Creation is idempotent (an existing ``pyvenv.cfg`` short-circuits it;
    an existing venv whose system site is the runner's own install has that
    switched off, #1322), but two fix-ups run **every** call so a venv created by another party is brought
    up to spec: (1) ``_ensure_venv_pip`` writes the ``<venv>/bin/pip`` shim if
    missing (so a bare ``pip`` / ``!pip`` uses the tool-venv, not system pip —
    also needs the AppArmor ``ix`` grant, see ``workspace_venv_bin_exec_rule``);
    (2) the pip bridge (see ``_write_pip_bridge``) is (re)written, tracking
    the runner's current pip.

    Args:
        venv_path: Absolute path where the venv lives (see ``resolve_venv_path``).
        base_python: Interpreter to build against.  Defaults to ``sys.executable``.

    Returns:
        ``venv_path`` (unchanged), for call-site chaining.

    Raises:
        subprocess.CalledProcessError: If ``python -m venv`` fails.
        RuntimeError: If the venv has no resolvable site-packages dir.
    """
    leaks_runner = _system_site_packages_are_the_runners(base_python)
    if not os.path.exists(os.path.join(venv_path, "pyvenv.cfg")):
        os.makedirs(os.path.dirname(venv_path) or ".", exist_ok=True)
        # --system-site-packages only when the system site is not the
        # runner's own install (#1322); see the module docstring.
        system_site = [] if leaks_runner else ["--system-site-packages"]
        subprocess.run(
            # --without-pip: venv creation runs ensurepip internally, which
            # extracts a wheel to /tmp — denied in the confined runner.  We
            # provide pip via the pip bridge (import) + a `bin/pip` shim
            # instead (_ensure_venv_pip), so the venv needs no pip of its own.
            [base_python or sys.executable, "-m", "venv",
             *system_site, "--without-pip", venv_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    elif leaks_runner:
        _exclude_system_site_packages(venv_path)
    _ensure_venv_pip(venv_path)
    _write_pip_bridge(venv_path)
    return venv_path


def apply_venv_to_env(env: MutableMapping[str, str], venv_path: str) -> None:
    """Activate ``venv_path`` in ``env`` in place.

    Prepends the venv ``bin`` dir to ``PATH`` and its ``site-packages`` to
    ``PYTHONPATH`` (both ahead of any existing entries, so the venv wins),
    sets ``VIRTUAL_ENV``, and clears ``PYTHONHOME`` (which, if set, would
    override the venv's ``pyvenv.cfg`` home and break resolution).  Existing
    ``PYTHONPATH`` entries are preserved after the venv's — the runner source
    tree stays importable.

    Args:
        env: The subprocess environment mapping to mutate.
        venv_path: Absolute venv path.
    """
    bin_dir = _bin_dir(venv_path)
    sep = os.pathsep
    existing_path = env.get("PATH", "")
    env["PATH"] = bin_dir + (sep + existing_path if existing_path else "")
    env["VIRTUAL_ENV"] = venv_path
    env.pop("PYTHONHOME", None)

    site = venv_site_packages(venv_path)
    if site:
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = site + (sep + existing_pp if existing_pp else "")


def effective_workspace_venv(
    cli_config: Optional[dict],
    workspace_path: Optional[str],
    managed_workspace_root: Optional[str],
) -> Optional[str]:
    """The managed-default ``workspace_venv`` for this session, or ``None``.

    ``None`` when a profile already decided (``plugin_configs.cli`` carries
    the key, a blank value being the opt-out) or when the workspace is not
    one the daemon manages -- a user's own checkout never grows a venv it
    did not ask for.  Otherwise :data:`DEFAULT_WORKSPACE_VENV`.
    """
    from .workspace_home import is_daemon_managed

    if cli_config is not None and _VENV_CONFIG_KEY in cli_config:
        return None
    if is_daemon_managed(workspace_path, managed_workspace_root):
        return DEFAULT_WORKSPACE_VENV
    return None


def inject_workspace_venv(
    plugin_configs: dict,
    workspace_path: Optional[str],
    managed_workspace_root: Optional[str],
) -> Optional[str]:
    """Fold the managed-default venv into every surface that honours it.

    Mutates ``plugin_configs`` in place (``setdefault`` per surface, so an
    explicit per-surface value is kept) and returns the value applied, or
    ``None`` when no default applies.  An explicit ``cli`` value is NOT
    mirrored to the other surfaces: that is the pre-#1274 meaning of the
    key, and a profile that set it only for ``cli`` keeps that meaning.
    """
    if not isinstance(plugin_configs, dict):
        return None
    cli_cfg = plugin_configs.get("cli")
    cli_cfg = cli_cfg if isinstance(cli_cfg, dict) else None
    raw = effective_workspace_venv(cli_cfg, workspace_path, managed_workspace_root)
    if not raw:
        return None
    for surface in VENV_SURFACES:
        section = dict(plugin_configs.get(surface) or {})
        section.setdefault(_VENV_CONFIG_KEY, raw)
        plugin_configs[surface] = section
    return raw
