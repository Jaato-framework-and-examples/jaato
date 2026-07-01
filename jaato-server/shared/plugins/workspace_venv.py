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

Importing jaato's own code from the tool-venv (the runner-import bridge)
------------------------------------------------------------------------
The notebook kernel is *jaato's own* module (``shared.plugins.notebook.
kernel_main``) run under the tool-venv interpreter, so that interpreter must
be able to ``import shared.*`` (and ``jaato_sdk``, which the kernel's
``tool_stubs`` needs).  ``--system-site-packages`` can NOT provide this: a venv
created from the runner venv resolves its base to the system prefix (``/usr``),
not the runner venv where jaato is installed; and for an **editable** install
``shared`` is wired through a meta-path finder registered by a ``.pth`` that
only runs when its dir is processed as a *site* dir (never for ``PYTHONPATH``).
The deployment-agnostic, least-privilege fix is ``_write_runner_bridge``: a
``.pth`` of **plain directory lines** (appended to ``sys.path`` verbatim,
without executing any ``.pth`` finder) listing the base site-packages
(installed third-party deps) plus the explicit jaato source roots
(``jaato_source_dirs`` → ``shared`` + ``jaato_sdk``).  This surfaces exactly
those packages — NOT the whole base venv's other editable installs (which a
naive ``addsitedir`` would drag in, leaking unrelated src dirs the model could
target with ``pip install --target``).  Runner-side only; ``cli`` /
``interactive_shell`` run the *model's* commands (which need only the
tool-venv), but the bridge is harmless there and keeps a single ``ensure``
path.

Contract
--------
- Empty / unset path = feature OFF.  There is no implicit default venv — the
  path is the explicit agreement between the tool subprocess and the client's
  in-process import path.
- Create-if-absent is idempotent, but the runner-import bridge is refreshed on
  every ``ensure`` (so a venv created elsewhere without it is fixed up).
- Relative paths resolve against the session workspace root; a relative path
  with no workspace root is a configuration error (raised, not defaulted).
"""

import glob
import importlib
import os
import site
import subprocess
import sys
from typing import List, MutableMapping, Optional


_BRIDGE_PTH = "_jaato_runner_bridge.pth"


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
    an import finder.  Bridging them into a tool-venv (see
    ``ensure_workspace_venv``) is what lets the tool-venv interpreter import
    ``shared.*`` regardless of install style.
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

    The subprocess kernel (``shared.plugins.notebook.kernel_main``) imports
    ``shared`` (jaato-server) and ``jaato_sdk`` (its ``tool_stubs`` needs
    ``ToolSchema``, which pulls in ``pydantic``).  For an **editable** install
    these are the external src dirs (e.g. ``.../jaato-server``,
    ``.../jaato-sdk``); for a **wheel** install they resolve into the base
    site-packages (deduped by the caller).  Deliberately NOT the whole base
    site-packages — that would surface UNRELATED editable installs
    (jaato_premium, a client's own editable package).
    """
    dirs: List[str] = []
    for mod_name in ("shared", "jaato_sdk"):
        mod = importlib.import_module(mod_name)
        dirs.append(os.path.dirname(mod.__path__[0]))
    return dirs


def _bridge_dirs() -> List[str]:
    """Dirs to append to the tool-venv ``sys.path`` (order-preserving, existing).

    Base site-packages (installed third-party deps) + the jaato source roots.
    """
    seen = set()
    out: List[str] = []
    for d in runner_site_dirs() + jaato_source_dirs():
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def _write_runner_bridge(venv_path: str) -> None:
    """(Re)write the ``.pth`` that lets the tool-venv import jaato's own code.

    A tool-venv created *from* the runner venv resolves its ``base`` to the
    system prefix (``/usr``), NOT the runner venv — so ``--system-site-packages``
    can never surface jaato's packages.  And for an **editable** install
    ``shared.*`` is wired through a meta-path finder registered by a ``.pth``
    that runs only when its dir is processed as a *site* dir (never for
    ``PYTHONPATH``).

    Rather than ``site.addsitedir(<base site-packages>)`` — which runs ALL of
    the base dir's editable ``.pth`` finders and so drags UNRELATED editable
    src dirs (jaato_premium, a client's own package) onto the tool-venv
    ``sys.path`` (least-privilege leak + a path the model can wrongly target
    with ``pip install --target``) — the bridge writes **plain directory
    lines**: a ``.pth`` line that is not an ``import`` statement is appended to
    ``sys.path`` verbatim, WITHOUT executing any ``.pth`` finder.  It lists the
    base site-packages (installed third-party deps like ``pydantic``) plus the
    explicit jaato source roots (``jaato_source_dirs``).  So the tool-venv can
    import exactly ``shared`` + ``jaato_sdk`` + installed deps — nothing else.

    Written into the tool-venv's own site-packages and refreshed on every
    ``ensure`` (so a venv created elsewhere is fixed up).  Entries are appended
    AFTER the tool-venv's own site-packages, so the model's ``pip install``
    there keeps priority for any shared package name.
    """
    site_dir = venv_site_packages(venv_path)
    if site_dir is None:
        raise RuntimeError(
            f"workspace venv at {venv_path} has no site-packages directory; "
            "cannot bridge runner imports")
    body = "\n".join(_bridge_dirs())
    with open(os.path.join(site_dir, _BRIDGE_PTH), "w", encoding="utf-8") as f:
        f.write(body + "\n")


def ensure_workspace_venv(venv_path: str, base_python: Optional[str] = None) -> str:
    """Create the workspace venv if absent + bridge runner imports; return path.

    Creation is idempotent (an existing ``pyvenv.cfg`` short-circuits it), but
    the runner-import bridge (see ``_write_runner_bridge``) is (re)written
    **every** call — so a venv created by another party without the bridge is
    fixed up here, and the bridge tracks the runner's current site dirs.

    Args:
        venv_path: Absolute path where the venv lives (see ``resolve_venv_path``).
        base_python: Interpreter to build against.  Defaults to ``sys.executable``.

    Returns:
        ``venv_path`` (unchanged), for call-site chaining.

    Raises:
        subprocess.CalledProcessError: If ``python -m venv`` fails.
        RuntimeError: If the venv has no resolvable site-packages dir.
    """
    if not os.path.exists(os.path.join(venv_path, "pyvenv.cfg")):
        os.makedirs(os.path.dirname(venv_path) or ".", exist_ok=True)
        subprocess.run(
            [base_python or sys.executable, "-m", "venv",
             "--system-site-packages", venv_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    _write_runner_bridge(venv_path)
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
