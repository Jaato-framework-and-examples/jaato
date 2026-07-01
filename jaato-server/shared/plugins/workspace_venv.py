"""Workspace-scoped virtualenv provisioning for tool subprocesses.

When a tool plugin (``cli`` / ``interactive_shell`` / ``notebook``) is
configured with a ``workspace_venv`` path, its subprocesses run with that
venv **activated** so the model's in-tool ``pip install X`` persists to the
venv and a later ``import X`` resolves.  The venv lives INSIDE the workspace
(the confined runner has rw there; the runner's own base environment stays
read-only), and is built with ``--system-site-packages`` so the runner's base
dependencies remain importable without duplication.

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

Contract
--------
- Empty / unset path = feature OFF.  There is no implicit default venv — the
  path is the explicit agreement between the tool subprocess and the client's
  in-process import path.
- Create-if-absent is idempotent: an existing ``pyvenv.cfg`` short-circuits.
- Relative paths resolve against the session workspace root; a relative path
  with no workspace root is a configuration error (raised, not defaulted).
"""

import glob
import os
import subprocess
import sys
from typing import MutableMapping, Optional


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


def ensure_workspace_venv(venv_path: str, base_python: Optional[str] = None) -> str:
    """Create the workspace venv if absent; return its path (idempotent).

    An existing ``pyvenv.cfg`` short-circuits creation.  The venv is built with
    ``--system-site-packages`` against ``base_python`` (default: the current
    runner interpreter) so base deps stay importable and read-only.

    Args:
        venv_path: Absolute path where the venv lives (see ``resolve_venv_path``).
        base_python: Interpreter to build against.  Defaults to ``sys.executable``.

    Returns:
        ``venv_path`` (unchanged), for call-site chaining.

    Raises:
        subprocess.CalledProcessError: If ``python -m venv`` fails.
    """
    if os.path.exists(os.path.join(venv_path, "pyvenv.cfg")):
        return venv_path
    os.makedirs(os.path.dirname(venv_path) or ".", exist_ok=True)
    subprocess.run(
        [base_python or sys.executable, "-m", "venv",
         "--system-site-packages", venv_path],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
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
