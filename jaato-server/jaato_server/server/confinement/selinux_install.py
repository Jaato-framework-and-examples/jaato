"""``jaato-selinux``: build and load the SELinux policy module this release needs.

Phase 5 of docs/design/selinux-backend.md (§9).  The module source
(``selinux_policy/jaato.{te,fc,if}``) ships in this package, so the module
an operator loads is the one this daemon's :data:`REQUIRED_POLICY_VERSION`
was written for.  The steps are the ones every kernel handoff ran by hand:

    jaato-selinux install [--home DIR ...]   build, load, label (as root)
    jaato-selinux uninstall                  remove the module and the label
    jaato-selinux status                     what a daemon started now gets
                                             (exit 0 ready, 1 not ready)

``install`` builds with the host's own ``selinux-policy-devel`` (a module
compiled against another policy release is not guaranteed to load), loads
it with ``semodule -i``, labels this interpreter's venv ``lib_t`` (a runner
must map the venv's libraries; a system interpreter is already labelled
and is left alone), relabels each ``--home``'s ``~/.jaato`` (default: this
process's ``$HOME``), and then runs the daemon's own readiness check.

Every step is a command run as given; the first that fails stops the
command and is named.  Nothing here is imported by the daemon.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path
from typing import Callable, List, Optional, Sequence

#: The reference-policy build system ``selinux-policy-devel`` installs.
DEVEL_MAKEFILE = "/usr/share/selinux/devel/Makefile"
MODULE_NAME = "jaato"
SOURCES = ("jaato.te", "jaato.fc", "jaato.if")

Runner = Callable[..., subprocess.CompletedProcess]


class InstallError(Exception):
    """A step that failed, named for the operator."""


def policy_source_dir() -> Path:
    """The packaged module source directory."""
    return Path(str(resources.files("jaato_server.server.confinement") / "selinux_policy"))


def venv_prefix() -> Optional[str]:
    """This interpreter's venv, or ``None`` for a system interpreter."""
    return sys.prefix if sys.prefix != sys.base_prefix else None


def _venv_spec(prefix: str) -> str:
    return f"{os.path.realpath(prefix)}(/.*)?"


def _run(run: Runner, argv: Sequence[str], **kw) -> subprocess.CompletedProcess:
    p = run(list(argv), capture_output=True, text=True, **kw)
    if p.returncode != 0:
        raise InstallError(f"{' '.join(argv)} failed ({p.returncode}): "
                           f"{(p.stderr or p.stdout).strip()}")
    return p


def build_module(run: Runner, source: Path, work: Path) -> Path:
    """``jaato.pp`` built from *source* in *work* with the host's devel tree."""
    if not os.path.isfile(DEVEL_MAKEFILE):
        raise InstallError(f"{DEVEL_MAKEFILE} not found: install selinux-policy-devel")
    for name in SOURCES:
        shutil.copy(source / name, work / name)
    _run(run, ["make", "-f", DEVEL_MAKEFILE, f"{MODULE_NAME}.pp"], cwd=str(work))
    return work / f"{MODULE_NAME}.pp"


def _fcontext_defined(run: Runner, spec: str) -> bool:
    out = _run(run, ["semanage", "fcontext", "-l", "-C"]).stdout
    return any(line.split()[:1] == [spec] for line in out.splitlines())


def label_venv(run: Runner, prefix: str) -> str:
    """Label the venv ``lib_t`` (a local fcontext rule, then restorecon)."""
    spec = _venv_spec(prefix)
    verb = "-m" if _fcontext_defined(run, spec) else "-a"
    _run(run, ["semanage", "fcontext", verb, "-t", "lib_t", spec])
    _run(run, ["restorecon", "-R", os.path.realpath(prefix)])
    return spec


def relabel_homes(run: Runner, homes: Sequence[str]) -> List[str]:
    """``restorecon -R`` each home's ``~/.jaato``; one line per home."""
    done = []
    for home in homes:
        user_dir = os.path.join(home, ".jaato")
        if os.path.isdir(user_dir):
            _run(run, ["restorecon", "-R", user_dir])
            done.append(f"relabelled {user_dir}")
        else:
            done.append(f"skipped {home}: no ~/.jaato yet (relabel it with "
                        f"restorecon -R once it exists)")
    return done


def readiness_line() -> str:
    """What the daemon's own readiness check says now."""
    from jaato_server.server.confinement.selinux import (
        REQUIRED_POLICY_VERSION, SELinuxBackend,
    )

    ready = SELinuxBackend().host_readiness()
    if ready.ready:
        return f"ready: policy module v{REQUIRED_POLICY_VERSION} is usable by this daemon"
    return f"not ready: {ready.reason}"


def install(homes: Sequence[str], run: Runner = subprocess.run) -> List[str]:
    """Build, load and label; return what was done, line by line."""
    done = []
    with tempfile.TemporaryDirectory(prefix="jaato-selinux-") as work:
        pp = build_module(run, policy_source_dir(), Path(work))
        _run(run, ["semodule", "-i", str(pp)])
        done.append(f"loaded module {MODULE_NAME} from {pp.name}")
    prefix = venv_prefix()
    if prefix is None:
        done.append("system interpreter: its label is the system's, left alone")
    else:
        done.append(f"labelled {label_venv(run, prefix)} lib_t")
    done += relabel_homes(run, homes)
    return done


def uninstall(run: Runner = subprocess.run) -> List[str]:
    """Remove the module and the venv's local fcontext rule."""
    _run(run, ["semodule", "-r", MODULE_NAME])
    done = [f"removed module {MODULE_NAME}"]
    prefix = venv_prefix()
    if prefix is not None and _fcontext_defined(run, _venv_spec(prefix)):
        _run(run, ["semanage", "fcontext", "-d", _venv_spec(prefix)])
        _run(run, ["restorecon", "-R", os.path.realpath(prefix)])
        done.append(f"removed the lib_t rule for {_venv_spec(prefix)}")
    return done


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="jaato-selinux", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    inst = sub.add_parser("install", help="build, load and label (as root)")
    inst.add_argument("--home", action="append", default=None,
                      help="a home whose ~/.jaato to relabel (repeatable; default $HOME)")
    sub.add_parser("uninstall", help="remove the module and the venv label (as root)")
    sub.add_parser("status", help="what a daemon started now would get")
    args = ap.parse_args(argv)
    if args.cmd == "status":
        line = readiness_line()
        print(line)
        return 0 if line.startswith("ready") else 1
    if os.geteuid() != 0:
        print(f"jaato-selinux {args.cmd}: run as root", file=sys.stderr)
        return 2
    try:
        done = (install(args.home or [str(Path.home())]) if args.cmd == "install"
                else uninstall())
    except InstallError as exc:
        print(f"jaato-selinux {args.cmd}: {exc}", file=sys.stderr)
        return 1
    print("\n".join(done))
    if args.cmd == "uninstall":
        return 0
    line = readiness_line()
    print(line)
    return 0 if line.startswith("ready") else 1


if __name__ == "__main__":
    sys.exit(main())
