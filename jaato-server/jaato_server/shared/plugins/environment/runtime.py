"""``get_environment(aspect="runtime")``: what this session can run (#1346).

An agent in a confined session used to find out what it could run by
probing, and reached wrong conclusions from the probes (a "command-name
blocklist" that was #1342, a "missing" ``which`` that was an unreadable
script, a git that "needed" ``GIT_CONFIG_GLOBAL=/dev/null``).  Every fact it
needed already existed runner-side.  This module reads them and reports
them; it derives none of them a second time:

| Field | Read from |
|---|---|
| confinement tier | the calling thread's AppArmor label, via :mod:`jaato_server.shared.apparmor_label` |
| profile, exec scope, exec roots | the session's ``//child`` grant record, via :mod:`jaato_server.shared.confinement_grants` (#1348) |
| subprocess ``PATH``, ``HOME``, ``XDG_*``, tool-venv | ``CLIToolPlugin._build_subprocess_env()`` on the session's own ``cli`` instance |
| bound toolchains | ``<workspace>/.jaato/environment.json`` (#1344), when present |

The ``PATH`` is the one ``cli`` builds for its next command, not a
re-assembly of it: a second assembly is how the report and the command
would start disagreeing (the #1171 shape).  When ``cli`` is not on the
session's surface there is no subprocess ``PATH`` to report, and the aspect
says so rather than showing the runner's own.

Pull-based on purpose (design ``web-coder-environment-bootstrap.md`` §8):
it costs nothing per request, and it is current after a mid-session change
and in a revived session whose persisted prompt predates one.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from jaato_server.shared.apparmor_label import (
    SANDBOX_MODE_APPARMOR,
    SANDBOX_MODE_APPARMOR_COMPLAIN,
    AppArmorLabel,
    read_thread_label,
)
from jaato_server.shared.confinement_grants import (
    ConfinementGrants,
    confinement_grants,
    parse_rules,
)

#: The tier a session with no AppArmor profile reports.  Never "unknown":
#: a label that could not be read claims no boundary, so it is this too.
TIER_UNCONFINED = "unconfined"

#: The manifest #1344 phase 1 will write.  Read, never written, here.
ENVIRONMENT_MANIFEST = os.path.join(".jaato", "environment.json")

_XDG_KEYS = ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME")


# ---------------------------------------------------------------------------
# Confinement
# ---------------------------------------------------------------------------


def confinement_tier(label: AppArmorLabel) -> str:
    """The tier for *label*: ``apparmor``, ``apparmor-complain`` or ``unconfined``.

    A profile in a mode the kernel names otherwise reports
    ``apparmor-<mode>``, never ``apparmor``: only an enforced profile
    blocks.  A label with no ``(mode)`` at all is not evidence of AppArmor
    (a host whose active LSM is another one reports a bare ``kernel``), so
    it is ``unconfined``; :func:`confinement_report` quotes the label.
    """
    if not label.confined or label.mode is None:
        return TIER_UNCONFINED
    if label.enforced:
        return SANDBOX_MODE_APPARMOR
    if label.complaining:
        return SANDBOX_MODE_APPARMOR_COMPLAIN
    return f"apparmor-{label.mode}"


def exec_rules(grants: ConfinementGrants) -> Dict[str, Any]:
    """The exec-granting rules of the ``//child`` body, as the record holds them.

    ``exec_roots`` are the globs of allow rules whose mode includes ``x``;
    ``exec_denied`` those of deny rules.  Nothing is re-derived from the
    template: a glob not in the record is not reported.
    """
    parsed = parse_rules(grants.rules)
    roots = [r.glob for r in parsed.rules if r.grants("x") and not r.deny]
    out: Dict[str, Any] = {"exec_roots": roots}
    denied = [r.glob for r in parsed.rules if r.grants("x") and r.deny]
    if denied:
        out["exec_denied"] = denied
    if parsed.grants_everything:
        out["grants_everything"] = True
    if parsed.unreadable:
        out["unreadable_rules"] = len(parsed.unreadable)
    return out


def confinement_report(
    label: AppArmorLabel, grants: Optional[ConfinementGrants],
) -> Dict[str, Any]:
    """The confinement block: tier, profile, exec scope and exec roots."""
    tier = confinement_tier(label)
    out: Dict[str, Any] = {"tier": tier}
    if tier == TIER_UNCONFINED:
        if not label.raw:
            out["note"] = ("the AppArmor label could not be read, so no "
                           "kernel boundary is claimed")
        elif label.confined:
            out["kernel_label"] = label.raw
            out["note"] = ("the label names no AppArmor mode, so no AppArmor "
                           "boundary is claimed")
        if grants is None:
            return out
    out["profile"] = grants.profile_name if grants else label.profile
    if grants is None:
        out["grant_record"] = "absent"
        out["note"] = ("this runner holds no record of what the profile "
                       "grants (for example after a daemon restart); exec "
                       "scope and exec roots are unknown")
        return out
    out["grant_record"] = "present"
    out["exec_scope"] = grants.exec_scope or "unrecorded"
    out["applies_to"] = "commands run through cli, in the //child sub-profile"
    out.update(exec_rules(grants))
    return out


# ---------------------------------------------------------------------------
# The subprocess environment, from cli's own builder
# ---------------------------------------------------------------------------


def cli_on_surface(registry: Any, session: Any) -> Optional[Any]:
    """The ``cli`` plugin instance when this session's model can call it."""
    if registry is None:
        return None
    try:
        if "cli" not in registry.list_exposed():
            return None
        plugin = registry.get_plugin("cli")
    except Exception:
        return None
    wanted = getattr(session, "_tool_plugins", None) if session is not None else None
    if wanted is not None and "cli" not in wanted:
        return None
    return plugin if hasattr(plugin, "_build_subprocess_env") else None


def _venv_report(venv_path: Optional[str]) -> Dict[str, Any]:
    from jaato_server.shared.plugins.workspace_venv import venv_python
    if not venv_path:
        return {"configured": False}
    python = venv_python(venv_path)
    return {
        "configured": True,
        "path": venv_path,
        "interpreter": python,
        # Created on the first command that passes containment.
        "created": os.path.exists(python),
    }


def subprocess_report(cli_plugin: Any) -> Dict[str, Any]:
    """PATH, HOME, TMPDIR, XDG_* and the tool-venv from ``cli``'s env builder.

    ``tmpdir`` is the session's own temp directory; in a confined session it
    is the only writable place under ``/tmp`` (#1361) -- unless the session
    has a private ``/tmp`` (#1381, :func:`private_tmp_report`), where it is
    ``/tmp`` itself and all of it is writable.
    """
    try:
        env, venv_path = cli_plugin._build_subprocess_env()
    except Exception as exc:  # e.g. a relative workspace_venv, no workspace
        return {"error": f"cli could not build its environment: {exc}"}
    path = env.get("PATH", "")
    return {
        "path": [p for p in path.split(os.pathsep) if p],
        "home": env.get("HOME"),
        "tmpdir": env.get("TMPDIR"),
        "xdg": {k: env[k] for k in _XDG_KEYS if k in env},
        "pycache_prefix": env.get("PYTHONPYCACHEPREFIX"),
        "virtual_env": env.get("VIRTUAL_ENV"),
        "tool_venv": _venv_report(venv_path),
    }


def private_tmp_report() -> Dict[str, Any]:
    """Whether ``/tmp`` is this workspace's own ``.tmp`` (#1381).

    Read from this process's mount namespace (device + inode), not from the
    envelope, so it never claims a private ``/tmp`` the task does not see.
    """
    from jaato_server.shared.private_tmp import (
        describe, private_shm_in_effect)
    private, backing = describe()
    if not private:
        return {
            "private": False,
            "note": "/tmp is the host's; in a confined session only $TMPDIR "
                    "under it is writable",
        }
    return {
        "private": True,
        "backing_dir": backing,
        "private_shm": private_shm_in_effect(),
        "note": "/tmp and /var/tmp are this workspace's own directory, "
                "shared by its sessions and invisible to other workspaces",
    }


# ---------------------------------------------------------------------------
# Bound toolchains
# ---------------------------------------------------------------------------


def toolchains_report(workspace: Optional[str]) -> Dict[str, Any]:
    """Bound toolchains from ``<workspace>/.jaato/environment.json``.

    Tolerant: a missing file is ``absent``, a file that cannot be read or
    parsed is ``unreadable`` with the reason.  Nothing is written.
    """
    if not workspace:
        return {"status": "absent", "note": "no workspace is known"}
    manifest = os.path.join(workspace, ENVIRONMENT_MANIFEST)
    try:
        with open(manifest, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {"status": "absent", "manifest": manifest}
    except (OSError, ValueError) as exc:
        return {"status": "unreadable", "manifest": manifest, "error": str(exc)}
    if not isinstance(data, dict):
        return {"status": "unreadable", "manifest": manifest,
                "error": "the manifest is not a JSON object"}
    return {"status": "present", "manifest": manifest,
            "toolchains": data.get("toolchains", [])}


# ---------------------------------------------------------------------------
# The aspect
# ---------------------------------------------------------------------------


def runtime_report(
    registry: Any,
    session: Any,
    workspace: Optional[str],
    *,
    read_label: Optional[Callable[[], AppArmorLabel]] = None,
    grants: Optional[Callable[[], Optional[ConfinementGrants]]] = None,
) -> Dict[str, Any]:
    """The full ``runtime`` aspect.

    Args:
        registry: The session's plugin registry, for its ``cli`` instance.
        session: The calling session; its ``_tool_plugins`` decides whether
            ``cli`` is on its surface.
        workspace: The workspace root, for the toolchain manifest.
        read_label: Reads the calling thread's AppArmor label; defaults to
            :func:`read_thread_label`, resolved at call time.
        grants: Returns the installed ``//child`` grants; defaults to
            :func:`confinement_grants`, resolved at call time.
    """
    label = (read_label or read_thread_label)()
    record = (grants or confinement_grants)()
    report: Dict[str, Any] = {"confinement": confinement_report(label, record)}
    cli = cli_on_surface(registry, session)
    if cli is None:
        report["subprocess"] = {
            "cli": "not loaded",
            "note": "cli is not on this session's tool surface, so there is "
                    "no subprocess PATH to report",
        }
    else:
        report["subprocess"] = subprocess_report(cli)
        workspace = workspace or getattr(cli, "_workspace_root", None)
    report["private_tmp"] = private_tmp_report()
    report["toolchains"] = toolchains_report(workspace)
    return report


def _confinement_line(block: Dict[str, Any]) -> str:
    parts = [block["tier"]]
    if "profile" in block:
        parts.append(f"profile {block['profile']}")
    if block.get("grant_record") == "absent":
        parts.append("grant record absent")
    elif "exec_scope" in block:
        parts.append(f"exec {block['exec_scope']}, "
                     f"{len(block.get('exec_roots', []))} exec roots")
    return ", ".join(parts)


def _subprocess_lines(block: Dict[str, Any]) -> List[Tuple[str, str]]:
    if "cli" in block:
        return [("path", "cli not loaded")]
    if "error" in block:
        return [("path", block["error"])]
    venv = block["tool_venv"]
    return [
        ("path", f"{len(block['path'])} entries: " + os.pathsep.join(block["path"])),
        ("home", block.get("home") or "unset"),
        ("tmpdir", block.get("tmpdir") or "unset"),
        ("tool_venv", venv["interpreter"] if venv["configured"] else "not configured"),
    ]


def _private_tmp_line(block: Dict[str, Any]) -> str:
    if block.get("private"):
        shm = ", private /dev/shm" if block.get("private_shm") else ""
        return f"yes ({block.get('backing_dir')} bound over /tmp, /var/tmp{shm})"
    return "no"


def runtime_summary(report: Dict[str, Any]) -> Dict[str, str]:
    """One line per field, for ``aspect="all"``."""
    lines = {"confinement": _confinement_line(report["confinement"])}
    lines.update(_subprocess_lines(report["subprocess"]))
    lines["private_tmp"] = _private_tmp_line(report.get("private_tmp") or {})
    chains = report["toolchains"]
    lines["toolchains"] = chains["status"]
    lines["detail"] = "get_environment(aspect='runtime') for the full report"
    return lines
