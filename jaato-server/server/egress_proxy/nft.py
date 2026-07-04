"""cgroup-v2-scoped nftables egress enforcement (Phase 5 §5.11d-v2).

The AppArmor ``peer=(ip=)`` approach for hard egress confinement does NOT work
on stock Ubuntu (kernel advertises ``network_v8/af_inet`` but does not enforce
it — see ``docs/design/phase5_5_11_egress_proxy_spike.md`` §11).  The proven
replacement enforces at netfilter: a per-session ``nft`` table gates the
session's cgroup so it may only reach the loopback egress proxy; every other
outbound connection is rejected by the kernel, regardless of whether the
process honours ``HTTPS_PROXY``.  Verified on the gate host
(``scripts/verify_egress_nft.sh``): a process in the cgroup reaches the proxy
but not ``8.8.8.8``; a process outside the cgroup is unaffected.

This module renders + installs/removes that ruleset per session.  It shells out
to ``nft`` (via ``sudo`` when the daemon is not root — mirroring the existing
``sudo apparmor_parser`` pattern; the deployment must grant ``nft`` in the same
NOPASSWD scope).  Rendering is a pure function so it is unit-testable without
touching the kernel.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# nft identifiers allow [A-Za-z0-9_]; sanitize the session id for the table name.
_ID_SANITIZE = re.compile(r"[^A-Za-z0-9_]")

# cgroup2 mount root — paths in `socket cgroupv2 "..."` are relative to this.
CGROUP2_ROOT = "/sys/fs/cgroup"


def _table_name(session_id: str) -> str:
    return "jaato_egress_" + _ID_SANITIZE.sub("_", session_id)


def cgroup_rel_and_level(cgroup_path: str) -> "tuple[str, int]":
    """Return (path-relative-to-cgroup2-root, level) for an absolute cgroup path.

    ``level`` is the depth below the cgroup2 root — the value the nft
    ``socket cgroupv2 level N`` match needs.  E.g. ``/sys/fs/cgroup/jaato/
    jaato-ws-abc`` → (``"jaato/jaato-ws-abc"``, 2).
    """
    rel = os.path.relpath(os.path.abspath(cgroup_path), CGROUP2_ROOT).strip("/")
    level = len([p for p in rel.split("/") if p])
    return rel, level


def render_ruleset(
    session_id: str,
    cgroup_path: str,
    proxy_port: int,
    *,
    allow_local_resolver: bool = False,
) -> str:
    """Render the per-session nft ruleset (pure — no side effects).

    Gates ONLY the session's cgroup: allow outbound to the loopback proxy
    (``127.0.0.1:<proxy_port>``) and IPv6 loopback, reject everything else.
    Non-matching host traffic falls through ``policy accept`` so only the
    session is constrained.  Direct DNS is denied by default (name resolution
    goes through the proxy's CONNECT); ``allow_local_resolver`` re-opens the
    systemd stub at ``127.0.0.53`` for deployments that need runner-side DNS.
    """
    table = _table_name(session_id)
    rel, level = cgroup_rel_and_level(cgroup_path)
    gate_lines = [
        f"    ip  daddr 127.0.0.1 tcp dport {int(proxy_port)} accept",
        "    ip6 daddr ::1 accept",
    ]
    if allow_local_resolver:
        gate_lines.append("    ip  daddr 127.0.0.53 accept")
    gate_lines.append("    counter reject")
    gate_body = "\n".join(gate_lines)
    return (
        f"table inet {table} {{\n"
        f"  chain out {{\n"
        f"    type filter hook output priority 0; policy accept;\n"
        f'    socket cgroupv2 level {level} "{rel}" jump gate\n'
        f"  }}\n"
        f"  chain gate {{\n"
        f"{gate_body}\n"
        f"  }}\n"
        f"}}\n"
    )


class EgressNftManager:
    """Installs/removes the per-session cgroup-scoped nft egress table."""

    def __init__(self, use_sudo: Optional[bool] = None):
        # Root daemons call nft directly; non-root shell out via sudo (the
        # deployment must grant nft NOPASSWD, like apparmor_parser).
        self._use_sudo = (os.geteuid() != 0) if use_sudo is None else use_sudo
        self._lock = threading.Lock()
        self._installed: Dict[str, str] = {}   # session_id -> table name

    @staticmethod
    def nft_available() -> bool:
        return shutil.which("nft") is not None

    def _nft(self, args: List[str], stdin: Optional[str] = None) -> subprocess.CompletedProcess:
        cmd = (["sudo", "-n", "nft"] if self._use_sudo else ["nft"]) + args
        return subprocess.run(
            cmd, input=stdin, capture_output=True, text=True, timeout=10,
        )

    def install(
        self, session_id: str, cgroup_path: str, proxy_port: int,
        *, allow_local_resolver: bool = False,
    ) -> bool:
        """Render + load the session's egress table.  Returns True on success.

        Requires the cgroup at ``cgroup_path`` to already exist (nft resolves
        the cgroup path to an id at load time).  Idempotent per session.
        """
        if not os.path.isdir(cgroup_path):
            logger.warning(
                "egress nft: cgroup %s does not exist — skipping enforcement "
                "for session %s (proxy-only confinement)", cgroup_path, session_id)
            return False
        ruleset = render_ruleset(
            session_id, cgroup_path, proxy_port,
            allow_local_resolver=allow_local_resolver)
        with self._lock:
            # Replace any stale table first (idempotent re-install).
            self._delete_table(_table_name(session_id))
            proc = self._nft(["-f", "-"], stdin=ruleset)
            if proc.returncode != 0:
                logger.warning("egress nft install failed for session %s: %s",
                               session_id, (proc.stderr or "").strip())
                return False
            self._installed[session_id] = _table_name(session_id)
            logger.info("egress nft: enforcing for session %s (cgroup %s -> "
                        "proxy :%d)", session_id, cgroup_path, proxy_port)
            return True

    def remove(self, session_id: str) -> None:
        """Delete the session's egress table.  Idempotent."""
        with self._lock:
            table = self._installed.pop(session_id, None) or _table_name(session_id)
            self._delete_table(table)

    def _delete_table(self, table: str) -> None:
        # `nft delete table` errors if absent; that's fine (idempotent).
        proc = self._nft(["delete", "table", "inet", table])
        if proc.returncode != 0 and "No such file" not in (proc.stderr or ""):
            logger.debug("egress nft delete table %s: %s", table,
                         (proc.stderr or "").strip())

    def shutdown(self) -> None:
        with self._lock:
            tables = list(self._installed.values())
            self._installed.clear()
        for t in tables:
            self._delete_table(t)
