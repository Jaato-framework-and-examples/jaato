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

import ipaddress
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


# The systemd-resolved stub resolver, re-opened by ``allow_local_resolver``.
_STUB_RESOLVER = "127.0.0.53"


def _daddr_match(host: str) -> "tuple[str, str]":
    """Return the ``(family_keyword, literal)`` nft pair for a destination IP.

    ``host`` must be an IP **literal** — the rendered ruleset is fed to
    ``nft -f -``, so accepting an arbitrary string here would let whatever
    produced it inject rules.  Raises ``ValueError`` for anything that is not
    a valid address.
    """
    addr = ipaddress.ip_address(host)
    return ("ip6" if addr.version == 6 else "ip "), addr.compressed


def render_ruleset(
    session_id: str,
    cgroup_path: str,
    proxy_port: int,
    *,
    proxy_host: str = "127.0.0.1",
    allow_local_resolver: bool = False,
) -> str:
    """Render the per-session nft ruleset (pure — no side effects).

    Gates ONLY the session's cgroup: the single accepted destination is the
    egress proxy itself — ``<proxy_host>:<proxy_port>``, TCP — and everything
    else is rejected.  Non-matching host traffic falls through ``policy
    accept`` so only the session is constrained.

    The accept rule's address family follows ``proxy_host``:
    :class:`~server.egress_proxy.proxy.ConnectAllowlistProxy` binds ``AF_INET``
    on ``127.0.0.1``, so in practice only the IPv4 rule is emitted and IPv6 is
    rejected wholesale.  Earlier revisions carried an unconditional
    ``ip6 daddr ::1 accept`` beside the IPv4 rule; because it had no port (nor
    protocol) predicate it opened *every* service on IPv6 loopback — local
    model providers, the WS server, the webhook listener — to a confined
    session, defeating the CONNECT allowlist on any dual-stack host (#696).
    The proxy has never listened on ``::1``, so the rule was unnecessary as
    well as over-broad; it is gone rather than narrowed, and the family is now
    derived from the bind address instead of assumed.

    UDP is denied for every destination, by omission: no rule matches it, so
    it falls through to ``counter reject``.  QUIC and direct DNS therefore do
    not leave the cgroup; name resolution rides the proxy's CONNECT.
    ``allow_local_resolver`` re-opens the systemd stub at ``127.0.0.53`` for
    deployments that need runner-side DNS — scoped to port 53 (udp + tcp),
    not the whole address.
    """
    table = _table_name(session_id)
    rel, level = cgroup_rel_and_level(cgroup_path)
    port = int(proxy_port)
    if not 1 <= port <= 65535:
        raise ValueError(f"proxy_port out of range: {proxy_port!r}")
    family, host = _daddr_match(proxy_host)
    gate_lines = [
        f"    {family} daddr {host} tcp dport {port} accept",
    ]
    if allow_local_resolver:
        gate_lines += [
            f"    ip  daddr {_STUB_RESOLVER} udp dport 53 accept",
            f"    ip  daddr {_STUB_RESOLVER} tcp dport 53 accept",
        ]
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
        *, proxy_host: str = "127.0.0.1", allow_local_resolver: bool = False,
    ) -> bool:
        """Render + load the session's egress table.  Returns True on success.

        ``proxy_host`` is the address the session's proxy actually bound, and
        is the *only* destination the gate accepts; it must be an IP literal
        (see :func:`_daddr_match`).  Requires the cgroup at ``cgroup_path`` to
        already exist (nft resolves the cgroup path to an id at load time).
        Idempotent per session.
        """
        if not os.path.isdir(cgroup_path):
            logger.warning(
                "egress nft: cgroup %s does not exist — skipping enforcement "
                "for session %s (proxy-only confinement)", cgroup_path, session_id)
            return False
        try:
            ruleset = render_ruleset(
                session_id, cgroup_path, proxy_port, proxy_host=proxy_host,
                allow_local_resolver=allow_local_resolver)
        except ValueError as exc:
            logger.warning("egress nft: refusing to install for session %s: %s",
                           session_id, exc)
            return False
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
                        "proxy %s:%d)", session_id, cgroup_path, proxy_host,
                        int(proxy_port))
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
