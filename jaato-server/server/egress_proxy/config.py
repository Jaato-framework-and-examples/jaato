"""Egress allowlist policy schema + validation (Phase 5 §5.11a).

The policy is intentionally minimal and strict: an allowlist of hostnames
(optionally with a single leading ``*.`` wildcard) and ports, deny-by-default.
No deny entries, no path/regex wildcards, no content rules — the CONNECT proxy
only ever sees the destination host:port, so that is all the policy governs.

Wildcard semantics (per the design doc §4.2): ``*.foo.com`` matches
``bar.foo.com`` and ``bar.baz.foo.com`` (one or more sub-labels) but NOT the
apex ``foo.com`` — list the apex explicitly if you need it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


DEFAULT_ALLOWED_PORTS = (443,)
MAX_HOSTS = 64

# Allow-listed hostname characters: letters, digits, dot, hyphen, and a single
# leading ``*`` for the wildcard.  Deliberately strict — a hostname that enters
# a CONNECT line or an AppArmor/egress decision must not carry control chars.
_HOST_CHARS = re.compile(r"^[A-Za-z0-9.*-]+$")


def _normalize_host(host: str) -> str:
    """Lowercase and strip a trailing dot (FQDN root) for comparison."""
    return (host or "").lower().rstrip(".")


def host_matches(host: str, pattern: str) -> bool:
    """Return True if ``host`` matches allowlist ``pattern``.

    ``pattern`` is either an exact hostname or ``*.suffix``.  The wildcard
    matches one or more sub-labels but not the apex (see module docstring).
    """
    host = _normalize_host(host)
    pattern = _normalize_host(pattern)
    if not host or not pattern:
        return False
    if pattern.startswith("*."):
        suffix = pattern[1:]  # ".foo.com"
        return host.endswith(suffix) and len(host) > len(suffix)
    return host == pattern


@dataclass
class AllowlistConfig:
    """A per-session egress policy.

    Attributes:
        allowed_hosts: Exact hostnames or ``*.suffix`` wildcards.
        allowed_ports: Destination ports permitted (default: 443).
        deny_on_default: Deny anything not matched (always True for now — the
            field exists so a future "audit-only" mode can flip it).
        log_denied: Emit a log line on each denied CONNECT.
    """

    allowed_hosts: List[str]
    allowed_ports: List[int] = field(default_factory=lambda: list(DEFAULT_ALLOWED_PORTS))
    deny_on_default: bool = True
    log_denied: bool = True

    def host_allowed(self, host: str) -> bool:
        return any(host_matches(host, p) for p in self.allowed_hosts)

    def port_allowed(self, port: int) -> bool:
        return port in self.allowed_ports

    def allows(self, host: str, port: int) -> bool:
        """The single decision the proxy asks: may this CONNECT proceed?"""
        if not self.deny_on_default:
            # Audit-only mode (not wired yet): allow, decision logged upstream.
            return True
        return self.host_allowed(host) and self.port_allowed(port)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AllowlistConfig":
        """Build from a validated dict.  Call :func:`validate_allowlist` first;
        this constructor assumes the shape is already sound and only coerces
        types (lowercasing hosts, int-casting ports)."""
        hosts = [_normalize_host(str(h)) for h in (data.get("allowed_hosts") or [])]
        ports = [int(p) for p in (data.get("allowed_ports") or DEFAULT_ALLOWED_PORTS)]
        return cls(
            allowed_hosts=hosts,
            allowed_ports=ports,
            deny_on_default=bool(data.get("deny_on_default", True)),
            log_denied=bool(data.get("log_denied", True)),
        )


def validate_allowlist(data: Any) -> List[str]:
    """Validate an ``egress_allowlist`` dict; return a list of error strings.

    Empty list means valid.  Mirrors the §5.9 sub-profile-tightening validator
    posture: strict character/port/wildcard checks, a bounded host count, and
    no silent coercion of malformed input.
    """
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["egress_allowlist must be an object"]

    hosts = data.get("allowed_hosts")
    if not isinstance(hosts, list) or not hosts:
        errors.append("egress_allowlist.allowed_hosts must be a non-empty array")
        hosts = []
    if len(hosts) > MAX_HOSTS:
        errors.append(
            f"egress_allowlist.allowed_hosts has {len(hosts)} entries "
            f"(max {MAX_HOSTS})")
    for i, h in enumerate(hosts):
        if not isinstance(h, str) or not h:
            errors.append(f"allowed_hosts[{i}] must be a non-empty string")
            continue
        if not _HOST_CHARS.match(h):
            errors.append(
                f"allowed_hosts[{i}]={h!r} has invalid characters "
                "(allowed: letters, digits, '.', '-', leading '*.')")
            continue
        if "*" in h and not h.startswith("*."):
            errors.append(
                f"allowed_hosts[{i}]={h!r}: '*' is only valid as a leading "
                "'*.' wildcard")
        if h.startswith("*.") and "*" in h[2:]:
            errors.append(f"allowed_hosts[{i}]={h!r}: only one leading '*.' allowed")
        if ".." in h:
            errors.append(f"allowed_hosts[{i}]={h!r}: empty label ('..')")

    ports = data.get("allowed_ports", list(DEFAULT_ALLOWED_PORTS))
    if not isinstance(ports, list):
        errors.append("egress_allowlist.allowed_ports must be an array")
    else:
        for i, p in enumerate(ports):
            if not isinstance(p, int) or isinstance(p, bool) or not (1 <= p <= 65535):
                errors.append(
                    f"allowed_ports[{i}]={p!r} must be an integer in 1..65535")

    for key in ("deny_on_default", "log_denied"):
        if key in data and not isinstance(data[key], bool):
            errors.append(f"egress_allowlist.{key} must be a boolean")

    return errors
