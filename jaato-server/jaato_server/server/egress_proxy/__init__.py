"""Per-session egress allowlist proxy (Phase 5 §5.11).

A confined runner has legitimate outbound network access (model-provider API,
package metadata, MCP transports) but AppArmor's network rules are
per-socket-family, not per-destination-host — so a model-driven subprocess can
also reach arbitrary hosts and exfiltrate data.  This package closes that gap
with a per-session **hostname allowlist** enforced by a small stdlib CONNECT
proxy the runner is pointed at via ``HTTPS_PROXY``.

See ``docs/design/phase5_5_11_egress_proxy_spike.md`` for the design rationale
(Shape B: custom stdlib CONNECT-allowlist proxy, chosen over mitmproxy for the
stdlib-only daemon-critical-path policy).

Public surface:
- :class:`AllowlistConfig` / :func:`validate_allowlist` — the policy schema.
- :class:`ConnectAllowlistProxy` — the per-session proxy thread.
- :class:`EgressProxyManager` — per-session lifecycle (spawn / teardown).
- :class:`EgressEnforcementError` — strict-posture enforcement failure.
"""

from .config import AllowlistConfig, validate_allowlist
from .errors import EgressEnforcementError
from .proxy import ConnectAllowlistProxy
from .manager import EgressProxyManager

__all__ = [
    "AllowlistConfig",
    "validate_allowlist",
    "ConnectAllowlistProxy",
    "EgressProxyManager",
    "EgressEnforcementError",
]
