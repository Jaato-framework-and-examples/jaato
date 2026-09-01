"""Egress-enforcement exceptions.

Kept in a leaf module (no imports from the rest of the package) so the
session-spawn path can import the exception type without pulling in the proxy
or nft machinery.
"""

from __future__ import annotations


class EgressEnforcementError(RuntimeError):
    """Hard egress enforcement was *required* but could not be installed.

    Raised only under the strict posture
    (``JAATO_EGRESS_NFT_ENFORCE=strict``).  Under the default best-effort
    posture the same conditions degrade to proxy-only confinement plus a
    warning.  The session-spawn path lets this propagate: a session that asked
    for a kernel-enforced egress gate and did not get one must not start
    unconfined (deny on failure to apply the policy).
    """
