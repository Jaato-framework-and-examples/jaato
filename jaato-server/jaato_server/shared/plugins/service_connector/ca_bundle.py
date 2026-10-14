"""Per-service CA bundles: one service trusts a private root, nothing else does (#1671).

A service definition may name the PEM bundle its TLS is verified against::

    # <tier>/.jaato/services/lda-wiki/_service.yaml
    name: lda-wiki
    base_url: https://wiki.lda
    ca_file: lda-root.pem          # beside this file

When ``ca_file`` is set, ``call_service`` verifies that service's
certificate chain against that bundle **alone** — the system store is not
consulted, so trusting a private root for one service does not widen trust
for any other host, any other service or any other session.  The existing
alternatives each fail one of those:

| Option | What it trusts |
|---|---|
| ``ssl_trusted: true`` | nothing is verified, any certificate passes |
| ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` | the root, for every session of every tenant on the daemon (host-scoped) |
| the system store | the root, for every process on the host |
| ``ca_file`` | the root, for this service's requests only |

This module is the one place the rules live, so ``call_service``, the
``services show`` command, ``explain services`` and ``validate`` cannot
disagree about what a ``ca_file`` value means.  It is stdlib-only.

Rules (:data:`CA_FILE_RULES` renders them):

- **Where it resolves.**  ``~`` is expanded; a relative path is resolved
  against the directory holding the definition (``<service>/`` for a
  ``_service.yaml``, ``_discovered/`` for a discovered service).  Absolute
  paths are taken as written.  ``${VAR}`` is not expanded: the bundle is a
  file of the definition, not a per-session value.
- **Read in the runner.**  The runner's confinement must be able to read
  it: the service's own directory in the user tier
  (``~/.jaato/services/**``, granted by this plugin) or the workspace
  (granted by the template).  A bundle elsewhere is refused by the kernel
  on a confined host, and reported as unreadable.
- **Never silently unverified.**  A bundle that is missing, unreadable or
  holds no certificate refuses the call, naming the file; it never falls
  back to the system store or to no verification.  With ``ca_file`` set,
  ``insecure=true`` is refused and ``ssl_trusted`` is ignored (validate
  warns about the pair): a service that names its trust root is verified
  against it.
"""

from __future__ import annotations

import os
import ssl
from pathlib import Path
from typing import List, Optional, Tuple

CA_FILE_KEY = "ca_file"

CA_FILE_RULES: Tuple[str, ...] = (
    "ca_file: <path to a PEM bundle> in <service>/_service.yaml",
    "  verifies this service's TLS against that bundle ALONE (the system store",
    "  is not consulted), so a private root is trusted for one service only.",
    "  A relative path resolves against the directory holding the definition;",
    "  ~ is expanded; ${VAR} is not.  Read in the runner: keep it beside",
    "  _service.yaml or in the workspace, which confinement already grants.",
    "  A missing, unreadable or empty bundle refuses the call; it never falls",
    "  back to the system store.  insecure=true is refused and ssl_trusted is",
    "  ignored for a service with a ca_file.  On Python 3.13+ the chain is",
    "  checked strictly (VERIFY_X509_STRICT, as for public hosts): the root",
    "  needs keyUsage keyCertSign and the leaf an Authority Key Identifier.",
)


class CABundleError(Exception):
    """A ``ca_file`` that cannot be used; the message names the file."""


def resolve_ca_file(raw: str, source_dir: Optional[str]) -> Path:
    """Resolve a ``ca_file`` value to an absolute path.

    Args:
        raw: The value as written in the definition.
        source_dir: Directory holding the definition, or ``None`` when the
            config did not come from a file (an in-memory discovered entry).

    Returns:
        The absolute path the bundle is read from.

    Raises:
        CABundleError: Empty value, or a relative path with no directory to
            resolve it against.
    """
    value = (raw or "").strip()
    if not value:
        raise CABundleError("ca_file is empty")
    path = Path(os.path.expanduser(value))
    if path.is_absolute():
        return path
    if not source_dir:
        raise CABundleError(
            f"ca_file '{value}' is relative and this service has no "
            f"definition directory to resolve it against; use an absolute path"
        )
    return Path(source_dir) / path


def ssl_context_for(path: Path) -> ssl.SSLContext:
    """Build a verifying context that trusts ``path`` and nothing else.

    ``ssl.create_default_context(cafile=...)`` does not load the system
    store when a ``cafile`` is given, which is what makes the trust
    per-service.  Hostname checking and ``CERT_REQUIRED`` stay on.

    Raises:
        CABundleError: The file is missing, unreadable, or holds no
            certificate.
    """
    if not path.is_file():
        raise CABundleError(f"ca_file {path} does not exist or is not a file")
    try:
        ctx = ssl.create_default_context(cafile=str(path))
    except PermissionError as exc:
        raise CABundleError(
            f"ca_file {path} is not readable by the runner ({exc.strerror}); "
            f"keep it beside _service.yaml or in the workspace"
        ) from exc
    except (ssl.SSLError, OSError, ValueError) as exc:
        raise CABundleError(
            f"ca_file {path} is not a usable PEM bundle: {exc}"
        ) from exc
    if not ctx.get_ca_certs():
        raise CABundleError(f"ca_file {path} holds no CA certificate")
    return ctx


def ca_file_problems(raw: object, source_dir: Optional[str]) -> List[str]:
    """What is wrong with a ``ca_file`` value, as validate reports it.

    Returns:
        Empty when the bundle resolves and loads; otherwise one sentence
        per problem.
    """
    if not isinstance(raw, str):
        return [f"ca_file must be a path string, got {type(raw).__name__}"]
    try:
        ssl_context_for(resolve_ca_file(raw, source_dir))
    except CABundleError as exc:
        return [str(exc)]
    return []


def readable_under_confinement(
    path: Path, workspace: Optional[str], user_services_dir: Optional[str],
) -> bool:
    """Whether a confined runner's profile grants a read of ``path``.

    True for a path inside the workspace (the template grants it) or inside
    ``~/.jaato/services/`` (this plugin's own contribution).  Anything else
    is refused by the kernel on a confined host; an unconfined runner reads
    it with its own uid.  Symlinks are resolved first, as the kernel does.
    """
    try:
        target = path.resolve()
    except OSError:
        return False
    for root in (workspace, user_services_dir):
        if not root:
            continue
        try:
            target.relative_to(Path(root).resolve())
            return True
        except (ValueError, OSError):
            continue
    return False
