"""Filesystem credential store for API-key providers — shared machinery.

Every API-key provider in this tree needs the same seven behaviours: find
the credential file under the config-root / workspace / home chain, write
it with mode 0600, read it back, distinguish *absent* from *present but
broken*, report which file was used, clear it, and build an httpx client
that honours the corporate proxy and CA bundle while validating a key.

That is ~250 lines, and it has been copied per provider.  The copies are
identical modulo a filename and a display name, which means every fix
lands in one of them: :func:`try_load_with_reason`'s distinction between
"no file" and "unreadable file" is the reason ``verify_auth`` can say
*why* a stored credential did not work, and a provider whose copy predates
it silently reports "not configured" for a corrupt file.

So the mechanics live here, parameterised by the two things that actually
differ, and a provider's ``auth.py`` keeps only what is genuinely its own:
the validation probe (which endpoint proves a key is live) and the login
flow.

Deliberately NOT a base class.  A provider's ``auth.py`` is imported as a
module by ``env.py`` and by the plugin's own command surface, and those
call sites want functions; an instance built at module scope and called
through is the smaller change and keeps the public shape
(``get_stored_api_key(workspace_path=..., config_root=...)``) identical to
the hand-written copies it replaces.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from shared.secret_repr import secret_safe_repr
from shared.session_context import get_config_root, get_workspace_root

logger = logging.getLogger(__name__)

#: Keys the store owns; everything else in the file is provider payload.
_RESERVED_KEYS = ("api_key", "created_at")


@dataclass
class StoredCredential:
    """One provider's stored API key, plus whatever else it saved with it.

    ``fields`` carries the provider-specific extras — a custom
    ``base_url``, an organization id, an endpoint — so the store does not
    need to know what any given provider persists alongside the key.
    """

    api_key: str
    created_at: float = 0.0
    fields: Dict[str, Any] = field(default_factory=dict)

    # Never print the key: a bare dataclass repr put a live ``sk-…`` into a
    # pytest failure message, and from there into scrollback and CI logs
    # (#721).  ``to_dict`` still returns the real value — this guards
    # display, not storage.
    __repr__ = secret_safe_repr("api_key")

    def get(self, name: str, default: Any = None) -> Any:
        """Read one provider-specific field."""
        return self.fields.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        """The on-disk form: the reserved keys, then the extras."""
        data: Dict[str, Any] = {
            "api_key": self.api_key,
            "created_at": self.created_at or time.time(),
        }
        data.update({k: v for k, v in self.fields.items() if v is not None})
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredCredential":
        """Parse the on-disk form.

        Raises:
            KeyError: when ``api_key`` is absent — a credential file
                without one is malformed, not empty, and the caller needs
                to hear the difference.
        """
        return cls(
            api_key=data["api_key"],
            created_at=data.get("created_at", time.time()),
            fields={k: v for k, v in data.items() if k not in _RESERVED_KEYS},
        )


class ApiKeyStore:
    """The credential file for one provider.

    Args:
        filename: The file's basename, e.g. ``"openai_auth.json"``.
        label: Human name used in log and error text, e.g. ``"OpenAI"``.
    """

    def __init__(self, filename: str, label: str) -> None:
        self.filename = filename
        self.label = label

    # ---------------------------------------------------------- location

    def path(
        self,
        for_write: bool = False,
        workspace_path: Optional[str] = None,
        config_root: Optional[str] = None,
    ) -> Path:
        """Resolve the credential file path.

        Follows the jaato convention:

        1. Project tier — ``<config_root>/<filename>`` when a config root
           is set, else ``<workspace>/.jaato/<filename>``.
        2. Home tier — ``~/.jaato/<filename>``.

        ``config_root`` / ``workspace_path`` are the explicit
        session-supplied values; when absent the ambient
        :mod:`shared.session_context` values are used, which is what makes
        a headless session read its own workspace rather than the
        daemon's cwd.

        Args:
            for_write: Choose where a save would go, rather than where a
                read would find something.
            workspace_path: Explicit workspace override.
            config_root: Explicit read-only-config root override.
        """
        workspace = workspace_path or get_workspace_root() or os.getcwd()
        effective_root = config_root or get_config_root()
        if effective_root:
            project = Path(effective_root).expanduser().resolve() / self.filename
        else:
            project = Path(workspace) / ".jaato" / self.filename
        home = Path.home() / ".jaato" / self.filename

        if for_write:
            return project if project.parent.exists() else home
        return project if project.exists() else home

    def display_path(
        self,
        workspace_path: Optional[str] = None,
        config_root: Optional[str] = None,
    ) -> Optional[str]:
        """The path of the file that WOULD be loaded, or ``None``.

        Home-relative paths are shortened to ``~/...`` so the string can
        be shown to a user without leaking their account name.
        """
        path = self.path(workspace_path=workspace_path, config_root=config_root)
        if not path.exists():
            return None
        home = Path.home()
        if path.is_relative_to(home):
            return "~/" + str(path.relative_to(home))
        return str(path)

    # -------------------------------------------------------------- I/O

    def save(
        self,
        credential: StoredCredential,
        workspace_path: Optional[str] = None,
        config_root: Optional[str] = None,
    ) -> Path:
        """Write the credential, 0600 on POSIX.  Returns the path written."""
        path = self.path(
            for_write=True,
            workspace_path=workspace_path,
            config_root=config_root,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(credential.to_dict(), handle, indent=2)
        if os.name == "posix":
            os.chmod(path, 0o600)
        return path

    def try_load_with_reason(
        self,
        workspace_path: Optional[str] = None,
        config_root: Optional[str] = None,
    ) -> Tuple[Optional[StoredCredential], Optional[str]]:
        """Load, distinguishing *absent* from *present but unreadable*.

        Returns ``(credential, reason)``:

        - ``(StoredCredential, None)`` — loaded.
        - ``(None, None)`` — no file exists.
        - ``(None, "<reason>")`` — the file exists and could not be used.

        The third case is the one that matters: without it ``verify_auth``
        reports "no key found" for a file that is right there and merely
        corrupt, and the user goes looking for the wrong problem.
        """
        path = self.path(workspace_path=workspace_path, config_root=config_root)
        if not path.exists():
            return None, None

        try:
            with open(path) as handle:
                data = json.load(handle)
        except (OSError, PermissionError) as exc:
            return None, self._warn(f"cannot read {path}: {exc}")
        except json.JSONDecodeError as exc:
            return None, self._warn(
                f"invalid JSON at {path}: {exc.msg} "
                f"(line {exc.lineno}, col {exc.colno})"
            )

        try:
            return StoredCredential.from_dict(data), None
        except (KeyError, TypeError) as exc:
            return None, self._warn(
                f"malformed credentials at {path}: missing or invalid "
                f"field ({exc})"
            )
        except Exception as exc:  # defensive — don't mask the unexpected
            return None, self._warn(
                f"unexpected error loading {path}: "
                f"{exc.__class__.__name__}: {exc}"
            )

    def _warn(self, reason: str) -> str:
        """Log a load failure at WARNING and return it for the caller."""
        logger.warning("Failed to load %s credentials: %s", self.label, reason)
        return reason

    def load(
        self,
        workspace_path: Optional[str] = None,
        config_root: Optional[str] = None,
    ) -> Optional[StoredCredential]:
        """Load, or ``None`` when absent or unreadable (logged at WARNING)."""
        credential, _ = self.try_load_with_reason(
            workspace_path=workspace_path, config_root=config_root,
        )
        return credential

    def api_key(
        self,
        workspace_path: Optional[str] = None,
        config_root: Optional[str] = None,
    ) -> Optional[str]:
        """The stored key, or ``None``."""
        credential = self.load(
            workspace_path=workspace_path, config_root=config_root,
        )
        return credential.api_key if credential else None

    def clear(
        self,
        workspace_path: Optional[str] = None,
        config_root: Optional[str] = None,
    ) -> None:
        """Delete the credential file if it exists."""
        path = self.path(workspace_path=workspace_path, config_root=config_root)
        if path.exists():
            path.unlink()


# ==================== Validation transport ====================

def create_validation_client():
    """An httpx client with proxy, Kerberos and corporate CA support.

    Key validation is the one network call a provider makes outside its
    SDK, and it must go through the same corporate plumbing as the rest —
    otherwise validation fails behind a proxy that the actual session
    would have traversed fine.
    """
    from shared.http.proxy import get_httpx_client
    from shared.ssl_helper import active_cert_bundle

    kwargs = {}
    ca_bundle = active_cert_bundle()
    if ca_bundle:
        kwargs["verify"] = ca_bundle
    return get_httpx_client(**kwargs)


def body_snippet(response: Any, limit: int = 300) -> str:
    """A short, single-line excerpt of a response body, for error detail."""
    try:
        text = response.text or ""
    except Exception:
        return ""
    text = text.strip().replace("\n", " ")
    return text[:limit] + "…" if len(text) > limit else text


def mask_key(key: Optional[str]) -> str:
    """Render a key for display: first 6 and last 4, or ``***`` if short."""
    if not key:
        return ""
    return f"{key[:6]}...{key[-4:]}" if len(key) > 12 else "***"
