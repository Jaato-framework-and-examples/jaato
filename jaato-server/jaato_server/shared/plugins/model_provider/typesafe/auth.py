"""Stored credentials for the TypeSafe decision provider (``typesafe-auth``).

The stored key is the LAST place the provider looks, beneath the profile
knob and the two environment variables (``env.py``).  It lets a daemon
deployment configure a key from the prompt.

Storage follows jaato convention:
1. ``<config_root>/typesafe_auth.json``, else
   ``<workspace>/.jaato/typesafe_auth.json`` (project tier);
2. ``~/.jaato/typesafe_auth.json`` (user tier).

A key is validated with one ``noul`` question to ``/v1/systemone``.
TypeSafe has no cheaper authenticated endpoint; the question is a few
hundred input tokens.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from jaato_server.shared.secret_repr import secret_safe_repr
from jaato_server.shared.session_context import get_config_root, get_workspace_root

from .env import DEFAULT_BASE_URL

logger = logging.getLogger(__name__)

_FILENAME = "typesafe_auth.json"

#: The path of the decision endpoint under the API root.
SYSTEMONE_PATH = "/v1/systemone"

#: The model the validation question is sent to.
VALIDATION_MODEL = "jev-latest"


@dataclass
class TypeSafeCredentials:
    """A stored TypeSafe key, and the API root it was validated against."""

    api_key: str
    created_at: float
    base_url: Optional[str] = None

    __repr__ = secret_safe_repr("api_key")

    def to_dict(self) -> dict:
        data = {"api_key": self.api_key, "created_at": self.created_at}
        if self.base_url:
            data["base_url"] = self.base_url
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TypeSafeCredentials":
        return cls(api_key=data["api_key"],
                   created_at=data.get("created_at", time.time()),
                   base_url=data.get("base_url"))


def _storage_path(for_write: bool = False, workspace_path: Optional[str] = None,
                  config_root: Optional[str] = None) -> Path:
    workspace = workspace_path or get_workspace_root() or os.getcwd()
    root = config_root or get_config_root()
    if root:
        project = Path(root).expanduser().resolve() / _FILENAME
    else:
        project = Path(workspace) / ".jaato" / _FILENAME
    home = Path.home() / ".jaato" / _FILENAME
    if for_write:
        return project if project.parent.exists() else home
    return project if project.exists() else home


def save_credentials(credentials: TypeSafeCredentials,
                     workspace_path: Optional[str] = None) -> Path:
    """Write the credentials (``0600`` on Unix); returns the path."""
    path = _storage_path(for_write=True, workspace_path=workspace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(credentials.to_dict(), f, indent=2)
    if os.name == "posix":
        os.chmod(path, 0o600)
    return path


def try_load_credentials_with_reason(
    workspace_path: Optional[str] = None, config_root: Optional[str] = None,
) -> Tuple[Optional[TypeSafeCredentials], Optional[str]]:
    """``(credentials, None)``, ``(None, None)`` when absent, or
    ``(None, reason)`` when a file exists and cannot be read."""
    path = _storage_path(workspace_path=workspace_path, config_root=config_root)
    if not path.exists():
        return None, None
    try:
        with open(path) as f:
            data = json.load(f)
        return TypeSafeCredentials.from_dict(data), None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        reason = f"cannot load {path}: {exc.__class__.__name__}: {exc}"
        logger.warning("TypeSafe credentials: %s", reason)
        return None, reason


def load_credentials(workspace_path: Optional[str] = None,
                     config_root: Optional[str] = None) -> Optional[TypeSafeCredentials]:
    """The stored credentials, or ``None`` when absent or unreadable."""
    return try_load_credentials_with_reason(workspace_path, config_root)[0]


def get_stored_api_key(workspace_path: Optional[str] = None,
                       config_root: Optional[str] = None) -> Optional[str]:
    creds = load_credentials(workspace_path, config_root)
    return creds.api_key if creds else None


def get_credential_file_path(workspace_path: Optional[str] = None,
                             config_root: Optional[str] = None) -> Optional[str]:
    """The file a stored key would be read from, or ``None``."""
    path = _storage_path(workspace_path=workspace_path, config_root=config_root)
    if not path.exists():
        return None
    home = Path.home()
    return ("~/" + str(path.relative_to(home))) if path.is_relative_to(home) else str(path)


def clear_credentials(workspace_path: Optional[str] = None,
                      config_root: Optional[str] = None) -> None:
    path = _storage_path(workspace_path=workspace_path, config_root=config_root)
    if path.exists():
        path.unlink()


def _classify_status(status: int, snippet: str) -> Tuple[bool, str]:
    if 200 <= status < 300:
        return True, ""
    if status == 401:
        return False, f"authentication_error: {status}: {snippet}"
    if status == 403:
        return False, f"forbidden: {status}: {snippet}"
    if status == 429:
        return False, f"rate_limit: {status}: {snippet}"
    if 500 <= status < 600:
        return False, f"server_error: {status}: {snippet}"
    return False, f"http_error: {status}: {snippet}"


def validate_api_key(api_key: str, base_url: Optional[str] = None) -> Tuple[bool, str]:
    """Send one ``noul`` question with ``api_key``; ``(valid, detail)``.

    ``detail`` starts with a code: ``authentication_error``,
    ``forbidden``, ``rate_limit``, ``server_error``, ``http_error`` or
    ``network_error``; it is empty when the key is valid.
    """
    from jaato_server.shared.http.proxy import get_httpx_client
    from jaato_server.shared.ssl_helper import active_cert_bundle

    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + SYSTEMONE_PATH
    body = {"model": VALIDATION_MODEL, "state": "jaato key check",
            "questions": {"ok": {"type": "noul",
                                 "instructions": "Is this text in English?"}}}
    try:
        kwargs = {}
        bundle = active_cert_bundle()
        if bundle:
            kwargs["verify"] = bundle
        client = get_httpx_client(**kwargs)
        response = client.post(url, json=body, timeout=30, headers={
            "Authorization": f"Bearer {api_key}", "Accept": "application/json"})
    except Exception as exc:
        return False, f"network_error: {exc}"
    snippet = (getattr(response, "text", "") or "").strip().replace("\n", " ")[:300]
    return _classify_status(response.status_code, snippet)


def login_with_key(api_key: str, base_url: Optional[str] = None,
                   on_message: Optional[Callable[[str], None]] = None,
                   workspace_path: Optional[str] = None) -> Optional[TypeSafeCredentials]:
    """Validate ``api_key`` and store it; ``None`` when validation fails."""
    valid, detail = validate_api_key(api_key, base_url)
    if not valid:
        if on_message:
            on_message(f"API key validation failed ({detail.split(':', 1)[0]}).")
        return None
    creds = TypeSafeCredentials(api_key=api_key, created_at=time.time(),
                                base_url=base_url if base_url != DEFAULT_BASE_URL else None)
    path = save_credentials(creds, workspace_path=workspace_path)
    if on_message:
        on_message(f"API key validated and saved to {path}.")
    return creds
