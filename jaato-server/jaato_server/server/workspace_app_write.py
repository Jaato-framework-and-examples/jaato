"""Write a binding's files into a workspace on an application's behalf (1.30).

An application that holds a per-user secret (the web coder's GitHub grants)
binds it to a workspace by putting a REFERENCE in the workspace ``.env``
(``GH_TOKEN=app://github``); the daemon resolves only references it finds
there.  The application writes that line itself when it can reach the
workspace.  When it runs as an account with no access to the daemon's
workspace root, it asks the daemon over its bind channel instead
(``workspace.app_write``), and this module is what the daemon does with the
request.

Everything here is policy the daemon enforces whatever the application sends:

* an ``env`` value is an ``app://`` reference or ``None``.  A literal is
  refused, so the verb can never write a secret to disk; a removal leaves a
  line whose value is not a reference alone (``kept-literal``), because a
  token the user typed there is theirs;
* a file path is on an allow-list (:func:`allowed_file_path`) and is resolved
  with symlinks followed, component by component, before anything is created
  or written: a workspace is model-writable, so a symlink planted there must
  not carry a root daemon's write outside it;
* a ``managed_by`` file is written only when absent or when its first line is
  that owner's ``jaato-managed:`` marker, and removed only then.

Validation is all-or-nothing and runs before any write
(:func:`validate_request`); what each item then did is reported per item.
Stdlib only, no daemon state: the caller checks the connection and the
ownership, and hands this module a resolved workspace directory.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from jaato_server.server.contained_write import PathLeavesRoot, atomic_write_bytes, contained_dir
from jaato_server.shared.workspace_ownership import inherit_owner
from jaato_server.shared.plugins.subagent.config import parse_app_secret_reference

#: An env name this verb may set: an ordinary upper-case variable name.
ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

#: The one home file the verb may write.
GITCONFIG_PATH = ".home/.gitconfig"
#: The directory an instruction file may be written into (one level, ``.md``).
INSTRUCTIONS_DIR = ".jaato/instructions"

_MARKER_RE = re.compile(r"^<!--\s*jaato-managed:\s*(\S+)\s+v(\d+)\b.*-->\s*$")

#: Bound on one file's content; the files this verb exists for are small.
MAX_FILE_BYTES = 256 * 1024


class AppWriteRefused(ValueError):
    """A request the rules refuse; nothing was written."""


def allowed_file_path(path: str) -> bool:
    """Whether ``path`` (workspace-relative) is one the verb may write."""
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
        return False
    parts = path.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return False
    if path == GITCONFIG_PATH:
        return True
    head, _, name = path.rpartition("/")
    return head == INSTRUCTIONS_DIR and name.endswith(".md") and len(name) > 3


def validate_request(env: Any, files: Any) -> Tuple[Dict[str, Optional[str]], List[Dict[str, Any]]]:
    """Check the whole request before anything is written.

    Returns the normalised ``(env, files)``.  Raises :class:`AppWriteRefused`
    naming the first item the rules refuse.
    """
    if not isinstance(env, dict):
        raise AppWriteRefused("env must be an object")
    if not isinstance(files, list):
        raise AppWriteRefused("files must be a list")
    clean_env = {name: _validate_env_item(name, value) for name, value in env.items()}
    return clean_env, [_validate_file_item(entry) for entry in files]


def _validate_env_item(name: Any, value: Any) -> Optional[str]:
    """One ``env`` entry: a variable name, and an ``app://`` reference or ``None``."""
    if not isinstance(name, str) or not ENV_NAME_RE.match(name):
        raise AppWriteRefused(f"env name {name!r} is not a variable name")
    if value is not None and parse_app_secret_reference(value) is None:
        raise AppWriteRefused(
            f"env {name}: only an app:// reference may be written (never a literal value)")
    return value


def _validate_file_item(entry: Any) -> Dict[str, Any]:
    """One ``files`` entry: an allow-listed path, bounded content, and an owner for a removal."""
    if not isinstance(entry, dict):
        raise AppWriteRefused("each file must be an object")
    path = entry.get("path")
    if not allowed_file_path(path):
        raise AppWriteRefused(f"file {path!r} is not on the allow-list")
    content = entry.get("content")
    managed_by = entry.get("managed_by")
    if content is not None and not isinstance(content, str):
        raise AppWriteRefused(f"file {path}: content must be a string")
    if content is not None and len(content.encode("utf-8")) > MAX_FILE_BYTES:
        raise AppWriteRefused(f"file {path}: content over {MAX_FILE_BYTES} bytes")
    if managed_by is not None and (not isinstance(managed_by, str) or not managed_by.strip()):
        raise AppWriteRefused(f"file {path}: managed_by must be a non-empty string")
    if content is None and managed_by is None:
        raise AppWriteRefused(f"file {path}: a removal needs managed_by")
    return {"path": path, "content": content, "managed_by": managed_by}


def upsert_env_line(body: str, key: str, value: Optional[str]) -> str:
    """``key=value`` upserted into (or removed from) a ``.env`` body.

    The same text transform as the web coder BFF's ``upsertEnvLine``: every
    existing assignment of ``key`` (``export`` or not) is dropped, the new one
    appended, other lines kept as written.
    """
    lines = body.split("\n") if body else []
    pattern = re.compile(r"^\s*(?:export\s+)?" + re.escape(key) + r"\s*=")
    kept = [ln for ln in lines if not pattern.match(ln)]
    if kept and kept[-1] == "":
        kept = kept[:-1]
    if value is not None:
        kept.append(f"{key}={value}")
    return "\n".join(kept) + "\n" if kept else ""


def _current_env_value(body: str, key: str) -> Optional[str]:
    pattern = re.compile(r"^\s*(?:export\s+)?" + re.escape(key) + r"\s*=(.*)$")
    value = None
    for ln in body.split("\n"):
        m = pattern.match(ln)
        if m:
            value = m.group(1).strip().strip('"').strip("'")
    return value


def _read_nofollow(path: str) -> Optional[str]:
    """The file's text, ``None`` when absent.  Refuses a symlink."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _atomic_write(path: str, body: str, mode: int, workspace: str) -> None:
    """Temp file in the same directory, then ``os.replace`` (shared helper).

    The replacement is a new file owned by the daemon, so it is handed to
    the workspace's owner (:mod:`jaato_server.shared.workspace_ownership`): a ``.env`` or a
    ``0600`` ``.gitconfig`` the workspace's runner could not read would
    break the very binding this verb exists to write.
    """
    atomic_write_bytes(path, body.encode("utf-8"), mode)
    inherit_owner(path, workspace)


def _dir_mode(part: str) -> int:
    return 0o700 if part == ".home" else 0o755


def _contained_dir(root: str, rel_dir: str, create: bool) -> Optional[str]:
    """The real directory ``root/rel_dir`` (see :func:`.contained_write.contained_dir`)."""
    try:
        return contained_dir(root, rel_dir, create, _dir_mode)
    except PathLeavesRoot as exc:
        raise AppWriteRefused(f"{rel_dir} leaves the workspace or is not a directory") from exc


def _marker_owner(body: str) -> Optional[str]:
    first = body.split("\n", 1)[0]
    m = _MARKER_RE.match(first.strip())
    return m.group(1) if m else None


def apply_env(root: str, env: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Apply ``env`` to ``root/.env``; one action per name."""
    if not env:
        return {}
    path = os.path.join(root, ".env")
    if os.path.islink(path):
        return {name: "error" for name in env}
    body = _read_nofollow(path) or ""
    actions: Dict[str, str] = {}
    new_body = body
    for name, value in env.items():
        current = _current_env_value(new_body, name)
        if value is None:
            if current is None:
                actions[name] = "absent"
                continue
            if parse_app_secret_reference(current) is None:
                actions[name] = "kept-literal"
                continue
            new_body = upsert_env_line(new_body, name, None)
            actions[name] = "removed"
            continue
        if current == value:
            actions[name] = "unchanged"
            continue
        new_body = upsert_env_line(new_body, name, value)
        actions[name] = "written"
    if new_body != body:
        mode = 0o600
        try:
            mode = os.stat(path).st_mode & 0o777
        except FileNotFoundError:
            pass
        _atomic_write(path, new_body, mode, root)
    return actions


def apply_file(root: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Write, refresh, leave or remove one allow-listed file under ``root``."""
    rel = entry["path"]
    content: Optional[str] = entry["content"]
    managed_by: Optional[str] = entry["managed_by"]
    rel_dir, _, name = rel.rpartition("/")
    try:
        directory = _contained_dir(root, rel_dir, create=content is not None)
        if directory is None:
            return {"path": rel, "action": "absent"}
        dest = os.path.join(directory, name)
        if os.path.islink(dest):
            return {"path": rel, "action": "error", "detail": "the destination is a symlink"}
        current = _read_nofollow(dest)
        if managed_by is not None and current is not None and _marker_owner(current) != managed_by:
            return {"path": rel, "action": "skipped-user-file"}
        if content is None:
            if current is None:
                return {"path": rel, "action": "absent"}
            os.unlink(dest)
            return {"path": rel, "action": "removed"}
        if current == content:
            return {"path": rel, "action": "unchanged"}
        mode = 0o600 if rel == GITCONFIG_PATH else 0o644
        _atomic_write(dest, content, mode, root)
        return {"path": rel, "action": "written"}
    except AppWriteRefused as exc:
        return {"path": rel, "action": "error", "detail": str(exc)}
    except OSError as exc:
        return {"path": rel, "action": "error", "detail": exc.strerror or str(exc)}


def apply_request(root: str, env: Any, files: Any) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Validate the whole request, then apply it under the resolved ``root``."""
    clean_env, clean_files = validate_request(env, files)
    root = os.path.realpath(root)
    env_actions = apply_env(root, clean_env)
    file_actions = [apply_file(root, entry) for entry in clean_files]
    return env_actions, file_actions
