"""Redact the secret VALUES this process holds from what it sends out (#1215).

The #863 scrub removes secret *names* from the environment a model-driven
subprocess inherits.  It cannot stop a command from reading a secret out of a
file (the workspace ``.env`` that ``config.update`` writes, a stored
``<provider>_auth.json``, a key a user pasted) and printing it.  Once printed,
the value reached every attached client, the model's history (replayed on
every later request), the session record and the traces, verbatim.

This module is the other half: a redactor built from the exact values the
runner holds, applied to text on its way out.  Matching the values the
process actually has, rather than pattern-matching things that look like
keys, does not guess.

**What is redacted** (:func:`collect_secrets`):

* every session-env value whose NAME matches :func:`redaction_patterns`.  That
  is :data:`~jaato_server.shared.secret_scrub.DEFAULT_SECRET_ENV_PATTERNS`
  plus every positive glob a profile or surface declared in
  ``scrub_secret_env``.  Exemptions (``!GH_TOKEN``), ``none`` and the #1228
  ``app://`` grants are deliberately NOT subtracted.  They decide what a
  subprocess may *inherit*, which is a different question from what may be
  *printed*: a granted ``GH_TOKEN`` stays in ``gh``'s environment and is still
  redacted if ``gh`` echoes it;
* the provider credential: ``plugin_configs.<provider>.api_key`` /
  ``oauth_token`` literals, and the live provider's ``_api_key`` /
  ``_oauth_token`` / ``_token`` once it exists;
* secret-named string leaves of every ``*_auth.json`` in the directories the
  providers read stored credentials from.

Each exact occurrence becomes ``‹redacted:NAME›``.  The name is kept so the
model still knows *which* credential was there.

**The floor.**  A value shorter than :data:`MIN_REDACT_LENGTH` is not
redacted.  Otherwise a ``*_TOKEN`` set to ``true`` or a port number would
mangle unrelated output.  Such a value is named, never shown, in one WARNING
per name per process, so it is a visible gap rather than a silent one.

**Stated limits.**  An encoded form (base64, URL-encoding, hex) is not caught,
nor is a value the model re-types differently (split with spaces, reversed,
partially quoted).  A secret split across two stream chunks IS caught, by
:class:`StreamCarry`.

**Binary payloads are left alone.**  :meth:`SecretRedactor.redact` skips
``bytes``, a ``data_b64`` key, and the ``data`` of a ``{mime_type, data}``
dict, because rewriting base64 would corrupt a payload rather than protect it.

The redactor is process-wide (:func:`current_redactor` /
:func:`install_redactor`), as the #1228 grant is, because the environment it
describes is process-wide: a runner serves one session at a time.  A process
that never installed one (the daemon, the embedded client) holds an empty
redactor, and every call is then a cheap no-op.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import (
    Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

from jaato_server.shared.secret_scrub import (
    DEFAULT_SECRET_ENV_PATTERNS,
    SCRUB_SURFACES,
    matches_secret,
    normalize_scrub_patterns,
)

logger = logging.getLogger(__name__)

#: Values shorter than this are not redacted (see the module docstring).
MIN_REDACT_LENGTH = 12

#: The replacement.  Stable, so a reader can tell two redactions of one
#: credential are the same credential.
MARKER_TEMPLATE = "‹redacted:{name}›"

#: Keys of a ``*_auth.json`` whose string values are credentials.  Matched
#: case-insensitively against the leaf's own key.
_AUTH_FILE_SECRET_KEYS = frozenset({
    "api_key", "apikey", "key", "token", "access_token", "refresh_token",
    "id_token", "oauth_token", "auth_token", "secret", "client_secret",
    "password", "copilot_token",
})

#: Provider attributes that hold a resolved credential.  Measured across
#: ``model_provider/*/provider.py``: these three are the only spellings.
_PROVIDER_CREDENTIAL_ATTRS = ("_api_key", "_oauth_token", "_token")

#: ``plugin_configs.<provider>`` keys that may carry a literal credential.
_PROVIDER_CONFIG_SECRET_KEYS = ("api_key", "oauth_token")

#: Dict keys whose string value is base64 binary, never text.
_BINARY_KEYS = frozenset({"data_b64"})

_warned_short: set = set()
_warned_lock = threading.Lock()


def _marker(name: str) -> str:
    return MARKER_TEMPLATE.format(name=name)


class SecretRedactor:
    """Replaces exact occurrences of known secret values with a marker.

    Immutable once built.  An empty redactor (``not redactor.active``) returns
    every input unchanged, and is what a process that never installed one
    holds.

    Args:
        secrets: ``(name, value)`` pairs.  The first name given for a value
            wins, so pass the most specific source first.  Values below
            :data:`MIN_REDACT_LENGTH` are skipped and warned about once.
    """

    def __init__(self, secrets: Iterable[Tuple[str, str]] = ()) -> None:
        by_value: Dict[str, str] = {}
        for name, value in secrets:
            if not isinstance(value, str) or not value:
                continue
            if len(value) < MIN_REDACT_LENGTH:
                _warn_short_once(name)
                continue
            by_value.setdefault(value, name)
        self._names: Dict[str, str] = by_value
        self._pattern: Optional[re.Pattern] = None
        self._by_first: Dict[str, List[str]] = {}
        self._max_len = 0
        if by_value:
            ordered = sorted(by_value, key=len, reverse=True)
            self._pattern = re.compile("|".join(re.escape(v) for v in ordered))
            for v in by_value:
                self._by_first.setdefault(v[0], []).append(v)
            self._max_len = max(len(v) for v in by_value)

    @property
    def active(self) -> bool:
        """True when there is at least one value to redact."""
        return self._pattern is not None

    @property
    def names(self) -> List[str]:
        """The names being redacted, sorted.  Never the values."""
        return sorted(set(self._names.values()))

    def redact_text(self, text: str) -> str:
        """Return *text* with every known secret value replaced by its marker."""
        if self._pattern is None or not text:
            return text
        return self._pattern.sub(lambda m: _marker(self._names[m.group(0)]), text)

    def redact(self, obj: Any) -> Any:
        """Redact every string inside a JSON-shaped structure.

        Walks dicts, lists and tuples; returns the SAME object when nothing
        changed, so an unaffected frame costs no copy.  Keys are not
        rewritten.  Binary payloads are skipped (see the module docstring),
        and so is any object that is not a str / dict / list / tuple.
        """
        if self._pattern is None:
            return obj
        return self._walk(obj)

    def _walk(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.redact_text(obj)
        if isinstance(obj, dict):
            return self._walk_dict(obj)
        if isinstance(obj, (list, tuple)):
            items = [self._walk(v) for v in obj]
            if all(a is b for a, b in zip(items, obj)):
                return obj
            return type(obj)(items) if isinstance(obj, tuple) else items
        return obj

    def _walk_dict(self, obj: Dict[Any, Any]) -> Dict[Any, Any]:
        media = "mime_type" in obj and "data" in obj
        changed: Dict[Any, Any] = {}
        for key, value in obj.items():
            if key in _BINARY_KEYS or (media and key == "data"):
                continue
            new = self._walk(value)
            if new is not value:
                changed[key] = new
        if not changed:
            return obj
        out = dict(obj)
        out.update(changed)
        return out

    def held_suffix_len(self, text: str) -> int:
        """Length of the longest suffix of *text* that could begin a secret.

        That suffix may be the first half of a value split across two stream
        chunks, so :class:`StreamCarry` holds it back until the next chunk
        decides.  Only a PROPER prefix counts: a complete value would already
        have been replaced.
        """
        if self._pattern is None or not text:
            return 0
        start = max(0, len(text) - (self._max_len - 1))
        for i in range(start, len(text)):
            candidates = self._by_first.get(text[i])
            if not candidates:
                continue
            tail = text[i:]
            if any(len(v) > len(tail) and v.startswith(tail) for v in candidates):
                return len(text) - i
        return 0


class StreamCarry:
    """Per-stream carry-over, so a secret split across chunks is still caught.

    A stream is any key the caller chooses (a tool call's id, a request's
    output source).  :meth:`feed` returns what is safe to emit now and holds
    back a suffix that could be the start of a secret.  :meth:`flush` returns
    what is held when the stream ends.  Thread-safe: the runner's output
    callbacks fire from worker threads.
    """

    def __init__(self) -> None:
        self._held: Dict[Hashable, str] = {}
        self._lock = threading.Lock()

    def feed(self, redactor: SecretRedactor, key: Hashable, text: str) -> str:
        """Redact *text* joined to what *key* held, and hold back a new tail."""
        with self._lock:
            joined = self._held.pop(key, "") + (text or "")
            out = redactor.redact_text(joined)
            keep = redactor.held_suffix_len(out)
            if keep:
                self._held[key] = out[-keep:]
                out = out[:-keep]
            return out

    def flush(self, key: Hashable) -> str:
        """Release (and forget) what *key* is holding."""
        with self._lock:
            return self._held.pop(key, "")

    def flush_where(self, predicate: Any) -> List[Tuple[Hashable, str]]:
        """Release every held stream whose key satisfies *predicate*."""
        with self._lock:
            keys = [k for k in self._held if predicate(k)]
            return [(k, self._held.pop(k)) for k in keys]


_EMPTY = SecretRedactor()
_current: SecretRedactor = _EMPTY


def current_redactor() -> SecretRedactor:
    """The redactor this process installed, or an empty one."""
    return _current


def install_redactor(redactor: Optional[SecretRedactor]) -> SecretRedactor:
    """Replace the process-wide redactor (``None`` installs the empty one)."""
    global _current
    _current = redactor if redactor is not None else _EMPTY
    return _current


def _warn_short_once(name: str) -> None:
    with _warned_lock:
        if name in _warned_short:
            return
        _warned_short.add(name)
    logger.warning(
        "secret redaction: %s is shorter than %d characters and is NOT "
        "redacted from tool output, model output or records (#1215); a short "
        "value would mangle unrelated text",
        name, MIN_REDACT_LENGTH,
    )


def redaction_patterns(declared: Iterable[Any] = ()) -> Tuple[str, ...]:
    """The name globs whose values are redacted.

    The framework default set plus every POSITIVE glob in *declared* (each a
    ``scrub_secret_env`` value in any accepted shape).  Exemptions and
    ``none`` narrow only what a subprocess inherits, so they are dropped
    here.  A malformed value contributes nothing; the default set always
    applies.
    """
    out: List[str] = list(DEFAULT_SECRET_ENV_PATTERNS)
    for value in declared:
        if value is None:
            continue
        try:
            patterns = normalize_scrub_patterns(value)
        except ValueError:
            continue
        for p in patterns:
            p = str(p).strip()
            if p and not p.startswith("!") and p not in out:
                out.append(p)
    return tuple(out)


def env_secrets(
    env: Optional[Mapping[str, str]], patterns: Sequence[str],
) -> List[Tuple[str, str]]:
    """``(name, value)`` for each env entry whose name matches *patterns*.

    Exemptions are not honoured (the patterns passed here carry none, see
    :func:`redaction_patterns`).
    """
    return [
        (k, v) for k, v in sorted((env or {}).items())
        if isinstance(v, str) and v and matches_secret(k, patterns)
    ]


def stored_auth_secrets(directories: Iterable[Optional[str]]) -> List[Tuple[str, str]]:
    """Secret-named string leaves of every ``*_auth.json`` in *directories*.

    Best-effort: a missing directory, an unreadable file or malformed JSON
    contributes nothing.  The name is ``<file>:<key>``.
    """
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for directory in directories:
        if not directory:
            continue
        try:
            files = sorted(Path(directory).glob("*_auth.json"))
        except OSError:
            continue
        for path in files:
            try:
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            _collect_auth_leaves(data, path.name, out)
    return out


def _collect_auth_leaves(obj: Any, fname: str, out: List[Tuple[str, str]]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                if str(key).lower() in _AUTH_FILE_SECRET_KEYS:
                    out.append((f"{fname}:{key}", value))
            else:
                _collect_auth_leaves(value, fname, out)
    elif isinstance(obj, list):
        for value in obj:
            _collect_auth_leaves(value, fname, out)


def provider_config_secrets(
    plugin_configs: Optional[Mapping[str, Any]], provider_name: Optional[str],
) -> List[Tuple[str, str]]:
    """Literal ``api_key`` / ``oauth_token`` in ``plugin_configs.<provider>``."""
    if not provider_name or not isinstance(plugin_configs, Mapping):
        return []
    cfg = plugin_configs.get(provider_name)
    if not isinstance(cfg, Mapping):
        return []
    return [
        (f"{provider_name}.{k}", cfg[k]) for k in _PROVIDER_CONFIG_SECRET_KEYS
        if isinstance(cfg.get(k), str)
    ]


def provider_credential_secrets(provider: Any, provider_name: str = "provider") -> List[Tuple[str, str]]:
    """The credential a live provider resolved in ``initialize()``, if any."""
    if provider is None:
        return []
    out: List[Tuple[str, str]] = []
    for attr in _PROVIDER_CREDENTIAL_ATTRS:
        value = getattr(provider, attr, None)
        if isinstance(value, str) and value:
            out.append((f"{provider_name}{attr}", value))
    return out


def stored_auth_directories(
    workspace_path: Optional[str], config_root: Optional[str],
) -> List[str]:
    """Where providers look for ``<provider>_auth.json``: config root, workspace, home."""
    dirs: List[str] = []
    if config_root:
        dirs.append(config_root)
    if workspace_path:
        dirs.append(os.path.join(workspace_path, ".jaato"))
    home = os.path.expanduser("~")
    if home and home != "~":
        dirs.append(os.path.join(home, ".jaato"))
    return dirs


def declared_scrub_values(
    profile_scrub: Any, plugin_configs: Optional[Mapping[str, Any]],
) -> List[Any]:
    """Every ``scrub_secret_env`` value a session declared, profile first."""
    values: List[Any] = [profile_scrub]
    if isinstance(plugin_configs, Mapping):
        for surface in SCRUB_SURFACES:
            cfg = plugin_configs.get(surface)
            if isinstance(cfg, Mapping) and "scrub_secret_env" in cfg:
                values.append(cfg["scrub_secret_env"])
    return values


# --------------------------------------------------------------------------
# The sources a runner builds its redactor from, and the two rebuild points.
#
# A runner configures the sources once per session env (bootstrap and every
# ``session.reload_env``), and every provider the session creates later --
# lazily on the first turn, on a tier switch, on a reload -- adds its
# resolved credential through :func:`note_provider_credential`.  A process
# that never configured sources (the daemon, the embedded client) ignores
# that call, so its redactor stays empty.
# --------------------------------------------------------------------------

_sources_lock = threading.Lock()
_sources: Optional[Dict[str, Any]] = None


def configure_redaction_sources(
    session_env: Optional[Mapping[str, str]],
    *,
    plugin_configs: Optional[Mapping[str, Any]] = None,
    provider_name: Optional[str] = None,
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
    profile_scrub: Any = None,
) -> SecretRedactor:
    """Record where this session's secrets come from, then build and install.

    Replaces what a previous call recorded (a reload is a REPLACEMENT, like
    ``apply_session_env``), except the provider credentials already noted,
    which describe providers that still exist until they are rebuilt.
    ``session_env=None`` keeps the previous env.
    """
    global _sources
    with _sources_lock:
        previous = _sources or {}
        _sources = {
            "session_env": dict(
                session_env if session_env is not None
                else previous.get("session_env") or {}
            ),
            "plugin_configs": plugin_configs if plugin_configs is not None
            else previous.get("plugin_configs"),
            "provider_name": provider_name or previous.get("provider_name"),
            "workspace_path": workspace_path or previous.get("workspace_path"),
            "config_root": config_root or previous.get("config_root"),
            "profile_scrub": profile_scrub if profile_scrub is not None
            else previous.get("profile_scrub"),
            "provider_secrets": list(previous.get("provider_secrets") or []),
        }
        return _rebuild_locked()


def note_provider_credential(
    provider: Any, provider_name: Optional[str], api_key: Optional[str] = None,
) -> None:
    """Add a freshly created provider's credential to the redactor.

    Called by ``JaatoRuntime.create_provider``.  A no-op in a process that
    never called :func:`configure_redaction_sources`.  Never raises.
    """
    try:
        name = provider_name or "provider"
        found = provider_credential_secrets(provider, name)
        if isinstance(api_key, str) and api_key:
            found.append((f"{name}.api_key", api_key))
        with _sources_lock:
            if _sources is None or not found:
                return
            known = _sources["provider_secrets"]
            fresh = [pair for pair in found if pair not in known]
            if not fresh:
                return
            known.extend(fresh)
            _rebuild_locked()
    except Exception:  # noqa: BLE001 -- redaction must never fail a provider
        logger.debug("secret redaction: could not note provider credential",
                     exc_info=True)


def reset_redaction_sources() -> None:
    """Forget the sources and install the empty redactor (tests, slot reuse)."""
    global _sources
    with _sources_lock:
        _sources = None
    install_redactor(None)


def _rebuild_locked() -> SecretRedactor:
    src = _sources or {}
    plugin_configs = src.get("plugin_configs")
    patterns = redaction_patterns(
        declared_scrub_values(src.get("profile_scrub"), plugin_configs),
    )
    secrets: List[Tuple[str, str]] = []
    secrets.extend(env_secrets(src.get("session_env"), patterns))
    secrets.extend(provider_config_secrets(plugin_configs, src.get("provider_name")))
    secrets.extend(src.get("provider_secrets") or [])
    secrets.extend(stored_auth_secrets(stored_auth_directories(
        src.get("workspace_path"), src.get("config_root"),
    )))
    redactor = install_redactor(SecretRedactor(secrets))
    if redactor.active:
        logger.info(
            "secret redaction: %d credential(s) redacted from output (#1215): %s",
            len(redactor.names), ", ".join(redactor.names),
        )
    return redactor
