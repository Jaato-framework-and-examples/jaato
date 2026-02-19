"""Content-addressed, thread-safe cache for instruction token counts.

Lives on SessionManager (daemon-scoped) and is passed through JaatoRuntime
so all sessions in the same process share counts.  This eliminates redundant
``provider.count_tokens()`` API calls for deterministic instruction texts
(system instructions, plugin instructions, framework constants) across
session creates and restores.

Key: ``(provider_name, sha256(text)[:16])`` → ``token_count``.
"""

import hashlib
import threading
from typing import Dict, Optional, Tuple


class InstructionTokenCache:
    """Thread-safe, content-addressed token count cache.

    Keyed by ``(provider_name, sha256_prefix)`` so identical text counted
    by the same provider returns instantly on subsequent calls.  The cache
    is scoped to the process lifetime (daemon mode) and shared across all
    sessions via ``JaatoRuntime.instruction_token_cache``.

    Thread safety: a ``threading.Lock`` guards all reads and writes so
    background token-counting threads can call ``put()`` concurrently
    with the main thread calling ``get()``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: Dict[Tuple[str, str], int] = {}

    @staticmethod
    def _key(provider_name: str, text: str) -> Tuple[str, str]:
        """Build a cache key from provider name and text content."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return (provider_name, digest)

    def get(self, provider_name: str, text: str) -> Optional[int]:
        """Look up a cached token count.

        Args:
            provider_name: Name of the model provider (e.g. ``'anthropic'``).
            text: The instruction text whose token count was previously stored.

        Returns:
            The cached token count, or ``None`` on cache miss.
        """
        key = self._key(provider_name, text)
        with self._lock:
            return self._store.get(key)

    def put(self, provider_name: str, text: str, token_count: int) -> None:
        """Store a token count in the cache.

        Args:
            provider_name: Name of the model provider.
            text: The instruction text.
            token_count: Accurate token count from ``provider.count_tokens()``.
        """
        key = self._key(provider_name, text)
        with self._lock:
            self._store[key] = token_count

    def __len__(self) -> int:
        """Return the number of cached entries."""
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        return f"InstructionTokenCache(entries={len(self)})"
