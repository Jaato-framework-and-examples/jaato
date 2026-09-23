"""A bounded, TTL'd record of deliveries already accepted, for replay refusal.

A freshness window (see :mod:`.signature_schemes`) narrows how long a captured
request stays useful; it does not stop a replay *inside* that window, because
every copy of the delivery carries the same signed timestamp and is therefore
equally fresh.  This is the part that does.

Three properties, each attached to a way a replay cache goes wrong:

* **It is bounded, and the bound is the point.**  Remembering every delivery
  forever is the naive form and it is a memory leak on a listener that is
  internet-reachable by design.  Entries expire after the route's freshness
  window, because past the window the timestamp check refuses the request
  anyway — the two mechanisms are sized to each other rather than each
  guessing.  A hard ``max_entries`` ceiling bounds the pathological case where
  a flood of *valid* deliveries arrives faster than the TTL retires them.
* **Only an AUTHENTICATED delivery is recorded.**  Recording on arrival would
  let anyone who can reach the port poison the cache with a guessed delivery
  id and have the genuine delivery refused as a replay — a denial of service
  built out of the anti-replay control.  Callers record as the last step, once
  the signature and the window have both passed.
* **The check and the record are one atomic step.**  Two copies of a delivery
  arriving concurrently on two server threads must not both find the key
  absent.  :meth:`ReplayCache.check_and_record` takes the lock once.

Under the ceiling, eviction is oldest-first, which is the safe direction: the
oldest entry is the closest to expiring anyway.  It is still an eviction, so a
sustained flood of valid deliveries can retire an entry before its TTL and
re-open a replay window for that one key — ``evictions`` is the counter that
says so, and raising ``replay_cache_size`` is the response.

stdlib only, like the rest of this plugin.
"""

import logging
import threading
from collections import OrderedDict
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Entries, not bytes.  A key is a short string (a delivery id or a hex digest)
# plus a float expiry, so ten thousand of them is well under a megabyte — and
# at the 300 s default window that is ~33 deliveries/second sustained before
# the ceiling starts evicting early.
DEFAULT_REPLAY_CACHE_SIZE = 10000


class ReplayCache:
    """Bounded, TTL'd set of delivery keys already accepted by this listener.

    One cache serves every route on a listener; keys are namespaced by route
    name by the caller (:func:`.routes.replay_key_for`) so two routes cannot
    collide on a shared delivery-id space.

    Thread safety: every public method takes ``_lock``.  The HTTP server hands
    requests to a handler on its own thread, and a deployment may run more than
    one listener thread, so the cache is written from several threads.

    Attributes:
        max_entries: Hard ceiling on retained keys.
        evictions: Count of entries dropped by the ceiling BEFORE expiring —
            nonzero means the cache is undersized for the delivery rate and a
            replay window has re-opened for those keys.
        replays_refused: Count of deliveries refused as replays.
    """

    def __init__(self, max_entries: int = DEFAULT_REPLAY_CACHE_SIZE):
        """Initialise an empty cache.

        Args:
            max_entries: Hard ceiling on retained keys.  Values below 1 are
                raised to 1 — a zero-size cache would silently accept every
                replay, which is the failure this class exists to prevent, so
                disabling it is a route-level decision (do not configure a
                replay key) rather than something a stray ``0`` achieves.
        """
        self.max_entries = max(1, int(max_entries))
        self.evictions = 0
        self.replays_refused = 0
        # key -> expiry (Unix seconds).  Ordered by insertion, which is also
        # expiry order whenever one TTL is in play, so popping the front is
        # both "oldest" and "closest to expiring".
        self._entries: "OrderedDict[str, float]" = OrderedDict()
        self._lock = threading.Lock()

    def check_and_record(self, key: str, now: float, ttl: float) -> bool:
        """Record ``key`` as seen, reporting whether it was a replay.

        Args:
            key: The delivery's replay key, already namespaced by route.
            now: Current Unix time.  Injected rather than read here so a test
                can state the TTL it means instead of sleeping through it.
            ttl: How long to remember this key, in seconds.  Normally the
                route's freshness window: past it, the timestamp check refuses
                the request without help from the cache.

        Returns:
            True when ``key`` was already present and unexpired — i.e. this
            delivery is a REPLAY and the caller must refuse it.  False when the
            key is new (and is now recorded).
        """
        with self._lock:
            self._purge_expired(now)

            expiry = self._entries.get(key)
            if expiry is not None and expiry > now:
                self.replays_refused += 1
                return True

            # An expired entry is re-inserted at the back rather than mutated
            # in place, so insertion order stays expiry order.
            if expiry is not None:
                del self._entries[key]

            self._entries[key] = now + max(0.0, float(ttl))
            self._enforce_ceiling()
            return False

    def _purge_expired(self, now: float) -> None:
        """Drop entries whose TTL has elapsed.  Caller holds ``_lock``.

        Walks from the front only, and stops at the first unexpired entry:
        insertion order is expiry order for a single TTL, so this is O(number
        actually expiring) rather than O(size) per request.  A route whose
        window was widened mid-run can leave a longer-lived entry ahead of a
        shorter-lived one; that entry is simply purged later, on a pass that
        reaches it, which costs memory and never correctness.

        Args:
            now: Current Unix time.
        """
        while self._entries:
            key, expiry = next(iter(self._entries.items()))
            if expiry > now:
                return
            del self._entries[key]

    def _enforce_ceiling(self) -> None:
        """Evict oldest-first until the cache is within ``max_entries``.

        Caller holds ``_lock``.  Each eviction is counted, because an entry
        dropped before its TTL has re-opened a replay window for that key and
        that is an operational signal, not a detail.
        """
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1
            if self.evictions == 1:
                logger.warning(
                    "Webhook replay cache is full (%d entries) and is now "
                    "evicting keys before their freshness window elapses — a "
                    "replay of an evicted delivery would be accepted. Raise "
                    "replay_cache_size.",
                    self.max_entries,
                )

    def __len__(self) -> int:
        """Number of retained keys, expired ones included until purged."""
        with self._lock:
            return len(self._entries)

    def get_stats(self) -> Dict[str, int]:
        """Return cache counters for ``webhook_status``.

        Returns:
            ``entries`` (retained keys), ``max_entries`` (the ceiling),
            ``replays_refused`` and ``evictions``.
        """
        with self._lock:
            return {
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "replays_refused": self.replays_refused,
                "evictions": self.evictions,
            }


def replay_key_for(
    route_name: str,
    delivery_id: Optional[str],
    replay_material: Optional[str],
    prefer_signature: bool,
) -> Optional[str]:
    """Choose the key a delivery is remembered by, namespaced to its route.

    Which source is right depends on the route's signature scheme, and the two
    regimes want opposite answers — which is why this takes an explicit flag
    rather than guessing from what happens to be present.

    **A timestamp-bound scheme** (``prefer_signature=True``) keys on the
    verified signature.  It is unforgeable, and it varies with the timestamp,
    so it names exactly one delivery and cannot collide with a later genuine
    one carrying the same body.  A delivery id would be *worse* here: no sender
    in this tree signs its headers, so an attacker replaying a captured request
    can simply rewrite the id and walk past a cache keyed on it.

    **The ``body`` scheme and ``token`` mode** (``prefer_signature=False``) key
    on a delivery id the route names via ``replay_key_header`` — GitHub's
    ``X-GitHub-Delivery``, GitLab's ``X-Gitlab-Event-UUID`` — or get no cache
    at all.  The signature is not usable as a key in this regime: with no
    timestamp in the signed payload it is a pure function of the body, so two
    genuine deliveries carrying byte-identical payloads would be
    indistinguishable from a replay and the second one refused.  The cost of
    the id is stated plainly in the plugin docs and in the startup warning: it
    dedupes a sender retry and a naive verbatim replay, and an attacker who
    rewrites the id defeats it.  Bounded protection, honestly labelled, is
    what is available when the sender signs no timestamp.

    Args:
        route_name: The matched route's config key, used to namespace the key
            so two routes cannot collide on a shared delivery-id space.
        delivery_id: The ``replay_key_header`` value, when the route names one.
        replay_material: ``SignatureVerdict.replay_material``.
        prefer_signature: True for a timestamp-bound scheme.

    Returns:
        The namespaced cache key, or None when the route has nothing to key on.
    """
    if prefer_signature:
        if replay_material:
            return f"{route_name}:sig:{replay_material}"
        return f"{route_name}:id:{delivery_id}" if delivery_id else None
    return f"{route_name}:id:{delivery_id}" if delivery_id else None
