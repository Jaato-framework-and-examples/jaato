"""Storage backend for the memory plugin.

The store is split into two physical layouts that match the actual
access patterns of producers (any agent calling ``store_memory``) and
the curator (the memory-advisor):

- **Raw queue** (``memories/raw/{id}.json``): one file per pending
  memory.  Producers write atomically via tempfile + rename — no write
  contention because each writer owns its file.  The curator drains
  the folder; raw memories are NEVER surfaced to agents as enrichment
  hints.
- **Curated store** (``memories/curated.jsonl``): single JSONL of
  validated/escalated memories.  Read-many (every enrichment pass),
  write-by-curator-only.  Atomic rewrites via tempfile + rename.

The ``MemoryStore`` facade exposes the operations the plugin and
curator need; it composes a ``RawStore`` and a ``CuratedStore`` over
a shared base directory.

The base path a caller supplies is a DIRECTORY (``.jaato/memories``).
A legacy ``*.jsonl`` path is still accepted and is rewritten to the
sibling directory named after its stem, so existing configs keep
working — see ``MemoryStore.__init__``.  That rewrite used to ignore
the named file in silence, which is outcome-indistinguishable from an
empty store (#912); a populated legacy file is now migrated into
``curated.jsonl`` on first use, or, when that would clobber an existing
store, announced at WARNING.
"""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime
from dataclasses import asdict, fields as dc_fields
from pathlib import Path
from typing import Iterable, List, Optional, Set

from .models import (
    ACTIVE_MATURITIES,
    MATURITY_DISMISSED,
    PROMOTES_OUT_OF_RAW,
    MATURITY_RAW,
    Memory,
)


logger = logging.getLogger(__name__)


# Fields that exist on the Memory dataclass.  Used to silently drop
# unknown keys from old or hand-edited files rather than crashing on
# ``TypeError: __init__() got an unexpected keyword argument``.
_MEMORY_FIELD_NAMES: Set[str] = {f.name for f in dc_fields(Memory)}


def _memory_from_dict(data: dict) -> Memory:
    """Construct a ``Memory`` from a raw JSON dict.

    Filters unknown keys and lets dataclass defaults fill any missing
    new fields.
    """
    filtered = {k: v for k, v in data.items() if k in _MEMORY_FIELD_NAMES}
    return Memory(**filtered)


def _atomic_write(path: Path, payload: str) -> None:
    """Write ``payload`` to ``path`` atomically via tempfile + rename.

    Readers either see the old file or the new file — never a partial
    write.  Works on the same filesystem (``os.rename`` is atomic on
    POSIX when source and destination share a filesystem).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ─────────────────────────────────────────────────────────────────────
# Raw queue
# ─────────────────────────────────────────────────────────────────────


class RawStore:
    """File-per-memory queue under ``<base>/raw/``.

    Each ``store_memory`` call writes one JSON file via tempfile +
    rename, eliminating contention between concurrent producers.
    Raw memories are never surfaced to agents — the curator drains
    them and either consolidates into the curated store or discards.
    """

    def __init__(self, base_dir: Path):
        self._dir = base_dir / "raw"

    @property
    def dir(self) -> Path:
        return self._dir

    def add(self, memory: Memory) -> None:
        """Atomically write a single raw memory file."""
        path = self._dir / f"{memory.id}.json"
        _atomic_write(path, json.dumps(asdict(memory)))

    def get(self, memory_id: str) -> Optional[Memory]:
        """Return a single raw memory or None if absent."""
        path = self._dir / f"{memory_id}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            return None
        return _memory_from_dict(data)

    def remove(self, memory_id: str) -> bool:
        """Unlink a raw memory.  Returns False if it didn't exist."""
        path = self._dir / f"{memory_id}.json"
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def list_all(self) -> List[Memory]:
        """Enumerate all raw memories, sorted by id (= timestamp prefix).

        Tolerates files disappearing mid-enumeration — the curator may
        be draining the folder concurrently.  A ``FileNotFoundError``
        on read just means another curator beat us to it; skip and
        continue.
        """
        if not self._dir.exists():
            return []
        result: List[Memory] = []
        for path in sorted(self._dir.iterdir()):
            if not path.suffix == ".json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                result.append(_memory_from_dict(data))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        return result

    def count(self) -> int:
        if not self._dir.exists():
            return 0
        return sum(
            1 for p in self._dir.iterdir() if p.suffix == ".json"
        )


# ─────────────────────────────────────────────────────────────────────
# Curated store
# ─────────────────────────────────────────────────────────────────────


class CuratedStore:
    """Single JSONL of curator-managed memories under ``<base>/curated.jsonl``.

    Single-writer (the curator) — concurrent writes from multiple
    curator instances would corrupt the file.  Cross-process
    coordination is handled by the reactor singleton mechanism (see
    backlog).  In-process the curator owns the writes serially.

    Readers can run concurrently without coordination: rewrites use
    tempfile + rename so a reader either sees the old file or the new
    file in full.
    """

    def __init__(self, base_dir: Path):
        self._path = base_dir / "curated.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def load_all(self) -> List[Memory]:
        """Read the entire curated store into memory.

        Returns empty list when the file doesn't exist yet.
        """
        if not self._path.exists():
            return []
        memories: List[Memory] = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    memories.append(_memory_from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    # Tolerate corruption: skip bad lines, never crash.
                    continue
        return memories

    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        for mem in self.load_all():
            if mem.id == memory_id:
                return mem
        return None

    def upsert(self, memory: Memory) -> None:
        """Insert or replace a memory by ID, then atomic rewrite."""
        all_memories = self.load_all()
        replaced = False
        for i, mem in enumerate(all_memories):
            if mem.id == memory.id:
                all_memories[i] = memory
                replaced = True
                break
        if not replaced:
            all_memories.append(memory)
        self._rewrite(all_memories)

    def remove(self, memory_id: str) -> bool:
        """Delete a memory by ID, atomic rewrite.  Returns False if absent."""
        all_memories = self.load_all()
        kept = [m for m in all_memories if m.id != memory_id]
        if len(kept) == len(all_memories):
            return False
        self._rewrite(kept)
        return True

    def count(self) -> int:
        return len(self.load_all())

    def replace_all(self, memories: List[Memory]) -> None:
        """Replace the whole file with ``memories``, atomically.

        The bulk counterpart of ``upsert``/``remove``, for a caller that
        already holds the complete desired contents — today only
        ``MemoryStore._recover_legacy_file``, which seeds a store that did
        not exist yet.  Unconditional: it does NOT merge with what is on
        disk, so a caller that could be racing another writer is
        responsible for deciding that it should win.
        """
        self._rewrite(memories)

    def _rewrite(self, memories: List[Memory]) -> None:
        payload = "".join(json.dumps(asdict(m)) + "\n" for m in memories)
        _atomic_write(self._path, payload)


# ─────────────────────────────────────────────────────────────────────
# Facade
# ─────────────────────────────────────────────────────────────────────


class MemoryStore:
    """Facade combining ``RawStore`` and ``CuratedStore``.

    Exposes the operations the plugin needs at the higher abstraction
    level (store, retrieve, curate) without leaking the split layout.

    Args:
        path: The store's base **directory** — it holds ``raw/`` and
            ``curated.jsonl``.  This is what every surface documents and
            defaults to (``.jaato/memories``).

            A legacy ``*.jsonl`` FILE path is also accepted, for profiles
            written before the layout split: it resolves to the sibling
            directory named after the file's stem
            (``.../memories.jsonl`` → ``.../memories/``), keeping the
            workspace and global stores distinct.  Explicit ``.jsonl``
            paths exist in the wild, so that rule is a tested contract —
            dropping it would repoint those deployments at a fresh empty
            store.

            Such a file is never itself the store.  Until #912 it was
            ignored in silence, which is outcome-indistinguishable from
            an empty store; it is now migrated or announced by
            ``_recover_legacy_file``.

    Attributes:
        _base_dir: The resolved directory both sub-stores live under —
            the value ``base_dir`` exposes.  Equal to ``path`` unless the
            legacy suffix rule rewrote it.
        _legacy_file: The ``*.jsonl`` path the caller passed, when the
            suffix rule fired; ``None`` for a directory path.  Read once,
            at construction, and never written to.
    """

    def __init__(self, path: str):
        p = Path(path)
        if p.suffix == ".jsonl":
            base = p.parent / p.stem  # e.g. ``.../memories.jsonl`` → ``.../memories/``
            self._legacy_file: Optional[Path] = p
        else:
            base = p
            self._legacy_file = None
        self._base_dir = base
        self._raw = RawStore(base)
        self._curated = CuratedStore(base)
        #: Serializes every read-route-write against THIS store.  The store
        #: had no lock of any kind while ``update`` routes between two
        #: sub-stores and ``CuratedStore.upsert``/``remove`` are
        #: read-modify-rewrite of one shared file -- and the memory tools
        #: have no parallel opt-out, so a curator emitting one retrieve plus
        #: several update_memory calls in a single response runs them
        #: CONCURRENTLY (up to 8).  Measured: 10 of 32 curator decisions
        #: silently undone, 1ms apart.
        #:
        #: The guard lives INSIDE the store so every caller inherits it --
        #: a guard at one call site is how #626's save race happened.
        #:
        #: IN-PROCESS ONLY, knowingly.  Sessions in separate runner
        #: processes sharing one workspace store can still interleave; full
        #: cross-process safety means flock on every operation.  Every
        #: observed contradiction was in-process (parallel tool threads in
        #: one runner), the curator pattern has ONE curator by design, and
        #: producers use the per-file atomic ``add`` path -- so flock waits
        #: for a cross-process interleaving to be OBSERVED rather than being
        #: built against a hypothesis.
        self._lock = threading.RLock()
        self._recover_legacy_file()

    # ── Legacy single-file recovery (#912) ──────────────────────────

    def _recover_legacy_file(self) -> None:
        """Migrate — or at minimum ANNOUNCE — a populated legacy ``*.jsonl``.

        ``.../memories.jsonl`` is reinterpreted as ``.../memories/``.  A
        real file at the given path used to be ignored without a word,
        which reads exactly like an empty store: the reported symptom was
        an agent waking up certain it knew nothing, with its memories on
        disk the whole time.

        Called once, at the end of ``__init__``.  Three properties, each
        attached to a way this could go wrong:

        - **Never clobber.**  Migration runs only when ``curated.jsonl``
          is ABSENT.  Two independently-populated stores are merged by a
          person, not by a constructor; that case warns and touches
          nothing.
        - **The legacy file is not deleted or renamed.**  This is a copy;
          the operator removes the original once satisfied.  A
          constructor that mutated the caller's file on import would be a
          worse bug than the one being fixed.
        - **Silence stays silent for the normal case.**  Directory paths,
          absent files, empty files and unreadable ones log nothing, so a
          WARNING here means something.

        Recovery can only GAIN data: a populated flat file can exist only
        on an install predating the layout split, where it has been
        unreadable ever since.
        """
        legacy = self._legacy_file
        if legacy is None:
            return
        memories = self._read_legacy_memories(legacy)
        if not memories:
            return
        if self._curated.path.exists():
            logger.warning(
                "memory: %s is a legacy single-file store and is NOT read; "
                "this store lives in %s, which already holds %s. Merge the "
                "two by hand, or delete the legacy file.",
                legacy, self._base_dir, self._curated.path,
            )
            return
        with self._lock:
            if self._curated.path.exists():
                return
            self._curated.replace_all(memories)
        logger.warning(
            "memory: migrated %d memories from the legacy single-file store "
            "%s into %s. The legacy file is left untouched — delete it once "
            "you have confirmed the migration.",
            len(memories), legacy, self._curated.path,
        )

    @staticmethod
    def _read_legacy_memories(legacy: Path) -> List[Memory]:
        """Parse a legacy single-file store, tolerating everything.

        Mirrors ``CuratedStore.load_all``'s line-by-line tolerance — a bad
        line is skipped, never fatal — and additionally swallows
        ``OSError``.  An unreadable path is the NORMAL case for a confined
        session: ``MemoryPlugin.initialize`` builds one store from a
        daemon-cwd-relative template and another from HOME, and a
        confined runner is correctly denied both.  Raising there would
        disable the plugin for the tier that IS reachable.

        Returns:
            The memories found, or ``[]`` when the file is absent, empty,
            unreadable or entirely corrupt.  The caller does not need to
            tell those apart: none of them is worth announcing.
        """
        try:
            if not legacy.is_file() or legacy.stat().st_size == 0:
                return []
            text = legacy.read_text(encoding="utf-8")
        except OSError:
            return []
        memories: List[Memory] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                memories.append(_memory_from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
        return memories

    @property
    def raw(self) -> RawStore:
        return self._raw

    @property
    def curated(self) -> CuratedStore:
        return self._curated

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    # ── Producer API (any agent) ────────────────────────────────────

    def save(self, memory: Memory) -> None:
        """Write a new memory.

        New memories always land in the raw queue regardless of the
        ``maturity`` field.  The curator promotes them to the curated
        store later.
        """
        with self._lock:
            self._raw.add(memory)

        # ── Consumer / curator API ──────────────────────────────────────

    def load_curated(self) -> List[Memory]:
        """All curated memories.  Source of truth for enrichment."""
        return self._curated.load_all()

    def list_raw(self) -> List[Memory]:
        """All raw memories awaiting curation.  Curator-only path."""
        return self._raw.list_all()

    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Look up a memory by ID across both stores (curated first)."""
        mem = self._curated.get_by_id(memory_id)
        if mem is not None:
            return mem
        return self._raw.get(memory_id)

    def update(self, memory: Memory) -> None:
        """Update a memory in place.

        Routing rules (the curator's perspective):

        - If the memory currently lives in the raw queue and is being
          marked validated/escalated → move it to the curated store.
        - If the memory currently lives in the raw queue and is being
          dismissed → unlink it (no curated trace).
        - If it's already curated → upsert into the curated store
          (covers maturity transitions, content edits, tag updates).
        """
        with self._lock:
            self._update_locked(memory)

    def _update_locked(self, memory: Memory) -> None:
        in_raw = self._raw.get(memory.id) is not None
        in_curated = self._curated.get_by_id(memory.id) is not None

        if in_raw:
            # PROMOTES_OUT_OF_RAW, not ACTIVE_MATURITIES.  The active set
            # contains RAW -- it answers "is this usable?" -- so testing
            # against it moved a memory out of the queue on ANY update,
            # including the usage-counter bump ``_execute_retrieve`` does on
            # every read.  The docstring above always said "validated/
            # escalated"; the condition did not.
            if memory.maturity in PROMOTES_OUT_OF_RAW:
                self._curated.upsert(memory)
                self._raw.remove(memory.id)
            elif memory.maturity == MATURITY_DISMISSED:
                self._raw.remove(memory.id)
            else:
                # STILL RAW -> stays in the queue, updated in place.  This
                # branch already existed and was unreachable for ``raw``.
                # ``RawStore.add`` is an atomic per-id file write, so this is
                # an idempotent upsert with no shared-file contention.
                self._raw.add(memory)
        elif in_curated:
            if memory.maturity == "dismissed":
                self._curated.remove(memory.id)
            else:
                self._curated.upsert(memory)
        else:
            # Not anywhere yet — treat as a new write.  Goes to raw.
            self._raw.add(memory)

    def record_usage(self, memory_id: str) -> None:
        """Bump usage stats on the CURRENT object, wherever it now lives.

        The usage write-back used to be ``update(stale_object)`` -- a full
        Memory carrying the maturity it had AT RETRIEVAL TIME, pushed through
        the routing logic after an arbitrary delay.  Under parallel tool
        execution a curator's decision could land in that window, and the
        stale write-back then either REVERTED it (stale-raw upserted over a
        now-validated curated entry) or RESURRECTED it (stale-raw re-added to
        the queue after a dismissal unlinked it -- via the "not anywhere yet,
        goes to raw" branch).  10 of 32 live curator decisions were undone
        that way, producing a re-decide livelock on the same ids.

        This method carries only the FACT ("this id was read"), not the
        object: it re-reads the current state under the store lock, mutates
        the usage fields alone, and writes back to the store it found the
        memory in.  A memory that was dismissed in the meantime is GONE, and
        recording usage on it would be resurrection -- so absent is a no-op,
        deliberately.
        """
        with self._lock:
            mem = self._raw.get(memory_id)
            if mem is not None:
                mem.usage_count += 1
                mem.last_accessed = datetime.now().isoformat()
                self._raw.add(mem)          # atomic per-id file: in-place
                return
            mem = self._curated.get_by_id(memory_id)
            if mem is not None:
                mem.usage_count += 1
                mem.last_accessed = datetime.now().isoformat()
                self._curated.upsert(mem)
            # else: dismissed/deleted since retrieval -- no-op, NOT a re-add.

    def delete(self, memory_id: str) -> bool:
        """Hard-delete a memory wherever it lives."""
        with self._lock:
            removed_raw = self._raw.remove(memory_id)
            removed_curated = self._curated.remove(memory_id)
            return removed_raw or removed_curated

        # ── Tag/maturity queries (used by tools, advisor) ───────────────

    def search_by_tags(
        self,
        tags: List[str],
        limit: int = 3,
        *,
        active_only: bool = True,
    ) -> List[Memory]:
        """Search **curated** memories by tag overlap.

        Raw memories are intentionally excluded — agents only see
        curator-vetted material as hints / via retrieval.
        """
        memories = self._curated.load_all()
        scored = []
        for mem in memories:
            if active_only and mem.maturity not in ACTIVE_MATURITIES:
                continue
            overlap = len(set(mem.tags) & set(tags))
            if overlap > 0:
                scored.append((overlap, mem))
        scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)
        return [mem for _, mem in scored[:limit]]

    def search_by_maturity(
        self,
        maturities: Iterable[str],
        limit: int = 50,
    ) -> List[Memory]:
        """Curator-facing maturity query.

        ``raw`` is sourced from the raw queue; everything else from
        the curated store.  Mixed queries combine both.
        """
        target = set(maturities)
        result: List[Memory] = []
        if MATURITY_RAW in target:
            result.extend(self._raw.list_all())
        non_raw = target - {MATURITY_RAW}
        if non_raw:
            result.extend(
                m for m in self._curated.load_all() if m.maturity in non_raw
            )
        result.sort(key=lambda m: m.timestamp, reverse=True)
        return result[:limit]

    def get_pending_curation(self, limit: int = 50) -> List[Memory]:
        """Convenience: return all raw memories awaiting curator review."""
        memories = self._raw.list_all()
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        return memories[:limit]

    def count_by_maturity(self) -> dict:
        """Return ``{maturity: count}`` across both stores."""
        counts: dict = {}
        counts[MATURITY_RAW] = self._raw.count()
        for mem in self._curated.load_all():
            counts[mem.maturity] = counts.get(mem.maturity, 0) + 1
        return counts

    def count(self) -> int:
        """Total memories across both stores."""
        return self._raw.count() + self._curated.count()

    def load_all(self) -> List[Memory]:
        """All memories across both stores (curator/diagnostic use only).

        Agents must NEVER call this for enrichment — it would surface
        unvetted raw memories.  Use ``load_curated()`` for that path.
        """
        return self._raw.list_all() + self._curated.load_all()


# Backwards-compatible alias so existing ``from .storage import
# MemoryStorage`` imports keep working without immediate refactoring.
# The class name is misleading now (the new layout is two stores), but
# the API surface matches.
MemoryStorage = MemoryStore
