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

Backward compatibility for the legacy single-file ``memories.jsonl``
layout is intentionally NOT provided — this is a development-stage
refactor with no migration commitment.
"""

import json
import os
import tempfile
from dataclasses import asdict, fields as dc_fields
from pathlib import Path
from typing import Iterable, List, Optional, Set

from .models import (
    ACTIVE_MATURITIES,
    MATURITY_RAW,
    Memory,
)


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
        path: Base path.  Either a directory that will hold ``raw/``
            and ``curated.jsonl``, or (for backwards compatibility with
            existing call sites that pass a ``.jsonl`` path) the file
            path's parent directory will be used.  In the latter case
            the legacy file is ignored — there is no migration.
    """

    def __init__(self, path: str):
        # Accept either a directory or a legacy ``*.jsonl`` path
        # (in which case we use a directory adjacent to the file,
        # named after its stem, so workspace and global stores stay
        # distinct).  This keeps the plugin's existing config keys
        # working — callers don't need to update their
        # ``storage_path`` / ``global_storage_path``.
        p = Path(path)
        if p.suffix == ".jsonl":
            base = p.parent / p.stem  # e.g. ``.../memories.jsonl`` → ``.../memories/``
        else:
            base = p
        self._base_dir = base
        self._raw = RawStore(base)
        self._curated = CuratedStore(base)

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
        in_raw = self._raw.get(memory.id) is not None
        in_curated = self._curated.get_by_id(memory.id) is not None

        if in_raw:
            if memory.maturity in ACTIVE_MATURITIES.union({"escalated"}):
                self._curated.upsert(memory)
                self._raw.remove(memory.id)
            elif memory.maturity == "dismissed":
                self._raw.remove(memory.id)
            else:
                # Unknown / new maturity: keep it as raw, replace file.
                self._raw.add(memory)
        elif in_curated:
            if memory.maturity == "dismissed":
                self._curated.remove(memory.id)
            else:
                self._curated.upsert(memory)
        else:
            # Not anywhere yet — treat as a new write.  Goes to raw.
            self._raw.add(memory)

    def delete(self, memory_id: str) -> bool:
        """Hard-delete a memory wherever it lives."""
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
