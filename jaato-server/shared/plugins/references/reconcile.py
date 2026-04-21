"""Per-bundle reconcile: bring a bundle's sidecar in sync with its catalog.

A **reconcile pass** computes the drift between the live catalog of
reference sources and what the bundle's sidecar matrix records, then
rewrites the sidecar + manifest + per-reference ``source_hash`` fields
atomically. It is what makes dropping a reference JSON into a bundle
directory "just work" without a manual regeneration step.

Design contract:
    * **Atomic swap**: the new sidecar matrix and manifest are written to
      ``*.tmp`` files alongside the originals, fsync'd, then renamed. A
      crash at any point leaves the old files intact.
    * **Advisory lock**: an ``flock(LOCK_EX | LOCK_NB)`` on
      ``<sidecar>.lock`` serializes concurrent daemons; if the lock is
      busy the reconcile returns a ``SKIPPED`` result rather than waiting.
    * **Best-effort**: a failure to embed one source skips that id and
      logs — the other sources still get reconciled. Only a fatal write
      error aborts the pass (without touching the on-disk state).
    * **No back-pressure on callers**: reconcile is invoked from
      ``initialize()`` (eager), from semantic matching entry points
      (lazy), or from the ``references reconcile`` user command
      (manual). Callers do not block on the lock; if another process has
      it, we return and try again next time.

Numpy is imported lazily so that the rest of the references plugin (and
``bundle.py``) work in environments without the embedding stack installed.
When numpy is missing, ``reconcile_bundle`` returns a result with
``status="unavailable"`` and changes nothing on disk.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .bundle import Bundle, DriftReport, detect_drift, metadata_hash, write_manifest
from .embedding_types import EmbeddingProviderProtocol
from .models import EmbeddingMetadata, ReferenceSource

logger = logging.getLogger(__name__)


class ReconcileStatus(str, Enum):
    """Outcome of a reconcile pass."""

    CLEAN = "clean"              # Nothing to do.
    UPDATED = "updated"          # Sidecar + manifest rewritten.
    SKIPPED_BUSY = "skipped_busy"  # Another process held the lock.
    UNAVAILABLE = "unavailable"  # No provider or numpy — can't reconcile.
    ERROR = "error"              # Unexpected failure; on-disk state untouched.


@dataclass
class ReconcileResult:
    """What reconcile did (or didn't) for a single bundle.

    Attributes:
        bundle_name: The bundle's name (``""`` for root).
        status: See :class:`ReconcileStatus`.
        added: Source ids that gained a new row in the sidecar.
        refreshed: Source ids whose row was re-embedded due to metadata drift.
        pruned: Source ids dropped because their reference disappeared.
        skipped: ``(id, reason)`` for sources we wanted to embed but couldn't.
        final_row_count: Row count of the sidecar after the pass.
        error: Human-readable error message when ``status == ERROR``.
    """

    bundle_name: str
    status: ReconcileStatus
    added: List[str] = field(default_factory=list)
    refreshed: List[str] = field(default_factory=list)
    pruned: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    final_row_count: int = 0
    error: Optional[str] = None

    def summary(self) -> str:
        """One-line summary suitable for a log or user-facing line."""
        if self.status == ReconcileStatus.CLEAN:
            return f"{self.bundle_name or '(root)'}: up-to-date"
        if self.status == ReconcileStatus.UPDATED:
            parts = []
            if self.added:
                parts.append(f"added {len(self.added)}")
            if self.refreshed:
                parts.append(f"refreshed {len(self.refreshed)}")
            if self.pruned:
                parts.append(f"pruned {len(self.pruned)}")
            if self.skipped:
                parts.append(f"skipped {len(self.skipped)}")
            return f"{self.bundle_name or '(root)'}: " + ", ".join(parts)
        return f"{self.bundle_name or '(root)'}: {self.status.value}"


def _try_acquire_lock(lock_path: Path):
    """Acquire a non-blocking exclusive flock on ``lock_path``.

    Returns an open file handle on success, or ``None`` when the lock is
    busy. The caller is responsible for closing the handle (which releases
    the lock). On platforms without ``fcntl.flock`` (Windows), this falls
    back to a no-op — the lock becomes advisory-only and assumes a single
    daemon per workspace.
    """
    try:
        import fcntl  # type: ignore
    except ImportError:
        # Windows fallback: no flock, single-daemon assumption.
        return open(lock_path, "w")

    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fh
    except BlockingIOError:
        fh.close()
        return None


def _release_lock(fh: Any, lock_path: Path) -> None:
    """Release and close a lock acquired by :func:`_try_acquire_lock`."""
    try:
        import fcntl  # type: ignore
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
    except ImportError:
        pass
    try:
        fh.close()
    except OSError:
        pass
    # Best-effort cleanup — not fatal if the file is still there.
    try:
        lock_path.unlink()
    except OSError:
        pass


def _write_sidecar_atomic(sidecar_path: Path, matrix: Any) -> None:
    """Write ``matrix`` to ``sidecar_path`` via a tmp file + rename.

    The tmp file is fsync'd before the rename so a crash between save and
    rename leaves the old sidecar intact.
    """
    import numpy as np  # local import — see module docstring

    tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, matrix, allow_pickle=False)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(sidecar_path)


def _update_source_hash_on_disk(
    source: ReferenceSource,
    new_hash: str,
    *,
    directory: Path,
) -> None:
    """Rewrite the reference JSON so its ``embedding.source_hash`` matches.

    Only the ``embedding`` block is touched; the rest of the file is
    preserved byte-for-byte apart from JSON re-encoding. Atomic write via
    tmp + rename. Silently no-ops when the file is missing or unreadable.
    """
    json_path = directory / f"{source.id}.json"
    if not json_path.is_file():
        return
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(raw, dict):
        return
    raw["embedding"] = {"source_hash": new_hash}
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(json_path)
    # Reflect the change in the in-memory object so the caller sees it.
    source.embedding = EmbeddingMetadata(source_hash=new_hash)


def _embed_text_for(source: ReferenceSource) -> str:
    """The canonical text fed to ``embed_batch`` for a reference.

    Mirrors :func:`metadata_hash` — we embed the same fields we fingerprint,
    so a stable hash implies a stable vector.
    """
    return "\n".join([
        source.name,
        source.description,
        ",".join(sorted(source.tags)),
        source.fetch_hint or "",
    ])


def reconcile_bundle(
    bundle: Bundle,
    sources: List[ReferenceSource],
    provider: Optional[EmbeddingProviderProtocol],
) -> ReconcileResult:
    """Bring ``bundle`` in sync with the live catalog and rewrite the sidecar.

    When there is no drift the function returns ``CLEAN`` without touching
    disk. When drift exists but no provider is available, returns
    ``UNAVAILABLE`` (tag matching still works; the operator can re-run
    once a provider is configured). When the advisory lock is busy
    returns ``SKIPPED_BUSY``.

    On success the bundle's manifest ``rows`` list and sidecar matrix are
    both rewritten atomically, and the in-memory ``bundle.embedding_rows``
    is updated to match.

    Args:
        bundle: The bundle to reconcile.
        sources: Full catalog (across all bundles). Filtered internally to
            sources whose ``bundle_name`` matches ``bundle.name``.
        provider: Embedding provider. Must be loaded and ``available``.
            When ``None`` or unavailable the function returns without
            touching disk.

    Returns:
        A populated :class:`ReconcileResult`. ``bundle.embedding_rows`` and
        ``bundle.matcher`` reflect the new state on ``UPDATED``.
    """
    drift = detect_drift(bundle, sources)
    if drift.is_clean():
        return ReconcileResult(
            bundle_name=bundle.name,
            status=ReconcileStatus.CLEAN,
            final_row_count=len(bundle.embedding_rows),
        )

    # Guard: anything beyond this point needs both numpy and a provider.
    if provider is None or not provider.available:
        logger.info(
            "Bundle '%s': %s but no embedding provider — skipping reconcile",
            bundle.display_name, drift.summary(),
        )
        return ReconcileResult(
            bundle_name=bundle.name,
            status=ReconcileStatus.UNAVAILABLE,
            final_row_count=len(bundle.embedding_rows),
        )

    try:
        import numpy as np
    except ImportError:
        logger.warning(
            "Bundle '%s': numpy not installed — cannot reconcile",
            bundle.display_name,
        )
        return ReconcileResult(
            bundle_name=bundle.name,
            status=ReconcileStatus.UNAVAILABLE,
            final_row_count=len(bundle.embedding_rows),
        )

    lock = _try_acquire_lock(bundle.lock_path)
    if lock is None:
        logger.info(
            "Bundle '%s': reconcile already in progress elsewhere — skipping",
            bundle.display_name,
        )
        return ReconcileResult(
            bundle_name=bundle.name,
            status=ReconcileStatus.SKIPPED_BUSY,
            final_row_count=len(bundle.embedding_rows),
        )

    try:
        result = _reconcile_locked(bundle, sources, drift, provider, np)
        return result
    except Exception as exc:  # pragma: no cover — defensive top-level guard
        logger.exception(
            "Bundle '%s': reconcile failed: %s", bundle.display_name, exc
        )
        return ReconcileResult(
            bundle_name=bundle.name,
            status=ReconcileStatus.ERROR,
            final_row_count=len(bundle.embedding_rows),
            error=str(exc),
        )
    finally:
        _release_lock(lock, bundle.lock_path)


def _reconcile_locked(
    bundle: Bundle,
    sources: List[ReferenceSource],
    drift: DriftReport,
    provider: EmbeddingProviderProtocol,
    np: Any,
) -> ReconcileResult:
    """Inner reconcile body executed while holding the advisory lock.

    Computes the new matrix + rows list, writes them atomically, and
    updates each affected reference JSON's ``source_hash``.
    """
    own_sources = {
        s.id: s for s in sources if s.bundle_name == bundle.name
    }

    # Ids that survive into the new sidecar: anything in the current rows
    # list that is still in the catalog and not stale.
    stale_set = set(drift.stale)
    orphan_set = set(drift.orphan)
    kept_ids: List[str] = [
        sid
        for sid in bundle.embedding_rows
        if sid not in stale_set and sid not in orphan_set
    ]

    new_ids: List[str] = []
    for sid in drift.missing:
        if sid in own_sources:
            new_ids.append(sid)
    for sid in drift.stale:
        if sid in own_sources:
            new_ids.append(sid)

    # Load the current matrix so we can preserve rows for kept_ids.
    old_matrix = None
    if bundle.sidecar_path.is_file():
        try:
            old_matrix = np.load(bundle.sidecar_path, allow_pickle=False)
        except (OSError, ValueError) as e:
            logger.warning(
                "Bundle '%s': failed to load existing sidecar %s: %s — "
                "treating all rows as new",
                bundle.display_name, bundle.sidecar_path, e,
            )
            old_matrix = None
            kept_ids = []
            # Everything owned by this bundle now needs to be re-embedded.
            new_ids = [sid for sid in own_sources]

    # Build the preserved block.
    if old_matrix is not None and kept_ids:
        old_index = {sid: i for i, sid in enumerate(bundle.embedding_rows)}
        keep_rows = [old_index[sid] for sid in kept_ids]
        preserved_block = old_matrix[keep_rows]
    else:
        preserved_block = None

    # Embed new_ids. Batch embed so a provider that supports batching
    # amortizes setup cost; skip individual failures rather than aborting.
    skipped: List[Tuple[str, str]] = []
    new_rows: List[str] = []
    new_vectors: List[Any] = []
    if new_ids:
        texts = [_embed_text_for(own_sources[sid]) for sid in new_ids]
        try:
            results = provider.embed_batch(texts)
        except Exception as e:
            logger.warning(
                "Bundle '%s': embed_batch failed: %s — falling back to per-source embeds",
                bundle.display_name, e,
            )
            results = [None] * len(texts)
        # Recover any None entries by trying one-at-a-time; this lets a
        # single bad input get recorded as skipped without torching the
        # whole batch.
        for sid, res in zip(new_ids, results):
            if res is None:
                try:
                    res = provider.embed_text(_embed_text_for(own_sources[sid]))
                except Exception as e:  # pragma: no cover — unusual path
                    logger.warning(
                        "Bundle '%s': embed_text failed for %r: %s",
                        bundle.display_name, sid, e,
                    )
                    res = None
            if res is None:
                skipped.append((sid, "provider returned None"))
                continue
            if res.dimensions != bundle.embedding_dimensions:
                skipped.append((
                    sid,
                    f"dimension mismatch ({res.dimensions} vs {bundle.embedding_dimensions})",
                ))
                continue
            new_rows.append(sid)
            new_vectors.append(res.embedding)

    # Assemble the new matrix.
    if preserved_block is not None and new_vectors:
        new_block = np.asarray(new_vectors, dtype=preserved_block.dtype)
        new_matrix = np.vstack([preserved_block, new_block])
    elif preserved_block is not None:
        new_matrix = preserved_block
    elif new_vectors:
        new_matrix = np.asarray(new_vectors, dtype=np.float32)
    else:
        # Everything is gone (all orphans, no new vectors). Empty matrix.
        new_matrix = np.zeros((0, bundle.embedding_dimensions), dtype=np.float32)

    final_rows = kept_ids + new_rows

    # Atomic swap: sidecar first, then manifest. If the sidecar write
    # fails the manifest still points at the previous rows; if the
    # manifest write fails the sidecar has new rows the manifest doesn't
    # know about — next reconcile will see them as orphans and prune.
    _write_sidecar_atomic(bundle.sidecar_path, new_matrix)
    write_manifest(bundle, rows=final_rows)

    # Stamp each new/refreshed reference JSON with its new source_hash so
    # future drift detection sees the metadata as fresh.
    refreshed_set = set(drift.stale) & set(new_rows)
    added_set = set(new_rows) - refreshed_set
    for sid in new_rows:
        source = own_sources.get(sid)
        if source is None:
            continue
        new_hash = metadata_hash(source)
        _update_source_hash_on_disk(
            source, new_hash, directory=bundle.directory,
        )

    return ReconcileResult(
        bundle_name=bundle.name,
        status=ReconcileStatus.UPDATED,
        added=sorted(added_set),
        refreshed=sorted(refreshed_set),
        pruned=sorted(orphan_set),
        skipped=skipped,
        final_row_count=len(final_rows),
    )
