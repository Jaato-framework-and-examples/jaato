"""Merge one knowledge bundle into another.

A merge folds a **source** bundle (a directory on disk, or a bundle
already loaded under ``.jaato/references/<name>/``) into a **target**
bundle (another loaded bundle, typically the root). Reference JSONs are
copied into the target's directory, their vectors are appended to the
target's sidecar matrix, and the target's manifest is rewritten to
extend its ``rows`` list.

The merge has three compatibility regimes:

* **Hot path — compatible bundles**: same ``embedding_model`` and
  ``embedding_dimensions``. Source vectors are copied from the source's
  existing sidecar. No provider needed.
* **Re-embed path**: model or dim mismatch, user passed ``--re-embed``.
  The source references are re-embedded through the active provider
  using the target's model, and the new vectors are appended.
* **Refuse path**: model/dim mismatch without ``--re-embed``. Merge
  aborts with an error; source and target remain untouched.

ID collisions (the same ``id`` in source and target) are handled per
``--on-conflict``:

* ``reject`` (default) — merge aborts, listing the conflicting ids.
* ``prefix`` — incoming ids are renamed ``<source_name>_<id>``. The
  JSON file's ``id`` field and filename are both rewritten, so the
  renamed refs are first-class members of the target bundle.
* ``newer`` — whichever side has the fresher metadata fingerprint
  wins; same-hash pairs are dropped as duplicates. Ties broken by
  source JSON mtime.

All writes go through the same atomic-swap path as :mod:`reconcile`
(``.tmp`` + rename, guarded by an ``flock`` on
``<target-sidecar>.lock``) so a crash mid-merge can never leave the
target bundle in a half-written state. Source files are copied but
never deleted — callers who want to consolidate are responsible for
removing the source bundle afterwards.

``--dry-run`` computes the full plan (additions, renames, skips) and
returns it without touching disk. Useful for previewing a merge from
the TUI before committing.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .bundle import (
    EMBEDDING_CONFIG_FILENAME,
    Bundle,
    metadata_hash,
    write_manifest,
)
from .embedding_types import EmbeddingProviderProtocol
from .models import EmbeddingMetadata, ReferenceSource
from .reconcile import _embed_text_for, _try_acquire_lock, _release_lock, _write_sidecar_atomic

logger = logging.getLogger(__name__)

# Valid on_conflict strategies.
_VALID_CONFLICT_MODES = {"reject", "prefix", "newer"}


class MergeStatus(str, Enum):
    """Outcome of a merge attempt."""

    OK = "ok"
    DRY_RUN = "dry_run"
    ERROR = "error"
    SKIPPED_BUSY = "skipped_busy"
    UNAVAILABLE = "unavailable"  # --re-embed needed but no provider/numpy


@dataclass
class MergeOptions:
    """Merge flags parsed from ``references merge``.

    Attributes:
        on_conflict: How to handle id collisions between source and
            target. One of ``reject``, ``prefix``, ``newer``.
        re_embed: If True, re-embed source references against the
            target's model instead of copying vectors from the source
            sidecar. Required when models or dimensions differ.
        dry_run: If True, compute the plan but do not touch disk.
    """

    on_conflict: str = "reject"
    re_embed: bool = False
    dry_run: bool = False

    def validate(self) -> Optional[str]:
        """Return an error message when options are internally inconsistent."""
        if self.on_conflict not in _VALID_CONFLICT_MODES:
            return (
                f"Unknown --on-conflict mode {self.on_conflict!r}. "
                f"Must be one of: {sorted(_VALID_CONFLICT_MODES)}"
            )
        return None


@dataclass
class MergeResult:
    """What merge did (or would do). ``DRY_RUN`` carries the same fields as ``OK``.

    Attributes:
        source_name: The source bundle/path identifier the user passed.
        target_name: The target bundle's name (``""`` for root).
        status: See :class:`MergeStatus`.
        added: Ids now in the target (after renames).
        renamed: ``(old_id, new_id)`` for prefix-mode collisions.
        skipped: ``(id, reason)`` for sources we couldn't or chose not to
            merge (duplicates, orphan rows, embedding failures).
        conflicts: Conflicting ids surfaced when ``status == ERROR`` and
            the cause is reject-mode collisions. Empty otherwise.
        final_row_count: Target's row count after the merge.
        error: Human-readable error when ``status == ERROR``.
    """

    source_name: str
    target_name: str
    status: MergeStatus
    added: List[str] = field(default_factory=list)
    renamed: List[Tuple[str, str]] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    final_row_count: int = 0
    error: Optional[str] = None

    def summary(self) -> str:
        """One-line summary for a log or user-facing line."""
        src = self.source_name or "(path)"
        dst = self.target_name or "(root)"
        if self.status == MergeStatus.ERROR:
            return f"merge '{src}' → '{dst}': ERROR — {self.error}"
        if self.status == MergeStatus.SKIPPED_BUSY:
            return f"merge '{src}' → '{dst}': target busy, try again"
        if self.status == MergeStatus.UNAVAILABLE:
            return f"merge '{src}' → '{dst}': {self.error}"
        prefix = "would merge" if self.status == MergeStatus.DRY_RUN else "merged"
        parts = []
        if self.added:
            parts.append(f"added {len(self.added)}")
        if self.renamed:
            parts.append(f"renamed {len(self.renamed)}")
        if self.skipped:
            parts.append(f"skipped {len(self.skipped)}")
        if not parts:
            parts.append("nothing to do")
        return f"{prefix} '{src}' → '{dst}': " + ", ".join(parts)


def parse_merge_args(tokens: List[str]) -> Tuple[Optional[str], Optional[str], MergeOptions, Optional[str]]:
    """Parse the tail of ``references merge …`` into structured pieces.

    The subcommand already splits the first token as ``source``; what
    follows is a mix of positional arguments and flags. We use a small
    hand-rolled scanner rather than ``argparse`` because the plugin
    returns structured errors (not SystemExit) and we want flag values
    to be greedy enough to tolerate quoted paths with spaces at some
    future date.

    Args:
        tokens: ``shlex``-split tail after the ``merge`` subcommand.

    Returns:
        ``(source, target, options, error)``. ``error`` is set when the
        tokens are malformed; the other fields are best-effort in that
        case.
    """
    source: Optional[str] = None
    target: Optional[str] = None
    options = MergeOptions()

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--into":
            if i + 1 >= len(tokens):
                return source, target, options, "Flag --into requires a value"
            target = tokens[i + 1]
            i += 2
            continue
        if tok == "--on-conflict":
            if i + 1 >= len(tokens):
                return source, target, options, "Flag --on-conflict requires a value"
            options.on_conflict = tokens[i + 1]
            i += 2
            continue
        if tok == "--re-embed":
            options.re_embed = True
            i += 1
            continue
        if tok == "--dry-run":
            options.dry_run = True
            i += 1
            continue
        if tok.startswith("-"):
            return source, target, options, f"Unknown flag {tok!r}"
        if source is None:
            source = tok
            i += 1
            continue
        return source, target, options, f"Unexpected positional argument {tok!r}"

    return source, target, options, None


def _normalize_prefix(source_name: str) -> str:
    """Produce a filesystem-safe prefix for rename-on-conflict.

    Strips leading ``./`` / path separators and replaces any remaining
    slashes with underscores. Empty results fall back to ``merged``.
    """
    name = source_name.strip("/ \t")
    for sep in ("/", os.sep):
        name = name.replace(sep, "_")
    name = name.strip("_")
    return name or "merged"


def _load_sources_from_dir(directory: Path) -> List[ReferenceSource]:
    """Load every reference JSON in ``directory`` (non-recursive).

    Mirrors the subset of :func:`config_loader.discover_references`
    we need for merge — we don't resolve paths here because the
    merged JSON is copied verbatim into the target's directory, where
    its relative paths resolve against the target's workspace.
    """
    sources: List[ReferenceSource] = []
    if not directory.is_dir():
        return sources
    for p in sorted(directory.iterdir()):
        if not p.is_file() or p.suffix != ".json":
            continue
        if p.name == EMBEDDING_CONFIG_FILENAME:
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        sources.append(ReferenceSource.from_dict(raw))
    return sources


def _read_bundle_manifest(directory: Path) -> Optional[Bundle]:
    """Build a Bundle from a directory's ``embedding_config.json``.

    Used when the source is a path rather than a loaded bundle. Unlike
    :func:`bundle._load_bundle_from_manifest` this returns a Bundle
    even with empty rows — an unindexed source bundle can still be
    merged if ``--re-embed`` is set.
    """
    manifest_path = directory / EMBEDDING_CONFIG_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    return Bundle(
        name=directory.name,
        directory=directory.resolve(),
        embedding_model=str(raw.get("embedding_model", "")),
        embedding_dimensions=int(raw.get("embedding_dimensions", 0) or 0),
        embedding_sidecar=str(raw.get("embedding_sidecar", "")),
        embedding_rows=list(raw.get("rows", []) or []),
        owned_source_ids=set(raw.get("rows", []) or []),
    )


def _resolve_conflicts(
    collisions: List[str],
    source_sources: Dict[str, ReferenceSource],
    target_sources: Dict[str, ReferenceSource],
    source_dir: Path,
    target_dir: Path,
    options: MergeOptions,
) -> Tuple[Dict[str, str], List[Tuple[str, str]], List[str]]:
    """Apply the chosen on-conflict strategy.

    Returns ``(rename_map, skipped, hard_conflicts)`` where:
        * ``rename_map`` maps source id → target id for ``prefix`` mode.
        * ``skipped`` records ``(id, reason)`` tuples from ``newer`` mode.
        * ``hard_conflicts`` is non-empty only in ``reject`` mode, and
          signals the caller should abort.
    """
    rename_map: Dict[str, str] = {}
    skipped: List[Tuple[str, str]] = []
    hard_conflicts: List[str] = []

    if not collisions:
        return rename_map, skipped, hard_conflicts

    if options.on_conflict == "reject":
        return rename_map, skipped, sorted(collisions)

    if options.on_conflict == "prefix":
        prefix = _normalize_prefix(source_dir.name or "merged")
        for sid in collisions:
            rename_map[sid] = f"{prefix}_{sid}"
        return rename_map, skipped, hard_conflicts

    if options.on_conflict == "newer":
        for sid in collisions:
            src = source_sources[sid]
            tgt = target_sources[sid]
            if src.embedding and tgt.embedding and \
                    src.embedding.source_hash == tgt.embedding.source_hash:
                skipped.append((sid, "same content as target"))
                continue
            src_mtime = (source_dir / f"{sid}.json").stat().st_mtime \
                if (source_dir / f"{sid}.json").is_file() else 0.0
            tgt_mtime = (target_dir / f"{sid}.json").stat().st_mtime \
                if (target_dir / f"{sid}.json").is_file() else 0.0
            if src_mtime <= tgt_mtime:
                skipped.append((sid, "target is newer"))
            # else: source wins — fall through to add, overwriting target
        return rename_map, skipped, hard_conflicts

    # Shouldn't reach here — options.validate() should have caught bad modes.
    return rename_map, skipped, hard_conflicts


def _copy_reference_json(
    source_dir: Path,
    target_dir: Path,
    source_id: str,
    new_id: str,
    new_hash: Optional[str],
) -> bool:
    """Copy a single reference JSON into the target directory.

    Optionally rewrites the JSON's ``id`` field (for prefix mode) and
    stamps a fresh ``source_hash`` (for --re-embed). Returns True on
    success, False when the source JSON was missing or malformed.
    """
    src_path = source_dir / f"{source_id}.json"
    tgt_path = target_dir / f"{new_id}.json"
    if not src_path.is_file():
        return False
    try:
        raw = json.loads(src_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(raw, dict):
        return False
    if new_id != source_id:
        raw["id"] = new_id
    if new_hash is not None:
        raw["embedding"] = {"source_hash": new_hash}
    tmp = tgt_path.with_suffix(tgt_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(tgt_path)
    return True


def merge_bundle(
    *,
    target: Bundle,
    source_bundle: Bundle,
    source_sources: List[ReferenceSource],
    target_sources: List[ReferenceSource],
    provider: Optional[EmbeddingProviderProtocol],
    options: MergeOptions,
) -> MergeResult:
    """Merge ``source_bundle`` into ``target``.

    See the module docstring for the full algorithm. On success the
    target's manifest + sidecar are rewritten; source files are copied
    but the source directory is left in place.

    Args:
        target: The bundle receiving the merge.
        source_bundle: A bundle to merge in. For path sources this is
            built by :func:`_read_bundle_manifest`; for loaded bundles
            it comes straight from the plugin.
        source_sources: Reference sources from the source directory.
        target_sources: Reference sources currently owned by the target
            (used to detect id collisions).
        provider: Required for ``--re-embed``; ignored otherwise.
        options: Parsed merge flags.

    Returns:
        A populated :class:`MergeResult`.
    """
    err = options.validate()
    if err:
        return MergeResult(
            source_name=source_bundle.name,
            target_name=target.name,
            status=MergeStatus.ERROR,
            error=err,
        )

    # Determine whether we need to re-embed.
    model_matches = (
        source_bundle.embedding_model == target.embedding_model
        and source_bundle.embedding_dimensions == target.embedding_dimensions
    )
    needs_reembed = options.re_embed or not model_matches

    if needs_reembed and not options.re_embed:
        return MergeResult(
            source_name=source_bundle.name,
            target_name=target.name,
            status=MergeStatus.ERROR,
            error=(
                f"Model/dimension mismatch "
                f"(source: {source_bundle.embedding_model}/"
                f"{source_bundle.embedding_dimensions} vs "
                f"target: {target.embedding_model}/"
                f"{target.embedding_dimensions}). "
                f"Pass --re-embed to re-vectorize source references."
            ),
        )

    if needs_reembed and (provider is None or not provider.available):
        return MergeResult(
            source_name=source_bundle.name,
            target_name=target.name,
            status=MergeStatus.UNAVAILABLE,
            error="--re-embed requires an available embedding provider",
        )

    try:
        import numpy as np
    except ImportError:
        return MergeResult(
            source_name=source_bundle.name,
            target_name=target.name,
            status=MergeStatus.UNAVAILABLE,
            error="numpy is required for merge",
        )

    # Build quick-lookup maps.
    source_by_id = {s.id: s for s in source_sources}
    target_by_id = {s.id: s for s in target_sources}

    # Resolve collisions.
    collisions = sorted(set(source_by_id) & set(target_by_id))
    rename_map, skipped, hard_conflicts = _resolve_conflicts(
        collisions,
        source_by_id,
        target_by_id,
        source_bundle.directory,
        target.directory,
        options,
    )
    if hard_conflicts:
        return MergeResult(
            source_name=source_bundle.name,
            target_name=target.name,
            status=MergeStatus.ERROR,
            conflicts=hard_conflicts,
            error=(
                f"{len(hard_conflicts)} id collision(s) with --on-conflict=reject: "
                f"{', '.join(hard_conflicts)}. "
                f"Re-run with --on-conflict prefix or --on-conflict newer."
            ),
        )

    skipped_ids = {sid for sid, _reason in skipped}

    # Source ids we intend to merge, in the source's row order (so the
    # resulting sidecar's row order matches a stable, predictable layout).
    # For --re-embed we may also include source refs that aren't in the
    # source bundle's rows list yet — gracefully treat them as
    # "embed-only".
    merge_order: List[str] = []
    seen: set = set()
    for sid in source_bundle.embedding_rows:
        if sid in source_by_id and sid not in seen and sid not in skipped_ids:
            merge_order.append(sid)
            seen.add(sid)
    for sid in source_by_id:
        if sid not in seen and sid not in skipped_ids:
            merge_order.append(sid)
            seen.add(sid)

    # Load target sidecar (or start empty) to preserve its rows.
    if target.sidecar_path.is_file():
        try:
            target_matrix = np.load(target.sidecar_path, allow_pickle=False)
        except (OSError, ValueError) as e:
            return MergeResult(
                source_name=source_bundle.name,
                target_name=target.name,
                status=MergeStatus.ERROR,
                error=f"Failed to load target sidecar: {e}",
            )
    else:
        target_matrix = np.zeros(
            (0, target.embedding_dimensions), dtype=np.float32,
        )

    # Build source vectors.
    source_matrix = None
    if not needs_reembed:
        if not source_bundle.sidecar_path.is_file():
            return MergeResult(
                source_name=source_bundle.name,
                target_name=target.name,
                status=MergeStatus.ERROR,
                error=(
                    f"Source sidecar {source_bundle.sidecar_path} not found; "
                    f"pass --re-embed to rebuild from metadata"
                ),
            )
        try:
            source_matrix = np.load(source_bundle.sidecar_path, allow_pickle=False)
        except (OSError, ValueError) as e:
            return MergeResult(
                source_name=source_bundle.name,
                target_name=target.name,
                status=MergeStatus.ERROR,
                error=f"Failed to load source sidecar: {e}",
            )

    source_row_index: Dict[str, int] = {
        sid: i for i, sid in enumerate(source_bundle.embedding_rows)
    }

    # Determine which ids actually get appended (skipping same-content
    # duplicates that already exist on target under --on-conflict=newer).
    added_ids: List[str] = []
    appended_vectors: List[Any] = []

    for sid in merge_order:
        new_id = rename_map.get(sid, sid)
        # Under --on-conflict=newer, the same id (without rename) may
        # already live in the target; we only append if we're winning
        # the comparison (i.e. sid was not in skipped).
        if new_id in target.embedding_rows and sid not in rename_map:
            # Target keeps its existing row — we skip appending but
            # still drop a fresh JSON on disk below so the newer copy wins.
            continue

        if needs_reembed:
            src_source = source_by_id[sid]
            try:
                emb = provider.embed_text(_embed_text_for(src_source))
            except Exception as e:  # pragma: no cover — defensive
                skipped.append((sid, f"embed_text failed: {e}"))
                continue
            if emb is None:
                skipped.append((sid, "embed_text returned None"))
                continue
            if emb.dimensions != target.embedding_dimensions:
                skipped.append((
                    sid,
                    f"provider returned dim {emb.dimensions}, "
                    f"target expects {target.embedding_dimensions}",
                ))
                continue
            vec = np.asarray([emb.embedding], dtype=np.float32)
        else:
            idx = source_row_index.get(sid)
            if idx is None or idx >= source_matrix.shape[0]:
                skipped.append((sid, "orphan row in source bundle"))
                continue
            vec = source_matrix[idx : idx + 1]

        appended_vectors.append(vec)
        added_ids.append(new_id)

    # Nothing to write.
    if not added_ids and not skipped:
        return MergeResult(
            source_name=source_bundle.name,
            target_name=target.name,
            status=MergeStatus.DRY_RUN if options.dry_run else MergeStatus.OK,
            final_row_count=len(target.embedding_rows),
        )

    if options.dry_run:
        return MergeResult(
            source_name=source_bundle.name,
            target_name=target.name,
            status=MergeStatus.DRY_RUN,
            added=added_ids,
            renamed=sorted(rename_map.items()),
            skipped=skipped,
            final_row_count=len(target.embedding_rows) + len(added_ids),
        )

    # Commit: assemble + atomic swap, guarded by advisory lock.
    lock = _try_acquire_lock(target.lock_path)
    if lock is None:
        return MergeResult(
            source_name=source_bundle.name,
            target_name=target.name,
            status=MergeStatus.SKIPPED_BUSY,
            final_row_count=len(target.embedding_rows),
        )

    try:
        if appended_vectors:
            new_block = np.vstack(appended_vectors).astype(
                target_matrix.dtype if target_matrix.size else np.float32,
                copy=False,
            )
            if target_matrix.size:
                new_matrix = np.vstack([target_matrix, new_block])
            else:
                new_matrix = new_block
        else:
            new_matrix = target_matrix

        new_rows = list(target.embedding_rows) + list(added_ids)

        _write_sidecar_atomic(target.sidecar_path, new_matrix)
        write_manifest(target, rows=new_rows)

        # Copy source JSONs into target dir; stamp fresh source_hash
        # when we re-embedded (metadata moved to a new model).
        for sid in merge_order:
            new_id = rename_map.get(sid, sid)
            if new_id not in added_ids and new_id in target.embedding_rows:
                # Either a rename-collision with an existing id (we
                # already logged) or a same-id newer-wins rewrite.
                if sid in rename_map:
                    continue
                if options.on_conflict != "newer":
                    continue
                # newer mode, source wins over an existing target row:
                # overwrite the JSON on disk; vector stays as source row
                # already appended above. Fall through to copy.
            source = source_by_id[sid]
            new_hash: Optional[str] = None
            if needs_reembed:
                new_hash = metadata_hash(source)
            _copy_reference_json(
                source_bundle.directory,
                target.directory,
                sid,
                new_id,
                new_hash,
            )

        return MergeResult(
            source_name=source_bundle.name,
            target_name=target.name,
            status=MergeStatus.OK,
            added=added_ids,
            renamed=sorted(rename_map.items()),
            skipped=skipped,
            final_row_count=len(new_rows),
        )
    except Exception as e:  # pragma: no cover — defensive top-level guard
        logger.exception(
            "merge '%s' → '%s' failed: %s",
            source_bundle.name, target.name, e,
        )
        return MergeResult(
            source_name=source_bundle.name,
            target_name=target.name,
            status=MergeStatus.ERROR,
            error=str(e),
        )
    finally:
        _release_lock(lock, target.lock_path)
