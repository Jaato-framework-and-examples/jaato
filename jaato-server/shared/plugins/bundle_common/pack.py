"""Pack one or more bundles into a self-contained archive (v2 format).

A v2 archive is a gzipped tarball that can hold bundles from any
combination of registered :class:`bundle_common.handler.BundleEntryHandler`
kinds. Layout::

    bundle_archive.json          envelope (format_version, source_tier,
                                 source_name, packed_at, kinds[], per_kind{})
    kinds/<kind>/
        bundle/                  the bundle directory contents for this kind
        payload/<slot>/<basename>  external payloads (deduped by canonical
                                 source path within the kind)

For backwards compatibility :mod:`bundle_common.unpack` also reads v1
archives (the references-only single-kind layout shipped before this
phase). v1 archives have ``bundle/`` and ``payload/`` at the top level
and an envelope with ``format_version=1`` and ``payload_paths``
populated. They are treated as a single ``"references"`` kind on read.

The packer is parametric over :class:`BundleEntryHandler`: it asks
each handler for the entries it owns, calls ``collect_pack_payloads``
to discover external payloads, and ``rewrite_for_pack`` to mutate the
staged copies of those entries before they're tar'd. Domains without
external payloads (URL/MCP/INLINE refs, profiles, services) need no
overrides — the protocol's default no-ops give a clean self-contained
archive for free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .bundle import (
    EMBEDDING_CONFIG_FILENAME,
    Bundle,
    discover_bundles,
    resolve_bundle_roots,
)
from .handler import (
    BundleEntry,
    BundleEntryHandler,
    BundleEntryRegistry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

# Top-level archive members. v1 archives put ``bundle/`` and ``payload/``
# directly at the root and use ``EMBEDDING_CONFIG_FILENAME`` as the
# manifest. v2 archives nest everything under ``kinds/<kind>/``.
ARCHIVE_ENVELOPE_FILENAME = "bundle_archive.json"
ARCHIVE_KINDS_DIR = "kinds"
ARCHIVE_BUNDLE_DIR = "bundle"
ARCHIVE_PAYLOAD_DIR = "payload"

# v1 layout (references-only) — read-only support inside :mod:`unpack`.
# These names mirror what the original references-only pack module
# wrote so a v1 archive remains unpackable by recent jaato builds.
ARCHIVE_FORMAT_VERSION_V1 = 1

# v2 layout (multi-kind) — what new packs produce.
ARCHIVE_FORMAT_VERSION = 2

# Default kind a v1 archive's contents are routed to. The pre-Phase-8
# pack module only ever emitted references content, so v1 archives
# always describe a single references bundle.
LEGACY_V1_KIND = "references"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class KindPackResult:
    """Per-kind contribution recorded in a :class:`PackResult`."""

    kind: str
    bundle_name: str
    bundle_directory: Path
    entry_count: int
    payload_count: int


@dataclass
class PackResult:
    """Outcome of a successful :func:`pack_bundles` call.

    Attributes:
        archive_path: Absolute path to the produced ``.tar.gz``.
        source_tier: Tier the source bundles were packed from.
        source_name: Bundle name common across kinds.
        kinds: One :class:`KindPackResult` per kind included.
        bytes_written: Size of the archive on disk in bytes.
        format_version: The v2 archive format version recorded in the
            envelope.
    """

    archive_path: Path
    source_tier: str
    source_name: str
    kinds: List[KindPackResult] = field(default_factory=list)
    bytes_written: int = 0
    format_version: int = ARCHIVE_FORMAT_VERSION


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def pack_bundle(
    handler: BundleEntryHandler,
    bundle: Bundle,
    archive_path: Path,
    *,
    jaato_version: Optional[str] = None,
) -> PackResult:
    """Pack a single bundle (one kind) into a v2 archive.

    Convenience wrapper for the common case of "pack just this one
    references bundle" or "pack just this one agents bundle". The
    archive layout is the same v2 format as :func:`pack_bundles` —
    even single-kind archives use ``kinds/<kind>/`` so the unpack side
    can stay uniform.

    Args:
        handler: Owns the bundle's entries and any external payloads.
        bundle: The bundle to pack. Must already be a known bundle to
            ``handler`` — the packer doesn't re-validate ownership.
        archive_path: Destination ``.tar.gz``. Parent directory is
            created if missing; the file is overwritten if present.
        jaato_version: Optional version string recorded in the
            envelope. ``None`` records ``"unknown"``.

    Returns:
        :class:`PackResult` summarizing what was written.
    """
    return pack_bundles(
        [(handler, bundle)],
        archive_path,
        source_name=bundle.name,
        source_tier=bundle.tier,
        jaato_version=jaato_version,
    )


def pack_bundle_set(
    name: str,
    tier: str,
    archive_path: Path,
    *,
    registry: BundleEntryRegistry,
    workspace_path: Optional[Path] = None,
    user_home: Optional[Path] = None,
    jaato_version: Optional[str] = None,
) -> PackResult:
    """Pack every bundle named ``name`` in ``tier`` across all
    registered handlers (composite archive).

    For each handler the packer discovers bundles under
    ``handler.domain_subpath`` (joined with ``workspace_path`` /
    ``user_home`` via :func:`resolve_bundle_roots`) and includes the
    one matching ``(name, tier)`` if present. Handlers with no matching
    bundle are silently skipped — a composite pack of ``teammate`` will
    happily produce an archive containing only the references and
    agents portions when the operator hasn't created a tasks-side
    ``teammate`` bundle.

    Args:
        name: Bundle name to look for in every handler's domain.
        tier: ``BUNDLE_TIER_WORKSPACE`` or ``BUNDLE_TIER_USER`` —
            both sides must use the same tier; cross-tier set-packing
            is intentionally unsupported (would surprise users).
        archive_path: Destination ``.tar.gz``.
        registry: The registry to walk. Production code passes the
            module-level singleton; tests pass an isolated instance.
        workspace_path: Workspace root for the workspace tier (None
            disables the workspace tier; the user tier is always
            available).
        user_home: Override for ``Path.home()`` (test seam).
        jaato_version: Optional version string recorded in the envelope.

    Returns:
        :class:`PackResult` listing each kind that contributed.

    Raises:
        ValueError: No handler had a bundle matching ``(name, tier)``.
    """
    pairs: List[Tuple[BundleEntryHandler, Bundle]] = []
    for handler in registry.all_handlers():
        roots = resolve_bundle_roots(
            workspace_path,
            domain_subpath=handler.domain_subpath,
            user_home=user_home,
        )
        bundles = discover_bundles(roots)
        match = next(
            (b for b in bundles if b.name == name and b.tier == tier), None,
        )
        if match is not None:
            pairs.append((handler, match))

    if not pairs:
        raise ValueError(
            f"no bundle named '{name}' in tier '{tier}' found across any "
            f"registered kinds ({', '.join(registry.kinds()) or '(none)'})"
        )

    return pack_bundles(
        pairs,
        archive_path,
        source_name=name,
        source_tier=tier,
        jaato_version=jaato_version,
    )


def pack_bundles(
    handler_bundle_pairs: List[Tuple[BundleEntryHandler, Bundle]],
    archive_path: Path,
    *,
    source_name: str,
    source_tier: str,
    jaato_version: Optional[str] = None,
) -> PackResult:
    """Pack the given ``(handler, bundle)`` pairs into one v2 archive.

    Lower-level entry point used by :func:`pack_bundle` (single-kind)
    and :func:`pack_bundle_set` (composite). Most callers should use
    one of those wrappers; this is exposed for the rare case where the
    bundle list has already been resolved (e.g. by a test or by a
    higher-level command that needs unusual filtering).

    Args:
        handler_bundle_pairs: Each pair contributes one
            ``kinds/<kind>/`` subtree to the archive.
        archive_path: Destination ``.tar.gz``.
        source_name: Bundle name recorded in the envelope (used by
            unpack as the default install name).
        source_tier: Tier recorded in the envelope.
        jaato_version: Optional version string for the envelope.

    Returns:
        :class:`PackResult` summarizing every kind written.
    """
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if not handler_bundle_pairs:
        raise ValueError("handler_bundle_pairs must contain at least one entry")

    per_kind_payload_paths: Dict[str, Dict[str, str]] = {}
    kind_results: List[KindPackResult] = []

    with tempfile.TemporaryDirectory(prefix="jaato-pack-") as staging_str:
        staging = Path(staging_str)
        kinds_root = staging / ARCHIVE_KINDS_DIR
        kinds_root.mkdir()

        for handler, bundle in handler_bundle_pairs:
            kind_root = kinds_root / handler.kind
            staged_bundle = kind_root / ARCHIVE_BUNDLE_DIR
            staged_payload = kind_root / ARCHIVE_PAYLOAD_DIR
            kind_root.mkdir()
            staged_payload.mkdir()

            # Step 1: copy the bundle directory verbatim. We rewrite
            # entries' files in step 3 — copying first means we don't
            # have to enumerate every non-JSON file (sidecar, README,
            # etc.) by hand.
            _copy_bundle_dir(bundle.directory, staged_bundle)

            # Step 2: collect external payloads via the handler hook.
            # Dedupe is keyed by canonical source path within this
            # kind; cross-kind dedupe is intentionally not done because
            # the archive's payload dirs are kind-scoped.
            entries_in_bundle = [
                e for e in handler.list_entries() if e.bundle_name == bundle.name
            ]
            payload_plan, payload_map = _plan_payloads(
                handler, entries_in_bundle,
            )

            # Step 3: rewrite each entry's staged copy via the handler.
            # The file must exist in the staged dir (we just copied
            # the bundle dir verbatim), so we rebind ``file_path`` to
            # the staged path before calling rewrite_for_pack.
            for entry in entries_in_bundle:
                staged_entry = _rebind_entry_to_staged(
                    entry, bundle.directory, staged_bundle,
                )
                if staged_entry is None:
                    continue
                handler.rewrite_for_pack(staged_entry, payload_map)

            # Step 4: copy the actual payloads into ``payload/<slot>/``.
            for slot_relpath, source_path in payload_plan.items():
                _copy_payload(source_path, staged_payload / slot_relpath)

            per_kind_payload_paths[handler.kind] = {
                # Map ref id -> archive-relative path inside this kind.
                entry_id: f"{ARCHIVE_PAYLOAD_DIR}/{slot}"
                for entry_id, slot in _entry_to_slot_map(
                    handler, entries_in_bundle, payload_map,
                ).items()
            }
            kind_results.append(KindPackResult(
                kind=handler.kind,
                bundle_name=bundle.name,
                bundle_directory=bundle.directory,
                entry_count=len(entries_in_bundle),
                payload_count=len(payload_plan),
            ))

        # Step 5: write the envelope.
        envelope = {
            "format_version": ARCHIVE_FORMAT_VERSION,
            "source_tier": source_tier,
            "source_name": source_name,
            "packed_at": datetime.now(timezone.utc).isoformat(),
            "jaato_version": jaato_version or "unknown",
            "kinds": sorted({kr.kind for kr in kind_results}),
            "per_kind": {
                kind: {"payload_paths": paths}
                for kind, paths in per_kind_payload_paths.items()
            },
        }
        envelope_path = staging / ARCHIVE_ENVELOPE_FILENAME
        envelope_path.write_text(
            json.dumps(envelope, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Step 6: tar.gz the staging dir into archive_path.
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(envelope_path, arcname=ARCHIVE_ENVELOPE_FILENAME)
            tf.add(kinds_root, arcname=ARCHIVE_KINDS_DIR)

    bytes_written = archive_path.stat().st_size
    return PackResult(
        archive_path=archive_path.resolve(),
        source_tier=source_tier,
        source_name=source_name,
        kinds=kind_results,
        bytes_written=bytes_written,
        format_version=ARCHIVE_FORMAT_VERSION,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _plan_payloads(
    handler: BundleEntryHandler,
    entries: List[BundleEntry],
) -> Tuple[Dict[str, Path], Dict[Path, str]]:
    """Build the dedupe plan for a kind's payloads.

    Returns two parallel maps:

    * ``slot_to_source``: ``{archive_relpath: canonical_source_path}``,
      used to drive the actual file copies into ``payload/<slot>/``.
    * ``source_to_slot``: ``{canonical_source_path: archive_relpath}``,
      passed to ``handler.rewrite_for_pack`` so entries can rewrite
      their on-disk path fields to the new archive-relative location.

    Slot names are derived from the SHA-256 of the canonical source
    path (first 16 hex chars) — short enough to read in archive
    listings, collision-free in practice for any realistic bundle.
    """
    slot_to_source: Dict[str, Path] = {}
    source_to_slot: Dict[Path, str] = {}
    for entry in entries:
        for source_path, archive_basename in handler.collect_pack_payloads(entry):
            canonical = source_path.resolve()
            if canonical in source_to_slot:
                continue  # already planned via another entry
            slot = hashlib.sha256(
                str(canonical).encode("utf-8"),
            ).hexdigest()[:16]
            relpath = f"{slot}/{archive_basename}"
            slot_to_source[relpath] = canonical
            source_to_slot[canonical] = f"{ARCHIVE_PAYLOAD_DIR}/{relpath}"
    return slot_to_source, source_to_slot


def _entry_to_slot_map(
    handler: BundleEntryHandler,
    entries: List[BundleEntry],
    payload_map: Dict[Path, str],
) -> Dict[str, str]:
    """Compute ``{entry_id: archive_payload_path}`` for the envelope.

    Mirrors ``payload_map`` (which is keyed by source path) but keyed
    by entry id so a recipient inspecting the envelope can quickly
    find which payload belongs to which entry. Entries without a
    payload are omitted.
    """
    result: Dict[str, str] = {}
    for entry in entries:
        for source_path, _basename in handler.collect_pack_payloads(entry):
            archive_path = payload_map.get(source_path.resolve())
            if archive_path is not None:
                # Strip the leading ``payload/`` prefix from the
                # archive_path; the envelope stores the kind-relative
                # form so the same key shape works for v1 + v2.
                if archive_path.startswith(f"{ARCHIVE_PAYLOAD_DIR}/"):
                    result[entry.id] = archive_path[len(ARCHIVE_PAYLOAD_DIR) + 1:]
                else:
                    result[entry.id] = archive_path
            break  # first payload wins; multi-payload entries are rare
    return result


def _copy_bundle_dir(src: Path, dst: Path) -> None:
    """Copy the bundle directory verbatim into the staging area.

    Symlinks inside the bundle dir are followed (we want concrete
    file content in the archive). Re-uses ``dirs_exist_ok=True`` so
    the pre-created ``dst`` is populated rather than recreated.
    """
    shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=False)


def _copy_payload(src: Path, dst: Path) -> None:
    """Copy a payload (file or directory) into staging.

    Args:
        src: Resolved source path (absolute, exists).
        dst: Destination path inside the staging payload dir; parent
            directories are created as needed.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for root, _dirs, files in os.walk(src, followlinks=True):
            root_path = Path(root)
            rel = root_path.relative_to(src)
            (dst / rel).mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(root_path / f, dst / rel / f, follow_symlinks=True)
    else:
        shutil.copy2(src, dst, follow_symlinks=True)


def _rebind_entry_to_staged(
    entry: BundleEntry,
    live_bundle_dir: Path,
    staged_bundle_dir: Path,
) -> Optional[BundleEntry]:
    """Return a copy of ``entry`` with ``file_path`` pointing at the
    staged copy of its file inside the archive's bundle dir.

    The handler's ``rewrite_for_pack`` mutates the file at the path
    it receives, so it must point at the *staging* copy — not the
    live workspace one — to avoid side-effects. Returns ``None`` if
    the entry's file doesn't appear under ``live_bundle_dir`` (which
    would mean the entry was free, not bundled).
    """
    try:
        rel = entry.file_path.relative_to(live_bundle_dir)
    except ValueError:
        return None
    staged_file = staged_bundle_dir / rel
    if not staged_file.is_file():
        return None
    return BundleEntry(
        id=entry.id,
        kind=entry.kind,
        file_path=staged_file,
        bundle_name=entry.bundle_name,
        bundle_tier=entry.bundle_tier,
    )
