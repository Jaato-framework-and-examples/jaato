"""Pack a knowledge bundle into a self-contained archive for distribution.

Layout of a packed archive (gzipped tar)::

    bundle_archive.json          envelope: format_version, source_tier,
                                 source_name, packed_at, payload_paths
    bundle/                      original bundle directory contents
        embedding_config.json    untouched
        references.embeddings.npy
        my-ref.json              path field rewritten to payload/...
        ...
    payload/                     LOCAL reference payloads, deduped by
        <hash>/<basename>        resolved source path

The archive is **self-contained**: every LOCAL reference's payload is
copied into ``payload/`` and the ref JSON's ``path`` field is rewritten
so that on unpack, the references resolve relative to the destination
bundle directory regardless of the original source's location. URL,
MCP, and INLINE references pass through untouched.

Pack does *not* re-embed: the original ``.npy`` sidecar is shipped as-is
so the recipient gets a deterministic roundtrip when they load the same
embedding model. Drift caused by a different recipient model is detected
and self-healed by the post-unpack reconcile pass (see :mod:`unpack`).
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

from .bundle import Bundle, EMBEDDING_CONFIG_FILENAME
from .models import ReferenceSource, SourceType

logger = logging.getLogger(__name__)


# Top-level archive members. Kept here (and in ``unpack``) so the two
# sides agree without depending on each other beyond the format.
ARCHIVE_ENVELOPE_FILENAME = "bundle_archive.json"
ARCHIVE_BUNDLE_DIR = "bundle"
ARCHIVE_PAYLOAD_DIR = "payload"

# Schema version of the envelope. Bumped only when the archive layout
# changes in a way that older :func:`unpack_bundle` calls would not
# understand. Backwards-compatible additions (new optional envelope
# fields, new top-level files) do not bump this.
ARCHIVE_FORMAT_VERSION = 1


@dataclass
class PackResult:
    """Outcome of a successful :func:`pack_bundle` call.

    Attributes:
        archive_path: Absolute path to the produced ``.tar.gz``.
        bundle_name: The packed bundle's name.
        source_tier: The bundle's tier at pack time (recorded in the
            envelope; the recipient may unpack into a different tier).
        ref_count: Total references included in the archive (LOCAL +
            URL + MCP + INLINE).
        local_payloads: Number of distinct payload entries written under
            ``payload/`` (deduped by resolved source path).
        bytes_written: Size of the archive on disk in bytes.
    """

    archive_path: Path
    bundle_name: str
    source_tier: str
    ref_count: int
    local_payloads: int
    bytes_written: int


def pack_bundle(
    bundle: Bundle,
    bundle_sources: List[ReferenceSource],
    output_path: Path,
    *,
    jaato_version: Optional[str] = None,
) -> PackResult:
    """Write a self-contained archive of ``bundle`` to ``output_path``.

    The archive contains the bundle's manifest + sidecar + reference
    JSONs (with LOCAL paths rewritten to bundle-relative ``payload/...``
    locations), every LOCAL reference's payload (file or directory tree,
    deduped by resolved source path), and a top-level
    ``bundle_archive.json`` envelope. Symlinks encountered while
    walking LOCAL directory references are followed; size limits are
    not enforced — the operator picks what to pack.

    Args:
        bundle: The bundle to pack. Provides the source directory,
            manifest path, and tier metadata.
        bundle_sources: All references that belong to ``bundle``
            (caller filters by ``source.bundle_name``). Determines
            which payloads need to be copied.
        output_path: Destination ``.tar.gz`` path. Parent directory
            must already exist; the file is overwritten if present.
        jaato_version: Optional version string recorded in the envelope
            for forensics. ``None`` records ``"unknown"``.

    Returns:
        :class:`PackResult` summarizing what was written.

    Raises:
        FileNotFoundError: A LOCAL reference points at a path that does
            not exist on disk. We surface this rather than silently
            shipping a broken archive.
        OSError: Underlying I/O failure (disk full, permissions, etc).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stage everything in a temp directory; once complete, write the tar.
    # Doing the rewrites in-place would mutate the live workspace.
    with tempfile.TemporaryDirectory(prefix="jaato-pack-") as staging_str:
        staging = Path(staging_str)
        staged_bundle = staging / ARCHIVE_BUNDLE_DIR
        staged_payload = staging / ARCHIVE_PAYLOAD_DIR
        staged_bundle.mkdir()
        staged_payload.mkdir()

        # Step 1: copy the bundle directory verbatim. We rewrite the
        # LOCAL reference JSONs in step 3 — copying first means we
        # don't have to enumerate every non-JSON file (sidecar, README,
        # etc.) by hand.
        _copy_bundle_dir(bundle.directory, staged_bundle)

        # Step 2: collect LOCAL payloads and assign archive-relative
        # destination paths. Dedupe is keyed by the *resolved* source
        # path, so two refs pointing at the same file yield one entry.
        payload_plan = _plan_local_payloads(bundle_sources)

        # Step 3: rewrite reference JSONs that need it. Walk the staged
        # bundle dir (not the live one), and for every JSON file whose
        # id is in ``payload_plan``, replace its ``path`` field with the
        # archive-relative payload path. Drop ``resolved_path`` since
        # the recipient resolves it at load time.
        _rewrite_local_paths_in_staged_bundle(staged_bundle, payload_plan)

        # Step 4: copy the actual payloads into ``payload/``.
        for entry in payload_plan.values():
            _copy_payload(entry.source_path, staged_payload / entry.archive_relpath)

        # Step 5: write the envelope.
        envelope = {
            "format_version": ARCHIVE_FORMAT_VERSION,
            "source_tier": bundle.tier,
            "source_name": bundle.name,
            "packed_at": datetime.now(timezone.utc).isoformat(),
            "jaato_version": jaato_version or "unknown",
            "payload_paths": {
                ref_id: f"{ARCHIVE_PAYLOAD_DIR}/{entry.archive_relpath}"
                for ref_id, entry in payload_plan.items()
            },
        }
        envelope_path = staging / ARCHIVE_ENVELOPE_FILENAME
        envelope_path.write_text(
            json.dumps(envelope, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Step 6: tar.gz the staging dir into output_path. Members are
        # added with their staging-relative names so the archive layout
        # matches the documented format exactly.
        with tarfile.open(output_path, "w:gz") as tf:
            tf.add(envelope_path, arcname=ARCHIVE_ENVELOPE_FILENAME)
            tf.add(staged_bundle, arcname=ARCHIVE_BUNDLE_DIR)
            # Only add the payload dir when there's something to ship —
            # avoids an empty directory entry for URL/MCP-only bundles.
            if any(staged_payload.iterdir()):
                tf.add(staged_payload, arcname=ARCHIVE_PAYLOAD_DIR)

    bytes_written = output_path.stat().st_size

    # ``payload_plan`` is keyed by ref id; multiple refs may share a
    # single :class:`_PayloadEntry` after dedupe. Count distinct entries
    # by object identity to report the actual number of bytes copied.
    distinct_payloads = {id(e): e for e in payload_plan.values()}

    return PackResult(
        archive_path=output_path.resolve(),
        bundle_name=bundle.name,
        source_tier=bundle.tier,
        ref_count=len(bundle_sources),
        local_payloads=len(distinct_payloads),
        bytes_written=bytes_written,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _PayloadEntry:
    """One physical payload destined for ``payload/`` in the archive.

    Multiple references may share an entry when they resolve to the
    same source path (dedupe). ``archive_relpath`` is the destination
    relative to ``payload/`` (e.g. ``a1b2.../api-spec.md``).
    """

    source_path: Path
    archive_relpath: str
    is_directory: bool
    referenced_by: List[str] = field(default_factory=list)


def _plan_local_payloads(
    sources: List[ReferenceSource],
) -> Dict[str, _PayloadEntry]:
    """Map each LOCAL reference id to its payload entry, deduped by path.

    A single archive-relative directory is allocated per unique resolved
    source path. References sharing the same path get distinct ids in
    the returned dict but point to the *same* :class:`_PayloadEntry`
    instance, so the JSON rewrite step rewrites both refs to the same
    archive-relative target and the copy step writes the bytes once.

    Args:
        sources: All references in the bundle (LOCAL and otherwise).

    Returns:
        Mapping from reference id to payload entry, including only
        LOCAL references whose path resolved successfully.

    Raises:
        FileNotFoundError: A LOCAL ref's path does not exist on disk.
    """
    plan: Dict[str, _PayloadEntry] = {}
    by_resolved: Dict[str, _PayloadEntry] = {}

    for source in sources:
        if source.type != SourceType.LOCAL:
            continue
        raw_path = source.resolved_path or source.path
        if not raw_path:
            # LOCAL source with neither resolved_path nor path is malformed;
            # skip rather than crash so URL/MCP refs in the same bundle
            # still pack cleanly. The ref's JSON is still shipped — the
            # recipient will see the same broken state, which is correct.
            logger.warning(
                "pack: skipping LOCAL ref %r — no path/resolved_path set",
                source.id,
            )
            continue
        resolved = Path(raw_path).expanduser()
        if not resolved.is_absolute():
            # Relative paths can't be resolved without the workspace
            # context, but ``resolved_path`` should already be absolute
            # by the time the catalog is loaded. If we still see a
            # relative one, we can't safely include it.
            raise FileNotFoundError(
                f"LOCAL reference {source.id!r} has a relative path "
                f"({raw_path!r}) that cannot be resolved during pack; "
                f"ensure the catalog has been loaded with workspace context"
            )
        # Resolve to follow symlinks and collapse ``..`` so dedupe works
        # by canonical identity, not surface form.
        try:
            canonical = resolved.resolve(strict=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"LOCAL reference {source.id!r} points at {resolved} "
                f"which does not exist"
            ) from e

        key = str(canonical)
        existing = by_resolved.get(key)
        if existing is not None:
            existing.referenced_by.append(source.id)
            plan[source.id] = existing
            continue

        # Allocate a fresh slot. The slot directory name is a short
        # SHA-256 of the canonical path; enough to avoid collisions
        # across hundreds of payloads while staying readable.
        slot = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        archive_relpath = f"{slot}/{canonical.name}"
        entry = _PayloadEntry(
            source_path=canonical,
            archive_relpath=archive_relpath,
            is_directory=canonical.is_dir(),
            referenced_by=[source.id],
        )
        by_resolved[key] = entry
        plan[source.id] = entry

    return plan


def _copy_bundle_dir(src: Path, dst: Path) -> None:
    """Copy the bundle directory verbatim into the staging area.

    Uses :func:`shutil.copytree` with ``dirs_exist_ok=True`` so the
    pre-created ``dst`` is populated rather than recreated. Symlinks
    inside the bundle dir are followed (we want concrete file content
    in the archive).
    """
    shutil.copytree(
        src, dst,
        dirs_exist_ok=True,
        symlinks=False,  # Follow symlinks: copy the target's bytes.
    )


def _copy_payload(src: Path, dst: Path) -> None:
    """Copy a LOCAL reference's payload (file or directory) into staging.

    Args:
        src: Resolved source path (absolute, exists).
        dst: Destination path inside the staging payload dir; parent
            directories are created as needed.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        # ``followlinks=True`` walks into symlinked subdirectories.
        # Symlinks pointing at files are still copied as concrete files
        # by ``shutil.copy2`` below.
        for root, _dirs, files in os.walk(src, followlinks=True):
            root_path = Path(root)
            rel = root_path.relative_to(src)
            (dst / rel).mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(root_path / f, dst / rel / f, follow_symlinks=True)
    else:
        shutil.copy2(src, dst, follow_symlinks=True)


def _rewrite_local_paths_in_staged_bundle(
    staged_bundle: Path,
    plan: Dict[str, _PayloadEntry],
) -> None:
    """Rewrite ``path`` in each LOCAL ref JSON to its archive-relative target.

    We walk the staged bundle directory (not the live one) and rewrite
    every JSON whose id is in ``plan``. The rewrite is intentionally
    conservative:

    * ``path`` is replaced with ``payload/<slot>/<basename>``.
    * ``resolved_path`` is dropped — the recipient resolves it at load.
    * No other fields are modified, so embedding metadata, tags,
      contents subfolders, etc. remain intact.

    Args:
        staged_bundle: The staging copy of the bundle dir.
        plan: Mapping returned by :func:`_plan_local_payloads`.
    """
    if not plan:
        return
    # Index payload entries by reference id for O(1) lookup.
    for json_path in staged_bundle.rglob("*.json"):
        if json_path.name == EMBEDDING_CONFIG_FILENAME:
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        ref_id = data.get("id")
        if ref_id not in plan:
            continue
        entry = plan[ref_id]
        # New ``path`` is bundle-relative. The recipient bundle's
        # ``directory`` is where the ``payload/`` sibling lives, so a
        # bundle-relative path of ``payload/<slot>/<basename>`` resolves
        # correctly when the catalog loader joins it with the ref JSON's
        # base path.
        data["path"] = f"{ARCHIVE_PAYLOAD_DIR}/{entry.archive_relpath}"
        data.pop("resolved_path", None)
        json_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
