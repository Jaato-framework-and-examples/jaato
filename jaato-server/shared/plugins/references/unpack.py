"""Unpack a self-contained bundle archive into a tier on the recipient.

The format is documented in :mod:`pack`. This module implements the
inverse: validate the envelope, extract the tarball into a staging
area (rejecting any path-traversal attempts), and lay the bundle down
under the chosen tier root.

Three modes determine what to do when the destination already holds a
bundle of the same name:

* ``error`` (default) — refuse to clobber; the operator decides.
* ``overwrite`` — delete the destination's contents and replace.
* ``merge`` — delegate to :func:`merge_bundle`, which handles id
  conflicts, dedupe, and re-embedding.

Auto-reconcile runs after a successful write (unless disabled) so a
recipient with a different embedding model than the packer self-heals
their sidecar instead of inheriting stale vectors.
"""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bundle import (
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    EMBEDDING_CONFIG_FILENAME,
    ROOT_BUNDLE_NAME,
    VALID_BUNDLE_TIERS,
    Bundle,
    _load_bundle_from_manifest,
)
from .pack import (
    ARCHIVE_BUNDLE_DIR,
    ARCHIVE_ENVELOPE_FILENAME,
    ARCHIVE_FORMAT_VERSION,
    ARCHIVE_PAYLOAD_DIR,
)

logger = logging.getLogger(__name__)


class UnpackMode(str, Enum):
    """Collision policy when the unpack target already exists.

    * ``ERROR`` — refuse the operation; the caller must pick a different
      target or escalate to ``OVERWRITE`` / ``MERGE``.
    * ``OVERWRITE`` — recursively delete the target and replace it with
      the archive contents. Selected references are lost.
    * ``MERGE`` — keep the existing target and run :func:`merge_bundle`
      with the unpacked archive as the source.
    """

    ERROR = "error"
    OVERWRITE = "overwrite"
    MERGE = "merge"


class UnpackError(ValueError):
    """Raised when unpack cannot proceed safely.

    Common causes: archive missing the envelope, format_version newer
    than this build supports, tar member with a path that escapes the
    staging directory, target collision without an explicit mode.
    """


@dataclass
class UnpackResult:
    """Outcome of a successful :func:`unpack_bundle` call.

    Attributes:
        archive_path: The archive that was unpacked.
        target_dir: Absolute path where the bundle now lives.
        target_tier: Tier the target lives under (workspace or user).
        target_name: Bundle name on the destination side (defaults to
            the archive's source_name; the caller may override).
        mode: The :class:`UnpackMode` actually used.
        ref_count: Reference count visible after unpack (best-effort —
            the caller should reload the catalog to refresh).
        reconciled: ``True`` when post-unpack reconcile ran.
    """

    archive_path: Path
    target_dir: Path
    target_tier: str
    target_name: str
    mode: UnpackMode
    ref_count: int
    reconciled: bool


def read_envelope(archive_path: Path) -> Dict[str, Any]:
    """Read and validate the ``bundle_archive.json`` envelope.

    Returns the parsed envelope dict. Raises :class:`UnpackError` when
    the archive doesn't look like a packed bundle or declares a format
    version this build can't read. This is exposed separately so a CLI
    "inspect" command can preview an archive without actually extracting
    it.

    Args:
        archive_path: Path to a ``.tar.gz`` produced by :func:`pack_bundle`.

    Raises:
        UnpackError: Archive missing/malformed envelope, or unknown
            format version.
        FileNotFoundError: Archive doesn't exist.
        tarfile.ReadError: Archive isn't a valid tar.
    """
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")

    with tarfile.open(archive_path, "r:*") as tf:
        try:
            member = tf.getmember(ARCHIVE_ENVELOPE_FILENAME)
        except KeyError:
            raise UnpackError(
                f"archive {archive_path.name} is missing "
                f"{ARCHIVE_ENVELOPE_FILENAME}; not a packed bundle"
            )
        extracted = tf.extractfile(member)
        if extracted is None:
            raise UnpackError(
                f"{ARCHIVE_ENVELOPE_FILENAME} in {archive_path.name} "
                f"is not a regular file"
            )
        try:
            envelope = json.loads(extracted.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise UnpackError(
                f"{ARCHIVE_ENVELOPE_FILENAME} in {archive_path.name} "
                f"is not valid JSON: {e}"
            ) from e

    if not isinstance(envelope, dict):
        raise UnpackError(f"{ARCHIVE_ENVELOPE_FILENAME} must be a JSON object")

    fmt = envelope.get("format_version")
    if fmt != ARCHIVE_FORMAT_VERSION:
        raise UnpackError(
            f"archive format_version {fmt!r} is not supported by this "
            f"jaato build (expected {ARCHIVE_FORMAT_VERSION}); upgrade "
            f"jaato or repack the bundle"
        )
    if "source_name" not in envelope:
        raise UnpackError("envelope missing 'source_name' field")
    return envelope


def unpack_bundle(
    archive_path: Path,
    *,
    target_root: Path,
    target_tier: str,
    target_name: Optional[str] = None,
    mode: UnpackMode = UnpackMode.ERROR,
) -> UnpackResult:
    """Unpack ``archive_path`` into ``target_root``.

    The destination bundle directory is ``target_root / target_name``.
    For the root bundle the destination is ``target_root`` itself
    (matches the discovery convention of ``embedding_config.json`` at
    the tier-root level).

    The function does **not** perform reconcile. The caller wires that
    in (so the live catalog and matchers can be updated alongside);
    see :meth:`ReferencesPlugin._cmd_references_unpack`. The returned
    ``reconciled`` field is always ``False`` here.

    Args:
        archive_path: ``.tar.gz`` produced by :func:`pack_bundle`.
        target_root: The tier root, e.g.
            ``<workspace>/.jaato/references`` or ``~/.jaato/references``.
        target_tier: Either ``BUNDLE_TIER_WORKSPACE`` or
            ``BUNDLE_TIER_USER``. Recorded on the result; the chosen
            root must already match this tier.
        target_name: Bundle name on this side. ``None`` means "use the
            archive's source_name" (the typical case). Pass an empty
            string to install as the root bundle.
        mode: Collision policy when ``target_root / target_name`` is
            already present.

    Returns:
        :class:`UnpackResult` (with ``reconciled=False``).

    Raises:
        UnpackError: Archive validation failed, target collision in
            ``ERROR`` mode, tar-traversal attempt, or merge mode invoked
            without a target to merge into.
        FileNotFoundError: Archive missing.
    """
    archive_path = Path(archive_path)
    target_root = Path(target_root)
    if target_tier not in VALID_BUNDLE_TIERS:
        raise UnpackError(
            f"unknown target_tier {target_tier!r}; "
            f"expected one of {', '.join(VALID_BUNDLE_TIERS)}"
        )

    envelope = read_envelope(archive_path)
    name_to_use = (
        target_name
        if target_name is not None
        else envelope.get("source_name", "")
    )
    if not isinstance(name_to_use, str):
        raise UnpackError("target_name must be a string")

    # Resolve the destination directory. Root bundle lands directly
    # under target_root; named bundles get their own subdirectory.
    if name_to_use == ROOT_BUNDLE_NAME:
        target_dir = target_root.resolve() if target_root.exists() else target_root
    else:
        target_dir = (target_root / name_to_use).resolve() \
            if (target_root / name_to_use).exists() else target_root / name_to_use

    target_exists = (
        target_dir.is_dir()
        and (target_dir / EMBEDDING_CONFIG_FILENAME).is_file()
    )
    if target_exists and mode == UnpackMode.ERROR:
        raise UnpackError(
            f"target {target_dir} already contains a bundle; pass "
            f"mode=overwrite to replace or mode=merge to merge"
        )
    if not target_exists and mode == UnpackMode.MERGE:
        raise UnpackError(
            f"merge mode requires an existing bundle at {target_dir}; "
            f"none found"
        )

    # Stage extraction so we can validate before touching the destination.
    with tempfile.TemporaryDirectory(prefix="jaato-unpack-") as staging_str:
        staging = Path(staging_str)
        _safe_extract(archive_path, staging)
        staged_bundle = staging / ARCHIVE_BUNDLE_DIR
        if not (staged_bundle / EMBEDDING_CONFIG_FILENAME).is_file():
            raise UnpackError(
                f"archive is missing {ARCHIVE_BUNDLE_DIR}/"
                f"{EMBEDDING_CONFIG_FILENAME}; not a valid bundle archive"
            )
        staged_payload = staging / ARCHIVE_PAYLOAD_DIR

        if mode == UnpackMode.MERGE:
            # Merge from the staged dir into the existing target. We
            # don't import merge_bundle here to avoid a hard dependency
            # on numpy at module import time — ``merge`` requires it,
            # but plain unpack does not.
            from .merge import MergeOptions, merge_bundle  # local import
            from .config_loader import discover_references

            existing = _load_bundle_from_manifest(
                target_dir / EMBEDDING_CONFIG_FILENAME,
                name=name_to_use,
                tier=target_tier,
            )
            staged_bundle_obj = _load_bundle_from_manifest(
                staged_bundle / EMBEDDING_CONFIG_FILENAME,
                name=name_to_use,
                tier=target_tier,
            )
            if existing is None or staged_bundle_obj is None:
                raise UnpackError(
                    "merge mode: failed to load manifests on either side"
                )
            # Place the payload inside the staged source bundle so
            # rewritten paths still resolve when merge inspects them.
            if staged_payload.is_dir() and any(staged_payload.iterdir()):
                shutil.move(
                    str(staged_payload),
                    str(staged_bundle / ARCHIVE_PAYLOAD_DIR),
                )
            source_sources = discover_references(
                str(staged_bundle),
                base_path=str(staging),
            )
            for s in source_sources:
                s.bundle_name = name_to_use
            target_sources = discover_references(
                str(target_dir),
                base_path=str(target_dir.parent),
            )
            for s in target_sources:
                s.bundle_name = name_to_use
            result = merge_bundle(
                target=existing,
                source_bundle=staged_bundle_obj,
                source_sources=source_sources,
                target_sources=target_sources,
                provider=None,
                options=MergeOptions(),
            )
            ref_count = result.final_row_count or len(target_sources)
        else:
            # ERROR (target absent) or OVERWRITE → write everything fresh.
            if target_exists and mode == UnpackMode.OVERWRITE:
                _safe_replace_bundle(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            # Move the staged bundle contents into target_dir.
            for item in staged_bundle.iterdir():
                shutil.move(str(item), str(target_dir / item.name))
            # Move the payload directory alongside the bundle so that
            # the rewritten ``payload/<slot>/<basename>`` paths in the
            # ref JSONs resolve correctly when the catalog loads.
            if staged_payload.is_dir() and any(staged_payload.iterdir()):
                payload_dst = target_dir / ARCHIVE_PAYLOAD_DIR
                payload_dst.mkdir(exist_ok=True)
                for item in staged_payload.iterdir():
                    shutil.move(str(item), str(payload_dst / item.name))

            # Best-effort ref count from the freshly-extracted bundle
            # (one JSON per ref).
            ref_count = sum(
                1 for p in target_dir.glob("*.json")
                if p.name != EMBEDDING_CONFIG_FILENAME
            )

    return UnpackResult(
        archive_path=archive_path.resolve(),
        target_dir=target_dir.resolve(),
        target_tier=target_tier,
        target_name=name_to_use,
        mode=mode,
        ref_count=ref_count,
        reconciled=False,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_extract(archive_path: Path, dest: Path) -> None:
    """Extract a tar archive into ``dest`` rejecting path-traversal attempts.

    Every member's normalized destination must remain inside ``dest``.
    Absolute paths, ``..`` segments, and symlinks/hardlinks pointing
    outside ``dest`` are rejected with :class:`UnpackError` *before* any
    bytes are written, so a malformed archive cannot leave stray files
    on disk after a partial extraction failure.
    """
    dest_resolved = dest.resolve()
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            target = (dest_resolved / member.name).resolve()
            try:
                target.relative_to(dest_resolved)
            except ValueError:
                raise UnpackError(
                    f"refusing to extract {member.name!r} — path escapes "
                    f"the unpack directory"
                )
            if member.issym() or member.islnk():
                # Resolve the link target relative to the member's parent
                # and verify it still lands inside dest. Symlinks pointing
                # at /etc/passwd would otherwise quietly succeed.
                link_target = member.linkname
                if Path(link_target).is_absolute():
                    raise UnpackError(
                        f"refusing to extract symlink {member.name!r} → "
                        f"absolute path {link_target!r}"
                    )
                resolved_link = (target.parent / link_target).resolve()
                try:
                    resolved_link.relative_to(dest_resolved)
                except ValueError:
                    raise UnpackError(
                        f"refusing to extract symlink {member.name!r} → "
                        f"{link_target!r} (escapes unpack directory)"
                    )
        # Validation passed; do the actual extraction. Python 3.12+
        # introduces ``filter='data'`` for sandboxing; fall back to the
        # plain extractall on older interpreters since we already did
        # the traversal check above.
        try:
            tf.extractall(dest, filter="data")  # type: ignore[arg-type]
        except TypeError:
            tf.extractall(dest)


def _safe_replace_bundle(target_dir: Path) -> None:
    """Remove an existing bundle directory's contents in ``OVERWRITE`` mode.

    We delete only files we recognize as bundle artefacts: the manifest,
    any sidecar ``.npy``, ``*.json`` reference files, and the
    ``payload/`` subtree. Stray non-bundle files (notes, READMEs) are
    left in place — operators sometimes drop project-specific files
    next to a bundle and clobbering them would be surprising.
    """
    if not target_dir.is_dir():
        return
    manifest = target_dir / EMBEDDING_CONFIG_FILENAME
    if manifest.is_file():
        manifest.unlink()
    for npy in target_dir.glob("*.npy"):
        npy.unlink()
    for npy_lock in target_dir.glob("*.npy.lock"):
        try:
            npy_lock.unlink()
        except FileNotFoundError:
            pass
    for js in target_dir.glob("*.json"):
        js.unlink()
    payload = target_dir / ARCHIVE_PAYLOAD_DIR
    if payload.is_dir():
        shutil.rmtree(payload)
