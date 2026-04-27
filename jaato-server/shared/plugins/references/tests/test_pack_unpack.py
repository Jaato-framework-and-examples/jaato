"""Tests for the pack/unpack subcommands.

The roundtrip cases cover the contract documented in :mod:`pack` and
:mod:`unpack`: LOCAL payloads are collected into ``payload/`` (deduped
by resolved source path, symlinks followed), reference JSONs have their
``path`` field rewritten to bundle-relative archive paths, the envelope
records the format version and source tier, and the resulting archive
unpacks cleanly into either tier with the rewritten paths intact.

URL/MCP/INLINE references are exercised to confirm they pass through
without payload copies, and tar-traversal hardening is verified by
hand-crafting a malicious archive.
"""

import io
import json
import tarfile
from pathlib import Path

import pytest

from shared.plugins.references.bundle import (
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    EMBEDDING_CONFIG_FILENAME,
    Bundle,
)
from shared.plugins.references.models import (
    InjectionMode,
    ReferenceSource,
    SourceType,
)
from shared.plugins.references.pack import (
    ARCHIVE_BUNDLE_DIR,
    ARCHIVE_ENVELOPE_FILENAME,
    ARCHIVE_FORMAT_VERSION,
    ARCHIVE_PAYLOAD_DIR,
    pack_bundle,
)
from shared.plugins.references.unpack import (
    UnpackError,
    UnpackMode,
    read_envelope,
    unpack_bundle,
)


def _bundle(tmp_path, name="teammate", tier=BUNDLE_TIER_WORKSPACE):
    """Build a minimal bundle directory + Bundle instance."""
    refs_dir = tmp_path / ".jaato" / "references"
    bundle_dir = refs_dir / name if name else refs_dir
    bundle_dir.mkdir(parents=True)
    (bundle_dir / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
        "embedding_model": "test-model",
        "embedding_dimensions": 4,
        "embedding_sidecar": "vectors.npy",
        "rows": [],
    }))
    # An empty .npy stand-in keeps the sidecar packed without numpy.
    (bundle_dir / "vectors.npy").write_bytes(b"\x93NUMPY-FAKE")
    bundle = Bundle(
        name=name,
        directory=bundle_dir.resolve(),
        embedding_model="test-model",
        embedding_dimensions=4,
        embedding_sidecar="vectors.npy",
        tier=tier,
    )
    return bundle, bundle_dir


def _local_ref(bundle_dir, ref_id, source_path, **kwargs):
    """Write a LOCAL reference JSON in the bundle and return the source dataclass."""
    ref_path = bundle_dir / f"{ref_id}.json"
    ref_path.write_text(json.dumps({
        "id": ref_id,
        "name": kwargs.get("name", ref_id),
        "description": kwargs.get("description", "test"),
        "type": "local",
        "mode": "selectable",
        "path": str(source_path),
        "tags": [],
    }))
    return ReferenceSource(
        id=ref_id,
        name=kwargs.get("name", ref_id),
        description=kwargs.get("description", "test"),
        type=SourceType.LOCAL,
        mode=InjectionMode.SELECTABLE,
        path=str(source_path),
        resolved_path=str(source_path),
        bundle_name=kwargs.get("bundle_name", ""),
    )


def _archive_members(archive_path):
    with tarfile.open(archive_path, "r:gz") as tf:
        return sorted(tf.getnames())


def _read_archive_member_text(archive_path, member_name):
    with tarfile.open(archive_path, "r:gz") as tf:
        f = tf.extractfile(member_name)
        assert f is not None
        return f.read().decode("utf-8")


class TestPack:
    """pack_bundle writes a self-contained archive with the documented layout."""

    def test_pure_metadata_bundle_packs_without_payload_dir(self, tmp_path):
        """A bundle with only URL/INLINE refs ships no payload directory."""
        bundle, bundle_dir = _bundle(tmp_path)
        url_ref = ReferenceSource(
            id="docs",
            name="Docs",
            description="External docs",
            type=SourceType.URL,
            mode=InjectionMode.SELECTABLE,
            url="https://example.com/docs",
            bundle_name="teammate",
        )
        (bundle_dir / "docs.json").write_text(json.dumps({
            "id": "docs",
            "name": "Docs",
            "description": "External docs",
            "type": "url",
            "mode": "selectable",
            "url": "https://example.com/docs",
            "tags": [],
        }))

        archive = tmp_path / "out.tar.gz"
        result = pack_bundle(bundle, [url_ref], archive)

        assert result.archive_path == archive.resolve()
        assert result.local_payloads == 0
        assert result.ref_count == 1
        members = _archive_members(archive)
        # Envelope and bundle dir present; payload dir absent.
        assert ARCHIVE_ENVELOPE_FILENAME in members
        assert any(m.startswith(f"{ARCHIVE_BUNDLE_DIR}/") for m in members)
        assert not any(m.startswith(f"{ARCHIVE_PAYLOAD_DIR}/") for m in members)

    def test_local_ref_path_is_rewritten(self, tmp_path):
        """LOCAL ref's path is rewritten to bundle-relative payload/<slot>/<basename>."""
        bundle, bundle_dir = _bundle(tmp_path)
        payload_src = tmp_path / "outside" / "api-spec.md"
        payload_src.parent.mkdir()
        payload_src.write_text("# API Spec\n")
        ref = _local_ref(bundle_dir, "api-spec", payload_src, bundle_name="teammate")

        archive = tmp_path / "out.tar.gz"
        pack_bundle(bundle, [ref], archive)

        # Read the rewritten ref JSON from the archive.
        rewritten_raw = _read_archive_member_text(
            archive, f"{ARCHIVE_BUNDLE_DIR}/api-spec.json",
        )
        rewritten = json.loads(rewritten_raw)
        assert rewritten["path"].startswith(f"{ARCHIVE_PAYLOAD_DIR}/")
        assert rewritten["path"].endswith("/api-spec.md")
        # resolved_path is dropped on pack so the recipient resolves at load.
        assert "resolved_path" not in rewritten

        # Envelope records the same archive-relative path.
        envelope = json.loads(_read_archive_member_text(archive, ARCHIVE_ENVELOPE_FILENAME))
        assert envelope["format_version"] == ARCHIVE_FORMAT_VERSION
        assert envelope["source_tier"] == BUNDLE_TIER_WORKSPACE
        assert envelope["source_name"] == "teammate"
        assert envelope["payload_paths"]["api-spec"] == rewritten["path"]

    def test_local_payloads_are_deduped_by_resolved_path(self, tmp_path):
        """Two refs pointing at the same file produce one payload entry."""
        bundle, bundle_dir = _bundle(tmp_path)
        target = tmp_path / "shared" / "doc.md"
        target.parent.mkdir()
        target.write_text("hello")
        ref_a = _local_ref(bundle_dir, "ref-a", target, bundle_name="teammate")
        ref_b = _local_ref(bundle_dir, "ref-b", target, bundle_name="teammate")

        archive = tmp_path / "out.tar.gz"
        result = pack_bundle(bundle, [ref_a, ref_b], archive)

        assert result.local_payloads == 1
        envelope = json.loads(_read_archive_member_text(archive, ARCHIVE_ENVELOPE_FILENAME))
        # Both refs map to the same archive-relative payload path.
        assert envelope["payload_paths"]["ref-a"] == envelope["payload_paths"]["ref-b"]

    def test_directory_payload_is_recursively_copied(self, tmp_path):
        """LOCAL ref pointing at a directory packs the whole subtree."""
        bundle, bundle_dir = _bundle(tmp_path)
        payload = tmp_path / "external" / "knowledge"
        (payload / "subdir").mkdir(parents=True)
        (payload / "subdir" / "page.md").write_text("page content")
        (payload / "top.md").write_text("top content")
        ref = _local_ref(bundle_dir, "kb", payload, bundle_name="teammate")

        archive = tmp_path / "out.tar.gz"
        pack_bundle(bundle, [ref], archive)

        members = _archive_members(archive)
        assert any(m.endswith("/knowledge/top.md") for m in members)
        assert any(m.endswith("/knowledge/subdir/page.md") for m in members)

    def test_symlinked_file_payload_is_followed(self, tmp_path):
        """A LOCAL ref pointing at a symlink ships the target's bytes."""
        bundle, bundle_dir = _bundle(tmp_path)
        real = tmp_path / "real.md"
        real.write_text("real content")
        link = tmp_path / "link.md"
        link.symlink_to(real)
        ref = _local_ref(bundle_dir, "linked", link, bundle_name="teammate")

        archive = tmp_path / "out.tar.gz"
        pack_bundle(bundle, [ref], archive)

        with tarfile.open(archive, "r:gz") as tf:
            # Find the file member (skip the slot directory entry).
            payload_member = next(
                m for m in tf.getmembers()
                if m.name.startswith(f"{ARCHIVE_PAYLOAD_DIR}/") and m.isfile()
            )
            extracted = tf.extractfile(payload_member)
            assert extracted is not None
            assert extracted.read() == b"real content"

    def test_missing_local_payload_raises(self, tmp_path):
        bundle, bundle_dir = _bundle(tmp_path)
        ghost = tmp_path / "does-not-exist.md"
        ref = _local_ref(bundle_dir, "ghost", ghost, bundle_name="teammate")

        archive = tmp_path / "out.tar.gz"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            pack_bundle(bundle, [ref], archive)


class TestUnpack:
    """unpack_bundle inverts pack and rejects malformed input."""

    def test_roundtrip_preserves_bundle_structure(self, tmp_path):
        """pack -> unpack reproduces the original bundle layout."""
        bundle, bundle_dir = _bundle(tmp_path, name="teammate")
        payload = tmp_path / "src" / "doc.md"
        payload.parent.mkdir()
        payload.write_text("the payload")
        ref = _local_ref(bundle_dir, "doc", payload, bundle_name="teammate")

        archive = tmp_path / "out.tar.gz"
        pack_bundle(bundle, [ref], archive)

        target_root = tmp_path / "recipient" / ".jaato" / "references"
        target_root.mkdir(parents=True)
        result = unpack_bundle(
            archive,
            target_root=target_root,
            target_tier=BUNDLE_TIER_WORKSPACE,
        )

        # The unpacked bundle directory has manifest, sidecar, ref JSON.
        target_bundle_dir = target_root / "teammate"
        assert (target_bundle_dir / EMBEDDING_CONFIG_FILENAME).is_file()
        assert (target_bundle_dir / "vectors.npy").is_file()
        assert (target_bundle_dir / "doc.json").is_file()
        # Payload landed alongside under payload/.
        payloads = list((target_bundle_dir / ARCHIVE_PAYLOAD_DIR).rglob("doc.md"))
        assert len(payloads) == 1
        assert payloads[0].read_text() == "the payload"
        # Rewritten ref JSON path resolves correctly relative to the bundle.
        ref_data = json.loads((target_bundle_dir / "doc.json").read_text())
        assert (target_bundle_dir / ref_data["path"]).is_file()
        assert result.target_tier == BUNDLE_TIER_WORKSPACE
        assert result.target_name == "teammate"
        assert result.mode == UnpackMode.ERROR
        assert result.reconciled is False

    def test_unpack_into_user_tier(self, tmp_path):
        bundle, bundle_dir = _bundle(tmp_path)
        archive = tmp_path / "out.tar.gz"
        pack_bundle(bundle, [], archive)

        user_root = tmp_path / "home" / ".jaato" / "references"
        user_root.mkdir(parents=True)
        result = unpack_bundle(
            archive,
            target_root=user_root,
            target_tier=BUNDLE_TIER_USER,
        )

        assert result.target_tier == BUNDLE_TIER_USER
        assert (user_root / "teammate" / EMBEDDING_CONFIG_FILENAME).is_file()

    def test_target_collision_error_by_default(self, tmp_path):
        bundle, _ = _bundle(tmp_path)
        archive = tmp_path / "out.tar.gz"
        pack_bundle(bundle, [], archive)

        target_root = tmp_path / "recipient" / ".jaato" / "references"
        target_root.mkdir(parents=True)
        unpack_bundle(
            archive, target_root=target_root, target_tier=BUNDLE_TIER_WORKSPACE,
        )
        # Second unpack hits the existing dir.
        with pytest.raises(UnpackError, match="already contains a bundle"):
            unpack_bundle(
                archive, target_root=target_root, target_tier=BUNDLE_TIER_WORKSPACE,
            )

    def test_overwrite_replaces_existing_bundle(self, tmp_path):
        bundle, bundle_dir = _bundle(tmp_path)
        payload = tmp_path / "src.md"
        payload.write_text("v1")
        ref = _local_ref(bundle_dir, "doc", payload, bundle_name="teammate")
        archive = tmp_path / "out.tar.gz"
        pack_bundle(bundle, [ref], archive)

        target_root = tmp_path / "recipient" / ".jaato" / "references"
        target_root.mkdir(parents=True)
        unpack_bundle(
            archive, target_root=target_root, target_tier=BUNDLE_TIER_WORKSPACE,
        )

        # Modify payload, repack, overwrite.
        payload.write_text("v2")
        archive2 = tmp_path / "out2.tar.gz"
        pack_bundle(bundle, [ref], archive2)
        unpack_bundle(
            archive2, target_root=target_root, target_tier=BUNDLE_TIER_WORKSPACE,
            mode=UnpackMode.OVERWRITE,
        )

        # The target now holds the v2 payload bytes.
        target_bundle_dir = target_root / "teammate"
        ref_data = json.loads((target_bundle_dir / "doc.json").read_text())
        assert (target_bundle_dir / ref_data["path"]).read_text() == "v2"

    def test_traversal_attempt_is_rejected(self, tmp_path):
        """A tar member whose path escapes the unpack dir is rejected."""
        bad_archive = tmp_path / "bad.tar.gz"
        # Hand-roll a malicious tarball alongside a valid envelope so
        # read_envelope() doesn't trip first. The envelope has the
        # correct format_version; the malicious member follows.
        with tarfile.open(bad_archive, "w:gz") as tf:
            envelope_bytes = json.dumps({
                "format_version": ARCHIVE_FORMAT_VERSION,
                "source_tier": BUNDLE_TIER_WORKSPACE,
                "source_name": "evil",
                "packed_at": "2026-04-27T00:00:00Z",
                "jaato_version": "test",
                "payload_paths": {},
            }).encode("utf-8")
            info = tarfile.TarInfo(ARCHIVE_ENVELOPE_FILENAME)
            info.size = len(envelope_bytes)
            tf.addfile(info, io.BytesIO(envelope_bytes))
            # Innocuous bundle dir + manifest so unpack would otherwise
            # proceed. The traversal member is what we expect to trip.
            mf_bytes = json.dumps({
                "embedding_model": "m", "embedding_dimensions": 1,
                "embedding_sidecar": "x.npy", "rows": [],
            }).encode("utf-8")
            mf_info = tarfile.TarInfo(
                f"{ARCHIVE_BUNDLE_DIR}/{EMBEDDING_CONFIG_FILENAME}"
            )
            mf_info.size = len(mf_bytes)
            tf.addfile(mf_info, io.BytesIO(mf_bytes))
            evil_bytes = b"GOTCHA"
            evil_info = tarfile.TarInfo("../escaped.txt")
            evil_info.size = len(evil_bytes)
            tf.addfile(evil_info, io.BytesIO(evil_bytes))

        target_root = tmp_path / "recipient" / ".jaato" / "references"
        target_root.mkdir(parents=True)
        with pytest.raises(UnpackError, match="escapes"):
            unpack_bundle(
                bad_archive,
                target_root=target_root,
                target_tier=BUNDLE_TIER_WORKSPACE,
                target_name="evil",
            )
        # And the file did not appear above the unpack dir.
        assert not (target_root.parent.parent / "escaped.txt").exists()


class TestReadEnvelope:
    """read_envelope rejects malformed archives without extracting."""

    def test_format_version_mismatch_rejected(self, tmp_path):
        bad = tmp_path / "bad.tar.gz"
        with tarfile.open(bad, "w:gz") as tf:
            payload = json.dumps({
                "format_version": 99,
                "source_name": "x",
            }).encode("utf-8")
            info = tarfile.TarInfo(ARCHIVE_ENVELOPE_FILENAME)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))

        with pytest.raises(UnpackError, match="format_version"):
            read_envelope(bad)

    def test_missing_envelope_rejected(self, tmp_path):
        bad = tmp_path / "noenv.tar.gz"
        with tarfile.open(bad, "w:gz") as tf:
            info = tarfile.TarInfo("random.txt")
            payload = b"hello"
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))

        with pytest.raises(UnpackError, match="missing"):
            read_envelope(bad)

    def test_missing_archive_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_envelope(tmp_path / "nope.tar.gz")
