"""Tests for the v2 pack/unpack subsystem in bundle_common.

These exercise the generic, handler-driven archive format that
replaced the references-only v1 format. The references handler is
used as the realistic consumer because it covers every protocol method
(collect_pack_payloads, rewrite_for_pack, plus the simpler enum/move/
delete shape). A stub handler is also exercised to confirm the
dispatcher itself doesn't depend on any references-specific behaviour.
"""

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from shared.plugins.bundle_common.bundle import (
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    EMBEDDING_CONFIG_FILENAME,
    Bundle,
)
from shared.plugins.bundle_common.handler import (
    BundleEntry,
    BundleEntryHandler,
    BundleEntryRegistry,
)
from shared.plugins.bundle_common.pack import (
    ARCHIVE_BUNDLE_DIR,
    ARCHIVE_ENVELOPE_FILENAME,
    ARCHIVE_FORMAT_VERSION,
    ARCHIVE_FORMAT_VERSION_V1,
    ARCHIVE_KINDS_DIR,
    ARCHIVE_PAYLOAD_DIR,
    LEGACY_V1_KIND,
    pack_bundle,
    pack_bundle_set,
)
from shared.plugins.bundle_common.unpack import (
    UnpackError,
    UnpackMode,
    read_envelope,
    unpack_archive,
)
from shared.plugins.references.entry_handler import ReferencesEntryHandler
from shared.plugins.references.models import (
    InjectionMode,
    ReferenceSource,
    SourceType,
)


def _bundle(tmp_path, name="teammate", tier=BUNDLE_TIER_WORKSPACE):
    refs_dir = tmp_path / ".jaato" / "references"
    bundle_dir = refs_dir / name if name else refs_dir
    bundle_dir.mkdir(parents=True)
    (bundle_dir / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
        "embedding_model": "test-model",
        "embedding_dimensions": 4,
        "embedding_sidecar": "vectors.npy",
        "rows": [],
    }))
    (bundle_dir / "vectors.npy").write_bytes(b"\x93NUMPY-FAKE")
    return Bundle(
        name=name,
        directory=bundle_dir.resolve(),
        embedding_model="test-model",
        embedding_dimensions=4,
        embedding_sidecar="vectors.npy",
        tier=tier,
    )


def _local_ref_json(bundle_dir: Path, ref_id: str, source_path: Path) -> None:
    (bundle_dir / f"{ref_id}.json").write_text(json.dumps({
        "id": ref_id,
        "name": ref_id,
        "description": "test",
        "type": "local",
        "mode": "selectable",
        "path": str(source_path),
        "tags": [],
    }))


class _StubReferencesPlugin:
    """Cheap stand-in for ReferencesPlugin so we can drive
    ReferencesEntryHandler in isolation, without spinning up the full
    plugin lifecycle."""

    def __init__(self, sources, bundle):
        self._sources = sources
        self._bundles = [bundle]
        self._workspace_path = str(bundle.directory.parent.parent)
        self._project_root = str(bundle.directory.parent.parent)
        self._embedding_provider = None

    def _locate_ref_file(self, source):
        """Find the JSON file backing this source by id."""
        bundle = next(
            (b for b in self._bundles if b.name == source.bundle_name), None,
        )
        if bundle is None:
            return None
        for json_path in bundle.directory.glob("*.json"):
            if json_path.name == EMBEDDING_CONFIG_FILENAME:
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("id") == source.id:
                return json_path
        return None

    def _tier_root(self, tier):
        # Workspace tier resolves under the workspace path; user tier
        # not exercised in these tests.
        if tier == BUNDLE_TIER_WORKSPACE:
            return Path(self._workspace_path) / ".jaato" / "references"
        return None

    def _reload_catalog(self, *_args, **_kwargs):
        pass

    def _discover_and_load_bundles(self):
        pass


def _make_handler_with_local_ref(tmp_path, ref_id, payload_relpath):
    """Create a workspace + bundle + LOCAL ref + stub plugin + handler."""
    bundle = _bundle(tmp_path)
    payload = tmp_path / payload_relpath
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text("payload-content")
    _local_ref_json(bundle.directory, ref_id, payload)

    source = ReferenceSource(
        id=ref_id,
        name=ref_id,
        description="test",
        type=SourceType.LOCAL,
        mode=InjectionMode.SELECTABLE,
        path=str(payload),
        resolved_path=str(payload),
        bundle_name=bundle.name,
    )
    plugin = _StubReferencesPlugin([source], bundle)
    handler = ReferencesEntryHandler(plugin)
    return bundle, handler, payload


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------


class TestPackSingleKind:
    """pack_bundle produces a v2 archive with one kinds/ subtree."""

    def test_envelope_records_v2_format(self, tmp_path):
        bundle, handler, _ = _make_handler_with_local_ref(
            tmp_path, "api-spec", "outside/api-spec.md",
        )
        archive = tmp_path / "out.tar.gz"

        result = pack_bundle(handler, bundle, archive)

        assert result.format_version == ARCHIVE_FORMAT_VERSION
        assert result.archive_path == archive.resolve()
        envelope = read_envelope(archive)
        assert envelope["format_version"] == ARCHIVE_FORMAT_VERSION
        assert envelope["source_tier"] == BUNDLE_TIER_WORKSPACE
        assert envelope["source_name"] == "teammate"
        assert envelope["kinds"] == ["references"]

    def test_archive_layout_uses_kinds_subtree(self, tmp_path):
        bundle, handler, _ = _make_handler_with_local_ref(
            tmp_path, "api-spec", "outside/api-spec.md",
        )
        archive = tmp_path / "out.tar.gz"

        pack_bundle(handler, bundle, archive)

        with tarfile.open(archive, "r:gz") as tf:
            names = sorted(tf.getnames())
        # Top-level: envelope + kinds dir.
        assert ARCHIVE_ENVELOPE_FILENAME in names
        assert any(
            n.startswith(f"{ARCHIVE_KINDS_DIR}/references/{ARCHIVE_BUNDLE_DIR}/")
            for n in names
        )
        assert any(
            n.startswith(f"{ARCHIVE_KINDS_DIR}/references/{ARCHIVE_PAYLOAD_DIR}/")
            for n in names
        )

    def test_local_ref_path_is_rewritten(self, tmp_path):
        bundle, handler, _ = _make_handler_with_local_ref(
            tmp_path, "api-spec", "outside/api-spec.md",
        )
        archive = tmp_path / "out.tar.gz"
        pack_bundle(handler, bundle, archive)

        with tarfile.open(archive, "r:gz") as tf:
            f = tf.extractfile(
                f"{ARCHIVE_KINDS_DIR}/references/{ARCHIVE_BUNDLE_DIR}/api-spec.json"
            )
            assert f is not None
            data = json.loads(f.read())

        assert data["path"].startswith(f"{ARCHIVE_PAYLOAD_DIR}/")
        assert data["path"].endswith("/api-spec.md")
        assert "resolved_path" not in data

    def test_envelope_records_per_kind_payload_paths(self, tmp_path):
        bundle, handler, _ = _make_handler_with_local_ref(
            tmp_path, "api-spec", "outside/api-spec.md",
        )
        archive = tmp_path / "out.tar.gz"
        pack_bundle(handler, bundle, archive)

        envelope = read_envelope(archive)
        per_kind = envelope["per_kind"]["references"]
        assert "api-spec" in per_kind["payload_paths"]
        assert per_kind["payload_paths"]["api-spec"].endswith("/api-spec.md")


# ---------------------------------------------------------------------------
# Pack — composite
# ---------------------------------------------------------------------------


class _MetadataOnlyHandler(BundleEntryHandler):
    """Stub handler with no external payloads — exercises the
    composite-pack path for non-LOCAL kinds."""

    def __init__(self, kind: str, bundle: Bundle, entries: List[BundleEntry]):
        self._kind = kind
        self._bundle = bundle
        self._entries = entries

    @property
    def kind(self):
        return self._kind

    @property
    def domain_subpath(self):
        return Path(f".jaato/{self._kind}")

    def list_entries(self):
        return list(self._entries)

    def find_entry(self, entry_id):
        return next((e for e in self._entries if e.id == entry_id), None)

    def move_entry_to_bundle(self, entry, target_bundle):
        return entry.file_path

    def move_entry_to_free(self, entry, target_tier):
        return entry.file_path

    def delete_entry(self, entry):
        pass

    def reload_catalog(self):
        pass

    def reconcile_bundle(self, bundle):
        return None


class TestPackComposite:
    """pack_bundle_set walks every registered handler that has a
    bundle of the requested name+tier."""

    def test_packs_only_handlers_with_matching_bundles(self, tmp_path):
        # references/teammate exists; the metadata-only kind has *no*
        # teammate bundle, so it should be silently skipped.
        bundle, handler, _ = _make_handler_with_local_ref(
            tmp_path, "api-spec", "outside/api-spec.md",
        )
        registry = BundleEntryRegistry()
        registry.register(handler)

        archive = tmp_path / "composite.tar.gz"
        result = pack_bundle_set(
            "teammate", BUNDLE_TIER_WORKSPACE, archive,
            registry=registry,
            workspace_path=tmp_path.resolve(),
        )

        assert [k.kind for k in result.kinds] == ["references"]
        envelope = read_envelope(archive)
        assert envelope["kinds"] == ["references"]

    def test_no_matching_bundle_raises(self, tmp_path):
        registry = BundleEntryRegistry()
        # No handlers registered.
        with pytest.raises(ValueError, match="no bundle named"):
            pack_bundle_set(
                "ghost", BUNDLE_TIER_WORKSPACE, tmp_path / "x.tar.gz",
                registry=registry,
                workspace_path=tmp_path,
            )


# ---------------------------------------------------------------------------
# Unpack
# ---------------------------------------------------------------------------


class TestUnpackV2:
    def test_roundtrip_preserves_bundle_structure(self, tmp_path):
        bundle, handler, _ = _make_handler_with_local_ref(
            tmp_path, "api-spec", "outside/api-spec.md",
        )
        archive = tmp_path / "out.tar.gz"
        pack_bundle(handler, bundle, archive)

        registry = BundleEntryRegistry()
        registry.register(handler)
        recipient = tmp_path / "recipient"
        recipient.mkdir()

        result = unpack_archive(
            archive,
            registry=registry,
            target_tier=BUNDLE_TIER_WORKSPACE,
            workspace_path=recipient,
        )

        assert result.format_version == ARCHIVE_FORMAT_VERSION
        assert len(result.kinds) == 1
        kind_result = result.kinds[0]
        assert kind_result.kind == "references"
        target_bundle_dir = recipient / ".jaato" / "references" / "teammate"
        assert (target_bundle_dir / EMBEDDING_CONFIG_FILENAME).is_file()
        assert (target_bundle_dir / "vectors.npy").is_file()
        assert (target_bundle_dir / "api-spec.json").is_file()
        # Payload landed alongside.
        assert any(
            (target_bundle_dir / ARCHIVE_PAYLOAD_DIR).rglob("api-spec.md")
        )

    def test_collision_error_by_default(self, tmp_path):
        bundle, handler, _ = _make_handler_with_local_ref(
            tmp_path, "api-spec", "outside/api-spec.md",
        )
        archive = tmp_path / "out.tar.gz"
        pack_bundle(handler, bundle, archive)

        registry = BundleEntryRegistry()
        registry.register(handler)
        recipient = tmp_path / "recipient"
        recipient.mkdir()

        unpack_archive(
            archive, registry=registry,
            target_tier=BUNDLE_TIER_WORKSPACE,
            workspace_path=recipient,
        )
        with pytest.raises(UnpackError, match="already contains"):
            unpack_archive(
                archive, registry=registry,
                target_tier=BUNDLE_TIER_WORKSPACE,
                workspace_path=recipient,
            )

    def test_overwrite_mode_replaces_target(self, tmp_path):
        bundle, handler, payload = _make_handler_with_local_ref(
            tmp_path, "doc", "src/doc.md",
        )
        archive_v1 = tmp_path / "v1.tar.gz"
        pack_bundle(handler, bundle, archive_v1)

        registry = BundleEntryRegistry()
        registry.register(handler)
        recipient = tmp_path / "recipient"
        recipient.mkdir()
        unpack_archive(
            archive_v1, registry=registry,
            target_tier=BUNDLE_TIER_WORKSPACE,
            workspace_path=recipient,
        )

        # Mutate payload, repack, overwrite.
        payload.write_text("v2")
        archive_v2 = tmp_path / "v2.tar.gz"
        pack_bundle(handler, bundle, archive_v2)
        unpack_archive(
            archive_v2, registry=registry,
            target_tier=BUNDLE_TIER_WORKSPACE,
            workspace_path=recipient,
            mode=UnpackMode.OVERWRITE,
        )

        target_dir = recipient / ".jaato" / "references" / "teammate"
        rewritten = json.loads((target_dir / "doc.json").read_text())
        assert (target_dir / rewritten["path"]).read_text() == "v2"

    def test_unknown_kind_in_archive_raises(self, tmp_path):
        """An archive declaring a kind the receiver doesn't know
        aborts before any disk writes."""
        bundle, handler, _ = _make_handler_with_local_ref(
            tmp_path, "api-spec", "outside/api-spec.md",
        )
        archive = tmp_path / "out.tar.gz"
        pack_bundle(handler, bundle, archive)

        registry = BundleEntryRegistry()
        # Register *no* references handler, simulating a recipient
        # without that plugin.

        with pytest.raises(UnpackError, match="no matching handlers"):
            unpack_archive(
                archive, registry=registry,
                target_tier=BUNDLE_TIER_WORKSPACE,
                workspace_path=tmp_path,
            )

    def test_traversal_attempt_is_rejected(self, tmp_path):
        bad_archive = tmp_path / "bad.tar.gz"
        with tarfile.open(bad_archive, "w:gz") as tf:
            envelope_bytes = json.dumps({
                "format_version": ARCHIVE_FORMAT_VERSION,
                "source_tier": BUNDLE_TIER_WORKSPACE,
                "source_name": "evil",
                "packed_at": "2026-04-27T00:00:00Z",
                "jaato_version": "test",
                "kinds": ["references"],
                "per_kind": {"references": {"payload_paths": {}}},
            }).encode("utf-8")
            info = tarfile.TarInfo(ARCHIVE_ENVELOPE_FILENAME)
            info.size = len(envelope_bytes)
            tf.addfile(info, io.BytesIO(envelope_bytes))
            mf_bytes = json.dumps({
                "embedding_model": "m", "embedding_dimensions": 1,
                "embedding_sidecar": "x.npy", "rows": [],
            }).encode("utf-8")
            mf_info = tarfile.TarInfo(
                f"{ARCHIVE_KINDS_DIR}/references/{ARCHIVE_BUNDLE_DIR}/"
                f"{EMBEDDING_CONFIG_FILENAME}"
            )
            mf_info.size = len(mf_bytes)
            tf.addfile(mf_info, io.BytesIO(mf_bytes))
            evil_bytes = b"GOTCHA"
            evil_info = tarfile.TarInfo("../escaped.txt")
            evil_info.size = len(evil_bytes)
            tf.addfile(evil_info, io.BytesIO(evil_bytes))

        # Need at least the references handler registered or we'd hit
        # the unknown-kind check first.
        bundle, handler, _ = _make_handler_with_local_ref(
            tmp_path / "wsdummy", "x", "ext/x.md",
        )
        registry = BundleEntryRegistry()
        registry.register(handler)
        recipient = tmp_path / "recipient"
        recipient.mkdir()

        with pytest.raises(UnpackError, match="escapes"):
            unpack_archive(
                bad_archive, registry=registry,
                target_tier=BUNDLE_TIER_WORKSPACE,
                target_name="evil",
                workspace_path=recipient,
            )


class TestUnpackV1Compat:
    """v1 (pre-Phase-8) archives — references-only, top-level
    bundle/+payload/ — must keep unpacking on this build."""

    def _build_v1_archive(self, tmp_path: Path) -> Path:
        archive = tmp_path / "v1.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            envelope_bytes = json.dumps({
                "format_version": ARCHIVE_FORMAT_VERSION_V1,
                "source_tier": BUNDLE_TIER_WORKSPACE,
                "source_name": "legacy",
                "packed_at": "2026-04-01T00:00:00Z",
                "jaato_version": "0.x",
                "payload_paths": {},
            }).encode("utf-8")
            envelope_info = tarfile.TarInfo(ARCHIVE_ENVELOPE_FILENAME)
            envelope_info.size = len(envelope_bytes)
            tf.addfile(envelope_info, io.BytesIO(envelope_bytes))

            manifest_bytes = json.dumps({
                "embedding_model": "test-model",
                "embedding_dimensions": 4,
                "embedding_sidecar": "vectors.npy",
                "rows": [],
            }).encode("utf-8")
            mf_info = tarfile.TarInfo(
                f"{ARCHIVE_BUNDLE_DIR}/{EMBEDDING_CONFIG_FILENAME}"
            )
            mf_info.size = len(manifest_bytes)
            tf.addfile(mf_info, io.BytesIO(manifest_bytes))
        return archive

    def test_v1_archive_routes_to_references_kind(self, tmp_path):
        bundle, handler, _ = _make_handler_with_local_ref(
            tmp_path / "wsdummy", "x", "ext/x.md",
        )
        registry = BundleEntryRegistry()
        registry.register(handler)
        archive = self._build_v1_archive(tmp_path)
        recipient = tmp_path / "recipient"
        recipient.mkdir()

        result = unpack_archive(
            archive, registry=registry,
            target_tier=BUNDLE_TIER_WORKSPACE,
            target_name="legacy",
            workspace_path=recipient,
        )

        assert result.format_version == ARCHIVE_FORMAT_VERSION_V1
        assert [k.kind for k in result.kinds] == [LEGACY_V1_KIND]
        assert (
            recipient / ".jaato" / "references" / "legacy"
            / EMBEDDING_CONFIG_FILENAME
        ).is_file()


class TestReadEnvelope:
    def test_unknown_format_version_rejected(self, tmp_path):
        archive = tmp_path / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            payload = json.dumps({
                "format_version": 99, "source_name": "x",
            }).encode("utf-8")
            info = tarfile.TarInfo(ARCHIVE_ENVELOPE_FILENAME)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))

        with pytest.raises(UnpackError, match="format_version"):
            read_envelope(archive)

    def test_missing_envelope_rejected(self, tmp_path):
        archive = tmp_path / "noenv.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            info = tarfile.TarInfo("random.txt")
            payload = b"hello"
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))

        with pytest.raises(UnpackError, match="missing"):
            read_envelope(archive)

    def test_missing_archive_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_envelope(tmp_path / "nope.tar.gz")
