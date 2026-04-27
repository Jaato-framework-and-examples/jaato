"""Tests for the Bundle abstraction: discovery, drift, and manifest I/O."""

import json
from pathlib import Path

import pytest

from shared.plugins.references.bundle import (
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    EMBEDDING_CONFIG_FILENAME,
    ROOT_BUNDLE_NAME,
    Bundle,
    detect_drift,
    discover_bundles,
    metadata_hash,
    resolve_bundle_roots,
    write_manifest,
)
from shared.plugins.references.models import (
    EmbeddingMetadata,
    InjectionMode,
    ReferenceSource,
    SourceType,
)


def _manifest(path: Path, *, rows, model="all-MiniLM-L6-v2", dim=384, sidecar="references.embeddings.npy", extra=None):
    payload = {
        "embedding_model": model,
        "embedding_dimensions": dim,
        "embedding_sidecar": sidecar,
        "rows": rows,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source(sid, *, name=None, tags=None, fetch_hint=None, hash_=None, bundle_name=""):
    src = ReferenceSource(
        id=sid,
        name=name or sid.replace("-", " ").title(),
        description=f"Description for {sid}",
        type=SourceType.LOCAL,
        mode=InjectionMode.SELECTABLE,
        path=f"/tmp/{sid}",
        tags=list(tags) if tags else [],
        fetch_hint=fetch_hint,
        bundle_name=bundle_name,
    )
    if hash_ is not None:
        src.embedding = EmbeddingMetadata(source_hash=hash_)
    return src


class TestMetadataHash:
    """The source_hash fingerprint is stable and covers the embedded fields."""

    def test_deterministic(self):
        s1 = _source("r1", name="R1", tags=["api", "auth"], fetch_hint="Read first")
        s2 = _source("r1", name="R1", tags=["api", "auth"], fetch_hint="Read first")
        assert metadata_hash(s1) == metadata_hash(s2)

    def test_tag_order_does_not_matter(self):
        s1 = _source("r1", tags=["api", "auth"])
        s2 = _source("r1", tags=["auth", "api"])
        assert metadata_hash(s1) == metadata_hash(s2)

    def test_name_change_flips_hash(self):
        base = _source("r1", name="Original")
        changed = _source("r1", name="Renamed")
        assert metadata_hash(base) != metadata_hash(changed)

    def test_description_change_flips_hash(self):
        base = _source("r1")
        changed = _source("r1")
        changed.description = "Something else entirely"
        assert metadata_hash(base) != metadata_hash(changed)

    def test_fetch_hint_none_and_empty_are_equivalent(self):
        """Missing fetch_hint and empty string both contribute nothing."""
        s1 = _source("r1", fetch_hint=None)
        s2 = _source("r1", fetch_hint="")
        assert metadata_hash(s1) == metadata_hash(s2)

    def test_format(self):
        s1 = _source("r1")
        h = metadata_hash(s1)
        assert h.startswith("sha256:")
        assert len(h) == len("sha256:") + 64


class TestDetectDrift:
    """detect_drift correctly categorizes missing/stale/orphan."""

    def _bundle(self, rows):
        return Bundle(
            name="",
            directory=Path("/tmp/fake"),
            embedding_model="m",
            embedding_dimensions=4,
            embedding_sidecar="x.npy",
            embedding_rows=list(rows),
            owned_source_ids=set(rows),
        )

    def test_clean_bundle(self):
        src = _source("a")
        src.embedding = EmbeddingMetadata(source_hash=metadata_hash(src))
        bundle = self._bundle(["a"])

        drift = detect_drift(bundle, [src])

        assert drift.is_clean()
        assert drift.summary() == "up-to-date"

    def test_missing_when_no_embedding_block(self):
        src = _source("a")
        bundle = self._bundle([])

        drift = detect_drift(bundle, [src])

        assert drift.missing == ["a"]
        assert drift.stale == []
        assert drift.orphan == []

    def test_stale_when_hash_differs(self):
        src = _source("a", hash_="sha256:old-value")
        bundle = self._bundle(["a"])

        drift = detect_drift(bundle, [src])

        assert drift.stale == ["a"]
        assert drift.missing == []

    def test_orphan_when_row_has_no_source(self):
        src = _source("a")
        src.embedding = EmbeddingMetadata(source_hash=metadata_hash(src))
        bundle = self._bundle(["a", "ghost"])

        drift = detect_drift(bundle, [src])

        assert drift.orphan == ["ghost"]

    def test_only_counts_own_sources(self):
        """A source in another bundle does not pollute the drift report."""
        own = _source("a", hash_="sha256:old", bundle_name="")
        other = _source("b", bundle_name="teammate")
        bundle = self._bundle(["a"])

        drift = detect_drift(bundle, [own, other])

        # 'b' lives in 'teammate' bundle, not in 'bundle' (root).
        # It should not appear as missing here.
        assert "b" not in drift.missing
        assert drift.stale == ["a"]


class TestDiscoverBundles:
    """Bundle discovery walks the references directory correctly."""

    def test_empty_directory_yields_no_bundles(self, tmp_path):
        refs = tmp_path / "references"
        refs.mkdir()
        assert discover_bundles(refs) == []

    def test_missing_directory_yields_no_bundles(self, tmp_path):
        assert discover_bundles(tmp_path / "nope") == []

    def test_root_bundle_only(self, tmp_path):
        refs = tmp_path / "references"
        refs.mkdir()
        _manifest(refs / EMBEDDING_CONFIG_FILENAME, rows=["a"])

        bundles = discover_bundles(refs)

        assert len(bundles) == 1
        assert bundles[0].name == ROOT_BUNDLE_NAME
        assert bundles[0].display_name == "(root)"
        assert bundles[0].embedding_rows == ["a"]
        assert bundles[0].owned_source_ids == {"a"}

    def test_sub_bundle_without_root(self, tmp_path):
        refs = tmp_path / "references"
        refs.mkdir()
        sub = refs / "teammate"
        sub.mkdir()
        _manifest(sub / EMBEDDING_CONFIG_FILENAME, rows=["x", "y"])

        bundles = discover_bundles(refs)

        assert [b.name for b in bundles] == ["teammate"]
        assert bundles[0].directory == sub.resolve()

    def test_root_is_listed_before_sub_bundles(self, tmp_path):
        refs = tmp_path / "references"
        refs.mkdir()
        _manifest(refs / EMBEDDING_CONFIG_FILENAME, rows=["r1"])
        (refs / "beta").mkdir()
        _manifest(refs / "beta" / EMBEDDING_CONFIG_FILENAME, rows=["b1"])
        (refs / "alpha").mkdir()
        _manifest(refs / "alpha" / EMBEDDING_CONFIG_FILENAME, rows=["a1"])

        bundles = discover_bundles(refs)

        assert [b.name for b in bundles] == [ROOT_BUNDLE_NAME, "alpha", "beta"]

    def test_subdir_without_manifest_is_ignored(self, tmp_path):
        refs = tmp_path / "references"
        refs.mkdir()
        (refs / "no-manifest").mkdir()
        (refs / "no-manifest" / "somefile.json").write_text("{}")

        bundles = discover_bundles(refs)

        assert bundles == []

    def test_malformed_manifest_is_skipped_not_fatal(self, tmp_path):
        """A broken sub-bundle manifest must not prevent the rest from loading."""
        refs = tmp_path / "references"
        refs.mkdir()
        _manifest(refs / EMBEDDING_CONFIG_FILENAME, rows=["r1"])
        bad = refs / "broken"
        bad.mkdir()
        (bad / EMBEDDING_CONFIG_FILENAME).write_text("not json {{{")
        good = refs / "good"
        good.mkdir()
        _manifest(good / EMBEDDING_CONFIG_FILENAME, rows=["g1"])

        bundles = discover_bundles(refs)

        assert [b.name for b in bundles] == [ROOT_BUNDLE_NAME, "good"]

    def test_manifest_missing_rows_is_skipped(self, tmp_path):
        refs = tmp_path / "references"
        refs.mkdir()
        (refs / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
            "embedding_model": "m",
            "embedding_dimensions": 4,
            "embedding_sidecar": "x.npy",
        }))

        assert discover_bundles(refs) == []

    def test_reconcile_mode_default_and_override(self, tmp_path):
        refs = tmp_path / "references"
        refs.mkdir()
        _manifest(refs / EMBEDDING_CONFIG_FILENAME, rows=[])
        (refs / "lazy-bundle").mkdir()
        _manifest(
            refs / "lazy-bundle" / EMBEDDING_CONFIG_FILENAME,
            rows=[],
            extra={"reconcile": "lazy"},
        )

        bundles = {b.name: b for b in discover_bundles(refs)}

        assert bundles[ROOT_BUNDLE_NAME].reconcile_mode == "eager"
        assert bundles["lazy-bundle"].reconcile_mode == "lazy"

    def test_reconcile_mode_unknown_falls_back_to_eager(self, tmp_path):
        refs = tmp_path / "references"
        refs.mkdir()
        _manifest(
            refs / EMBEDDING_CONFIG_FILENAME,
            rows=[],
            extra={"reconcile": "something-else"},
        )

        bundles = discover_bundles(refs)

        assert bundles[0].reconcile_mode == "eager"


class TestResolveBundleRoots:
    """resolve_bundle_roots returns workspace-then-user in deterministic order."""

    def test_workspace_first_then_user(self, tmp_path):
        ws = tmp_path / "ws"
        home = tmp_path / "home"
        roots = resolve_bundle_roots(str(ws), user_home=home)

        assert roots == [
            (ws.resolve() / ".jaato" / "references", BUNDLE_TIER_WORKSPACE),
            (home / ".jaato" / "references", BUNDLE_TIER_USER),
        ]

    def test_workspace_omitted_when_unknown(self, tmp_path):
        """A None workspace yields user-tier-only roots — no synthetic ws root."""
        home = tmp_path / "home"
        roots = resolve_bundle_roots(None, user_home=home)

        assert roots == [(home / ".jaato" / "references", BUNDLE_TIER_USER)]

    def test_workspace_path_resolved(self, tmp_path):
        """Relative ``..`` segments in the workspace path are normalized."""
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "sub").mkdir()
        # Pass a path with a trailing ``sub/..`` to force resolve() to collapse it.
        roots = resolve_bundle_roots(str(ws / "sub" / ".."), user_home=tmp_path / "home")
        assert roots[0][0] == ws.resolve() / ".jaato" / "references"


class TestDiscoverBundlesMultiTier:
    """discover_bundles walks multiple roots with workspace-shadows-user precedence."""

    def test_legacy_single_path_arg_still_works(self, tmp_path):
        """A bare ``Path`` argument keeps the pre-tiering API working."""
        refs = tmp_path / ".jaato" / "references"
        refs.mkdir(parents=True)
        _manifest(refs / EMBEDDING_CONFIG_FILENAME, rows=["a"])

        bundles = discover_bundles(refs)

        assert len(bundles) == 1
        assert bundles[0].tier == BUNDLE_TIER_WORKSPACE

    def test_each_bundle_tagged_with_its_tier(self, tmp_path):
        ws_refs = tmp_path / "ws" / ".jaato" / "references"
        ws_refs.mkdir(parents=True)
        _manifest(ws_refs / EMBEDDING_CONFIG_FILENAME, rows=["w1"])
        (ws_refs / "ws-only").mkdir()
        _manifest(ws_refs / "ws-only" / EMBEDDING_CONFIG_FILENAME, rows=["w2"])

        user_refs = tmp_path / "home" / ".jaato" / "references"
        user_refs.mkdir(parents=True)
        (user_refs / "personal").mkdir()
        _manifest(user_refs / "personal" / EMBEDDING_CONFIG_FILENAME, rows=["u1"])

        bundles = discover_bundles([
            (ws_refs, BUNDLE_TIER_WORKSPACE),
            (user_refs, BUNDLE_TIER_USER),
        ])

        by_name = {b.name: b for b in bundles}
        assert by_name[ROOT_BUNDLE_NAME].tier == BUNDLE_TIER_WORKSPACE
        assert by_name["ws-only"].tier == BUNDLE_TIER_WORKSPACE
        assert by_name["personal"].tier == BUNDLE_TIER_USER

    def test_workspace_root_shadows_user_root(self, tmp_path):
        """Same name (root) in both tiers — workspace wins, user invisible."""
        ws_refs = tmp_path / "ws" / ".jaato" / "references"
        ws_refs.mkdir(parents=True)
        _manifest(ws_refs / EMBEDDING_CONFIG_FILENAME, rows=["ws-row"])

        user_refs = tmp_path / "home" / ".jaato" / "references"
        user_refs.mkdir(parents=True)
        _manifest(user_refs / EMBEDDING_CONFIG_FILENAME, rows=["user-row"])

        bundles = discover_bundles([
            (ws_refs, BUNDLE_TIER_WORKSPACE),
            (user_refs, BUNDLE_TIER_USER),
        ])

        assert len(bundles) == 1
        assert bundles[0].embedding_rows == ["ws-row"]
        assert bundles[0].tier == BUNDLE_TIER_WORKSPACE

    def test_workspace_named_bundle_shadows_user_named_bundle(self, tmp_path):
        """A workspace ``teammate`` hides a user ``teammate`` entirely."""
        ws_refs = tmp_path / "ws" / ".jaato" / "references"
        (ws_refs / "teammate").mkdir(parents=True)
        _manifest(ws_refs / "teammate" / EMBEDDING_CONFIG_FILENAME, rows=["w"])

        user_refs = tmp_path / "home" / ".jaato" / "references"
        (user_refs / "teammate").mkdir(parents=True)
        _manifest(user_refs / "teammate" / EMBEDDING_CONFIG_FILENAME, rows=["u"])

        bundles = discover_bundles([
            (ws_refs, BUNDLE_TIER_WORKSPACE),
            (user_refs, BUNDLE_TIER_USER),
        ])

        assert len(bundles) == 1
        assert bundles[0].embedding_rows == ["w"]
        assert bundles[0].tier == BUNDLE_TIER_WORKSPACE

    def test_user_bundle_visible_when_no_workspace_collision(self, tmp_path):
        """Distinct bundle names from each tier coexist; only collisions shadow."""
        ws_refs = tmp_path / "ws" / ".jaato" / "references"
        (ws_refs / "project").mkdir(parents=True)
        _manifest(ws_refs / "project" / EMBEDDING_CONFIG_FILENAME, rows=["p"])

        user_refs = tmp_path / "home" / ".jaato" / "references"
        (user_refs / "personal").mkdir(parents=True)
        _manifest(user_refs / "personal" / EMBEDDING_CONFIG_FILENAME, rows=["x"])

        bundles = discover_bundles([
            (ws_refs, BUNDLE_TIER_WORKSPACE),
            (user_refs, BUNDLE_TIER_USER),
        ])

        names = sorted((b.name, b.tier) for b in bundles)
        assert names == [
            ("personal", BUNDLE_TIER_USER),
            ("project", BUNDLE_TIER_WORKSPACE),
        ]

    def test_missing_root_is_skipped_silently(self, tmp_path):
        """Either tier root may not exist on disk — discovery returns the rest."""
        user_refs = tmp_path / "home" / ".jaato" / "references"
        (user_refs / "personal").mkdir(parents=True)
        _manifest(user_refs / "personal" / EMBEDDING_CONFIG_FILENAME, rows=["x"])

        bundles = discover_bundles([
            (tmp_path / "ws-does-not-exist" / ".jaato" / "references", BUNDLE_TIER_WORKSPACE),
            (user_refs, BUNDLE_TIER_USER),
        ])

        assert len(bundles) == 1
        assert bundles[0].name == "personal"
        assert bundles[0].tier == BUNDLE_TIER_USER

    def test_qualified_ref_format(self, tmp_path):
        """qualified_ref follows ``<tier>:<display_name>``; root renders as (root)."""
        refs = tmp_path / ".jaato" / "references"
        refs.mkdir(parents=True)
        _manifest(refs / EMBEDDING_CONFIG_FILENAME, rows=[])
        (refs / "teammate").mkdir()
        _manifest(refs / "teammate" / EMBEDDING_CONFIG_FILENAME, rows=[])

        bundles = discover_bundles([(refs, BUNDLE_TIER_WORKSPACE)])
        by_name = {b.name: b for b in bundles}

        assert by_name[ROOT_BUNDLE_NAME].qualified_ref == "workspace:(root)"
        assert by_name["teammate"].qualified_ref == "workspace:teammate"


class TestWriteManifest:
    """write_manifest performs an atomic write and updates the Bundle in-memory."""

    def test_writes_rows_and_updates_bundle(self, tmp_path):
        refs = tmp_path / "references"
        refs.mkdir()
        _manifest(refs / EMBEDDING_CONFIG_FILENAME, rows=["old"])
        bundle = discover_bundles(refs)[0]

        write_manifest(bundle, rows=["new-a", "new-b"])

        raw = json.loads((refs / EMBEDDING_CONFIG_FILENAME).read_text())
        assert raw["rows"] == ["new-a", "new-b"]
        assert bundle.embedding_rows == ["new-a", "new-b"]
        assert bundle.owned_source_ids == {"new-a", "new-b"}

    def test_includes_reconcile_when_not_eager(self, tmp_path):
        refs = tmp_path / "references"
        refs.mkdir()
        _manifest(
            refs / EMBEDDING_CONFIG_FILENAME,
            rows=[],
            extra={"reconcile": "lazy"},
        )
        bundle = discover_bundles(refs)[0]

        write_manifest(bundle, rows=["r"])

        raw = json.loads((refs / EMBEDDING_CONFIG_FILENAME).read_text())
        assert raw["reconcile"] == "lazy"

    def test_omits_reconcile_when_eager(self, tmp_path):
        """Default mode doesn't clutter the manifest."""
        refs = tmp_path / "references"
        refs.mkdir()
        _manifest(refs / EMBEDDING_CONFIG_FILENAME, rows=[])
        bundle = discover_bundles(refs)[0]

        write_manifest(bundle, rows=["r"])

        raw = json.loads((refs / EMBEDDING_CONFIG_FILENAME).read_text())
        assert "reconcile" not in raw
