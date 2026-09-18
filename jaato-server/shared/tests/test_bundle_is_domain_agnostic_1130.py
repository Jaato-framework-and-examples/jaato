"""A bundle is a directory a domain claims — not a references vector index.

Regression tests for #1130.  ``bundle_common/handler.py`` states the
principle the layer is built on: the bundle commands "don't know
anything about references, agents, tasks, profiles, or services on
their own".  :class:`Bundle` was the one thing that reached through
that seam — of its ten fields, seven belonged to the references
domain, including ``matcher``, a LIVE semantic-matcher instance typed
``Any`` because the generic module could not name its type.

Three of those were REQUIRED, so the type could not represent a bundle
without a vector index and the whole chain below it inherited that:

    Bundle requires embedding_model/dimensions/sidecar
      -> the loader enforced them
        -> discover_bundles returned nothing for a vectorless directory
          -> pack_bundle(handler, bundle: Bundle) was unreachable for it

The index moved to
:class:`shared.plugins.references.bundle.ReferenceBundle`.  What must
survive the move, and is what these tests hold:

* a directory that declares a bundle and no vector index is discovered
  (the issue's two reproductions);
* a directory with NO manifest is still ignored entirely — the
  anti-pollution guard the docstring promises;
* a bundle already on disk, carrying only the references-era
  ``embedding_config.json``, keeps loading;
* the archive path is reachable for a vectorless bundle.
"""

import json
from dataclasses import fields
from pathlib import Path
from typing import List

import pytest

from shared.plugins.bundle_common.bundle import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_MARKER_FILENAMES,
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    ROOT_BUNDLE_NAME,
    Bundle,
    bundle_marker_path,
    discover_bundles,
    is_bundle_directory,
    load_bundle,
    write_bundle_manifest,
)
from shared.plugins.bundle_common.handler import (
    BundleEntry,
    BundleEntryHandler,
)
from shared.plugins.bundle_common.pack import pack_bundle
from shared.plugins.bundle_common.unpack import read_envelope
from shared.plugins.references.bundle import (
    EMBEDDING_CONFIG_FILENAME,
    ReferenceBundle,
    discover_bundles as discover_reference_bundles,
)
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


REVERSIONS = [
    Reversion(
        target="jaato-server/shared/plugins/bundle_common/bundle.py",
        find=(
            "    return Bundle(\n"
            "        name=name,\n"
            "        directory=directory.resolve(),\n"
            "        tier=tier,\n"
            "    )\n"
        ),
        replace=(
            "    _raw = json.loads(marker.read_text(encoding='utf-8'))\n"
            "    if not (_raw.get('embedding_model')\n"
            "            and _raw.get('embedding_dimensions')\n"
            "            and _raw.get('embedding_sidecar')):\n"
            "        return None\n"
            "    return Bundle(\n"
            "        name=name,\n"
            "        directory=directory.resolve(),\n"
            "        tier=tier,\n"
            "    )\n"
        ),
        test="TestVectorlessBundles::test_a_definitions_only_sibling_is_discovered",
        because="the loader enforcing the three embedding fields, which is "
                "what made a references directory with definitions and no "
                "sidecar invisible to the catalog with no error and no "
                "warning",
    ),
    Reversion(
        target="jaato-server/shared/plugins/bundle_common/bundle.py",
        find=(
            "BUNDLE_MARKER_FILENAMES: Tuple[str, ...] = (\n"
            "    (BUNDLE_MANIFEST_FILENAME,) + LEGACY_BUNDLE_MARKER_FILENAMES\n"
            ")"
        ),
        replace=(
            "BUNDLE_MARKER_FILENAMES: Tuple[str, ...] = "
            "LEGACY_BUNDLE_MARKER_FILENAMES"
        ),
        test="TestVectorlessBundles::test_a_generic_manifest_declares_a_bundle",
        because="a manifest saying only what a distribution manifest should "
                "say -- a name and a description -- not being recognised as "
                "declaring a bundle at all",
    ),
    Reversion(
        target="jaato-server/shared/plugins/bundle_common/bundle.py",
        find=(
            "    for filename in BUNDLE_MARKER_FILENAMES:\n"
            "        candidate = directory / filename\n"
        ),
        replace=(
            "    if directory.is_dir():\n"
            "        return directory\n"
            "    for filename in BUNDLE_MARKER_FILENAMES:\n"
            "        candidate = directory / filename\n"
        ),
        test="TestAntiPollutionGuardSurvives::test_a_directory_without_a_manifest_is_not_a_bundle",
        because="dropping an unrelated directory into a tier root polluting "
                "the catalog -- the requirement discover_bundles' docstring "
                "calls out and that this change had to preserve",
    ),
]


# The seven fields that used to sit on the generic dataclass and now
# belong to the references domain.  Named here rather than derived so
# the guard states the contract instead of restating the code.
_DOMAIN_FIELD_NAMES = frozenset({
    "embedding_model",
    "embedding_dimensions",
    "embedding_sidecar",
    "embedding_rows",
    "reconcile_mode",
    "owned_source_ids",
    "matcher",
})

_BUNDLE_COMMON = Path(__file__).resolve().parents[2] / "shared" / "plugins" / "bundle_common"


def _write_generic_manifest(directory: Path, **payload) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / BUNDLE_MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )


def _write_embedding_config(directory: Path, *, rows=(), model="m", dim=4) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
        "embedding_model": model,
        "embedding_dimensions": dim,
        "embedding_sidecar": "vectors.npy",
        "rows": list(rows),
    }), encoding="utf-8")


def _write_definition(directory: Path, sid: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{sid}.json").write_text(json.dumps({
        "id": sid,
        "name": sid,
        "description": f"Desc for {sid}",
        "type": "local",
        "path": f"./dummy/{sid}.md",
        "mode": "selectable",
    }), encoding="utf-8")


class TestTheGenericTypeHoldsNoDomainFields:
    """The seam the issue reports on: ``Bundle``'s own shape."""

    def test_bundle_declares_exactly_three_generic_fields(self):
        assert {f.name for f in fields(Bundle)} == {"name", "directory", "tier"}

    def test_no_domain_field_survives_on_the_generic_type(self):
        present = _DOMAIN_FIELD_NAMES & {f.name for f in fields(Bundle)}
        assert present == set(), (
            f"{sorted(present)} belong to a domain, not to a directory a "
            f"domain claims"
        )

    def test_the_generic_package_names_no_domain_field_anywhere(self):
        """No module under ``bundle_common`` reads or writes one.

        A field can be removed from the dataclass and still be reached
        through ``getattr`` or an annotation elsewhere in the package,
        which would put the coupling back one indirection out.  This
        scans the package's own source rather than the type.
        """
        import ast

        offenders = []
        for path in sorted(_BUNDLE_COMMON.rglob("*.py")):
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                name = None
                if isinstance(node, ast.Attribute):
                    name = node.attr
                elif isinstance(node, ast.Name):
                    name = node.id
                elif isinstance(node, ast.keyword):
                    name = node.arg
                if name in _DOMAIN_FIELD_NAMES:
                    offenders.append(f"{path.name}:{node.lineno} {name}")
        assert offenders == [], offenders

    def test_the_references_type_is_a_bundle_that_carries_them(self):
        names = {f.name for f in fields(ReferenceBundle)}
        assert _DOMAIN_FIELD_NAMES <= names
        assert issubclass(ReferenceBundle, Bundle)


class TestVectorlessBundles:
    """The issue's two reproductions."""

    def test_a_generic_manifest_declares_a_bundle(self, tmp_path):
        """Reproduction 2 — what an author writing a distribution
        manifest would put in it.

        Before the split this logged "manifest missing required fields
        (embedding_model, embedding_dimensions, embedding_sidecar)" and
        discovered nothing, so an author following the message added
        embeddings they never wanted.
        """
        refs = tmp_path / "refs"
        _write_generic_manifest(
            refs / "my-pack",
            name="my-pack",
            description="a shippable set of references",
        )

        bundles = discover_bundles(refs)

        assert [b.name for b in bundles] == ["my-pack"]
        assert bundles[0].tier == BUNDLE_TIER_WORKSPACE

    def test_a_definitions_only_sibling_is_discovered(self, tmp_path):
        """Reproduction 1 — two sibling bundles, one with vectors.

        ``no-vectors`` used to contribute nothing: no error, no
        warning, its sources simply never reached the catalog.
        """
        refs = tmp_path / "refs"
        _write_generic_manifest(refs / "no-vectors", name="no-vectors")
        _write_definition(refs / "no-vectors", "plain-ref")
        _write_generic_manifest(refs / "with-vectors", name="with-vectors")
        _write_embedding_config(refs / "with-vectors", rows=["vec-ref"])
        _write_definition(refs / "with-vectors", "vec-ref")

        discovered = sorted(b.name for b in discover_bundles(refs))

        assert discovered == ["no-vectors", "with-vectors"]

    def test_the_references_domain_reports_which_one_has_an_index(self, tmp_path):
        refs = tmp_path / "refs"
        _write_generic_manifest(refs / "no-vectors", name="no-vectors")
        _write_generic_manifest(refs / "with-vectors", name="with-vectors")
        _write_embedding_config(refs / "with-vectors", rows=["vec-ref"])

        by_name = {b.name: b for b in discover_reference_bundles(refs)}

        assert by_name["no-vectors"].has_index is False
        assert by_name["no-vectors"].embedding_rows == []
        assert by_name["no-vectors"].sidecar_path is None
        assert by_name["no-vectors"].lock_path is None
        assert by_name["with-vectors"].has_index is True
        assert by_name["with-vectors"].embedding_rows == ["vec-ref"]
        assert by_name["with-vectors"].embedding_model == "m"

    def test_a_root_bundle_may_also_be_vectorless(self, tmp_path):
        refs = tmp_path / "refs"
        _write_generic_manifest(refs, name="root-pack")

        bundles = discover_bundles(refs)

        assert [b.name for b in bundles] == [ROOT_BUNDLE_NAME]
        assert bundles[0].display_name == "(root)"

    def test_write_bundle_manifest_declares_one(self, tmp_path):
        target = tmp_path / "refs" / "fresh"

        written = write_bundle_manifest(
            target, name="fresh", description="declared, not vectorised",
        )

        assert written.name == BUNDLE_MANIFEST_FILENAME
        assert is_bundle_directory(target)
        assert json.loads(written.read_text()) == {
            "name": "fresh", "description": "declared, not vectorised",
        }


class TestAntiPollutionGuardSurvives:
    """"Only the manifest's schema changes, not the need for one." """

    def test_a_directory_without_a_manifest_is_not_a_bundle(self, tmp_path):
        refs = tmp_path / "refs"
        stray = refs / "somebody-elses-folder"
        stray.mkdir(parents=True)
        (stray / "notes.md").write_text("not ours")
        _write_definition(stray, "looks-like-a-ref")

        assert discover_bundles(refs) == []
        assert is_bundle_directory(stray) is False
        assert bundle_marker_path(stray) is None
        assert load_bundle(stray, name="somebody-elses-folder") is None

    def test_an_empty_tier_root_yields_nothing(self, tmp_path):
        refs = tmp_path / "refs"
        refs.mkdir()

        assert discover_bundles(refs) == []


class TestBundlesAlreadyOnDisk:
    """"Every manifest already carries the embedding fields, so loading
    is unchanged for everything already produced." """

    def test_a_legacy_embedding_config_still_marks_a_bundle(self, tmp_path):
        refs = tmp_path / "refs"
        _write_embedding_config(refs / "teammate", rows=["a", "b"])

        bundles = discover_bundles(refs)

        assert [b.name for b in bundles] == ["teammate"]
        assert bundle_marker_path(refs / "teammate").name == (
            EMBEDDING_CONFIG_FILENAME
        )

    def test_its_index_still_loads_on_the_references_side(self, tmp_path):
        refs = tmp_path / "refs"
        _write_embedding_config(refs / "teammate", rows=["a", "b"], dim=8)

        bundle = discover_reference_bundles(refs)[0]

        assert bundle.has_index is True
        assert bundle.embedding_dimensions == 8
        assert bundle.embedding_rows == ["a", "b"]
        assert bundle.owned_source_ids == {"a", "b"}
        assert bundle.sidecar_path == (refs / "teammate" / "vectors.npy")

    def test_the_canonical_manifest_wins_when_a_bundle_carries_both(self, tmp_path):
        refs = tmp_path / "refs"
        _write_generic_manifest(refs / "teammate", name="teammate")
        _write_embedding_config(refs / "teammate", rows=["a"])

        marker = bundle_marker_path(refs / "teammate")

        assert marker.name == BUNDLE_MANIFEST_FILENAME
        assert BUNDLE_MARKER_FILENAMES[0] == BUNDLE_MANIFEST_FILENAME
        # ...and the index is still read.
        assert discover_reference_bundles(refs)[0].embedding_rows == ["a"]

    def test_workspace_still_shadows_user(self, tmp_path):
        ws = tmp_path / "ws" / ".jaato" / "references"
        home = tmp_path / "home" / ".jaato" / "references"
        _write_generic_manifest(ws / "teammate", name="teammate")
        _write_embedding_config(home / "teammate", rows=["u"])

        bundles = discover_bundles([
            (ws, BUNDLE_TIER_WORKSPACE), (home, BUNDLE_TIER_USER),
        ])

        assert [(b.name, b.tier) for b in bundles] == [
            ("teammate", BUNDLE_TIER_WORKSPACE),
        ]


class _DefinitionsOnlyHandler(BundleEntryHandler):
    """A domain with no external payloads — the shape ``pack.py``'s own
    docstring says gets a clean archive for free, and that could not
    reach the packer while its entry point took a type that could not
    express a vectorless bundle."""

    def __init__(self, bundle: Bundle, entries: List[BundleEntry]) -> None:
        self._bundle = bundle
        self._entries = entries

    @property
    def kind(self) -> str:
        return "definitions"

    @property
    def domain_subpath(self) -> Path:
        return Path(".jaato/definitions")

    def list_entries(self) -> List[BundleEntry]:
        return list(self._entries)

    def list_bundles(self) -> List[Bundle]:
        return [self._bundle]

    def find_entry(self, entry_id):
        return next((e for e in self._entries if e.id == entry_id), None)

    def move_entry_to_bundle(self, entry, target_bundle):
        return entry.file_path

    def move_entry_to_free(self, entry, target_tier):
        return entry.file_path

    def delete_entry(self, entry) -> None:
        pass

    def reload_catalog(self) -> None:
        pass

    def reconcile_bundle(self, bundle):
        return None


class TestTheArchivePathIsReachable:
    """The end of the consequence chain: "a references set with no
    embeddings cannot be packed at all"."""

    def test_a_vectorless_bundle_packs(self, tmp_path):
        bundle_dir = tmp_path / ".jaato" / "definitions" / "my-pack"
        _write_generic_manifest(
            bundle_dir, name="my-pack", description="no vectors here",
        )
        _write_definition(bundle_dir, "plain-ref")
        bundle = Bundle(
            name="my-pack",
            directory=bundle_dir.resolve(),
            tier=BUNDLE_TIER_WORKSPACE,
        )
        entry = BundleEntry(
            id="plain-ref",
            kind="definitions",
            file_path=bundle_dir / "plain-ref.json",
            bundle_name="my-pack",
            bundle_tier=BUNDLE_TIER_WORKSPACE,
        )
        handler = _DefinitionsOnlyHandler(bundle, [entry])

        archive = tmp_path / "my-pack.tar.gz"
        result = pack_bundle(handler, bundle, archive)

        assert archive.is_file()
        assert [k.kind for k in result.kinds] == ["definitions"]
        assert read_envelope(archive)["source_name"] == "my-pack"
