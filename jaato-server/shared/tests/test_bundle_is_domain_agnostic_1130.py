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
* an ``embedding_config.json`` on its own marks NOTHING: there is one
  marker, and a domain's optional index descriptor is not it;
* an indexed bundle carries the two files independently, and both are
  read by the side that owns them;
* the archive path is reachable for a vectorless bundle.
"""

import json
from dataclasses import fields
from pathlib import Path
from typing import List

import pytest

from shared.plugins.bundle_common.bundle import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    ROOT_BUNDLE_NAME,
    Bundle,
    discover_bundles,
    is_bundle_directory,
    load_bundle,
    write_bundle_manifest,
)
from shared.plugins.bundle_common.handler import (
    BundleEntry,
    BundleEntryHandler,
    BundleEntryRegistry,
)
from shared.plugins.bundle_common.pack import pack_bundle
from shared.plugins.bundle_common.unpack import read_envelope, unpack_archive
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
        find='BUNDLE_MANIFEST_FILENAME = "bundle.json"',
        replace='BUNDLE_MANIFEST_FILENAME = "embedding_config.json"',
        test="TestVectorlessBundles::test_a_generic_manifest_declares_a_bundle",
        because="a manifest saying only what a distribution manifest should "
                "say -- a name and a description -- not being recognised as "
                "declaring a bundle at all",
    ),
    Reversion(
        target="jaato-server/shared/plugins/bundle_common/bundle.py",
        find=(
            "    try:\n"
            "        return (directory / BUNDLE_MANIFEST_FILENAME).is_file()\n"
            "    except OSError:\n"
            "        return False"
        ),
        replace=(
            "    try:\n"
            "        return directory.is_dir()\n"
            "    except OSError:\n"
            "        return False"
        ),
        test="TestAntiPollutionGuardSurvives::test_a_directory_without_a_manifest_is_not_a_bundle",
        because="dropping an unrelated directory into a tier root polluting "
                "the catalog -- the requirement discover_bundles' docstring "
                "calls out and that this change had to preserve",
    ),
    Reversion(
        target="jaato-server/shared/plugins/bundle_common/bundle.py",
        find=(
            "    ONE marker: ``bundle.json``.  A directory is a bundle because a\n"
            "    domain claimed it, never because of what it happens to contain --\n"
            "    which is the whole of #1130.  In particular the references plugin's\n"
            "    ``embedding_config.json`` is an index descriptor that sits beside\n"
            "    the manifest and marks nothing.\n"
            '    """\n'
            "    try:\n"
            "        return (directory / BUNDLE_MANIFEST_FILENAME).is_file()\n"
        ),
        replace=(
            "    Recognises the references-era name as a second marker.\n"
            '    """\n'
            "    try:\n"
            "        if (directory / 'embedding_config.json').is_file():\n"
            "            return True\n"
            "        return (directory / BUNDLE_MANIFEST_FILENAME).is_file()\n"
        ),
        test="TestAntiPollutionGuardSurvives::test_an_index_descriptor_alone_marks_nothing",
        because="a domain's optional index descriptor being read as a second "
                "bundle marker, which is the coupling #1130 exists to undo: "
                "the directory would be a bundle because of what it happens "
                "to contain rather than because a domain claimed it",
    ),
    Reversion(
        target="jaato-server/shared/plugins/bundle_common/unpack.py",
        find="    skip = {BUNDLE_MANIFEST_FILENAME} | set(non_entry_filenames)",
        replace="    skip = {BUNDLE_MANIFEST_FILENAME}",
        test="TestTheGenericLayerAsksRatherThanKnows"
             "::test_a_declared_non_entry_file_is_not_counted_as_an_entry",
        because="the generic layer ignoring what a domain says its own "
                "metadata files are, so every one of them is counted as an "
                "installed entry -- the count a person reads to decide "
                "whether an unpack did what they asked",
    ),
    Reversion(
        target="jaato-server/shared/plugins/references/entry_handler.py",
        find="        return REFERENCE_NON_SOURCE_FILENAMES\n",
        replace="        return ()\n",
        test="TestTheGenericLayerAsksRatherThanKnows"
             "::test_the_references_handler_answers_with_its_index",
        because="the one domain that HAS such a file not declaring it, which "
                "the Protocol's empty default makes silent: the seam exists, "
                "the layer asks, and the answer is wrong",
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


# The marker's name, spelled out rather than read from the constant.
# A guard whose fixtures write to ``BUNDLE_MANIFEST_FILENAME`` follows
# the constant wherever it is pointed, so it would keep passing with the
# name pointed back at the index descriptor -- which is one of the two
# shapes #1130 had to end.  Stating the name is what makes that
# observable.
_MARKER_NAME = "bundle.json"


def _write_generic_manifest(directory: Path, **payload) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / _MARKER_NAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )


def _write_embedding_config(directory: Path, *, rows=(), model="m", dim=4) -> None:
    """An INDEXED bundle: the manifest that marks it, plus its index.

    Two files on purpose.  ``bundle.json`` is the domain's claim on the
    directory; ``embedding_config.json`` describes a vector index that
    may or may not be there.  Independent things, separate files.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _write_generic_manifest(directory, name=directory.name)
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
        assert load_bundle(stray, name="somebody-elses-folder") is None

    def test_an_empty_tier_root_yields_nothing(self, tmp_path):
        refs = tmp_path / "refs"
        refs.mkdir()

        assert discover_bundles(refs) == []

    def test_an_index_descriptor_alone_marks_nothing(self, tmp_path):
        """There is ONE marker, and the references index is not it.

        ``embedding_config.json`` was read as a bundle marker for as
        long as it WAS the manifest.  Keeping it as a second marker
        would preserve exactly the coupling #1130 exists to undo: a
        directory would be a bundle because of what it happens to
        contain, rather than because a domain claimed it.  The generic
        layer does not know this filename at all -- the assertion below
        holds for any file a domain might drop in.
        """
        refs = tmp_path / "refs"
        indexed = refs / "teammate"
        indexed.mkdir(parents=True)
        (indexed / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
            "embedding_model": "m",
            "embedding_dimensions": 4,
            "embedding_sidecar": "vectors.npy",
            "rows": ["a"],
        }), encoding="utf-8")

        assert is_bundle_directory(indexed) is False
        assert load_bundle(indexed, name="teammate") is None
        assert discover_bundles(refs) == []
        assert discover_reference_bundles(refs) == []


class TestAnIndexedBundle:
    """A bundle that has a vector index -- two files, independently."""

    def test_its_index_still_loads_on_the_references_side(self, tmp_path):
        refs = tmp_path / "refs"
        _write_embedding_config(refs / "teammate", rows=["a", "b"], dim=8)

        bundle = discover_reference_bundles(refs)[0]

        assert bundle.has_index is True
        assert bundle.embedding_dimensions == 8
        assert bundle.embedding_rows == ["a", "b"]
        assert bundle.owned_source_ids == {"a", "b"}
        assert bundle.sidecar_path == (refs / "teammate" / "vectors.npy")

    def test_the_manifest_marks_and_the_index_describes(self, tmp_path):
        """The two files are independent, and both are read.

        This is the shape #1130 argues for: ``bundle.json`` is the
        claim, ``embedding_config.json`` is a vector index sitting
        beside it.  Neither implies the other.
        """
        refs = tmp_path / "refs"
        _write_embedding_config(refs / "teammate", rows=["a"])
        teammate = refs / "teammate"

        assert (teammate / BUNDLE_MANIFEST_FILENAME).is_file()
        assert (teammate / EMBEDDING_CONFIG_FILENAME).is_file()
        assert is_bundle_directory(teammate) is True
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

    def __init__(
        self,
        bundle: Bundle,
        entries: List[BundleEntry],
        non_entry: tuple = (),
    ) -> None:
        self._bundle = bundle
        self._entries = entries
        self._non_entry = non_entry

    def non_entry_filenames(self):
        return self._non_entry

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


class TestTheGenericLayerAsksRatherThanKnows:
    """A domain's metadata is named by the domain, and by nobody else.

    ``bundle.json`` is the generic layer's own file, so it skips that
    one on its own authority.  Anything else a domain keeps beside it
    -- the references plugin's ``embedding_config.json`` -- is a name
    the layer must not hold, which is the same rule that moved the
    seven embedding fields off :class:`Bundle`.  So it asks
    :meth:`BundleEntryHandler.non_entry_filenames`.
    """

    def _pack_with(self, tmp_path: Path, non_entry: tuple) -> Path:
        bundle_dir = tmp_path / ".jaato" / "definitions" / "my-pack"
        _write_generic_manifest(bundle_dir, name="my-pack")
        _write_definition(bundle_dir, "plain-ref")
        # A domain metadata file that is *.json and is not an entry.
        (bundle_dir / "domain_index.json").write_text(
            json.dumps({"rows": []}), encoding="utf-8",
        )
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
        handler = _DefinitionsOnlyHandler(bundle, [entry], non_entry)
        archive = tmp_path / "my-pack.tar.gz"
        pack_bundle(handler, bundle, archive)
        return archive, handler

    def test_a_declared_non_entry_file_is_not_counted_as_an_entry(
        self, tmp_path,
    ):
        archive, handler = self._pack_with(tmp_path, ("domain_index.json",))
        registry = BundleEntryRegistry()
        registry.register(handler)
        recipient = tmp_path / "recipient"
        recipient.mkdir()

        result = unpack_archive(
            archive, registry=registry,
            target_tier=BUNDLE_TIER_WORKSPACE,
            target_name="my-pack",
            workspace_path=recipient,
        )

        installed = recipient / ".jaato" / "definitions" / "my-pack"
        # The file is INSTALLED -- it is the domain's metadata, not
        # something to drop -- and it is not one of the entries.
        assert (installed / "domain_index.json").is_file()
        assert [k.entry_count for k in result.kinds] == [1]

    def test_without_the_declaration_it_is_counted(self, tmp_path):
        """The control: the same archive, the same file, no declaration.

        Without it the layer would have to know the name itself, and
        the count is wrong by exactly the metadata files the domain
        keeps -- which is what makes this a seam rather than a
        formality.
        """
        archive, handler = self._pack_with(tmp_path, ())
        registry = BundleEntryRegistry()
        registry.register(handler)
        recipient = tmp_path / "recipient"
        recipient.mkdir()

        result = unpack_archive(
            archive, registry=registry,
            target_tier=BUNDLE_TIER_WORKSPACE,
            target_name="my-pack",
            workspace_path=recipient,
        )

        assert [k.entry_count for k in result.kinds] == [2]

    def test_the_references_handler_answers_with_its_index(self):
        """And the domain that has such a file answers with it.

        Called unbound: the answer is a property of the DOMAIN, not of
        any particular catalog, so it must not need a loaded plugin to
        be obtained.  The default the Protocol supplies is ``()``, so
        a handler that simply never overrode this would look correct
        and have its index counted as a reference.
        """
        from shared.plugins.references.entry_handler import (
            ReferencesEntryHandler,
        )

        declared = ReferencesEntryHandler.non_entry_filenames(None)

        assert EMBEDDING_CONFIG_FILENAME in declared
        assert BUNDLE_MANIFEST_FILENAME in declared
