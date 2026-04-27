"""Tests for the :class:`BundleEntryRegistry` semantics.

These tests exercise the registry against a stub handler so the
registration / lookup contract is verified independently of any real
plugin's adapter. The references-side smoke test in
``shared/plugins/references/tests/test_entry_handler.py`` covers the
real adapter; here we validate the dispatcher itself.
"""

from pathlib import Path
from typing import List, Optional

import pytest

from shared.plugins.bundle_common.bundle import (
    BUNDLE_TIER_WORKSPACE,
    Bundle,
)
from shared.plugins.bundle_common.handler import (
    BundleEntry,
    BundleEntryHandler,
    BundleEntryRegistry,
)


class _StubHandler(BundleEntryHandler):
    """Minimal handler that returns canned data for registry tests."""

    def __init__(
        self,
        kind: str,
        entries: Optional[List[BundleEntry]] = None,
        domain_subpath: Path = Path(".jaato/stub"),
    ) -> None:
        self._kind = kind
        self._entries = entries or []
        self._domain_subpath = domain_subpath
        # Trace which methods the registry routes to us.
        self.move_calls: List[str] = []
        self.delete_calls: List[str] = []
        self.reload_calls = 0

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def domain_subpath(self) -> Path:
        return self._domain_subpath

    def list_entries(self) -> List[BundleEntry]:
        return list(self._entries)

    def find_entry(self, entry_id):
        return next((e for e in self._entries if e.id == entry_id), None)

    def move_entry_to_bundle(self, entry, target_bundle):
        self.move_calls.append(f"to-bundle:{entry.id}->{target_bundle.name}")
        return entry.file_path

    def move_entry_to_free(self, entry, target_tier):
        self.move_calls.append(f"to-free:{entry.id}->{target_tier}")
        return entry.file_path

    def delete_entry(self, entry):
        self.delete_calls.append(entry.id)

    def reload_catalog(self):
        self.reload_calls += 1

    def reconcile_bundle(self, bundle):
        return None


def _entry(eid: str, kind: str, *, bundle_name: str = "") -> BundleEntry:
    return BundleEntry(
        id=eid,
        kind=kind,
        file_path=Path(f"/tmp/{kind}/{eid}.json"),
        bundle_name=bundle_name,
        bundle_tier=BUNDLE_TIER_WORKSPACE,
    )


class TestRegistration:
    def test_register_and_get(self):
        reg = BundleEntryRegistry()
        h = _StubHandler("references")
        reg.register(h)
        assert reg.get("references") is h

    def test_get_unknown_returns_none(self):
        reg = BundleEntryRegistry()
        assert reg.get("missing") is None

    def test_register_replaces_existing(self):
        """Re-registration is intentional — supports plugin restart."""
        reg = BundleEntryRegistry()
        first = _StubHandler("references")
        second = _StubHandler("references")
        reg.register(first)
        reg.register(second)
        assert reg.get("references") is second

    def test_register_rejects_empty_kind(self):
        reg = BundleEntryRegistry()
        with pytest.raises(ValueError, match="non-empty"):
            reg.register(_StubHandler(""))

    def test_unregister_idempotent(self):
        reg = BundleEntryRegistry()
        reg.register(_StubHandler("references"))
        reg.unregister("references")
        # Second unregister is a no-op.
        reg.unregister("references")
        assert reg.get("references") is None


class TestEnumeration:
    def test_kinds_are_sorted(self):
        reg = BundleEntryRegistry()
        reg.register(_StubHandler("references"))
        reg.register(_StubHandler("agents"))
        reg.register(_StubHandler("tasks"))
        assert reg.kinds() == ["agents", "references", "tasks"]

    def test_all_handlers_returned_in_kind_order(self):
        reg = BundleEntryRegistry()
        h_refs = _StubHandler("references")
        h_agents = _StubHandler("agents")
        reg.register(h_refs)
        reg.register(h_agents)
        assert reg.all_handlers() == [h_agents, h_refs]

    def test_list_all_entries_aggregates_across_kinds(self):
        reg = BundleEntryRegistry()
        reg.register(_StubHandler("references", [
            _entry("api-spec", "references"),
            _entry("changelog", "references"),
        ]))
        reg.register(_StubHandler("agents", [
            _entry("reviewer", "agents"),
        ]))

        all_entries = reg.list_all_entries()

        # Sorted by kind first (agents before references), then by
        # handler-defined order within kind.
        assert [(e.kind, e.id) for e in all_entries] == [
            ("agents", "reviewer"),
            ("references", "api-spec"),
            ("references", "changelog"),
        ]


class TestFindEntry:
    def test_find_known_entry(self):
        reg = BundleEntryRegistry()
        reg.register(_StubHandler("references", [_entry("x", "references")]))
        found = reg.find_entry("references", "x")
        assert found is not None
        assert found.id == "x"
        assert found.kind == "references"

    def test_find_unknown_kind(self):
        reg = BundleEntryRegistry()
        assert reg.find_entry("references", "x") is None

    def test_find_unknown_id_within_known_kind(self):
        reg = BundleEntryRegistry()
        reg.register(_StubHandler("references", [_entry("x", "references")]))
        assert reg.find_entry("references", "ghost") is None

    def test_same_id_under_different_kinds_resolves_separately(self):
        """`<kind>` is the disambiguator — `references:weekly` and
        `tasks:weekly` are distinct entries."""
        reg = BundleEntryRegistry()
        reg.register(_StubHandler("references", [_entry("weekly", "references")]))
        reg.register(_StubHandler("tasks", [_entry("weekly", "tasks")]))

        ref_hit = reg.find_entry("references", "weekly")
        task_hit = reg.find_entry("tasks", "weekly")

        assert ref_hit is not None and ref_hit.kind == "references"
        assert task_hit is not None and task_hit.kind == "tasks"


class TestProtocolConformance:
    """Sanity-check that BundleEntryHandler is a runtime-checkable Protocol."""

    def test_stub_passes_isinstance(self):
        h = _StubHandler("references")
        assert isinstance(h, BundleEntryHandler)

    def test_non_handler_fails_isinstance(self):
        class _NotAHandler:
            pass
        assert not isinstance(_NotAHandler(), BundleEntryHandler)
