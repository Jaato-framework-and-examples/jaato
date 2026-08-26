"""Result store — append-only JSONL, one record per arm.

JSONL because a sweep is long, arms finish out of order, and a run that
dies halfway should leave every completed arm readable.  Appending each
arm as it lands means a crashed sweep is resumable and a running one is
inspectable with ``tail -f``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from .arm import ArmResult


class ResultStore:
    """Append-only JSONL writer/reader for :class:`ArmResult` records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: ArmResult) -> None:
        """Write one record and flush.

        Flushed per record deliberately: an unflushed buffer is
        indistinguishable from an arm that never ran if the process dies,
        and the whole point of the three-valued verdict is not to confuse
        those two.
        """
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.to_dict(), sort_keys=True, default=str))
            fh.write("\n")
            fh.flush()

    def read(self) -> List[Dict[str, Any]]:
        """Load every record.  Missing file reads as empty."""
        return list(self.iter_records())

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        """Stream records, skipping a truncated trailing line.

        A sweep killed mid-write leaves a partial final line; that is a
        known, benign state and must not make the whole results file
        unreadable.
        """
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def completed_arm_ids(self) -> set:
        """Arm ids already recorded — the resume key."""
        return {r.get("arm_id") for r in self.iter_records() if r.get("arm_id")}


def canonical_hash(payload: Any) -> str:
    """Canonical-JSON sha256 of a completion payload.

    Mirrors ``orchestrator/sdk_harness.py::hash_payload`` in
    jaato-cascade-based-prototype, which uses the same canonicalisation
    for its determinism tests.  Sorting keys and eliding whitespace makes
    the hash depend on the payload's content rather than on dict ordering
    the provider happened to emit.
    """
    import hashlib

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def merge(stores: Iterable["ResultStore"]) -> List[Dict[str, Any]]:
    """Concatenate several shards' records, later shards winning on arm_id."""
    merged: Dict[str, Dict[str, Any]] = {}
    for store in stores:
        for record in store.iter_records():
            merged[record.get("arm_id", "")] = record
    return list(merged.values())
