#!/usr/bin/env python3
"""Codegen pipeline: ``jaato_sdk.events`` (pydantic) → JSON Schema → TS.

Phase 2 of the SDK feature parity workstream — see
``project_backlog_sdk_feature_parity.md``.  Single source of truth
for wire-protocol shapes is ``jaato-sdk/jaato_sdk/events.py``;
this script regenerates ``jaato-sdk-ts/src/events.ts`` from the
pydantic model definitions so the TS SDK never drifts.

Pipeline:

1. Import every Event subclass listed in ``_EVENT_CLASSES``.
2. Build a unified JSON Schema using pydantic's ``TypeAdapter``
   over a ``Union`` of all event classes — this nests every event
   under ``$defs`` and produces a top-level ``anyOf`` discriminator
   on the ``type`` field.
3. Write the schema to a temp file, invoke ``npx
   --yes json-schema-to-typescript`` to convert.
4. Post-process the output to add a stable header (so re-runs are
   diffable) and a ``JaatoEvent`` discriminated-union alias for
   ergonomic client-side dispatch.
5. Either write the result to disk (default) or compare against
   the existing file and exit non-zero on mismatch (``--check``,
   used by the CI staleness gate).

Usage::

    .venv/bin/python scripts/codegen_ts_events.py            # regenerate
    .venv/bin/python scripts/codegen_ts_events.py --check    # CI gate
"""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Union

# Make jaato-sdk importable when invoked from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
SDK_PATH = REPO_ROOT / "jaato-sdk"
if str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))

from pydantic import TypeAdapter  # noqa: E402

from jaato_sdk.events import _EVENT_CLASSES  # noqa: E402

OUTPUT_FILE = REPO_ROOT / "jaato-sdk-ts" / "src" / "events.ts"

GENERATED_HEADER = """\
/*
 * GENERATED FILE — DO NOT EDIT.
 *
 * Mirror of jaato-sdk/jaato_sdk/events.py (pydantic), regenerated
 * via scripts/codegen_ts_events.py.  CI fails if this file is stale
 * relative to events.py — re-run the codegen and commit the result.
 *
 * The wire protocol is the contract: every event below has a
 * matching pydantic model on the Python side and a baseline JSON
 * snapshot in jaato-sdk/jaato_sdk/tests/baselines/events_wire_format/.
 */

/* eslint-disable */
"""


def _build_root_schema() -> dict[str, Any]:
    """Produce a single JSON Schema with $defs for every Event class.

    Uses pydantic's ``TypeAdapter`` over the union of all event
    classes so nested types (e.g. ``StagedFileSpec``) are emitted
    once under ``$defs`` rather than duplicated per event.  The
    top-level ``anyOf`` enumerates every event type — that becomes
    the ``JaatoEvent`` discriminated union on the TS side.
    """
    # The values of _EVENT_CLASSES are the concrete Event subclasses.
    # We build a Union dynamically — pydantic's TypeAdapter accepts
    # the constructed type and emits a schema with $defs for every
    # member plus a top-level anyOf.
    classes = tuple(_EVENT_CLASSES.values())
    if not classes:
        raise RuntimeError("_EVENT_CLASSES is empty — cannot codegen")
    union_type = Union[classes]  # type: ignore[valid-type]
    adapter = TypeAdapter(union_type)
    return adapter.json_schema(mode="serialization")


def _run_json2ts(schema: dict[str, Any]) -> str:
    """Invoke ``json-schema-to-typescript`` on the schema.

    Returns the generated TypeScript source.  Uses ``npx --yes`` so
    the tool is fetched on demand if not present in
    ``jaato-sdk-ts/node_modules``.
    """
    # Filename becomes the root type name (json2ts derives from basename),
    # so use ``jaato_events.json`` to produce ``JaatoEvents`` — that's the
    # discriminated union root the JaatoEvent alias re-exports.
    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = Path(tmpdir) / "jaato_events.json"
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

        cmd = [
            "npx", "--yes", "json-schema-to-typescript@^15.0.0",
            "--no-additionalProperties",
            "--no-bannerComment",
            str(schema_path),
        ]
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT / "jaato-sdk-ts",
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            sys.stderr.write(
                f"json-schema-to-typescript failed (exit {result.returncode}):\n"
                f"  STDOUT: {result.stdout}\n"
                f"  STDERR: {result.stderr}\n"
            )
            raise SystemExit(1)
        return result.stdout


def _build_jaato_event_alias() -> str:
    """Emit the ``JaatoEvent`` discriminated-union alias.

    The schema's top-level anyOf becomes a synthetic root type
    named after the schema file (``JaatoEvents`` from
    ``jaato_events.json``); we re-export it under the singular
    ``JaatoEvent`` for ergonomic narrowing.
    """
    return """
/**
 * Discriminated union of every wire-protocol event.
 *
 * Use the ``type`` field (an ``EventType`` value) to narrow.
 * Generated from the union of every Event subclass declared in
 * ``jaato-sdk/jaato_sdk/events.py``; mirrors the Python-side
 * dispatch table in ``_EVENT_CLASSES``.
 */
export type JaatoEvent = JaatoEvents;
"""


def _compose(generated_body: str) -> str:
    """Combine header + generated body + JaatoEvent alias."""
    return GENERATED_HEADER + "\n" + generated_body.rstrip() + "\n" + _build_jaato_event_alias()


def _generate() -> str:
    schema = _build_root_schema()
    body = _run_json2ts(schema)
    return _compose(body)


def _emit(content: str) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(content, encoding="utf-8")
    print(f"Wrote {OUTPUT_FILE.relative_to(REPO_ROOT)} ({len(content):,} bytes)")


def _check(content: str) -> int:
    if not OUTPUT_FILE.exists():
        sys.stderr.write(
            f"--check failed: {OUTPUT_FILE.relative_to(REPO_ROOT)} does not exist.\n"
            f"Run scripts/codegen_ts_events.py and commit the result.\n"
        )
        return 1
    existing = OUTPUT_FILE.read_text(encoding="utf-8")
    if existing == content:
        print(f"OK: {OUTPUT_FILE.relative_to(REPO_ROOT)} is up to date.")
        return 0
    sys.stderr.write(
        f"--check failed: {OUTPUT_FILE.relative_to(REPO_ROOT)} is stale.\n"
        f"Diff (existing → freshly generated):\n"
    )
    diff = difflib.unified_diff(
        existing.splitlines(keepends=True),
        content.splitlines(keepends=True),
        fromfile="committed",
        tofile="regenerated",
        n=3,
    )
    sys.stderr.writelines(diff)
    sys.stderr.write(
        "\nRun scripts/codegen_ts_events.py and commit the result.\n"
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Exit non-zero if the generated file would change (CI gate).",
    )
    args = parser.parse_args()

    content = _generate()
    if args.check:
        return _check(content)
    _emit(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
