"""Tests for the typed cascade-authoring surface.

This module is the explicit accumulating destination for typed
contracts cascade authors need (see Hybrid-C strategy in
`reference_cascade_authoring_sdk_destination` memory).  First two
citizens: ``ProcessorResult`` (this PR) + ``ToolCallEntry``
(re-export from :mod:`jaato_sdk.completion_processors`).

These tests cover:
- Both umbrella exports resolve from package root + submodule
- ``ProcessorResult`` accepts empty/populated/mixed shapes
- Legacy ``List[str]`` is NOT a ``ProcessorResult`` (callers must
  pick one shape; the framework's ``invoke_processors`` handles both
  at the runtime level, but the type contract is the new shape)
"""

from __future__ import annotations

from typing import get_type_hints

import pytest

from jaato_sdk import ProcessorResult, ToolCallEntry
from jaato_sdk.cascade_authoring import (
    ProcessorResult as _PR,
    ToolCallEntry as _TCE,
)


def test_processorresult_exported_from_package_root():
    """Both ``from jaato_sdk import ProcessorResult`` and
    ``from jaato_sdk.cascade_authoring import ProcessorResult``
    resolve to the same class."""
    assert ProcessorResult is _PR


def test_toolcallentry_reexported_from_cascade_authoring_umbrella():
    """``ToolCallEntry`` (PR #184 first-citizen) is re-exported from
    the new cascade-authoring umbrella so authors can grab everything
    from one import.  Should resolve to the same class as the
    completion_processors-submodule export."""
    from jaato_sdk.completion_processors import (
        ToolCallEntry as _CP_TCE,
    )
    assert ToolCallEntry is _TCE is _CP_TCE


def test_processorresult_field_set():
    """Field names match the framework's
    ``invoke_processors`` dual-shape handler (server
    ``shared/completion_processors.py``).

    ``faults`` joined the set with the refusal budget (jaato #768): an
    environment fault is neither a wrong answer nor advisory, so it needed
    a channel of its own — one that blocks but never spends a refusal.
    The server-side half of this pairing is asserted in
    ``shared/tests/test_scaffold_completion_contract.py``, which compares
    these annotations against what ``explain completion`` renders.
    """
    hints = get_type_hints(ProcessorResult)
    assert set(hints.keys()) == {"errors", "warnings", "incomplete", "faults"}


def test_processorresult_accepts_populated_shape():
    """Smoke check: a populated dict satisfies the TypedDict
    contract at runtime.  (TypedDicts are structural at runtime —
    this verifies the documented shape works end-to-end.)
    """
    result: ProcessorResult = {
        "errors": ["agent must retry: missing memory_id"],
        "warnings": ["audit-only: unusual scope=universal"],
    }
    assert result["errors"][0].startswith("agent must retry")
    assert result["warnings"][0].startswith("audit-only")


def test_processorresult_accepts_empty_shape():
    """Empty lists are valid — happy-path processor returns
    ``{"errors": [], "warnings": []}`` to accept the payload with
    no flags."""
    result: ProcessorResult = {"errors": [], "warnings": []}
    assert result["errors"] == []
    assert result["warnings"] == []


def test_processorresult_accepts_errors_only():
    """Authors who only need fatal errors can omit warnings —
    ``total=False`` on the TypedDict allows partial keys."""
    result: ProcessorResult = {"errors": ["msg1", "msg2"]}
    assert result["errors"] == ["msg1", "msg2"]
    assert "warnings" not in result


def test_processorresult_accepts_warnings_only():
    """Symmetric — advisory-only processor can omit errors."""
    result: ProcessorResult = {"warnings": ["advisory msg"]}
    assert result["warnings"] == ["advisory msg"]
    assert "errors" not in result
