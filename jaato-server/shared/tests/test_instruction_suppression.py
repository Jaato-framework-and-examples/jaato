"""Tests for the granular ``suppress_base_instructions`` knob.

Covers the canonical normalizer (``instruction_suppression``) and the two
assembly paths' gating:

* the wire prompt (``JaatoRuntime.get_system_instructions`` — ``include_base`` /
  ``include_constants`` / ``include_security``), and
* the budget accounting (``instruction_budget_builder.collect_instruction_texts``
  — ``suppress_base`` / ``suppress_constants``).

The design: ``true`` drops {disk, constants} (the security untrusted-content
boundary is KEPT — it is the injection defense, dropped only when named); a
dict/list gives per-piece control; unknown pieces fail loud.
"""
import pytest

from shared.instruction_suppression import (
    PIECE_CONSTANTS,
    PIECE_DISK,
    PIECE_SECURITY,
    SUPPRESSION_PIECES,
    normalize_suppression,
    suppression_to_wire,
)


class TestNormalize:
    def test_false_and_none_suppress_nothing(self):
        assert normalize_suppression(False) == frozenset()
        assert normalize_suppression(None) == frozenset()

    def test_true_drops_disk_and_constants_keeps_security(self):
        # The blanket true never drops the injection-defense boundary.
        assert normalize_suppression(True) == frozenset({PIECE_DISK, PIECE_CONSTANTS})
        assert PIECE_SECURITY not in normalize_suppression(True)

    def test_dict_absent_key_is_keep(self):
        assert normalize_suppression({"constants": True}) == frozenset({PIECE_CONSTANTS})
        assert normalize_suppression(
            {"disk": True, "constants": False}
        ) == frozenset({PIECE_DISK})
        # Falsy values keep the piece.
        assert normalize_suppression({"disk": False}) == frozenset()

    def test_dict_all_true_drops_everything_incl_security(self):
        assert normalize_suppression(
            {"disk": True, "constants": True, "security": True}
        ) == SUPPRESSION_PIECES

    def test_list_form_and_all_token(self):
        assert normalize_suppression(["disk", "constants"]) == frozenset(
            {PIECE_DISK, PIECE_CONSTANTS}
        )
        assert normalize_suppression(["all"]) == SUPPRESSION_PIECES
        assert normalize_suppression([]) == frozenset()

    def test_frozenset_is_idempotent(self):
        fs = frozenset({PIECE_DISK, PIECE_SECURITY})
        assert normalize_suppression(fs) == fs
        # Re-normalizing the output is stable.
        assert normalize_suppression(normalize_suppression(True)) == normalize_suppression(True)

    def test_unknown_dict_key_fails_loud(self):
        with pytest.raises(ValueError, match="unknown piece"):
            normalize_suppression({"bogus": True})

    def test_unknown_list_item_fails_loud(self):
        with pytest.raises(ValueError, match="unknown piece"):
            normalize_suppression(["disk", "nope"])

    def test_unsupported_type_fails_loud(self):
        with pytest.raises(ValueError, match="must be a bool"):
            normalize_suppression("disk")  # bare string is rejected
        with pytest.raises(ValueError):
            normalize_suppression(42)

    def test_to_wire_is_sorted_list(self):
        assert suppression_to_wire(normalize_suppression(True)) == ["constants", "disk"]
        assert suppression_to_wire(frozenset()) == []
        # Round-trips back through normalize.
        wire = suppression_to_wire(SUPPRESSION_PIECES)
        assert normalize_suppression(wire) == SUPPRESSION_PIECES


class _StubRuntime:
    """Minimal runtime for exercising the builder's SYSTEM-children gates.

    ``collect_instruction_texts`` only touches ``get_base_system_instructions``
    on the runtime when ``suppress_base`` is False and there is no registry.
    """

    registry = None  # no plugins — the builder skips the PLUGIN-children walk

    def __init__(self, base="BASE DISK LAYER"):
        self._base = base

    def get_base_system_instructions(self):
        return self._base


def _child_keys(requests):
    return {r.child_key for r in requests}


class TestBuilderGating:
    """``collect_instruction_texts`` drops the right SYSTEM children."""

    def _collect(self, **kw):
        from shared.instruction_budget_builder import (
            collect_instruction_texts,
            SystemChildType,
        )
        requests, _ = collect_instruction_texts(
            _StubRuntime(), session_instructions="PERSONA", **kw
        )
        return _child_keys(requests), SystemChildType

    def test_default_includes_base_and_framework(self):
        keys, T = self._collect()
        assert T.BASE.value in keys
        assert T.FRAMEWORK.value in keys
        assert T.CLIENT.value in keys  # the persona

    def test_suppress_base_drops_disk_keeps_framework(self):
        keys, T = self._collect(suppress_base=True)
        assert T.BASE.value not in keys
        assert T.FRAMEWORK.value in keys  # constants still counted
        assert T.CLIENT.value in keys

    def test_suppress_constants_drops_framework_keeps_base(self):
        keys, T = self._collect(suppress_constants=True)
        assert T.FRAMEWORK.value not in keys
        assert T.BASE.value in keys
        assert T.CLIENT.value in keys

    def test_suppress_both_drops_disk_and_framework(self):
        keys, T = self._collect(suppress_base=True, suppress_constants=True)
        assert T.BASE.value not in keys
        assert T.FRAMEWORK.value not in keys
        assert T.CLIENT.value in keys  # persona survives — always kept


class TestRuntimeAssemblyGating:
    """``get_system_instructions`` drops constants / security on demand.

    Uses a bare ``JaatoRuntime`` (``__new__``) with the minimal attributes the
    method reads on the ``plugin_names=[]`` path, and a stubbed base loader.
    """

    def _runtime(self):
        from shared.jaato_runtime import JaatoRuntime
        rt = JaatoRuntime.__new__(JaatoRuntime)
        rt._registry = None
        rt._system_instructions = None
        rt._permission_plugin = None
        rt._formatter_pipeline = None
        rt.get_base_system_instructions = lambda: "BASE DISK LAYER"
        return rt

    def _assemble(self, **kw):
        return self._runtime().get_system_instructions(
            plugin_names=[], additional="PERSONA", **kw
        )

    def test_full_assembly_has_all_layers(self):
        from jaato_sdk.plugins.model_provider.types import untrusted_boundary_instruction
        text = self._assemble()
        assert "BASE DISK LAYER" in text
        assert "PERSONA" in text
        # A framework-constant phrase + the security boundary are present.
        assert untrusted_boundary_instruction()[:24] in text

    def test_include_base_false_drops_disk_only(self):
        text = self._assemble(include_base=False)
        assert "BASE DISK LAYER" not in text
        assert "PERSONA" in text

    def test_include_constants_false_drops_constants(self):
        from shared.jaato_runtime import _TASK_COMPLETION_INSTRUCTION
        full = self._assemble()
        gated = self._assemble(include_constants=False)
        # The task-completion constant is present in full, gone when gated.
        assert _TASK_COMPLETION_INSTRUCTION in full
        assert _TASK_COMPLETION_INSTRUCTION not in gated
        assert "PERSONA" in gated

    def test_include_security_false_drops_boundary(self):
        from jaato_sdk.plugins.model_provider.types import untrusted_boundary_instruction
        boundary = untrusted_boundary_instruction()
        assert boundary in self._assemble()
        assert boundary not in self._assemble(include_security=False)

    def test_blanket_true_semantics_keep_security(self):
        """The end-to-end contract: suppression=={disk,constants} (what ``true``
        normalizes to) drops disk + constants but KEEPS the security boundary."""
        from jaato_sdk.plugins.model_provider.types import untrusted_boundary_instruction
        supp = normalize_suppression(True)
        text = self._assemble(
            include_base=PIECE_DISK not in supp,
            include_constants=PIECE_CONSTANTS not in supp,
            include_security=PIECE_SECURITY not in supp,
        )
        assert "BASE DISK LAYER" not in text
        assert untrusted_boundary_instruction() in text  # kept
        assert "PERSONA" in text
