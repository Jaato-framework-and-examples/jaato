"""Tests for the hoisted ``server.subagent_id.validate_subagent_id``
helper (Phase 4 §4.3.5).

§4.3.4 anchored the validation on ``AppArmorManager._validate_subagent_id``.
§4.3.5 hoisted the body to ``server/subagent_id.py`` so the sub-cgroup
validator can share the same implementation.

These tests pin the module-level helper directly.  The
``AppArmorManager._validate_subagent_id`` classmethod still works
(delegates to this helper); its existing test suite in
``shared/tests/test_apparmor_sub_profile.py::TestSubagentIdSanitization``
exercises the public surface and continues to pass — that's the
backwards-compat guarantee for §4.3.5's refactor.

Divergence between sub-AppArmor + sub-cgroup validation would be a
latent confused-deputy bug — both surfaces accept the same runner-
supplied id and feed it into kernel-visible names.  This shared
helper is the single source of truth.
"""

from __future__ import annotations

import pytest

from server.subagent_id import (
    SUBAGENT_ID_MAX_LEN,
    validate_subagent_id,
)


class TestValidateSubagentIdAcceptance:
    """Valid ids — alphanumeric + hyphen + underscore."""

    def test_alphanumeric_accepted(self):
        assert validate_subagent_id("agent1") is None

    def test_hyphen_accepted(self):
        assert validate_subagent_id("agent-1") is None

    def test_underscore_accepted(self):
        assert validate_subagent_id("agent_1") is None

    def test_mixed_accepted(self):
        assert validate_subagent_id("agent-1_v2") is None

    def test_single_char_accepted(self):
        assert validate_subagent_id("a") is None

    def test_max_length_accepted(self):
        """Boundary case: exactly SUBAGENT_ID_MAX_LEN chars."""
        assert validate_subagent_id("a" * SUBAGENT_ID_MAX_LEN) is None


class TestValidateSubagentIdRejection:
    """Invalid ids — strict allow-list + length cap + non-empty."""

    def test_empty_rejected(self):
        err = validate_subagent_id("")
        assert err is not None
        assert "empty" in err.lower()

    def test_non_str_int_rejected(self):
        err = validate_subagent_id(42)
        assert err is not None
        assert "must be a str" in err

    def test_non_str_none_rejected(self):
        err = validate_subagent_id(None)
        assert err is not None
        assert "must be a str" in err

    def test_non_str_bytes_rejected(self):
        err = validate_subagent_id(b"agent-1")
        assert err is not None
        assert "must be a str" in err

    def test_oversize_rejected(self):
        err = validate_subagent_id("a" * (SUBAGENT_ID_MAX_LEN + 1))
        assert err is not None
        assert "length" in err.lower()
        assert "cap" in err.lower()

    def test_newline_rejected(self):
        """Newline would let an attacker break out of a profile-name
        or cgroup-directory string."""
        err = validate_subagent_id("agent\nprofile evil { }")
        assert err is not None
        assert "outside" in err

    def test_path_traversal_dot_dot_rejected(self):
        err = validate_subagent_id("../escape")
        assert err is not None

    def test_slash_rejected(self):
        """Slash would target other directories in cgroup tree, or
        clash with AppArmor's ``//`` separator."""
        err = validate_subagent_id("agent/1")
        assert err is not None

    def test_space_rejected(self):
        err = validate_subagent_id("my agent")
        assert err is not None

    def test_dot_rejected(self):
        """Dot is not in the allow-list — strict rejection (REJECT,
        not collapse, per Audit 6)."""
        err = validate_subagent_id("agent.1")
        assert err is not None

    def test_null_byte_rejected(self):
        """Null bytes can truncate kernel-side names."""
        err = validate_subagent_id("agent\x00evil")
        assert err is not None

    def test_unicode_rejected(self):
        """Non-ASCII chars are outside the allow-list."""
        err = validate_subagent_id("agent-éé")
        assert err is not None


class TestErrorMessageQuality:
    """Error messages must surface the problem precisely so callers
    can fix bugs quickly.  These pin the message content."""

    def test_oversize_mentions_actual_length(self):
        oversize = "a" * (SUBAGENT_ID_MAX_LEN + 5)
        err = validate_subagent_id(oversize)
        assert err is not None
        assert str(len(oversize)) in err

    def test_bad_chars_mentions_id_repr(self):
        err = validate_subagent_id("agent.1")
        assert err is not None
        # Repr keeps the id visible in logs/traces.
        assert "agent.1" in err

    def test_non_str_mentions_actual_type(self):
        err = validate_subagent_id(42)
        assert err is not None
        assert "int" in err
