"""Tests for Phase 5 §5.9 — supervisor-declared sub-profile
tightening flags allow-list / validator.

Pins the daemon-side validation contract on
``agent_params.isolated_*`` keys that flow runner → daemon through
``subagent.spawn_isolated_runner``.  Each test surfaces a single
shape violation; the validator must raise ``ValueError`` with a
message naming the offending key + the specific violation.

Design doc:
``docs/design/phase5_5_9_sub_profile_tightening_flags_audit.md``.
"""

from __future__ import annotations

import pytest

from server.runner_rpc_handlers.sub_profile_tightenings_schema import (
    SUB_PROFILE_TIGHTENING_KEYS,
    extract_and_validate_tightenings,
)


# ──────────────────────────────────────────────────────────────────
# Happy paths
# ──────────────────────────────────────────────────────────────────


class TestHappyPaths:
    def test_none_input_returns_empty(self):
        """Pin: ``None`` agent_params (the common case for
        supervisors that don't declare tightenings) returns an
        empty dict — never raises."""
        assert extract_and_validate_tightenings(None) == {}

    def test_empty_dict_returns_empty(self):
        assert extract_and_validate_tightenings({}) == {}

    def test_no_tightening_keys_returns_empty(self):
        """Pin: agent_params with only non-tightening keys
        (template data + the ``isolated`` control flag) returns
        an empty tightenings dict.  Template data is NOT
        propagated through the validator."""
        result = extract_and_validate_tightenings({
            "isolated": True,
            "name_template_arg": "alice",
            "task_context": "review-pr-67",
        })
        assert result == {}

    def test_workspace_subpath_validates(self):
        result = extract_and_validate_tightenings({
            "isolated_workspace_subpath": "scratch",
        })
        assert result == {"isolated_workspace_subpath": "scratch"}

    def test_workspace_subpath_with_nested_dir(self):
        """Pin: subpath with internal ``/`` separators is
        allowed (nested directory under the workspace)."""
        result = extract_and_validate_tightenings({
            "isolated_workspace_subpath": "scratch/agent-1",
        })
        assert result == {
            "isolated_workspace_subpath": "scratch/agent-1",
        }

    def test_read_only_workspace_validates(self):
        result = extract_and_validate_tightenings({
            "isolated_read_only_workspace": True,
        })
        assert result == {"isolated_read_only_workspace": True}

    def test_read_only_workspace_false_validates(self):
        """Pin: explicit False is preserved (lets supervisors
        say "no, definitely not read-only" without omitting the
        key — useful for profile-template overrides)."""
        result = extract_and_validate_tightenings({
            "isolated_read_only_workspace": False,
        })
        assert result == {"isolated_read_only_workspace": False}

    def test_both_flags_combined(self):
        result = extract_and_validate_tightenings({
            "isolated_workspace_subpath": "scratch",
            "isolated_read_only_workspace": True,
        })
        assert result == {
            "isolated_workspace_subpath": "scratch",
            "isolated_read_only_workspace": True,
        }


# ──────────────────────────────────────────────────────────────────
# Contract: validator returns NEW dict, doesn't mutate input
# ──────────────────────────────────────────────────────────────────


class TestNonMutating:
    def test_input_dict_not_mutated(self):
        """Pin: validator returns a NEW dict — caller is
        responsible for stripping the tightening keys from
        their copy of agent_params before forwarding."""
        original = {
            "isolated": True,
            "isolated_workspace_subpath": "scratch",
            "template_arg": "x",
        }
        snapshot = dict(original)
        extract_and_validate_tightenings(original)
        assert original == snapshot, (
            "extract_and_validate_tightenings mutated the input "
            "dict — violates the non-mutating contract"
        )


# ──────────────────────────────────────────────────────────────────
# Top-level type rejection
# ──────────────────────────────────────────────────────────────────


class TestTopLevelType:
    def test_non_dict_rejected(self):
        with pytest.raises(ValueError, match="agent_params must be a dict"):
            extract_and_validate_tightenings([1, 2, 3])  # type: ignore[arg-type]

    def test_string_input_rejected(self):
        with pytest.raises(ValueError, match="agent_params must be a dict"):
            extract_and_validate_tightenings("isolated=true")  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────
# isolated_workspace_subpath value validation
# ──────────────────────────────────────────────────────────────────


class TestWorkspaceSubpath:
    def test_non_str_rejected(self):
        with pytest.raises(ValueError, match="must be a str"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": 123,
            })

    def test_bool_rejected(self):
        """Pin: bool is rejected explicitly even though it would
        ``isinstance(True, int)``-but-not-str — the message
        names the str requirement."""
        with pytest.raises(ValueError, match="must be a str"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": True,
            })

    def test_none_rejected(self):
        with pytest.raises(ValueError, match="must be a str"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": None,
            })

    def test_empty_str_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "",
            })

    def test_oversized_rejected(self):
        with pytest.raises(ValueError, match="length cap"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "A" * 257,
            })

    def test_at_length_cap_ok(self):
        result = extract_and_validate_tightenings({
            "isolated_workspace_subpath": "A" * 256,
        })
        assert result["isolated_workspace_subpath"] == "A" * 256

    # ── Path-traversal rejection ───────────────────────────────

    def test_dot_dot_segment_rejected(self):
        """Pin: a single ``..`` segment is rejected — this is
        the canonical path-traversal attack the renderer must
        be defended against."""
        with pytest.raises(ValueError, match=r"\.\."):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "..",
            })

    def test_dot_dot_in_middle_rejected(self):
        with pytest.raises(ValueError, match=r"\.\."):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "a/../b",
            })

    def test_dot_dot_at_start_rejected(self):
        with pytest.raises(ValueError, match=r"\.\."):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "../escape",
            })

    def test_dot_dot_at_end_rejected(self):
        with pytest.raises(ValueError, match=r"\.\."):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "scratch/..",
            })

    def test_backslash_dot_dot_rejected(self):
        """Pin: Windows-style separators normalized before the
        ``..`` check so ``a\\..\\b`` is also caught."""
        with pytest.raises(ValueError, match=r"forbidden characters|\.\."):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "a\\..\\b",
            })

    # ── Absolute path rejection ────────────────────────────────

    def test_posix_absolute_rejected(self):
        with pytest.raises(ValueError, match="absolute"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "/etc/passwd",
            })

    def test_windows_drive_rejected(self):
        with pytest.raises(ValueError, match="drive-letter"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "C:foo",
            })

    def test_leading_backslash_rejected(self):
        with pytest.raises(ValueError, match="absolute"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "\\windows\\path",
            })

    # ── Trailing-slash rejection ───────────────────────────────

    def test_trailing_slash_rejected(self):
        """Pin: trailing slash would render as ``{workspace}/X//``
        — reject to keep the rendered path canonical."""
        with pytest.raises(ValueError, match="trailing slash"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "scratch/",
            })

    def test_trailing_backslash_rejected(self):
        with pytest.raises(ValueError, match="trailing slash"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "scratch\\",
            })

    # ── Forbidden chars: glob + AppArmor-quote-breaking ───────

    def test_asterisk_glob_rejected(self):
        with pytest.raises(ValueError, match="forbidden characters"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "*",
            })

    def test_glob_inside_path_rejected(self):
        with pytest.raises(ValueError, match="forbidden characters"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "scratch/a*b",
            })

    def test_question_mark_glob_rejected(self):
        with pytest.raises(ValueError, match="forbidden characters"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "scratch/a?b",
            })

    def test_bracket_glob_rejected(self):
        with pytest.raises(ValueError, match="forbidden characters"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "scratch/a[bc]",
            })

    def test_brace_glob_rejected(self):
        with pytest.raises(ValueError, match="forbidden characters"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "scratch/a{b,c}",
            })

    def test_double_quote_rejected(self):
        """Pin: double-quote would break the rendered AppArmor
        profile body's path quoting (rendered as
        ``"<sub_profile_name>"``).  Defense against profile
        injection."""
        with pytest.raises(ValueError, match="forbidden characters"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": 'a"b',
            })

    def test_newline_rejected(self):
        """Pin: newline would inject new rules into the
        rendered profile body."""
        with pytest.raises(ValueError, match="forbidden characters"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "a\nb",
            })

    def test_nul_rejected(self):
        with pytest.raises(ValueError, match="forbidden characters"):
            extract_and_validate_tightenings({
                "isolated_workspace_subpath": "a\x00b",
            })

    def test_shell_metacharacter_rejected(self):
        """Pin: defense in depth — shell metacharacters are
        rejected.  AppArmor profiles don't execute shell, but
        their values flow into render templates and we don't
        want surprises."""
        for ch in [";", "|", "&", "`", "$", "<", ">"]:
            with pytest.raises(
                ValueError, match="forbidden characters",
            ):
                extract_and_validate_tightenings({
                    "isolated_workspace_subpath": f"a{ch}b",
                })


# ──────────────────────────────────────────────────────────────────
# isolated_read_only_workspace value validation
# ──────────────────────────────────────────────────────────────────


class TestReadOnlyWorkspace:
    def test_str_rejected(self):
        with pytest.raises(ValueError, match="must be a bool"):
            extract_and_validate_tightenings({
                "isolated_read_only_workspace": "true",
            })

    def test_int_one_rejected(self):
        """Pin: int ``1`` is rejected even though it's truthy
        + ``isinstance(1, int)`` — explicit-type check rules
        out the int-truthy case."""
        with pytest.raises(ValueError, match="must be a bool"):
            extract_and_validate_tightenings({
                "isolated_read_only_workspace": 1,
            })

    def test_int_zero_rejected(self):
        with pytest.raises(ValueError, match="must be a bool"):
            extract_and_validate_tightenings({
                "isolated_read_only_workspace": 0,
            })

    def test_none_rejected(self):
        with pytest.raises(ValueError, match="must be a bool"):
            extract_and_validate_tightenings({
                "isolated_read_only_workspace": None,
            })


# ──────────────────────────────────────────────────────────────────
# Allow-list invariant
# ──────────────────────────────────────────────────────────────────


class TestAllowListInvariant:
    def test_allow_list_matches_expected_v1_keys(self):
        """Pin: the v1 allow-list contains exactly these two
        keys.  A future PR that adds a tightening flag MUST
        update this test in lockstep — the failure is the
        forcing function for design-doc + renderer + handler
        wire-up to all move together."""
        assert SUB_PROFILE_TIGHTENING_KEYS == frozenset({
            "isolated_workspace_subpath",
            "isolated_read_only_workspace",
        })
