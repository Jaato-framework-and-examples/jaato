"""Tests for the permission policy evaluation engine."""

import pytest
from ..policy import PermissionPolicy, PermissionDecision, PolicyMatch


class TestPermissionDecision:
    """Tests for PermissionDecision enum."""

    def test_decision_values(self):
        assert PermissionDecision.ALLOW.value == "allow"
        assert PermissionDecision.DENY.value == "deny"
        assert PermissionDecision.ASK_CHANNEL.value == "ask_channel"


class TestPolicyMatch:
    """Tests for PolicyMatch dataclass."""

    def test_basic_match(self):
        match = PolicyMatch(
            decision=PermissionDecision.ALLOW,
            reason="Test reason"
        )
        assert match.decision == PermissionDecision.ALLOW
        assert match.reason == "Test reason"
        assert match.matched_rule is None
        assert match.rule_type is None

    def test_match_with_rule_details(self):
        match = PolicyMatch(
            decision=PermissionDecision.DENY,
            reason="Blocked by blacklist",
            matched_rule="sudo *",
            rule_type="blacklist"
        )
        assert match.matched_rule == "sudo *"
        assert match.rule_type == "blacklist"


class TestPermissionPolicyBlacklist:
    """Tests for blacklist evaluation."""

    def test_blacklist_tool_exact_match(self):
        policy = PermissionPolicy(
            blacklist_tools={"dangerous_tool", "admin_tool"}
        )
        match = policy.check("dangerous_tool", {})
        assert match.decision == PermissionDecision.DENY
        assert match.rule_type == "blacklist"

    def test_blacklist_tool_no_match(self):
        policy = PermissionPolicy(
            default_policy="allow",
            blacklist_tools={"dangerous_tool"}
        )
        match = policy.check("safe_tool", {})
        assert match.decision == PermissionDecision.ALLOW

    def test_blacklist_pattern_glob_match(self):
        policy = PermissionPolicy(
            blacklist_patterns=["rm -rf *", "sudo *"]
        )
        # CLI tool signature is just the command
        match = policy.check("cli_based_tool", {"command": "sudo apt update"})
        assert match.decision == PermissionDecision.DENY
        assert "sudo *" in match.reason

    def test_blacklist_pattern_no_match(self):
        policy = PermissionPolicy(
            default_policy="allow",
            blacklist_patterns=["rm -rf *"]
        )
        match = policy.check("cli_based_tool", {"command": "rm file.txt"})
        assert match.decision == PermissionDecision.ALLOW

    def test_blacklist_arguments_starts_with(self):
        policy = PermissionPolicy(
            blacklist_arguments={
                "cli_based_tool": {"command": ["sudo", "rm -rf"]}
            }
        )
        match = policy.check("cli_based_tool", {"command": "sudo apt install"})
        assert match.decision == PermissionDecision.DENY

    def test_blacklist_arguments_contains(self):
        policy = PermissionPolicy(
            blacklist_arguments={
                "cli_based_tool": {"command": ["--force"]}
            }
        )
        match = policy.check("cli_based_tool", {"command": "git push --force origin"})
        assert match.decision == PermissionDecision.DENY


class TestPermissionPolicyWhitelist:
    """Tests for whitelist evaluation."""

    def test_whitelist_tool_exact_match(self):
        policy = PermissionPolicy(
            default_policy="deny",
            whitelist_tools={"safe_tool", "read_tool"}
        )
        match = policy.check("safe_tool", {})
        assert match.decision == PermissionDecision.ALLOW
        assert match.rule_type == "whitelist"

    def test_whitelist_pattern_glob_match(self):
        policy = PermissionPolicy(
            default_policy="deny",
            whitelist_patterns=["git *", "npm test"]
        )
        match = policy.check("cli_based_tool", {"command": "git status"})
        assert match.decision == PermissionDecision.ALLOW

    def test_whitelist_arguments_starts_with(self):
        policy = PermissionPolicy(
            default_policy="deny",
            whitelist_arguments={
                "cli_based_tool": {"command": ["git", "npm", "pip"]}
            }
        )
        match = policy.check("cli_based_tool", {"command": "git push origin main"})
        assert match.decision == PermissionDecision.ALLOW


class TestPermissionPolicyPriority:
    """Tests for blacklist/whitelist priority."""

    def test_blacklist_takes_priority_over_whitelist(self):
        """Blacklist should always win over whitelist."""
        policy = PermissionPolicy(
            blacklist_tools={"dangerous_tool"},
            whitelist_tools={"dangerous_tool"}  # Also whitelisted
        )
        match = policy.check("dangerous_tool", {})
        assert match.decision == PermissionDecision.DENY

    def test_blacklist_pattern_beats_whitelist_tool(self):
        policy = PermissionPolicy(
            blacklist_patterns=["rm *"],
            whitelist_tools={"cli_based_tool"}
        )
        match = policy.check("cli_based_tool", {"command": "rm file.txt"})
        assert match.decision == PermissionDecision.DENY

    def test_session_blacklist_highest_priority(self):
        policy = PermissionPolicy(
            default_policy="allow",
            whitelist_tools={"some_tool"}
        )
        policy.add_session_blacklist("some_tool")
        match = policy.check("some_tool", {})
        assert match.decision == PermissionDecision.DENY
        assert match.rule_type == "session_blacklist"

    def test_session_whitelist_after_static_blacklist(self):
        policy = PermissionPolicy(
            default_policy="deny"
        )
        policy.add_session_whitelist("dynamic_tool")
        match = policy.check("dynamic_tool", {})
        assert match.decision == PermissionDecision.ALLOW
        assert match.rule_type == "session_whitelist"


class TestPermissionPolicyDefaultPolicy:
    """Tests for default policy behavior."""

    def test_default_policy_allow(self):
        policy = PermissionPolicy(default_policy="allow")
        match = policy.check("unknown_tool", {})
        assert match.decision == PermissionDecision.ALLOW
        assert match.rule_type == "default"

    def test_default_policy_deny(self):
        policy = PermissionPolicy(default_policy="deny")
        match = policy.check("unknown_tool", {})
        assert match.decision == PermissionDecision.DENY
        assert match.rule_type == "default"

    def test_default_policy_ask(self):
        policy = PermissionPolicy(default_policy="ask")
        match = policy.check("unknown_tool", {})
        assert match.decision == PermissionDecision.ASK_CHANNEL
        assert match.rule_type == "default"


class TestPermissionPolicySessionRules:
    """Tests for session-level dynamic rules."""

    def test_add_session_blacklist(self):
        policy = PermissionPolicy(default_policy="allow")
        policy.add_session_blacklist("blocked_tool")

        match = policy.check("blocked_tool", {})
        assert match.decision == PermissionDecision.DENY

    def test_add_session_whitelist(self):
        policy = PermissionPolicy(default_policy="deny")
        policy.add_session_whitelist("allowed_tool")

        match = policy.check("allowed_tool", {})
        assert match.decision == PermissionDecision.ALLOW

    def test_clear_session_rules(self):
        policy = PermissionPolicy(default_policy="deny")
        policy.add_session_whitelist("tool")
        policy.clear_session_rules()

        match = policy.check("tool", {})
        assert match.decision == PermissionDecision.DENY

    def test_session_blacklist_with_pattern(self):
        policy = PermissionPolicy(default_policy="allow")
        policy.add_session_blacklist("git push*")

        match = policy.check("cli_based_tool", {"command": "git push origin main"})
        assert match.decision == PermissionDecision.DENY

    def test_explicit_session_whitelist_overrides_blacklist_pattern(self):
        """Explicit session whitelist entry should override blacklist patterns.

        This tests the scenario where:
        - Session blacklist has pattern "create*" (blocks createPlan, createIssue, etc.)
        - Session whitelist has explicit "createPlan"

        The explicit whitelist entry should take precedence over the pattern.
        """
        policy = PermissionPolicy(default_policy="deny")
        policy.add_session_blacklist("create*")  # Pattern blocking all create* tools
        policy.add_session_whitelist("createPlan")  # Explicit override

        # createPlan should be ALLOWED (explicit whitelist overrides pattern)
        match = policy.check("createPlan", {})
        assert match.decision == PermissionDecision.ALLOW
        assert match.rule_type == "session_whitelist"

        # createIssue should still be DENIED (no explicit whitelist)
        match = policy.check("createIssue", {})
        assert match.decision == PermissionDecision.DENY
        assert match.rule_type == "session_blacklist"

    def test_explicit_session_blacklist_beats_explicit_whitelist(self):
        """Explicit blacklist entry should still beat explicit whitelist.

        If both are explicit (not patterns), blacklist wins.
        """
        policy = PermissionPolicy(default_policy="allow")
        policy.add_session_whitelist("dangerousTool")
        policy.add_session_blacklist("dangerousTool")  # Exact match

        match = policy.check("dangerousTool", {})
        assert match.decision == PermissionDecision.DENY
        assert match.rule_type == "session_blacklist"

    def test_session_default_policy_allow(self):
        """Session default policy should override base default."""
        policy = PermissionPolicy(default_policy="deny")
        policy.set_session_default_policy("allow")

        match = policy.check("unknown_tool", {})
        assert match.decision == PermissionDecision.ALLOW

    def test_session_default_policy_deny(self):
        """Session default policy deny should override base allow."""
        policy = PermissionPolicy(default_policy="allow")
        policy.set_session_default_policy("deny")

        match = policy.check("unknown_tool", {})
        assert match.decision == PermissionDecision.DENY

    def test_session_default_policy_ask(self):
        """Session default policy ask should override base."""
        policy = PermissionPolicy(default_policy="allow")
        policy.set_session_default_policy("ask")

        match = policy.check("unknown_tool", {})
        assert match.decision == PermissionDecision.ASK_CHANNEL

    def test_session_default_policy_cleared_on_clear(self):
        """clear_session_rules should also clear session default policy."""
        policy = PermissionPolicy(default_policy="deny")
        policy.set_session_default_policy("allow")

        match = policy.check("unknown_tool", {})
        assert match.decision == PermissionDecision.ALLOW

        policy.clear_session_rules()

        match = policy.check("unknown_tool", {})
        assert match.decision == PermissionDecision.DENY

    def test_get_session_default_policy(self):
        """get_session_default_policy should return current value."""
        policy = PermissionPolicy(default_policy="deny")
        assert policy.get_session_default_policy() is None

        policy.set_session_default_policy("allow")
        assert policy.get_session_default_policy() == "allow"

    def test_set_session_default_policy_validation(self):
        """set_session_default_policy should reject invalid values."""
        policy = PermissionPolicy()

        with pytest.raises(ValueError) as exc_info:
            policy.set_session_default_policy("invalid")
        assert "Invalid policy" in str(exc_info.value)

    def test_set_session_default_policy_none_clears(self):
        """Setting session default to None should clear it."""
        policy = PermissionPolicy(default_policy="deny")
        policy.set_session_default_policy("allow")
        assert policy.session_default_policy == "allow"

        policy.set_session_default_policy(None)
        assert policy.session_default_policy is None

        match = policy.check("unknown_tool", {})
        assert match.decision == PermissionDecision.DENY


class TestPermissionPolicyFromConfig:
    """Tests for creating policy from config dict."""

    def test_from_config_basic(self):
        config = {
            "defaultPolicy": "ask",
            "blacklist": {
                "tools": ["dangerous"],
                "patterns": ["rm -rf *"]
            },
            "whitelist": {
                "tools": ["safe"],
                "patterns": ["git *"]
            }
        }
        policy = PermissionPolicy.from_config(config)

        assert policy.default_policy == "ask"
        assert "dangerous" in policy.blacklist_tools
        assert "rm -rf *" in policy.blacklist_patterns
        assert "safe" in policy.whitelist_tools
        assert "git *" in policy.whitelist_patterns

    def test_from_config_with_arguments(self):
        config = {
            "defaultPolicy": "deny",
            "blacklist": {
                "arguments": {
                    "cli_based_tool": {"command": ["sudo", "rm"]}
                }
            },
            "whitelist": {
                "arguments": {
                    "cli_based_tool": {"command": ["git", "npm"]}
                }
            }
        }
        policy = PermissionPolicy.from_config(config)

        assert "cli_based_tool" in policy.blacklist_arguments
        assert "cli_based_tool" in policy.whitelist_arguments

    def test_from_config_empty(self):
        policy = PermissionPolicy.from_config({})
        assert policy.default_policy == "deny"  # Default


class TestPermissionPolicySignatureBuilding:
    """Tests for command signature building."""

    def test_cli_tool_signature(self):
        policy = PermissionPolicy(
            whitelist_patterns=["git status"]
        )
        match = policy.check("cli_based_tool", {"command": "git status"})
        assert match.decision == PermissionDecision.ALLOW

    def test_cli_tool_with_args_list(self):
        policy = PermissionPolicy(
            whitelist_patterns=["git status --short"]
        )
        match = policy.check("cli_based_tool", {
            "command": "git status",
            "args": ["--short"]
        })
        assert match.decision == PermissionDecision.ALLOW

    def test_non_cli_tool_signature(self):
        """Non-CLI tools get signature like tool_name(arg=val, ...)"""
        policy = PermissionPolicy(
            whitelist_patterns=["search_issues(query=bug*)"]
        )
        match = policy.check("search_issues", {"query": "bug123"})
        # The signature format is tool_name(arg1=val1, arg2=val2, ...)
        # Sorted alphabetically
        assert match.decision == PermissionDecision.ALLOW
