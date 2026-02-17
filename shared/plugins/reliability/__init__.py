"""Reliability plugin for tracking tool failures and adaptive trust.

This plugin monitors tool execution and maintains reliability profiles
for each tool+parameters combination. When failures exceed thresholds,
tools are escalated to require explicit user approval.
"""

from .plugin import (
    ReliabilityPlugin,
    ReliabilityPermissionWrapper,
    create_plugin,
    wrap_permission_plugin,
)
from .types import (
    # Core types
    EscalationInfo,
    EscalationRule,
    FailureKey,
    FailureRecord,
    FailureSeverity,
    PrerequisitePolicy,
    ReliabilityConfig,
    ToolReliabilityState,
    TrustState,
    classify_failure,
    # Model reliability types
    ModelBehavioralProfile,
    ModelSwitchConfig,
    ModelSwitchStrategy,
    ModelSwitchSuggestion,
    ModelToolProfile,
    # Behavioral pattern types
    BehavioralPattern,
    BehavioralPatternType,
    PatternDetectionConfig,
    PatternSeverity,
    ToolCall,
    # Nudge injection types
    Nudge,
    NudgeConfig,
    NudgeLevel,
    NudgeType,
)
from .persistence import (
    ReliabilityPersistence,
    SessionSettings,
    SessionReliabilityState,
    WorkspaceReliabilityData,
    UserReliabilityData,
)
from .patterns import PatternDetector
from .nudge import NudgeInjector, NudgeStrategy
from .policy_config import (
    generate_default_config_safe,
    get_default_policy_config_path,
    load_policy_config,
    resolve_policy_config_path,
)

__all__ = [
    # Plugin
    "ReliabilityPlugin",
    "ReliabilityPermissionWrapper",
    "create_plugin",
    "wrap_permission_plugin",
    # Core types
    "EscalationInfo",
    "EscalationRule",
    "FailureKey",
    "FailureRecord",
    "FailureSeverity",
    "ReliabilityConfig",
    "ToolReliabilityState",
    "TrustState",
    "classify_failure",
    # Model reliability types
    "ModelBehavioralProfile",
    "ModelSwitchConfig",
    "ModelSwitchStrategy",
    "ModelSwitchSuggestion",
    "ModelToolProfile",
    # Behavioral pattern types
    "BehavioralPattern",
    "BehavioralPatternType",
    "PatternDetectionConfig",
    "PatternSeverity",
    "ToolCall",
    # Persistence
    "ReliabilityPersistence",
    "SessionSettings",
    "SessionReliabilityState",
    "WorkspaceReliabilityData",
    "UserReliabilityData",
    # Pattern detection
    "PatternDetector",
    # Prerequisite policies
    "PrerequisitePolicy",
    # Nudge injection
    "Nudge",
    "NudgeConfig",
    "NudgeInjector",
    "NudgeLevel",
    "NudgeStrategy",
    "NudgeType",
    # Policy config file
    "generate_default_config_safe",
    "get_default_policy_config_path",
    "load_policy_config",
    "resolve_policy_config_path",
]
