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
    ReliabilityConfig,
    ToolReliabilityState,
    TrustState,
    classify_failure,
    # Model reliability types
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
)
from .persistence import (
    ReliabilityPersistence,
    SessionSettings,
    SessionReliabilityState,
    WorkspaceReliabilityData,
    UserReliabilityData,
)
from .patterns import PatternDetector

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
]
