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
    EscalationInfo,
    EscalationRule,
    FailureKey,
    FailureRecord,
    FailureSeverity,
    ReliabilityConfig,
    ToolReliabilityState,
    TrustState,
    classify_failure,
)

__all__ = [
    # Plugin
    "ReliabilityPlugin",
    "ReliabilityPermissionWrapper",
    "create_plugin",
    "wrap_permission_plugin",
    # Types
    "EscalationInfo",
    "EscalationRule",
    "FailureKey",
    "FailureRecord",
    "FailureSeverity",
    "ReliabilityConfig",
    "ToolReliabilityState",
    "TrustState",
    "classify_failure",
]
