"""File-based policy configuration for the reliability plugin.

Loads ``PatternDetectionConfig`` overrides and ``PrerequisitePolicy``
definitions from a JSON file so that users can tune loop-prevention
rules without touching code.

Config file resolution (first found wins):
    1. ``.jaato/reliability-policies.json``  (workspace)
    2. ``~/.jaato/reliability-policies.json`` (user home)

Schema
------
::

    {
      "pattern_detection": {
        "repetitive_call_threshold": 3,
        "introspection_loop_threshold": 2,
        "max_reads_before_action": 5,
        "max_turn_duration_seconds": 120.0,
        "introspection_tool_names": ["list_tools", "get_tool_schemas"],
        "read_only_tools": ["readFile", "glob", "grep"],
        "action_tools": ["writeFile", "bash"],
        "announce_phrases": ["let me", "proceeding now"]
      },
      "prerequisite_policies": [
        {
          "policy_id": "plan_before_update",
          "prerequisite_tool": "createPlan",
          "gated_tools": ["updateStep"],
          "lookback_turns": 5,
          "nudge_templates": {
            "minor": ["direct", "Call createPlan before updateStep."],
            "moderate": ["direct", "You MUST createPlan before updateStep (violation #{count})."],
            "severe": ["interrupt", "BLOCKED: updateStep requires a plan. Call createPlan first."]
          },
          "expected_action_template": "Call {prerequisite_tool} before using {tool_name}"
        }
      ]
    }

All fields are optional.  Missing fields retain the built-in defaults.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    NudgeType,
    PatternDetectionConfig,
    PatternSeverity,
    PrerequisitePolicy,
)

logger = logging.getLogger(__name__)

# Sentinel config file name
POLICY_CONFIG_FILENAME = "reliability-policies.json"

# Maps JSON string keys to enum members
_NUDGE_TYPE_MAP = {
    "gentle": NudgeType.GENTLE_REMINDER,
    "direct": NudgeType.DIRECT_INSTRUCTION,
    "interrupt": NudgeType.INTERRUPT,
}

_SEVERITY_MAP = {
    "minor": PatternSeverity.MINOR,
    "moderate": PatternSeverity.MODERATE,
    "severe": PatternSeverity.SEVERE,
}


def resolve_policy_config_path(workspace_path: Optional[str] = None) -> Optional[Path]:
    """Find the first existing policy config file.

    Resolution order:
        1. ``<workspace>/.jaato/reliability-policies.json``
        2. ``~/.jaato/reliability-policies.json``

    Returns:
        Path to the config file, or None if no file exists.
    """
    candidates: List[Path] = []
    if workspace_path:
        candidates.append(Path(workspace_path) / ".jaato" / POLICY_CONFIG_FILENAME)
    candidates.append(Path.home() / ".jaato" / POLICY_CONFIG_FILENAME)

    for path in candidates:
        if path.is_file():
            return path

    return None


def get_default_policy_config_path(workspace_path: Optional[str] = None) -> Path:
    """Return the preferred path for a new policy config file.

    Defaults to the workspace-level location if *workspace_path* is set,
    otherwise falls back to the user-level location.
    """
    if workspace_path:
        return Path(workspace_path) / ".jaato" / POLICY_CONFIG_FILENAME
    return Path.home() / ".jaato" / POLICY_CONFIG_FILENAME


def load_policy_config(
    workspace_path: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> Tuple[Optional[PatternDetectionConfig], List[PrerequisitePolicy], List[str]]:
    """Load policy configuration from JSON file.

    Args:
        workspace_path: Workspace root for resolving the config file.
        config_path: Explicit path override (skips resolution).

    Returns:
        A 3-tuple of:
        - ``PatternDetectionConfig`` if the file contained ``pattern_detection``
          overrides, else None (caller keeps defaults).
        - List of ``PrerequisitePolicy`` objects parsed from
          ``prerequisite_policies``.
        - List of human-readable warning/error strings encountered during
          parsing (non-fatal).  An empty list means everything parsed cleanly.
    """
    path = config_path or resolve_policy_config_path(workspace_path)
    if path is None:
        return None, [], []

    warnings: List[str] = []

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [], [f"Cannot read {path}: {exc}"]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [], [f"Invalid JSON in {path}: {exc}"]

    if not isinstance(data, dict):
        return None, [], [f"{path}: root element must be a JSON object"]

    pattern_config = _parse_pattern_detection(data.get("pattern_detection"), warnings)
    policies = _parse_prerequisite_policies(data.get("prerequisite_policies"), warnings)

    if warnings:
        for w in warnings:
            logger.warning("reliability-policies.json: %s", w)

    return pattern_config, policies, warnings


def generate_default_config() -> str:
    """Return a pretty-printed JSON string with annotated defaults.

    Used when the user runs ``reliability policies edit`` and no config
    file exists yet — gives them a commented starting point.
    """
    default = {
        "pattern_detection": {
            "repetitive_call_threshold": 3,
            "introspection_loop_threshold": 2,
            "max_reads_before_action": 5,
            "max_turn_duration_seconds": 120.0,
            "introspection_tool_names": sorted(PatternDetectionConfig.introspection_tool_names.__func__()),
            "read_only_tools": sorted(PatternDetectionConfig.read_only_tools.__func__()),
            "action_tools": sorted(PatternDetectionConfig.action_tools.__func__()),
            "announce_phrases": PatternDetectionConfig.announce_phrases.__func__(),
        },
        "prerequisite_policies": [
            {
                "policy_id": "example_plan_before_update",
                "prerequisite_tool": "createPlan",
                "gated_tools": ["updateStep"],
                "lookback_turns": 5,
                "nudge_templates": {
                    "minor": ["direct", "Call {prerequisite_tool} before using {tool_name}."],
                    "moderate": ["direct", "You MUST call {prerequisite_tool} before {tool_name} (violation #{count})."],
                    "severe": ["interrupt", "BLOCKED: {tool_name} requires a plan. Call {prerequisite_tool} first."],
                },
                "expected_action_template": "Call {prerequisite_tool} before using {tool_name}",
            },
        ],
    }
    return json.dumps(default, indent=2) + "\n"


def generate_default_config_safe() -> str:
    """Return a pretty-printed JSON string with defaults.

    Unlike ``generate_default_config`` this does not access field
    default_factory internals and is safe to call in any context.
    """
    defaults = PatternDetectionConfig()
    default = {
        "pattern_detection": {
            "repetitive_call_threshold": defaults.repetitive_call_threshold,
            "introspection_loop_threshold": defaults.introspection_loop_threshold,
            "max_reads_before_action": defaults.max_reads_before_action,
            "max_turn_duration_seconds": defaults.max_turn_duration_seconds,
            "introspection_tool_names": sorted(defaults.introspection_tool_names),
            "read_only_tools": sorted(defaults.read_only_tools),
            "action_tools": sorted(defaults.action_tools),
            "announce_phrases": list(defaults.announce_phrases),
        },
        "prerequisite_policies": [
            {
                "policy_id": "example_plan_before_update",
                "prerequisite_tool": "createPlan",
                "gated_tools": ["updateStep"],
                "lookback_turns": 5,
                "nudge_templates": {
                    "minor": ["direct", "Call {prerequisite_tool} before using {tool_name}."],
                    "moderate": ["direct", "You MUST call {prerequisite_tool} before {tool_name} (violation #{count})."],
                    "severe": ["interrupt", "BLOCKED: {tool_name} requires a plan. Call {prerequisite_tool} first."],
                },
                "expected_action_template": "Call {prerequisite_tool} before using {tool_name}",
            },
        ],
    }
    return json.dumps(default, indent=2) + "\n"


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------


def _parse_pattern_detection(
    section: Any,
    warnings: List[str],
) -> Optional[PatternDetectionConfig]:
    """Parse the ``pattern_detection`` section into a PatternDetectionConfig."""
    if section is None:
        return None

    if not isinstance(section, dict):
        warnings.append("'pattern_detection' must be an object; ignored")
        return None

    kwargs: Dict[str, Any] = {}

    int_fields = {
        "repetitive_call_threshold": (1, 100),
        "introspection_loop_threshold": (1, 100),
        "max_reads_before_action": (1, 100),
    }
    for field, (lo, hi) in int_fields.items():
        if field in section:
            val = section[field]
            if isinstance(val, int) and lo <= val <= hi:
                kwargs[field] = val
            else:
                warnings.append(f"'{field}' must be an integer in [{lo}, {hi}]; got {val!r}")

    float_fields = {
        "max_turn_duration_seconds": (1.0, 3600.0),
    }
    for field, (lo, hi) in float_fields.items():
        if field in section:
            val = section[field]
            if isinstance(val, (int, float)) and lo <= val <= hi:
                kwargs[field] = float(val)
            else:
                warnings.append(f"'{field}' must be a number in [{lo}, {hi}]; got {val!r}")

    set_fields = ["introspection_tool_names", "read_only_tools", "action_tools"]
    for field in set_fields:
        if field in section:
            val = section[field]
            if isinstance(val, list) and all(isinstance(v, str) for v in val):
                kwargs[field] = set(val)
            else:
                warnings.append(f"'{field}' must be a list of strings; got {type(val).__name__}")

    if "announce_phrases" in section:
        val = section["announce_phrases"]
        if isinstance(val, list) and all(isinstance(v, str) for v in val):
            kwargs["announce_phrases"] = val
        else:
            warnings.append("'announce_phrases' must be a list of strings")

    if not kwargs:
        return None

    return PatternDetectionConfig(**kwargs)


def _parse_prerequisite_policies(
    section: Any,
    warnings: List[str],
) -> List[PrerequisitePolicy]:
    """Parse the ``prerequisite_policies`` array into PrerequisitePolicy objects."""
    if section is None:
        return []

    if not isinstance(section, list):
        warnings.append("'prerequisite_policies' must be an array; ignored")
        return []

    policies: List[PrerequisitePolicy] = []
    for i, entry in enumerate(section):
        policy = _parse_single_policy(entry, i, warnings)
        if policy is not None:
            policies.append(policy)

    return policies


def _parse_single_policy(
    entry: Any,
    index: int,
    warnings: List[str],
) -> Optional[PrerequisitePolicy]:
    """Parse a single prerequisite policy object."""
    prefix = f"prerequisite_policies[{index}]"

    if not isinstance(entry, dict):
        warnings.append(f"{prefix}: must be an object; skipped")
        return None

    # Required fields
    policy_id = entry.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id:
        warnings.append(f"{prefix}: 'policy_id' is required and must be a non-empty string; skipped")
        return None

    prerequisite_tool = entry.get("prerequisite_tool")
    if not isinstance(prerequisite_tool, str) or not prerequisite_tool:
        warnings.append(f"{prefix}: 'prerequisite_tool' is required and must be a non-empty string; skipped")
        return None

    gated_tools_raw = entry.get("gated_tools")
    if not isinstance(gated_tools_raw, list) or not gated_tools_raw:
        warnings.append(f"{prefix}: 'gated_tools' is required and must be a non-empty list; skipped")
        return None
    if not all(isinstance(t, str) for t in gated_tools_raw):
        warnings.append(f"{prefix}: 'gated_tools' must contain only strings; skipped")
        return None
    gated_tools = set(gated_tools_raw)

    # Optional fields
    lookback_turns = entry.get("lookback_turns", 2)
    if not isinstance(lookback_turns, int) or lookback_turns < 0:
        warnings.append(f"{prefix}: 'lookback_turns' must be a non-negative integer; using default 2")
        lookback_turns = 2

    expected_action_template = entry.get(
        "expected_action_template",
        "Call {prerequisite_tool} before using {tool_name}",
    )
    if not isinstance(expected_action_template, str):
        warnings.append(f"{prefix}: 'expected_action_template' must be a string; using default")
        expected_action_template = "Call {prerequisite_tool} before using {tool_name}"

    # Nudge templates
    nudge_templates: Dict[PatternSeverity, Tuple[NudgeType, str]] = {}
    raw_nudge = entry.get("nudge_templates")
    if raw_nudge is not None:
        if isinstance(raw_nudge, dict):
            for sev_key, value in raw_nudge.items():
                severity = _SEVERITY_MAP.get(sev_key.lower())
                if severity is None:
                    warnings.append(
                        f"{prefix}.nudge_templates: unknown severity '{sev_key}'; "
                        f"expected one of: {', '.join(_SEVERITY_MAP)}"
                    )
                    continue
                if not isinstance(value, list) or len(value) != 2:
                    warnings.append(
                        f"{prefix}.nudge_templates.{sev_key}: "
                        "must be a [nudge_type, message] pair"
                    )
                    continue
                nudge_type_str, message = value
                nudge_type = _NUDGE_TYPE_MAP.get(str(nudge_type_str).lower())
                if nudge_type is None:
                    warnings.append(
                        f"{prefix}.nudge_templates.{sev_key}: "
                        f"unknown nudge type '{nudge_type_str}'; "
                        f"expected one of: {', '.join(_NUDGE_TYPE_MAP)}"
                    )
                    continue
                if not isinstance(message, str):
                    warnings.append(
                        f"{prefix}.nudge_templates.{sev_key}: "
                        "message must be a string"
                    )
                    continue
                nudge_templates[severity] = (nudge_type, message)
        else:
            warnings.append(f"{prefix}: 'nudge_templates' must be an object")

    return PrerequisitePolicy(
        policy_id=policy_id,
        prerequisite_tool=prerequisite_tool,
        gated_tools=gated_tools,
        lookback_turns=lookback_turns,
        nudge_templates=nudge_templates,
        expected_action_template=expected_action_template,
        owner_plugin="policy_config",
    )
