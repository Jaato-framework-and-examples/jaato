"""The ``decide`` tool: ask a bound decision model typed questions.

Step 4 of docs/design/decision-models.md (§8.1).  The tool names a
``decision_models:`` entry of the calling session's profile, never a
model, and returns the decision model's typed answers.
"""

PLUGIN_KIND = "tool"
PLUGIN_TIER = "runner"

from .plugin import DecisionPlugin, create_plugin  # noqa: E402

__all__ = ["DecisionPlugin", "create_plugin", "PLUGIN_KIND", "PLUGIN_TIER"]
