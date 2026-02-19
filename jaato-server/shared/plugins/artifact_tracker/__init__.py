"""Artifact Tracker plugin for tracking created/modified artifacts.

This plugin helps the model keep track of artifacts (documents, tests, configs,
etc.) it creates or modifies during a session. The key feature is the system
instructions that remind the model to review related artifacts when making
changes, ensuring consistency across related files.

Example usage:

    from shared.plugins.artifact_tracker import ArtifactTrackerPlugin, create_plugin

    # Create and initialize plugin
    plugin = create_plugin()
    plugin.initialize({
        "storage_path": ".jaato/.artifact_tracker.json",  # default location
    })

    # Use via tool executors (for LLM)
    executors = plugin.get_executors()
    result = executors["trackArtifact"]({
        "path": "docs/README.md",
        "artifact_type": "document",
        "description": "Main project documentation",
        "related_to": ["src/main.py", "config.json"],
    })

    # Check what artifacts are related to a file before modifying it
    related = executors["checkRelated"]({"path": "src/main.py"})
"""

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

from .models import (
    ArtifactType,
    ReviewStatus,
    ArtifactRecord,
    ArtifactRegistry,
)
from .plugin import ArtifactTrackerPlugin, create_plugin

__all__ = [
    # Models
    'ArtifactType',
    'ReviewStatus',
    'ArtifactRecord',
    'ArtifactRegistry',
    # Plugin
    'ArtifactTrackerPlugin',
    'create_plugin',
]
