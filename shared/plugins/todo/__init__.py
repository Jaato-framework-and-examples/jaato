"""TODO plugin for plan registration and progress reporting.

This plugin enables LLMs to register execution plans with ordered steps
and progressively report progress through configurable transport protocols.

Supports three reporter types (matching the permissions plugin pattern):
- ConsoleReporter: Renders progress to terminal with visual indicators
- WebhookReporter: Sends progress events to HTTP endpoints
- FileReporter: Writes progress to filesystem for external monitoring

Example usage:

    from shared.plugins.todo import TodoPlugin, create_plugin

    # Create and initialize plugin
    plugin = create_plugin()
    plugin.initialize({
        "reporter_type": "console",
        "storage_type": "memory",
    })

    # Use via tool executors (for LLM)
    executors = plugin.get_executors()
    result = executors["createPlan"]({
        "title": "Refactor auth module",
        "steps": ["Analyze code", "Design changes", "Implement", "Test"]
    })

    # Or use programmatically
    plan = plugin.create_plan(
        title="Deploy feature",
        steps=["Run tests", "Build", "Deploy", "Verify"]
    )
"""

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

from .models import (
    StepStatus,
    PlanStatus,
    TodoStep,
    TodoPlan,
    ProgressEvent,
)
from .storage import (
    TodoStorage,
    InMemoryStorage,
    FileStorage,
    HybridStorage,
    create_storage,
)
from .actors import (
    TodoReporter,
    ConsoleReporter,
    WebhookReporter,
    FileReporter,
    MultiReporter,
    create_reporter,
)
from .config_loader import (
    TodoConfig,
    ConfigValidationError,
    load_config,
    validate_config,
    create_default_config,
)
from .plugin import TodoPlugin, create_plugin

__all__ = [
    # Models
    'StepStatus',
    'PlanStatus',
    'TodoStep',
    'TodoPlan',
    'ProgressEvent',
    # Storage
    'TodoStorage',
    'InMemoryStorage',
    'FileStorage',
    'HybridStorage',
    'create_storage',
    # Reporters (Actors)
    'TodoReporter',
    'ConsoleReporter',
    'WebhookReporter',
    'FileReporter',
    'MultiReporter',
    'create_reporter',
    # Config
    'TodoConfig',
    'ConfigValidationError',
    'load_config',
    'validate_config',
    'create_default_config',
    # Plugin
    'TodoPlugin',
    'create_plugin',
]
