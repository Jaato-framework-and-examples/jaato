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

Lazy loading: All imports are deferred via __getattr__ to allow partial
installation (e.g., TUI distribution only needs models and channels).
"""

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

_LAZY_IMPORTS = {
    # Models (from SDK)
    "StepStatus": ("jaato_sdk.plugins.todo.models", "StepStatus"),
    "PlanStatus": ("jaato_sdk.plugins.todo.models", "PlanStatus"),
    "TodoStep": ("jaato_sdk.plugins.todo.models", "TodoStep"),
    "TodoPlan": ("jaato_sdk.plugins.todo.models", "TodoPlan"),
    "ProgressEvent": ("jaato_sdk.plugins.todo.models", "ProgressEvent"),
    # Storage
    "TodoStorage": (".storage", "TodoStorage"),
    "InMemoryStorage": (".storage", "InMemoryStorage"),
    "FileStorage": (".storage", "FileStorage"),
    "HybridStorage": (".storage", "HybridStorage"),
    "create_storage": (".storage", "create_storage"),
    # Reporters (Channels) — ABC from SDK, concrete from local
    "TodoReporter": ("jaato_sdk.plugins.todo.channels", "TodoReporter"),
    "ConsoleReporter": (".channels", "ConsoleReporter"),
    "WebhookReporter": (".channels", "WebhookReporter"),
    "FileReporter": (".channels", "FileReporter"),
    "MemoryReporter": (".channels", "MemoryReporter"),
    "MultiReporter": (".channels", "MultiReporter"),
    "create_reporter": (".channels", "create_reporter"),
    # Config
    "TodoConfig": (".config_loader", "TodoConfig"),
    "ConfigValidationError": (".config_loader", "ConfigValidationError"),
    "load_config": (".config_loader", "load_config"),
    "validate_config": (".config_loader", "validate_config"),
    "create_default_config": (".config_loader", "create_default_config"),
    # Plugin
    "TodoPlugin": (".plugin", "TodoPlugin"),
    "create_plugin": (".plugin", "create_plugin"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        package = __name__ if module_path.startswith(".") else None
        module = importlib.import_module(module_path, package)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    # Reporters (Channels)
    'TodoReporter',
    'ConsoleReporter',
    'WebhookReporter',
    'FileReporter',
    'MemoryReporter',
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
