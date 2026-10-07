"""Python notebook plugin with GPU support via Kaggle.

This plugin provides interactive Python notebook capabilities with:
- Local Jupyter kernel for quick iterations (no GPU)
- Kaggle backend for free GPU compute (30h/week)
- Lightning.ai backend as alternative (35h/month)
- Notebook Tool Bindings for tool access from notebook scripts

The model can execute Python code in a stateful environment, preserving
variables across multiple executions within a session.
"""

PLUGIN_KIND = "tool"

PLUGIN_TIER = "runner"

#: Modules this package imports lazily (imported when the subprocess kernel backend is built).
#: The pool template imports them before it forks, so a session's
#: slot inherits them shared instead of importing them privately.
#: A literal tuple of strings, read without importing the package;
#: nothing listed may read session-scoped state at import time.
#: See jaato_server/server/runner/template_preload.py.
PLUGIN_PRELOAD = (
    "jaato_server.shared.plugins.notebook.backends.subprocess_kernel",
    "jaato_server.shared.plugins.notebook.kernel_protocol",
)
from .plugin import NotebookPlugin, create_plugin
from .tool_stubs import ToolBridge, ToolExecutionError, generate_tools_module, generate_tool_signatures

__all__ = [
    'NotebookPlugin', 'create_plugin',
    'ToolBridge', 'ToolExecutionError', 'generate_tools_module', 'generate_tool_signatures',
]
