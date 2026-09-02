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
from .plugin import NotebookPlugin, create_plugin
from .tool_stubs import ToolBridge, ToolExecutionError, generate_tools_module, generate_tool_signatures

__all__ = [
    'NotebookPlugin', 'create_plugin',
    'ToolBridge', 'ToolExecutionError', 'generate_tools_module', 'generate_tool_signatures',
]
