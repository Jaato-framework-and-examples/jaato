# shared/plugins/notebook_output_formatter/__init__.py
"""Notebook output formatter plugin.

Transforms <notebook-cell> markers into a structured table layout for
Jupyter-style notebook cell presentation. The layout is a borderless
2-column table where the first column contains cell labels (In[n]:, Out[n]:)
and the second column contains the cell content.

Borders are presentation layer and should be managed by the client.
Code content uses fenced blocks so code_block_formatter can apply syntax highlighting.
"""

from .plugin import NotebookOutputFormatterPlugin, create_plugin

__all__ = ["NotebookOutputFormatterPlugin", "create_plugin"]
