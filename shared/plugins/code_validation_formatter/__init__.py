# shared/plugins/code_validation_formatter/__init__.py
"""Code validation formatter plugin for LSP diagnostics on output code blocks.

This plugin validates code blocks in model output via LSP and appends
diagnostic warnings/errors, enabling the model to see issues before
code is written to files.

Example:
    from shared.plugins.code_validation_formatter import create_plugin

    validator = create_plugin()
    validator.set_lsp_plugin(lsp_plugin)
    validator.initialize({"enabled": True})

    # Use in formatter pipeline
    pipeline.register(validator)

    # Or use standalone
    formatted = validator.format_output(text_with_code_blocks)
"""

from .plugin import CodeValidationFormatterPlugin, create_plugin

__all__ = [
    "CodeValidationFormatterPlugin",
    "create_plugin",
]
