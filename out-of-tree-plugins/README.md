# Out-of-Tree Plugins

This directory contains example plugins that are built as **external installable packages**, demonstrating how third-party developers can extend jaato without modifying the core repository.

## What are Out-of-Tree Plugins?

Out-of-tree plugins are plugins that:
- Live outside the `shared/plugins/` directory
- Are packaged as standalone Python packages with `pyproject.toml`
- Use Python entry points for automatic discovery
- Can be installed via `pip install`
- Are developed and distributed independently

This is the recommended approach for:
- Third-party plugin developers
- Community contributions
- Organization-specific plugins
- Commercial plugin packages

## Examples in This Directory

### moon-phase/

A complete moon phase calculator plugin that demonstrates:
- Full plugin protocol implementation
- Tool schema definition
- Proper packaging with entry points
- Comprehensive testing
- Documentation

**Built to verify:** That the API documentation in `docs/api/` is sufficient for external developers to build plugins.

See `moon-phase/README.md` for usage instructions.

## Plugin Verification Report

See `PLUGIN_VERIFICATION_REPORT.md` for a detailed analysis of:
- Documentation sufficiency for external plugin development
- Identified documentation gaps
- Recommendations for improvements
- Test results

## Creating Your Own Out-of-Tree Plugin

Follow the guide in `docs/api/guides/tool-plugins.html` (Step 6: Package with pyproject.toml).

The `moon-phase/` example provides a complete reference implementation you can use as a template.

### Quick Start

1. **Create package structure:**
   ```bash
   mkdir -p my-plugin/src/my_plugin
   ```

2. **Implement plugin:** See `moon-phase/src/jaato_moon_phase/plugin.py`

3. **Add entry point in pyproject.toml:**
   ```toml
   [project.entry-points."jaato.plugins"]
   my_plugin = "my_plugin:create_plugin"
   ```

4. **Install and use:**
   ```bash
   pip install -e .
   ```

## Directory Structure

```
out-of-tree-plugins/
├── README.md                           # This file
├── PLUGIN_VERIFICATION_REPORT.md       # Verification results
└── moon-phase/                         # Example plugin package
    ├── pyproject.toml
    ├── README.md
    ├── test_moon_phase.py
    └── src/
        └── jaato_moon_phase/
            ├── __init__.py
            └── plugin.py
```

## Contributing

If you've built an interesting out-of-tree plugin and would like to share it as an example, please submit a PR adding your plugin to this directory!
