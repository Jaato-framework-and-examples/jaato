# Documentation Gaps Found During Plugin Development

This document lists gaps and issues found in the jaato API documentation while building the moon phase calculator plugin as an external developer.

## Issues Found

### 1. Missing Dependency Information for Out-of-Tree Plugins

**Location:** `docs/api/guides/tool-plugins.html` - Step 6: Package with pyproject.toml

**Issue:** The documentation shows how to create a `pyproject.toml` for an out-of-tree plugin with entry points, but does not specify that the plugin needs jaato itself as a dependency.

**Impact:**
- Plugins cannot import types like `ToolSchema` from `shared.plugins.model_provider.types`
- Testing standalone plugins fails because `shared` module is not available
- Unclear whether plugins should depend on jaato or if there's a separate types package

**What's Missing:**
```toml
[project]
dependencies = [
    "jaato>=1.0.0",  # This is not mentioned in the docs
]
```

**Recommendation:** The documentation should clarify:
1. Should out-of-tree plugins declare jaato as a dependency?
2. Is there a separate `jaato-types` or `jaato-plugin-sdk` package for type definitions?
3. How should plugins be tested without installing jaato?
4. Example of a complete working pyproject.toml with all necessary dependencies

### 2. Import Path Ambiguity

**Location:** Multiple locations in the documentation

**Issue:** The documentation shows different import patterns for the same types:

From `docs/api/api-reference/types.html`:
```python
from shared import ToolSchema, Message, Part, ...
```

From `docs/api/guides/tool-plugins.html`:
```python
from shared.plugins.model_provider.types import ToolSchema
```

**Impact:**
- External developers don't know which import path to use
- Inconsistency makes it unclear what the correct pattern is
- Both may work, but which is preferred?

**Recommendation:**
- Standardize on one import pattern throughout documentation
- Explain if both are valid and when to use each
- Show the canonical import location for each type

### 3. Testing Guidance for External Plugins

**Location:** `docs/api/guides/tool-plugins.html` - Step 7: Testing Your Plugin

**Issue:** The testing example assumes the plugin is in `shared/plugins/` and can import from `shared` directly:

```python
from shared.plugins.my_plugin import create_plugin, PLUGIN_INFO
```

This doesn't work for out-of-tree plugins installed as separate packages.

**What's Missing:**
- How to test a plugin before installing it
- How to set up a test environment with jaato available
- Whether to mock types or require jaato installation
- Example test that works for external packages

**Recommendation:** Provide two test examples:
1. Testing in-tree plugins (current example)
2. Testing out-of-tree plugins with proper setup

### 4. Plugin Discovery Mechanism Details

**Location:** `docs/api/core-concepts/plugins.html` - Plugin Discovery section

**Issue:** While the documentation mentions entry points and shows the pyproject.toml format, it doesn't explain:
- When plugin discovery happens (at import time? at runtime?)
- How to verify a plugin was discovered correctly
- What happens if entry point is malformed
- How to debug discovery issues

**Recommendation:** Add troubleshooting section with:
- How to verify plugin is discoverable
- Common discovery errors and solutions
- Debug logging for plugin discovery

### 5. Minimal Example That Works End-to-End

**Gap:** The documentation provides excellent reference material and step-by-step guides, but lacks a complete, minimal working example that can be copy-pasted and run immediately.

**Recommendation:** Add a "Quick Start: External Plugin" guide with:
- Complete file structure
- All files with full content (no ellipsis)
- Installation commands
- Test script that actually runs
- Expected output

## What Worked Well

Despite these gaps, the documentation was generally comprehensive and well-structured:

1. **Clear Structure:** The progression from concepts → guides → API reference is logical
2. **Good Examples:** Code examples are helpful and show multiple patterns
3. **Type System Documentation:** The provider-agnostic type system is well documented
4. **Plugin Protocol:** The required vs optional methods are clearly explained
5. **Tool Schema Format:** JSON Schema format is well documented with examples

## Workarounds Used

For this plugin development, I used the following workaround:

1. Added the jaato repository path to Python path for testing:
```python
import sys
import os
# Add jaato repo to path
jaato_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../jaato'))
sys.path.insert(0, jaato_path)
```

This allows importing from `shared` during testing without requiring jaato to be installed as a package.

## Overall Assessment

The documentation is **good enough** for an experienced Python developer to build a plugin, but requires:
- Some trial and error
- Understanding of Python packaging
- Ability to work around missing dependency information
- Inferring some details not explicitly stated

For a novice developer, the missing dependency and testing guidance would be blocking issues.
