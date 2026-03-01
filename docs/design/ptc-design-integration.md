# PTC Design Integration — Codebase Analysis & Brainstorm

## TL;DR

We already have **~70% of the PTC architecture** built. The notebook plugin IS the
PTC executor (with a different name), the ToolExecutor IS the bridge, the
CodeAnalyzer IS the security layer, and the introspection plugin IS the tool
discovery mechanism. What we're missing is the **tool stubs layer** (the
`jaato.tools` importable module) and the **wiring** to connect it into the notebook
execution namespace. This is not a green-field build — it's an integration task
with a thin new layer on top of existing infrastructure.

---

## Component-by-Component Mapping

### 1. PTC Executor → ALREADY EXISTS: Notebook Plugin

The proposed "Embedded IPython Executor" is almost exactly what
`shared/plugins/notebook/` already provides:

| PTC Proposal | What We Have | Gap |
|---|---|---|
| Embedded IPython InteractiveShell | `LocalJupyterBackend` using `exec()` with namespace isolation | No gap (exec+AST is actually lighter than IPython) |
| Pre-inject `jaato.tools` into namespace | Namespace `{'__name__': '__main__', '__builtins__': __builtins__}` | Need to inject tool stubs module |
| Capture stdout/stderr/return value | `redirect_stdout/redirect_stderr` + `_execute_and_capture()` (AST-based last-expression capture mimicking Jupyter) | No gap |
| Timeout enforcement | `timeout_seconds` parameter on `execute()` | Shell commands have it; need to add for Python exec too |
| Output truncation | `MAX_OUTPUT_LENGTH = 10000` | No gap |
| Clean namespace per invocation | `reset_notebook()` or create new notebook | No gap |
| `await` support at top level | Not supported (exec is sync) | Gap — but see below |

**Key insight**: The PTC design proposes IPython primarily for `await` support at
top level. But our tool stubs don't *need* to be async — they can be sync wrappers
that call through to the ToolExecutor (which is sync). The IPython dependency is
unnecessary baggage. We can keep using `exec()` with our existing
`_execute_and_capture()`.

If async is truly needed later, we can add it as a backend option without
refactoring — the `NotebookBackend` ABC already supports multiple backends
(local, Kaggle). An "IPythonBackend" could be added alongside without touching
the existing local backend.

**Recommendation**: **Don't introduce IPython**. The existing local backend is
lightweight, zero-dependency, battle-tested, and sufficient. Add tool stubs
injection to the existing namespace setup and call it done.

---

### 2. Tool Bridge → ALREADY EXISTS: ToolExecutor

The proposed `ToolBridge.execute(tool_name, arguments)` maps directly to
`ToolExecutor.execute(name, args)` in `ai_tool_runner.py`:

| PTC Proposal | What We Have |
|---|---|
| `bridge.execute(tool_name, args) → result` | `executor.execute(name, args) → (success, result)` |
| Pre-execution hooks (validate, permissions, rate limit) | `PermissionPlugin.check_permission()` |
| Post-execution hooks (filter, log) | Permission metadata injection, ledger recording |
| Rejection (raise exception in script) | Returns `(False, {'error': ...})` |
| Approval flow (pause for human approval) | Permission plugin with interactive prompts |
| Call logging | Ledger `_record('permission-check', ...)` |
| Route to MCP or custom handlers | Registry fallback: tries `_map`, then `registry.get_plugin_for_tool()` |

**Gap**: The bridge needs a *thin sync wrapper* that:
1. Accepts tool name + args dict
2. Calls `ToolExecutor.execute()`
3. Raises an exception on failure (so the script's try/except works naturally)
4. Returns just the result dict (strips the success bool)

This is literally ~10 lines of code.

---

### 3. Tool Stub Generator → NEW (but small)

This is the one genuinely new piece. We need to auto-generate Python functions
from `ToolSchema` objects. But the building blocks exist:

**What we have**:
- `PluginRegistry.get_exposed_tool_schemas()` → all registered `ToolSchema` objects
- Each `ToolSchema` has: `name`, `description`, `parameters` (JSON Schema dict)
- The introspection plugin already does a version of this — `list_tools()` and
  `get_tool_schemas()` present tool info to the model as structured data

**What we need**:
- A function that takes a `List[ToolSchema]` and produces a Python module
  (`types.ModuleType`) with one function per tool
- JSON Schema → Python type mapping (simple dict: `string→str`, `integer→int`, etc.)
- Each generated function calls `bridge.execute(tool_name, args_dict)`

**Design choice**: Generate the module dynamically with `types.ModuleType` rather
than writing temp files. It's cleaner, no file cleanup needed, and we control the
namespace completely.

```python
import types
from typing import Any, Dict

def generate_tools_module(
    schemas: list,
    bridge: 'ToolBridge',
) -> types.ModuleType:
    """Generate a Python module with one function per tool.

    Each function calls bridge.execute() under the hood.
    """
    module = types.ModuleType('jaato_tools')
    module.__doc__ = "Auto-generated tool functions for programmatic access."

    for schema in schemas:
        fn = _make_tool_function(schema, bridge)
        setattr(module, schema.name, fn)

    # Add discovery helper
    module.list_tools = lambda: {s.name: s.description for s in schemas}

    return module
```

---

### 4. Security Layer → ALREADY EXISTS: CodeAnalyzer

The proposed namespace restrictions map directly to
`shared/plugins/notebook/code_analyzer.py`:

| PTC Proposal | What We Have |
|---|---|
| Block `os`, `subprocess`, `shutil`, `socket`, ... imports | `DANGEROUS_MODULES` dict with risk levels |
| Block `open`, `exec`, `eval`, `compile`, `__import__` | `DANGEROUS_CALLS` dict |
| Block `os.system()`, `subprocess.run()`, etc. | `DANGEROUS_ATTRS` dict (50+ patterns) |
| External path detection | `EXTERNAL_PATH_PATTERNS` + `_is_external_path()` |
| Configurable strictness | `SandboxMode` enum: `DISABLED`, `WARN`, `BLOCK_CRITICAL`, `STRICT` |
| Code review mode (present to user before exec) | Permission plugin already does this for tool calls |

The CodeAnalyzer is *more thorough* than what the PTC design proposes. It does
full AST walking, import alias resolution, attribute chain analysis, and
introspection attack detection (`__builtins__`, `__subclasses__`, `__globals__`).

**Recommendation**: Reuse CodeAnalyzer as-is. Maybe adjust the risk level for
`open()` when PTC mode is active (since we *want* scripts to use jaato.tools
instead of direct file I/O).

---

### 5. Integration with Agent Loop → ALREADY EXISTS: Plugin Pattern

The proposal's "Option A: PTC as a tool" is exactly how jaato works — every
capability is a tool plugin. The `notebook_execute` tool already does this:

```python
# Existing: model calls notebook_execute(code="...")
ToolSchema(
    name="notebook_execute",
    description="Execute Python code in a persistent notebook environment.",
    parameters={"code": {"type": "string"}, "notebook_id": {"type": "string"}},
)
```

**For PTC, we either**:
1. **Extend `notebook_execute`** to auto-inject tool stubs into the namespace, or
2. **Add a separate `ptc_execute` tool** alongside `notebook_execute`

Option 1 is simpler and avoids tool proliferation. The notebook plugin already
has the concept of preparing a namespace — we just need to add tool stubs to it.

---

### 6. System Prompt Integration → ALREADY EXISTS: Plugin `get_system_instructions()`

Each plugin provides system instructions via `get_system_instructions()`. The
notebook plugin already does this (lines 361-405 of plugin.py). For PTC, we'd
extend these instructions to include:

- The tool function signatures (auto-generated from schemas)
- Usage patterns (loops, aggregation, cross-referencing)
- What goes through tool stubs vs. direct Python

The introspection plugin already generates tool listings dynamically — we can
reuse that logic.

---

### 7. Observability → ALREADY EXISTS: Multiple Systems

| PTC Proposal | What We Have |
|---|---|
| `PTCExecutionLog.script` | `ExecutionResult` already captures the code |
| `PTCExecutionLog.tool_calls` | ToolExecutor logs to `TokenLedger` |
| `PTCExecutionLog.total_duration` | `ExecutionResult.duration_seconds` |
| `PTCExecutionLog.output` | `ExecutionResult.outputs` (list of `CellOutput`) |
| `PTCExecutionLog.error` | `ExecutionResult.error_name`, `error_message`, `traceback` |
| Token savings estimate | New — but can compute from tool_calls count × avg tool schema tokens |
| Per-tool-call timing | ToolExecutor already tracks this in ledger |
| OTel spans | `jaato.tool` spans already exist |

---

## What's Actually New

When we strip away everything we already have, the net-new work is:

### A. Tool Stub Generator (~100-150 lines)

A module that takes `List[ToolSchema]` and produces a `types.ModuleType` with
callable functions. Key decisions:

1. **Sync functions** (not async) — the ToolExecutor is sync, the exec environment
   is sync, and we avoid the IPython dependency entirely
2. **Minimal type hints** — JSON Schema → Python type mapping for documentation,
   but at runtime everything is `dict` anyway
3. **Error propagation** — stub raises `ToolExecutionError` on failure so scripts
   can use try/except naturally
4. **`list_tools()` helper** — returns available tools with descriptions (mirrors
   what introspection plugin already does, but callable from within the script)

### B. Namespace Injection (~20-30 lines)

In `LocalJupyterBackend.create_notebook()` (or a new method), inject the tool
stubs module into the namespace:

```python
namespace = {
    '__name__': '__main__',
    '__builtins__': __builtins__,
    'tools': tools_module,       # jaato.tools
    'json': __import__('json'),  # Commonly needed in scripts
    're': __import__('re'),
}
```

### C. Bridge Wrapper (~10-15 lines)

A thin adapter between the stub functions and `ToolExecutor.execute()`:

```python
class ToolBridge:
    def __init__(self, executor: ToolExecutor):
        self.executor = executor
        self.call_log = []

    def call(self, tool_name: str, args: dict) -> Any:
        success, result = self.executor.execute(tool_name, args)
        if not success:
            raise ToolExecutionError(tool_name, result.get('error', 'Unknown error'))
        return result
```

### D. System Prompt Extension (~50 lines)

Extend the notebook plugin's `get_system_instructions()` to include PTC-specific
guidance when tool stubs are available. Auto-generate the tool function signatures
from the schemas.

### E. Configuration (~10 lines)

An env var like `JAATO_PTC_ENABLED=true` to control whether tool stubs are
injected into notebook namespaces.

**Total estimated new code: ~200-250 lines** (not including tests).

---

## Open Questions — Recommendations

### 1. Should PTC scripts call sub-agents?

**Recommendation: Not in v1.** Sub-agents need their own `JaatoSession` (via
`runtime.create_session()`). Exposing this in a script creates complex lifecycle
management. Start with tool calls only. If needed later, expose a
`tools.delegate(prompt, model=...)` function that creates a subagent session
internally.

### 2. Persistent state across PTC invocations?

**Already solved.** The notebook plugin already supports persistent state via
`notebook_id`. Variables survive across `notebook_execute` calls in the same
notebook. The tool stubs would persist too since they're in the namespace.

### 3. Reusable functions across PTC calls?

**Already solved.** Same as above — functions defined in one `notebook_execute`
call persist in the notebook namespace for subsequent calls.

### 4. How to handle streaming?

**Recommendation: Don't.** Tool stubs return the final result. If a tool
supports streaming (`:stream` suffix), the stub calls the non-streaming variant.
Streaming within a script is an optimization that adds complexity for marginal
benefit — the whole point of PTC is that intermediate results stay in the script.

---

## Integration Path

### Phase 1: Stub Generator + Bridge (MVP)

1. Add `shared/plugins/notebook/tool_stubs.py`:
   - `generate_tools_module(schemas, bridge)` → `types.ModuleType`
   - `ToolBridge` wrapper around `ToolExecutor`
   - `ToolExecutionError` exception class

2. Modify `NotebookPlugin._execute_notebook()` to inject tool stubs into the
   notebook namespace when `JAATO_PTC_ENABLED=true`

3. Extend `get_system_instructions()` to include tool function signatures

4. Tests: stub generation for various JSON Schema types, bridge error handling,
   end-to-end script execution with mocked tools

### Phase 2: Observability

5. Add `call_log` tracking to `ToolBridge` — record each tool call with timing
6. Include call_log in the `notebook_execute` result dict
7. Add token savings estimate (count tool calls × avg schema size)

### Phase 3: Security Hardening

8. Adjust `CodeAnalyzer` for PTC context (e.g., warn when script uses `open()`
   instead of `tools.file_read()`)
9. Add PTC-specific risk patterns (e.g., `tools` module attribute tampering)
10. Consider: should sandbox mode default to `BLOCK_CRITICAL` for PTC?

---

## What We're NOT Doing (and why)

| Proposal | Decision | Rationale |
|---|---|---|
| IPython dependency | Skip | exec() + AST capture already works; no `await` needed since ToolExecutor is sync |
| Separate `execute_script` tool | Skip | Reuse `notebook_execute` — it's the same tool with a different namespace |
| `sys.modules['jaato.tools']` registration | Skip | Inject into namespace directly; `import` is a security risk we'd rather avoid |
| Async tool stubs | Skip | ToolExecutor is sync; adding async adds complexity and requires IPython |
| Namespace restriction (blocking builtins) | Defer | CodeAnalyzer already warns/blocks; hard restrictions break legitimate scripts |
| Separate `PTC` plugin | Skip | Notebook plugin already has the right structure; adding a new plugin fragments the code execution story |

---

## Architecture Diagram (How it fits)

```
┌─────────────────────────────────────────────────────────┐
│  Agent                                                   │
│  "I'll write a script to do this..."                    │
│  Calls: notebook_execute(code="from tools import ...")   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  NotebookPlugin._execute_notebook()                      │
│                                                          │
│  1. CodeAnalyzer.analyze(code)  ◄── EXISTING             │
│  2. Permission check            ◄── EXISTING             │
│  3. Prepare namespace:                                   │
│     - __builtins__              ◄── EXISTING             │
│     - tools = <generated module> ◄── NEW (stub gen)      │
│     - json, re, etc.            ◄── NEW (stdlib inject)  │
│  4. exec(code, namespace)       ◄── EXISTING             │
│  5. Capture output              ◄── EXISTING             │
│  6. Return ExecutionResult      ◄── EXISTING             │
└──────────────────┬──────────────────────────────────────┘
                   │ tools.web_search(...) called in script
                   ▼
┌─────────────────────────────────────────────────────────┐
│  ToolBridge (NEW, ~10 lines)                             │
│  bridge.call("web_search", {"query": "..."})            │
│                    │                                     │
│                    ▼                                     │
│  ToolExecutor.execute("web_search", {...})  ◄── EXISTING │
│    → Permission check                       ◄── EXISTING │
│    → Plugin dispatch                        ◄── EXISTING │
│    → MCP/custom handler                     ◄── EXISTING │
│    → Return (success, result)               ◄── EXISTING │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Script continues with result...                         │
│  Only final print()/return goes back to agent context    │
└─────────────────────────────────────────────────────────┘
```

---

## Summary

The PTC design is good but over-engineers the solution relative to what we
already have. Our notebook plugin + ToolExecutor + CodeAnalyzer give us 70% of
the functionality. The remaining 30% is:

1. **Stub generator**: ~100-150 lines of new code
2. **Bridge wrapper**: ~10 lines
3. **Namespace injection**: ~20 lines
4. **System prompt extension**: ~50 lines

Total: **~200-250 lines of new code** to get PTC working, compared to the
~1000+ lines the proposal implies for the executor and security layer alone.

The key architectural insight is: **PTC is not a new system — it's a feature
flag on the notebook plugin that enriches the execution namespace with tool
stubs.**
