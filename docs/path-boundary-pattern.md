# Path Boundary Pattern: Python Runtime vs Shell Environment

When native Windows Python runs inside a Unix-like shell (MSYS2, Git Bash), there is a
fundamental mismatch: Python uses Windows APIs and produces `C:\Users\foo` paths, but the
shell expects `/c/Users/foo`. This pattern defines **where and how** to convert between the two.

Reference: [MSYS2 Python docs](https://www.msys2.org/docs/python/) |
[MSYS2 Filesystem Paths](https://www.msys2.org/docs/filesystem-paths/)

---

## Core Insight

On Windows, **regardless of the shell** (cmd, PowerShell, MSYS2, Git Bash), Python's path
storage is always Windows-native:

```
os.getcwd()          → "C:\Users\foo\project"
os.path.join(a, b)   → "C:\Users\foo\file.py"
Path.resolve()        → WindowsPath("C:/Users/foo/file.py")
os.path.realpath()    → "C:\Users\foo\file.py"
```

The `sys.platform` is always `"win32"`. Python is a native Windows executable — the MSYS2
shell is just a terminal wrapper around it. The conversion problem only exists at the
**boundary** between Python's internal path handling and what the user/shell sees.

---

## Three Conversion Layers

All path handling falls into exactly three layers. Each layer has a single responsibility
and a single function to call.

### Layer 1: Input (Shell → Python)

**When**: A path arrives from the user, the model, or any external source that might have
typed it in an MSYS2 shell.

**What**: Convert `/c/Users/foo` → `C:/Users/foo` so Python can open it.

**Function**: `msys2_to_windows_path(path)`

**Where to apply**: At the **entry point** of any function that receives a path string
and will pass it to Python's file APIs (`open()`, `Path()`, `os.path.*`, etc.).

```python
from shared.path_utils import msys2_to_windows_path

def _resolve_path(self, path: str) -> Path:
    path = msys2_to_windows_path(path)  # ← First line, before any Path() or os.path call
    p = Path(path)
    ...
```

**Current entry points**:
- `file_edit/plugin.py` → `_resolve_path()`
- `filesystem_query/plugin.py` → `_resolve_path()`
- `cli/plugin.py` → `_is_path_allowed()`
- `sandbox_utils.py` → `check_path_with_jaato_containment()`
- `permission/sanitization.py` → `resolve_path()`

### Layer 2: Comparison (Python ↔ Python)

**When**: Comparing two paths that may have been constructed with different separators
(e.g., sandbox prefix matching where one path used `\` and another used `/`).

**What**: Normalize `\` → `/` for consistent string comparison. Does **not** convert
drive letters — both paths are already in Python's native format.

**Function**: `normalize_for_comparison(path)`

**Where to apply**: At any **string comparison** between two paths on Windows (prefix
checks, equality checks, startswith).

```python
from shared.path_utils import normalize_for_comparison

# Prefix check (is path inside workspace?)
norm_path = normalize_for_comparison(real_path)
norm_root = normalize_for_comparison(workspace_root).rstrip('/') + '/'
if norm_path.startswith(norm_root):
    ...
```

**Important**: This activates on **all** Windows (`sys.platform == 'win32'`), not just
MSYS2. Mixed separators can cause comparison failures even on standard Windows, because
`os.path.join()` uses `\` but user input or config files may use `/`.

**Convenience wrappers**:
- `normalized_startswith(path, prefix)` — prefix match with normalization
- `normalized_equals(path1, path2)` — equality check with normalization

### Layer 3: Output (Python → Shell)

**When**: A path produced by Python will be displayed to the user or returned in tool
results that the model or user might reference.

**What**: Convert `C:\Users\foo` → `/c/Users/foo` so it's copy-pasteable in the MSYS2 shell.

**Function**: `normalize_result_path(path)` (or `normalize_path(path)`)

**Where to apply**: At the **exit point** — wrap any path value in tool result dicts,
error messages, or display strings.

```python
from shared.path_utils import normalize_result_path

return {
    "success": True,
    "path": normalize_result_path(path),           # ← Tool result
    "backup": normalize_result_path(str(backup)),   # ← Tool result
}
```

**Important**: This only activates under MSYS2 (`is_msys2_environment() == True`).
On standard Windows or Unix, paths are returned unchanged.

---

## Decision Table by Platform

| Platform | Input (`msys2_to_windows_path`) | Comparison (`normalize_for_comparison`) | Output (`normalize_result_path`) |
|----------|------|------|------|
| **Linux / macOS** | no-op | no-op | no-op |
| **Win + cmd/PowerShell** | no-op (no `/c/` paths) | `\` → `/` | no-op |
| **Win + MSYS2/Git Bash** | `/c/foo` → `C:/foo` | `\` → `/` | `C:\foo` → `/c/foo` |

All three functions are safe to call unconditionally — they are no-ops when not applicable.

---

## Pattern: Adding Path Handling to a New Component

When implementing a new plugin or component that handles file paths, follow this checklist:

### 1. Identify entry points

Any function parameter named `path`, `file_path`, `source_path`, `destination_path`,
or similar that comes from the model or user:

```python
from shared.path_utils import msys2_to_windows_path

def my_tool_executor(self, args: Dict[str, Any]) -> Dict[str, Any]:
    path = args.get("path", "")
    path = msys2_to_windows_path(path)  # ← Convert before using
    file_path = Path(path)
    ...
```

### 2. Identify comparison points

Any place where two paths are compared as strings (not via `os.path.samefile()`):

```python
from shared.path_utils import normalize_for_comparison

norm_target = normalize_for_comparison(target_path)
norm_allowed = normalize_for_comparison(allowed_root)
if norm_target.startswith(norm_allowed + '/'):
    ...
```

### 3. Identify exit points

Any path that appears in the return value, error message, or display output:

```python
from shared.path_utils import normalize_result_path

return {
    "path": normalize_result_path(path),
    "error": f"File not found: {normalize_result_path(path)}",
}
```

### 4. Do NOT normalize internally

Paths that stay inside Python (passed to `open()`, `Path.read_text()`,
`os.path.exists()`, etc.) should remain in Python's native format. Only convert
at boundaries.

---

## Roundtrip Safety

The input and output conversions form a safe roundtrip:

```
User types:    /c/Users/foo/file.py     (MSYS2 format)
    ↓ msys2_to_windows_path()
Python uses:   C:/Users/foo/file.py     (Windows native)
    ↓ normalize_result_path()
Tool returns:  /c/Users/foo/file.py     (MSYS2 format)
    ↓ model passes it back
    ↓ msys2_to_windows_path()
Python uses:   C:/Users/foo/file.py     (Windows native again)
```

Both path formats work as input. Users can provide either `C:\Users\foo` or `/c/Users/foo`
and both resolve to the same internal representation.

---

## MSYS2 Detection

Detection is performed once via `is_msys2_environment()` (lru_cached) by checking:

1. `sys.platform == 'win32'` (native Windows Python, not Cygwin)
2. `MSYSTEM` env var in `{MINGW64, MINGW32, MSYS, UCRT64, CLANG64, CLANGARM64}`
3. `TERM_PROGRAM == 'mintty'` (Git Bash default terminal)

If none match, all conversion functions become no-ops.

---

## False Positive Protection

The drive letter regex only matches **single-letter** path components:

```
/c/Users/foo    → C:/Users/foo     ✓  (single letter = drive)
/config/foo     → /config/foo      ✗  (multi-letter = not a drive)
/cache/bar      → /cache/bar       ✗  (multi-letter = not a drive)
/tmp/test       → /tmp/test        ✗  (multi-letter = not a drive)
```

Regex: `^/([a-zA-Z])(?:/|$)` — must be exactly one letter after the leading `/`,
followed by `/` or end of string.

---

## Implementation

All functions live in `shared/path_utils.py`. See that module for the full API:

| Function | Layer | Purpose |
|----------|-------|---------|
| `is_msys2_environment()` | Detection | Cached MSYS2 check |
| `msys2_to_windows_path(path)` | Input | `/c/foo` → `C:/foo` |
| `windows_to_msys2_path(path)` | Output | `C:\foo` → `/c/foo` |
| `normalize_path(path)` | Output | Full MSYS2 normalization for display |
| `normalize_for_comparison(path)` | Comparison | `\` → `/` on all Windows |
| `normalized_startswith(path, prefix)` | Comparison | Prefix match wrapper |
| `normalized_equals(path1, path2)` | Comparison | Equality wrapper |
| `normalize_result_path(path)` | Output | For tool result dicts |
| `get_display_separator()` | Output | `/` under MSYS2, `os.sep` otherwise |
