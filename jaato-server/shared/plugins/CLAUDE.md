# Plugin Implementation Guide

## Critical: `PLUGIN_KIND` in `__init__.py`

Every plugin **must** declare `PLUGIN_KIND` in its `__init__.py`. Without it, the
plugin is **silently skipped** during directory discovery and never loaded.

```python
# shared/plugins/my_plugin/__init__.py

PLUGIN_KIND = "tool"  # REQUIRED - "tool", "enrichment", "gc", "session", or "model_provider"

from .plugin import MyPlugin, create_plugin

__all__ = ["MyPlugin", "create_plugin", "PLUGIN_KIND"]
```

## Critical: `PLUGIN_TIER` for the Confined-Runner Partition

Every plugin that declares `PLUGIN_KIND` **must** also declare
`PLUGIN_TIER` (Phase 3 §3.3.5).  Tier classification per the parent
design `docs/design/per_session_confined_runner.md` §4.2:

```python
# shared/plugins/my_plugin/__init__.py

PLUGIN_KIND = "tool"
PLUGIN_TIER = "runner"  # or "daemon"

from .plugin import MyPlugin, create_plugin

__all__ = ["MyPlugin", "create_plugin", "PLUGIN_KIND", "PLUGIN_TIER"]
```

**Tier rules:**
- `"daemon"` — provider clients, OAuth tokens, GC over session history,
  cache state, formatters, telemetry forwarding, `*_auth` plugins.
- `"runner"` — workspace FS access, subprocess spawn (cli, lsp, mcp,
  interactive_shell, notebook), per-session in-memory state
  (permission, references, memory, todo, etc.).
- `"daemon_callable"` — **cross-tier**: the tool schema must surface
  to the model (which runs runner-side post-seat-flip) BUT the body
  must execute daemon-side because it needs daemon-only state
  (canonical case: `SessionManager` access for cross-session
  introspection).  The runner-side instance is a thin
  `DaemonForwardingMixin` stub; the daemon-side instance holds the
  real state and executes the body.  See "Cross-tier plugins" below.

**Why this matters:** `PluginRegistry.discover(tier_filter="...")`
filters discovery by tier.  Without `PLUGIN_TIER`, a plugin is
silently excluded when a tier filter is set — same footgun as
missing `PLUGIN_KIND`.

The build-fail gate is in `shared/tests/test_plugin_tier_partition.py`:
new plugins without an explicit tier fail
`test_every_plugin_with_kind_has_tier`.  Daemon-runner overlap
fails `test_daemon_and_runner_tiers_are_disjoint`.  Cross-tier
plugins without `DaemonForwardingMixin` (or `"runner"`-tier plugins
that declare `set_session_manager` without the mixin) fail the
`test_daemon_callable_plugins_extend_daemon_forwarding_mixin` /
`test_runner_tier_plugins_dont_secretly_need_daemon_state` gates.

### Cross-tier plugins (`PLUGIN_TIER = "daemon_callable"`)

The post-seat-flip architecture runs the model loop runner-side; tool
schemas surface from the runner-side registry (loaded via
`discover(tier_filter="runner")`).  For most plugins this aligns with
where the body should run (runner-tier = filesystem / subprocess /
per-session state).  **Cross-tier plugins** are the exception: they
need the schema visible runner-side AND the body executed daemon-side
because the body depends on daemon-only state (e.g. the
`SessionManager` for cross-session introspection).

**Pattern (mirror of `RunnerForwardingMixin` but reversed):**

```python
# __init__.py
PLUGIN_KIND = "tool"
PLUGIN_TIER = "daemon_callable"  # discovered both sides

# plugin.py
from shared.plugins.daemon_forwarding import DaemonForwardingMixin

class MyCrossTierPlugin(DaemonForwardingMixin):
    def __init__(self):
        self._session_manager = None  # wired daemon-side only

    def set_session_manager(self, sm) -> None:
        self._session_manager = sm

    def get_executors(self) -> Dict[str, Callable]:
        raw = {
            "my_tool": self._execute_my_tool,
        }
        # The wrapper self-routes:
        # - runner-side: forwards via daemon.plugin_execute RPC
        # - daemon-side: calls _execute_my_tool in-process
        return self.wrap_executors_for_daemon_forwarding(raw)
```

**Why both instances?**  Discovery loads the plugin BOTH sides
(daemon-side `discover()` with no filter + runner-side
`discover(tier_filter="runner")` accepts `daemon_callable`).  The
mixin self-routes by inspecting `registry.runner_rpc_client`:
- Runner-side: attribute exists (set by `runner/rpc.py:1114`); mixin
  forwards via `daemon.plugin_execute` RPC.
- Daemon-side: attribute does NOT exist (the symmetric daemon→runner
  client lives at `registry.runner_rpc`); mixin calls the in-process
  body — including when re-entered by the daemon-side
  `daemon.plugin_execute` handler.

The mixin's mirror is `RunnerForwardingMixin`
(`shared/plugins/runner_forwarding.py`) — same wrap-point pattern,
opposite direction.

**Gap #1 trap (saved-lesson history):** when adding a plugin that
needs daemon-only state, the wrong move is to declare
`PLUGIN_TIER = "runner"` and stop there.  The runner-side instance
will execute against a `None` reference because daemon-side wire-up
(`session_manager.py:4379` for `session_ops`) only mutates the
daemon-side instance.  The empirical symptom is the runner-side body
crashing or the model hallucinating that the tool is unavailable
(empty schema from premature failure).  The fix is:

1. Declare `PLUGIN_TIER = "daemon_callable"` (NOT `"runner"`).
2. Extend `DaemonForwardingMixin` on the plugin class.
3. Wrap `get_executors()` via `wrap_executors_for_daemon_forwarding(...)`.

The build-fail gate
`test_runner_tier_plugins_dont_secretly_need_daemon_state` catches the
half-step variant (`"runner"` + `set_session_manager` without the
mixin) so the next plugin to need this pattern can't silently land it.

## Critical: `SESSION_INDEPENDENT` for Auth Plugins

Auth plugins must also declare `SESSION_INDEPENDENT = True` in `__init__.py`.
Without it, the plugin's commands only appear after a session is loaded — but
auth plugins exist to establish credentials *before* connecting to a provider.

```python
# shared/plugins/my_auth/__init__.py

PLUGIN_KIND = "tool"
SESSION_INDEPENDENT = True  # REQUIRED for auth plugins

from .plugin import MyAuthPlugin, create_plugin

__all__ = ["MyAuthPlugin", "create_plugin", "PLUGIN_KIND", "SESSION_INDEPENDENT"]
```

**Why this matters:** The daemon (`server/__main__.py`) has two command sources:

1. **Session-bound plugins** — discovered per-session, gated behind `session.is_loaded`
2. **Daemon-level plugins** — discovered at daemon startup via `SESSION_INDEPENDENT`

Without `SESSION_INDEPENDENT`, auth commands are invisible until a session exists,
creating a chicken-and-egg problem: users can't authenticate because the auth
command requires a session, but connecting a session may require authentication.

**Why this matters:** `PluginRegistry._discover_via_directory()` (in `registry.py`)
checks every module with:

```python
module_kind = getattr(module, 'PLUGIN_KIND', None)
if module_kind != plugin_kind:
    continue  # silently skipped — no error, no warning
```

Missing `PLUGIN_KIND` means `None != "tool"` → plugin never loads → its user
commands never register → clients send the input to the model as prompt text
instead of executing it as a command. No autocompletion either.

## Three Plugin Patterns

### Pattern 1: Model Tools (e.g., `cli/`, `todo/`, `file_edit/`)

Plugins that provide tools the AI model invokes via function calling.

```python
# plugin.py
class MyToolPlugin:
    @property
    def name(self) -> str:
        return "my_tool"

    def get_tool_schemas(self) -> List[ToolSchema]:
        return [ToolSchema(name="my_function", description="...", parameters={...})]

    def get_executors(self) -> Dict[str, Any]:
        return {"my_function": self._execute_my_function}

    def get_user_commands(self) -> List[UserCommand]:
        return []  # No user commands

    def get_auto_approved_tools(self) -> List[str]:
        return []  # Model tools typically require permission
```

### Pattern 2: User Commands Only (e.g., `anthropic_auth/`, `github_auth/`, `zhipuai_auth/`)

Plugins that provide commands users invoke directly (not through the model).

```python
# plugin.py
class MyAuthPlugin:
    @property
    def name(self) -> str:
        return "my_auth"

    def get_tool_schemas(self) -> List[ToolSchema]:
        return []  # No model tools

    def get_executors(self) -> Dict[str, Any]:
        return {"my-auth": lambda args: self.execute_user_command("my-auth", args)}

    def get_user_commands(self) -> List[UserCommand]:
        return [
            UserCommand(
                name="my-auth",
                description="Manage My Service authentication",
                share_with_model=False,
                parameters=[
                    CommandParameter(
                        name="action",
                        description="Action: login, logout, status, or help",
                        required=True,
                        capture_rest=True,
                    ),
                ],
            ),
        ]

    def get_auto_approved_tools(self) -> List[str]:
        return ["my-auth"]  # User commands should be auto-approved

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        if command != "my-auth":
            return []
        actions = [
            CommandCompletion("login", "Authenticate with My Service"),
            CommandCompletion("logout", "Clear stored credentials"),
            CommandCompletion("status", "Show authentication status"),
            CommandCompletion("help", "Show detailed help"),
        ]
        if not args:
            return actions
        if len(args) == 1:
            partial = args[0].lower()
            return [a for a in actions if a.value.startswith(partial)]
        return []

    def execute_user_command(self, command: str, args: Dict[str, Any]) -> str:
        raw_action = args.get("action", "").strip()
        action_lower = raw_action.lower()
        if action_lower == "login":
            return self._cmd_login()
        elif action_lower == "help":
            return self._cmd_help()
        # ...

    def _cmd_help(self) -> HelpLines:
        """Return HelpLines (not str) for pager display."""
        return HelpLines(lines=[
            ("My Auth Command", "bold"),
            ("", ""),
            ("USAGE", "bold"),
            ("    my-auth <action>", ""),
            # ...
        ])
```

### Pattern 3: Enrichment Only (e.g., auto-steering, context cleanup)

Plugins that only enrich prompts, system instructions, or tool results — no tools, no commands.
These implement the `EnrichmentPlugin` protocol instead of `ToolPlugin`.

```python
# __init__.py
PLUGIN_KIND = "enrichment"  # NOT "tool" — discovered as enrichment-only

from .plugin import MyEnrichmentPlugin, create_plugin

__all__ = ["MyEnrichmentPlugin", "create_plugin", "PLUGIN_KIND"]
```

```python
# plugin.py
from jaato_sdk.plugins.base import EnrichmentPlugin, PromptEnrichmentResult

class MyEnrichmentPlugin:
    """Enrichment-only plugin that injects context hints into prompts.

    Implements EnrichmentPlugin protocol — no tools or commands needed.
    Discovered via PLUGIN_KIND = "enrichment" and automatically registered
    as enrichment-only by the registry.
    """

    @property
    def name(self) -> str:
        return "my_enrichment"

    def initialize(self, config=None) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def subscribes_to_prompt_enrichment(self) -> bool:
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        # Inject hints based on prompt content
        enhanced = prompt + "\n\n[Context hint from my_enrichment]"
        return PromptEnrichmentResult(prompt=enhanced, metadata={"injected": True})

    def get_enrichment_priority(self) -> int:
        return 60  # Optional, default is 50


def create_plugin():
    return MyEnrichmentPlugin()
```

**Key differences from Pattern 1/2:**
- `PLUGIN_KIND = "enrichment"` (not `"tool"`)
- Implements `EnrichmentPlugin` protocol (not `ToolPlugin`)
- No `get_tool_schemas()`, `get_executors()`, `get_user_commands()`, etc.
- Automatically registered as enrichment-only — participates in enrichment pipeline only
- Discovered alongside tool plugins during `registry.discover()`

## Tool Traits

Tools can declare semantic **traits** on their `ToolSchema` via the `traits` field
(`FrozenSet[str]`). Traits drive cross-cutting behavior (enrichment routing,
permission defaults, etc.) without hardcoding tool names in session or plugin code.

### Currently Defined Traits

| Constant | Value | Contract |
|----------|-------|----------|
| `TRAIT_FILE_WRITER` | `"file_writer"` | Tool writes/modifies files. Result must include `path` (str), `files_modified` (list), or `changes[].file`. Triggers full-JSON enrichment (LSP diagnostics, artifact tracking). |
| `TRAIT_GREPPABLE_CONTENT` | `"greppable_content"` | Tool returns bulk content eligible for result-rewriting. Routes the full JSON result through the same full-dict enrichment path as `TRAIT_FILE_WRITER` so result-rewriter plugins (`result_grep`) can shrink structured payloads the text-field path never sees (e.g. `call_service.body`). Marks eligibility only; filtering is done by the subscribed rewriter. |

### How to Declare Traits

```python
from ..model_provider.types import ToolSchema, TRAIT_FILE_WRITER

ToolSchema(
    name="myWriteTool",
    description="...",
    parameters={...},
    traits=frozenset({TRAIT_FILE_WRITER}),
)
```

### How Traits Are Consumed

- **Session** (`jaato_session.py`): Calls `registry.get_tool_traits(tool_name)` to
  decide enrichment strategy. Tools with `TRAIT_FILE_WRITER` get full-JSON
  enrichment (LSP diagnostics, artifact tracking).
- **Enrichment plugins** (LSP, artifact_tracker): Receive all tool results that the
  session routes to them. They extract file paths generically from the result dict
  using the standard keys (`path`, `files_modified`, `changes[].file`).

### Adding a New Trait

1. Add a `TRAIT_*` constant in `shared/plugins/model_provider/types.py` with a
   docstring documenting the contract.
2. Update consumers (session, plugins) to query `registry.get_tool_traits()` for
   the new trait.

## Configuration Schema (`get_config_schema`)

Plugins that accept configuration in `initialize(config)` should declare their
settings via `get_config_schema()`. This enables profile managers, TUI settings
forms, and documentation generators to introspect available settings.

```python
from jaato_sdk.plugins.base import PluginSetting

def get_config_schema(self) -> List[PluginSetting]:
    return [
        PluginSetting(
            name="max_results",
            type="int",
            default=100,
            description="Maximum matches to return",
        ),
        PluginSetting(
            name="region",
            type="str",
            default="wt-wt",
            description="Region for search results",
            choices=["wt-wt", "us-en", "uk-en"],
        ),
        PluginSetting(
            name="timeout",
            type="int",
            default=30,
            description="Request timeout in seconds",
            env_var="MY_PLUGIN_TIMEOUT",
        ),
    ]
```

**PluginSetting fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | str | yes | Config key as used in `config.get(name, default)` |
| `type` | str | yes | Type hint: `"int"`, `"str"`, `"bool"`, `"float"`, `"list[str]"`, `"dict"` |
| `default` | Any | yes | Default value when key is absent |
| `description` | str | yes | Human-readable description |
| `required` | bool | no | If True, key must be in config (default: False) |
| `choices` | list | no | Restricts valid values to this list |
| `env_var` | str | no | Environment variable that can override |

**What to include:** Only user/profile-configurable settings.

**What to exclude:** Internal wiring keys set by the framework — `agent_name`,
`workspace_root`, `workspace_path`, `session_id`, `base_path`, `config_path`.

**Registry access:** `registry.get_plugin_config_schema("cli")` returns the list
of `PluginSetting` for a plugin, or `[]` if unimplemented.

## Critical: Model-Supplied Paths Go Through `path_safety`

A plugin that reads or writes a path the **model** chose must not use the
`check(path)` → `open(path)` pattern. `sandbox_utils.check_path_with_jaato_containment`
answers the question by canonicalising with `os.path.realpath`; re-opening by
path afterwards makes the kernel resolve the symlinks a *second* time, so a
link swapped in between validates one object and acts on another. Use
`shared/plugins/path_safety.py`, which resolves once and proves the descriptor
it hands back is the object that was validated.

```python
from ..path_safety import read_text_verified, write_text_verified

def _validator(self, mode: str):
    return lambda resolved: self._is_path_allowed(resolved, mode=mode)

content = read_text_verified(path, validate=self._validator("read"))
write_text_verified(path, new, validate=self._validator("write"), exclusive=True)
```

| Helper | Use for |
|--------|---------|
| `open_verified(path, flags, validate=...)` | Anything the higher-level helpers don't cover. |
| `read_text_verified` / `read_bytes_verified` | Reads. |
| `write_text_verified(..., exclusive=True)` | New files — `O_EXCL`/`O_NOFOLLOW` relative to the resolved parent, so a symlink planted at the target is not followed. |
| `unlink_verified` / `move_verified` | Deletes and moves, pinned to the resolved parent. |
| `describe_special(path)` | Skip FIFOs/sockets/devices during a **directory walk** — opening a named pipe blocks the worker until a writer appears. |
| `ensure_private_dir(path)` | Any directory the plugin composes under a shared, world-writable `/tmp`; refuses a pre-planted symlink or another user's directory. |

Everything raises `UnsafePathError`, an `OSError` subclass, so existing
`except OSError` handlers degrade to an ordinary tool error.

**Search tools have a second obligation**: containment must be re-applied to
every *result*, not just the search root. `Path.glob` follows symlinked
directories, so a link committed into a repository (`data/logs -> /etc`) turns
a workspace-scoped search into a read outside it. See
`FilesystemQueryPlugin._make_result_guard` for the pattern (cache the verdict
per parent directory; check the leaf individually only when it is a symlink).

**Allow rules must resolve too, not only deny rules.** The `/tmp` allowance in
`check_path_with_jaato_containment` short-circuits the workspace check beneath
it, so deciding it on the path *as written* admitted a symlink for where the
link lives rather than where it points — `/tmp/x -> ~/.ssh/id_rsa` read as
"under /tmp". Any new allowlist branch must compare **resolved against
resolved** (resolve the configured roots as well, or macOS's
`/tmp -> /private/tmp` rejects every real temp path).

**Testing note:** on Linux `tmp_path` is itself under `/tmp`, so a fixture that
puts its "outside" target there is inside the temp allowance and the escape
test passes for the wrong reason — in both directions, since it also makes
non-escapes look like leaks. Substitute `SYSTEM_TEMP_PATHS` with a directory
under `tmp_path` instead of disabling `allow_tmp`, which masks this whole class
of bug.

## Checklist for New Plugins

1. `__init__.py` has `PLUGIN_KIND = "tool"` or `"enrichment"` (or other appropriate kind)
2. `__init__.py` exports `PLUGIN_KIND` in `__all__`
3. `plugin.py` has `create_plugin()` factory function
4. Plugin class implements `ToolPlugin` or `EnrichmentPlugin` protocol (see `base.py`)
5. User commands listed in `get_auto_approved_tools()` (prevents permission prompts)
6. `get_command_completions()` implemented for subcommand autocompletion
7. Help command returns `HelpLines` (not `str`) for pager display
8. `get_config_schema()` implemented if plugin has configurable settings
9. **Auth plugins:** `__init__.py` has `SESSION_INDEPENDENT = True`
10. **Model providers:** `verify_auth()` works before `initialize()` (no `self._client` access)
11. **File-writing tools:** Declare `traits=frozenset({TRAIT_FILE_WRITER})` and include `path`/`files_modified` in result
12. **Model-supplied paths:** Read/write via `path_safety` helpers, never `check()` then `open()`; search tools re-check every result, not just the root

## Critical: `verify_auth()` in Model Provider Plugins

Model provider plugins (`shared/plugins/model_provider/<name>/provider.py`) must
implement `verify_auth()` so it works **before `initialize()` is called**.

The runtime calls `verify_auth()` on a **fresh, uninitialized** provider instance
to check if credentials exist before creating a session (see
`jaato_runtime.py:verify_auth()` — "Create a temporary provider instance just for
auth verification. We don't call initialize() yet.").

This means `verify_auth()` must **never** use `self._client` or any state set by
`initialize()`. It should only check whether credentials are available — not
whether they are valid (that happens later during `initialize()` + first request).

```python
# CORRECT — checks credential availability without needing initialized state
def verify_auth(self, allow_interactive=False, on_message=None) -> bool:
    api_key = resolve_api_key() or get_stored_api_key()
    if api_key:
        if on_message:
            on_message("Found API key")
        return True
    if on_message:
        on_message("No credentials found")
    return False
```

```python
# WRONG — crashes with 'NoneType' has no attribute 'messages' because
# self._client is None on an uninitialized provider instance
def verify_auth(self, allow_interactive=False, on_message=None) -> bool:
    self._client.messages.create(...)  # self._client is None!
    return True
```

**Reference:** See `AnthropicProvider.verify_auth()` for the canonical pattern —
it checks PKCE tokens, OAuth tokens, and API keys without touching `self._client`.

## IPC Completion Flow

In IPC (daemon) mode, completions come from two sources:

**Session-independent plugins** (always available):
1. `_discover_daemon_plugins()` at daemon startup scans for `SESSION_INDEPENDENT = True`
2. `_get_command_list()` iterates `self._daemon_plugins` unconditionally
3. Subcommands pre-expanded into `CommandListEvent`

**Session-bound plugins** (only when session loaded):
1. `_get_command_list()` iterates loaded sessions → `registry.list_exposed()`
2. Calls `plugin.get_user_commands()` → `plugin.get_command_completions()`
3. Pre-expanded into same `CommandListEvent`

Deduplication (by command name) ensures no duplicates when both sources provide
the same command.

**Command execution** follows matching priority:
1. Static commands (session.*, tools.*) → daemon handles directly
2. Daemon-level plugins → `_execute_daemon_command()` (no session required)
3. Session plugins → `session_manager.handle_request()` → `server.execute_command()`

If `PLUGIN_KIND` is missing, the plugin is never discovered at either level.
If `SESSION_INDEPENDENT` is missing from an auth plugin, it only works at level 3.
