# .jaato Configuration Examples

This folder contains example configuration files for **jaato**. Initialize your
project with:

```bash
# Quick start: creates .jaato/ in the current directory
jaato init
```

Or copy manually if you prefer:

```bash
cp -r .jaato.example /path/to/your/project/.jaato
```

The `.jaato/` directory is gitignored by default, so your local configuration
won't be committed. If you want to share specific configs with your team, you
can selectively un-ignore subdirectories in `.gitignore`:

```gitignore
.jaato/*
!.jaato/profiles/
!.jaato/keybindings/
!.jaato/instructions/
!.jaato/prompts/
!.jaato/themes/
```

## Directory Layout

```
.jaato/
├── gc.json                     # Context budget / garbage collection settings
├── permissions.json            # Tool access policies (whitelist/blacklist)
├── sandbox.json                # Filesystem access boundaries
├── keybindings.json            # TUI keyboard shortcuts (default profile)
├── reliability-policies.json   # Loop prevention and pattern detection
├── filesystem_query.json       # File search exclusions and limits
├── .sessions.json              # Session persistence behavior
│
├── keybindings/                # Terminal-specific keybinding profiles
│   ├── vscode.json
│   └── tmux.json
│
├── themes/                     # Color themes (override built-in or add custom)
│   ├── dark.json
│   ├── light.json
│   ├── high-contrast.json
│   ├── dracula.json
│   ├── latte.json
│   ├── mocha.json
│   └── custom-example.json     # Starting point for your own theme
│
├── profiles/                   # Subagent profile definitions
│   ├── analyst-research.json
│   └── validator-tier1-universal.json
│
├── instructions/               # System instructions (markdown)
│   └── 00-system-instructions.md
│
├── prompts/                    # Reusable prompt templates
│   ├── weather-forecast-spain.md
│   └── code-review.md
│
└── references/                 # Reference sources for agent context
    └── project-docs.json
```

## Configuration Files

### gc.json

Controls the **garbage collection** strategy for the context window. When the
conversation grows too long, jaato trims old turns to stay within the model's
context limit.

- **type**: `"truncate"` | `"summarize"` | `"hybrid"` | `"budget"`
- **threshold_percent**: Trigger GC when context usage exceeds this percentage
- **target_percent**: Target usage after GC runs
- **pressure_percent**: When exceeded, even PRESERVABLE content can be evicted.
  Set to `0` for continuous mode (GC every turn, preservable never touched).

See the [GC config example](gc.json) for all fields and defaults.

### permissions.json

Defines which tools the model can invoke and under what conditions:

- **defaultPolicy**: `"allow"` | `"deny"` | `"ask"` (default: `"ask"`)
- **blacklist**: Tools and command patterns that are always blocked
- **whitelist**: Tools and patterns that are always allowed
- **channel**: How permission prompts are delivered (console, webhook, etc.)

See the [permissions example](permissions.json).

### sandbox.json

Controls filesystem access boundaries. Paths can be allowed or denied at the
workspace level. Session-level overrides live in `.jaato/sessions/<id>/sandbox.json`.

See the [sandbox example](sandbox.json).

### keybindings.json

Customize TUI keyboard shortcuts. Keys use prompt_toolkit syntax:

- Simple keys: `"enter"`, `"space"`, `"tab"`, `"q"`
- Control: `"c-c"`, `"c-d"`, `"c-p"` (Ctrl+C, Ctrl+D, Ctrl+P)
- Function keys: `"f1"` through `"f12"`
- Multi-key sequences: `["escape", "enter"]`

Terminal-specific profiles (VS Code, tmux, etc.) live in `keybindings/`.
Auto-detected from your terminal, or set `JAATO_KEYBINDING_PROFILE=<name>`.

See the [keybindings example](keybindings.json) and [keybindings/README.md](keybindings/README.md).

### reliability-policies.json

Prevents the model from getting stuck in loops: repeated tool calls, error
retries, introspection without action. Also defines prerequisite policies
(e.g., "must create a plan before updating step status").

See the [reliability-policies example](reliability-policies.json).

### themes/

Six built-in themes are included, plus a custom example. Override any built-in
theme by placing a file with the same name in `.jaato/themes/`. Create your own
by copying `custom-example.json` and adjusting the colors.

Switch themes at runtime: `/theme dark`, `/theme dracula`, etc.

See [themes/README.md](themes/README.md).

### profiles/

Subagent profiles define reusable configurations for specialized agents. Each
profile specifies which plugins are available, which references to preload, and
custom system instructions.

See [profiles/README.md](profiles/README.md).

### prompts/

Reusable prompt templates with optional YAML frontmatter for parameters. Used
with the `/prompt-library` command or loaded programmatically.

See [prompts/README.md](prompts/README.md).

## User-Level Configuration

Some configs also support a user-level fallback in `~/.jaato/`:

| File | Project (`.jaato/`) | User (`~/.jaato/`) |
|------|--------------------|--------------------|
| gc.json | Yes | Yes |
| permissions.json | Yes | Yes |
| sandbox paths | `sandbox.json` | `sandbox_paths.json` |
| keybindings.json | Yes | Yes |
| reliability-policies.json | Yes | Yes |
| theme.json | Yes | Yes |
| themes/*.json | Yes | Yes |
| preferences.json | No | Yes |

Project-level configs take precedence over user-level ones.

## Environment Variable Overrides

Many settings can be overridden via environment variables. See the project's
main `CLAUDE.md` for the full list, including:

- `JAATO_GC_THRESHOLD` / `JAATO_GC_TARGET` / `JAATO_GC_PRESSURE`
- `JAATO_KEY_<ACTION>` (e.g., `JAATO_KEY_SUBMIT=enter`)
- `JAATO_KEYBINDING_PROFILE=<profile>`
- `JAATO_THEME=<theme_name>`
- `JAATO_DEFERRED_TOOLS`, `JAATO_PARALLEL_TOOLS`
