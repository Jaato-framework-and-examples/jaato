# OpenRouter Auth Plugin

Authentication plugin for OpenRouter via API key.

## Session-Independent

This plugin is **session-independent** (`SESSION_INDEPENDENT = True`). Its
commands are available at daemon startup, before any session or provider
connection exists. This is essential because authentication must happen
*before* connecting to the OpenRouter provider.

## Commands

| Command | Description |
|---------|-------------|
| `openrouter-auth login` | Show instructions for getting your OpenRouter API key |
| `openrouter-auth key <api_key>` | Validate and store your API key |
| `openrouter-auth logout` | Clear stored API credentials |
| `openrouter-auth status` | Show current authentication status |
| `openrouter-auth help` | Show detailed help |

### Authentication Flow

1. User runs `openrouter-auth login` to see instructions
2. User visits https://openrouter.ai/settings/keys and creates a key
   (`sk-or-...`)
3. User runs `openrouter-auth key <api_key>` to validate and store it
4. Plugin hits `GET /api/v1/key` to validate, persists the key with mode 0600

Alternatively, set the `JAATO_OPENROUTER_API_KEY` environment variable
directly.

## Default Models

Curated entries returned by `get_default_models()`:

- `openrouter/openrouter/auto` — auto-router (OpenRouter picks per request)
- `openrouter/anthropic/claude-3.5-sonnet`
- `openrouter/openai/gpt-4o`
- `openrouter/google/gemini-2.0-flash-exp:free`
- `openrouter/deepseek/deepseek-r1`
- `openrouter/meta-llama/llama-3.3-70b-instruct`

The full catalog (300+ models) is available via `list_models()` on the
provider, which queries `GET https://openrouter.ai/api/v1/models`. Browse
the catalog at https://openrouter.ai/models.

## Plugin Protocol

```python
class OpenRouterAuthPlugin:
    provider_name = "openrouter"
    provider_display_name = "OpenRouter"

    def get_default_models() -> List[Dict[str, str]]   # Curated model list
    def verify_credentials() -> bool                     # Check API key in stored / env
    def get_user_commands() -> List[UserCommand]          # Command declarations
    def get_command_completions(cmd, args) -> List[...]   # Subcommand autocompletion
    def execute_user_command(cmd, args) -> str | HelpLines
```

## Attribution Headers

OpenRouter uses two optional headers for app rankings on
https://openrouter.ai/rankings:

- `HTTP-Referer` — site URL
- `X-OpenRouter-Title` — site / app title

These are sent automatically with sensible defaults; override via
`JAATO_OPENROUTER_HTTP_REFERER` / `JAATO_OPENROUTER_APP_TITLE`.

## File Structure

```
shared/plugins/openrouter_auth/
├── __init__.py      # PLUGIN_KIND, SESSION_INDEPENDENT, exports
├── plugin.py        # OpenRouterAuthPlugin implementation
└── README.md        # This documentation
```
