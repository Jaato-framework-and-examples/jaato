# Anthropic Auth Plugin

Authentication plugin for Anthropic Claude API (Pro/Max subscriptions and API keys).

## Session-Independent

This plugin is **session-independent** (`SESSION_INDEPENDENT = True`). Its commands
are available at daemon startup, before any session or provider connection exists.
This is essential because authentication must happen *before* connecting to the
Anthropic provider.

## Commands

| Command | Description |
|---------|-------------|
| `anthropic-auth login` | Open browser for OAuth PKCE authentication |
| `anthropic-auth code <code>` | Complete login with authorization code from browser |
| `anthropic-auth logout` | Clear stored OAuth tokens |
| `anthropic-auth status` | Show current authentication status |
| `anthropic-auth help` | Show detailed help |

### Authentication Flow

1. User runs `anthropic-auth login`
2. Browser opens to Anthropic's OAuth consent page
3. User authorizes and receives an authorization code
4. User runs `anthropic-auth code <authorization_code>`
5. Plugin exchanges code for OAuth tokens and stores them securely

Alternatively, set the `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` environment
variable directly (see main CLAUDE.md for details).

## Post-Auth Session Bootstrap

After successful authentication, the daemon detects valid credentials via
`verify_credentials()` and offers an interactive session setup wizard:

1. **Connect prompt** — *"Connect to Anthropic (Claude) now? [y/n]"*
   (or *"Switch to..."* if a session with a different provider is already active)
2. **Model selection** — Presents the default model list:
   - `anthropic/claude-sonnet-4-5-20250929` — Claude Sonnet 4.5
   - `anthropic/claude-opus-4-6` — Claude Opus 4.6
   - `anthropic/claude-haiku-4-5-20251001` — Claude Haiku 4.5
3. **Persistence prompt** — *"Save provider/model to .env? [y/n]"*
   Writes `JAATO_PROVIDER=anthropic` and `MODEL_NAME=<model>` to the workspace
   `.env` file, preserving existing values for other keys.

This flow is handled by `PostAuthSetupEvent` / `PostAuthSetupResponse` events
between the daemon and the rich client. See `server/__main__.py` for the daemon
side and `rich-client/rich_client.py` for the client-side wizard.

## Plugin Protocol

```python
class AnthropicAuthPlugin:
    provider_name = "anthropic"
    provider_display_name = "Anthropic (Claude)"

    def get_default_models() -> List[Dict[str, str]]   # Curated model list
    def verify_credentials() -> bool                     # Check OAuth tokens / API key
    def get_user_commands() -> List[UserCommand]          # Command declarations
    def get_command_completions(cmd, args) -> List[...]   # Subcommand autocompletion
    def execute_user_command(cmd, args) -> str | HelpLines
```

## File Structure

```
shared/plugins/anthropic_auth/
├── __init__.py      # PLUGIN_KIND, SESSION_INDEPENDENT, exports
├── plugin.py        # AnthropicAuthPlugin implementation
└── README.md        # This documentation
```
