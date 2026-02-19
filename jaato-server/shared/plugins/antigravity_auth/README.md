# Antigravity Auth Plugin

Authentication plugin for Google Antigravity IDE backend (Google OAuth PKCE flow).
Provides access to Gemini 3 and Claude models through Google's infrastructure.

## Session-Independent

This plugin is **session-independent** (`SESSION_INDEPENDENT = True`). Its commands
are available at daemon startup, before any session or provider connection exists.
This is essential because authentication must happen *before* connecting to the
Antigravity provider.

## Commands

| Command | Description |
|---------|-------------|
| `antigravity-auth login` | Open browser for Google OAuth authentication |
| `antigravity-auth code <code>` | Complete login with authorization code from browser |
| `antigravity-auth logout` | Clear stored OAuth tokens for all accounts |
| `antigravity-auth status` | Show current authentication status |
| `antigravity-auth accounts` | List all authenticated Google accounts |
| `antigravity-auth help` | Show detailed help |

### Authentication Flow

1. User runs `antigravity-auth login`
2. Browser opens to Google's OAuth consent page
3. User authorizes and receives an authorization code
4. User runs `antigravity-auth code <authorization_code>`
5. Plugin exchanges code for OAuth tokens and stores them securely

Supports multi-account rotation (`JAATO_ANTIGRAVITY_AUTO_ROTATE=true`) —
authenticate multiple Google accounts and the provider rotates between them.

## Post-Auth Session Bootstrap

After successful authentication, the daemon detects valid credentials via
`verify_credentials()` and offers an interactive session setup wizard:

1. **Connect prompt** — *"Connect to Google Antigravity now? [y/n]"*
   (or *"Switch to..."* if a session with a different provider is already active)
2. **Model selection** — Presents the default model list:
   - `antigravity-gemini-3-pro` — Gemini 3 Pro via Antigravity
   - `antigravity-gemini-3-flash` — Gemini 3 Flash via Antigravity
   - `antigravity-claude-sonnet-4-5` — Claude Sonnet 4.5 via Antigravity
3. **Persistence prompt** — *"Save provider/model to .env? [y/n]"*
   Writes `JAATO_PROVIDER=antigravity` and `MODEL_NAME=<model>` to the workspace
   `.env` file, preserving existing values for other keys.

This flow is handled by `PostAuthSetupEvent` / `PostAuthSetupResponse` events
between the daemon and the rich client. See `server/__main__.py` for the daemon
side and `jaato-tui/rich_client.py` for the client-side wizard.

## Plugin Protocol

```python
class AntigravityAuthPlugin:
    provider_name = "antigravity"
    provider_display_name = "Google Antigravity"

    def get_default_models() -> List[Dict[str, str]]   # Curated model list
    def verify_credentials() -> bool                     # Check OAuth tokens
    def get_user_commands() -> List[UserCommand]          # Command declarations
    def get_command_completions(cmd, args) -> List[...]   # Subcommand autocompletion
    def execute_user_command(cmd, args) -> str | HelpLines
```

## File Structure

```
shared/plugins/antigravity_auth/
├── __init__.py      # PLUGIN_KIND, SESSION_INDEPENDENT, exports
├── plugin.py        # AntigravityAuthPlugin implementation
└── README.md        # This documentation
```
