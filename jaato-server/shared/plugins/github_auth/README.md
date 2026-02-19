# GitHub Auth Plugin

Authentication plugin for GitHub Models API (device code OAuth flow).

## Session-Independent

This plugin is **session-independent** (`SESSION_INDEPENDENT = True`). Its commands
are available at daemon startup, before any session or provider connection exists.
This is essential because authentication must happen *before* connecting to the
GitHub Models provider.

## Commands

| Command | Description |
|---------|-------------|
| `github-auth login` | Start device code OAuth flow with GitHub |
| `github-auth poll` | Poll for authorization after entering code |
| `github-auth logout` | Clear stored OAuth tokens |
| `github-auth status` | Show current authentication status |
| `github-auth help` | Show detailed help |

### Authentication Flow

1. User runs `github-auth login`
2. Plugin displays a device code and verification URL
3. User opens the URL in a browser and enters the device code
4. User runs `github-auth poll` to check if authorization completed
5. Plugin stores the OAuth token securely

Alternatively, set the `GITHUB_TOKEN` environment variable with a Personal Access
Token that has `models: read` permission.

## Post-Auth Session Bootstrap

After successful authentication, the daemon detects valid credentials via
`verify_credentials()` and offers an interactive session setup wizard:

1. **Connect prompt** — *"Connect to GitHub Models now? [y/n]"*
   (or *"Switch to..."* if a session with a different provider is already active)
2. **Model selection** — Presents the default model list:
   - `github/gpt-4o` — GPT-4o
   - `github/gpt-4o-mini` — GPT-4o mini
   - `github/o3-mini` — O3 mini
3. **Persistence prompt** — *"Save provider/model to .env? [y/n]"*
   Writes `JAATO_PROVIDER=github_models` and `MODEL_NAME=<model>` to the workspace
   `.env` file, preserving existing values for other keys.

This flow is handled by `PostAuthSetupEvent` / `PostAuthSetupResponse` events
between the daemon and the rich client. See `server/__main__.py` for the daemon
side and `jaato-tui/rich_client.py` for the client-side wizard.

## Plugin Protocol

```python
class GitHubAuthPlugin:
    provider_name = "github_models"
    provider_display_name = "GitHub Models"

    def get_default_models() -> List[Dict[str, str]]   # Curated model list
    def verify_credentials() -> bool                     # Check OAuth token / env var
    def get_user_commands() -> List[UserCommand]          # Command declarations
    def get_command_completions(cmd, args) -> List[...]   # Subcommand autocompletion
    def execute_user_command(cmd, args) -> str | HelpLines
```

## File Structure

```
shared/plugins/github_auth/
├── __init__.py      # PLUGIN_KIND, SESSION_INDEPENDENT, exports
├── plugin.py        # GitHubAuthPlugin implementation
└── README.md        # This documentation
```
