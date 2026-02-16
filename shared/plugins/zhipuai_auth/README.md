# Zhipu AI Auth Plugin

Authentication plugin for Zhipu AI (Z.AI Coding Plan) via API key.

## Session-Independent

This plugin is **session-independent** (`SESSION_INDEPENDENT = True`). Its commands
are available at daemon startup, before any session or provider connection exists.
This is essential because authentication must happen *before* connecting to the
Zhipu AI provider.

## Commands

| Command | Description |
|---------|-------------|
| `zhipuai-auth login` | Show instructions for getting your Z.AI API key |
| `zhipuai-auth key <api_key>` | Validate and store your API key |
| `zhipuai-auth logout` | Clear stored API credentials |
| `zhipuai-auth status` | Show current authentication status |
| `zhipuai-auth help` | Show detailed help |

### Authentication Flow

1. User runs `zhipuai-auth login` to see instructions
2. User visits https://open.bigmodel.cn/ and creates an API key
3. User runs `zhipuai-auth key <api_key>` to store it
4. Plugin validates and persists the key securely

Alternatively, set the `ZHIPUAI_API_KEY` environment variable directly.

## Post-Auth Session Bootstrap

After successful authentication, the daemon detects valid credentials via
`verify_credentials()` and offers an interactive session setup wizard:

1. **Connect prompt** — *"Connect to Zhipu AI (Z.AI) now? [y/n]"*
   (or *"Switch to..."* if a session with a different provider is already active)
2. **Model selection** — Presents the default model list:
   - `zhipuai/glm-4.7` — Latest flagship with native CoT reasoning (200K)
   - `zhipuai/glm-4.7-flash` — Fast inference variant (200K)
   - `zhipuai/glm-4.6` — Previous flagship, strong coding (200K)
3. **Persistence prompt** — *"Save provider/model to .env? [y/n]"*
   Writes `JAATO_PROVIDER=zhipuai` and `MODEL_NAME=<model>` to the workspace
   `.env` file, preserving existing values for other keys.

This flow is handled by `PostAuthSetupEvent` / `PostAuthSetupResponse` events
between the daemon and the rich client. See `server/__main__.py` for the daemon
side and `jaato-tui/rich_client.py` for the client-side wizard.

## Plugin Protocol

```python
class ZhipuAIAuthPlugin:
    provider_name = "zhipuai"
    provider_display_name = "Zhipu AI (Z.AI)"

    def get_default_models() -> List[Dict[str, str]]   # Curated model list
    def verify_credentials() -> bool                     # Check API key in keychain / env
    def get_user_commands() -> List[UserCommand]          # Command declarations
    def get_command_completions(cmd, args) -> List[...]   # Subcommand autocompletion
    def execute_user_command(cmd, args) -> str | HelpLines
```

## File Structure

```
shared/plugins/zhipuai_auth/
├── __init__.py      # PLUGIN_KIND, SESSION_INDEPENDENT, exports
├── plugin.py        # ZhipuAIAuthPlugin implementation
└── README.md        # This documentation
```
