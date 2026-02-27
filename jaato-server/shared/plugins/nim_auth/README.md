# NVIDIA NIM Auth Plugin

Authentication plugin for NVIDIA NIM via API key.

## Session-Independent

This plugin is **session-independent** (`SESSION_INDEPENDENT = True`). Its commands
are available at daemon startup, before any session or provider connection exists.
This is essential because authentication must happen *before* connecting to the
NIM provider.

## Commands

| Command | Description |
|---------|-------------|
| `nim-auth login` | Show instructions for getting your NIM API key |
| `nim-auth key <api_key>` | Validate and store your API key |
| `nim-auth logout` | Clear stored API credentials |
| `nim-auth status` | Show current authentication status |
| `nim-auth help` | Show detailed help |

### Authentication Flow

1. User runs `nim-auth login` to see instructions
2. User visits https://build.nvidia.com/ and generates an API key (nvapi-...)
3. User runs `nim-auth key <api_key>` to store it
4. Plugin validates and persists the key securely

Alternatively, set the `JAATO_NIM_API_KEY` environment variable directly.

For self-hosted NIM containers, no API key is needed — set `JAATO_NIM_BASE_URL`
to your container endpoint.

## Post-Auth Session Bootstrap

After successful authentication, the daemon detects valid credentials via
`verify_credentials()` and offers an interactive session setup wizard:

1. **Connect prompt** — *"Connect to NVIDIA NIM now? [y/n]"*
   (or *"Switch to..."* if a session with a different provider is already active)
2. **Model selection** — Presents the default model list:
   - `nim/meta/llama-3.3-70b-instruct` — Llama 3.3 70B, strong general-purpose reasoning
   - `nim/meta/llama-3.1-405b-instruct` — Llama 3.1 405B, largest open model
   - `nim/deepseek/deepseek-r1` — DeepSeek-R1, reasoning with chain-of-thought
   - `nim/nvidia/llama-3.1-nemotron-70b-instruct` — Nemotron 70B, NVIDIA-tuned Llama
3. **Persistence prompt** — *"Save provider/model to .env? [y/n]"*
   Writes `JAATO_PROVIDER=nim` and `MODEL_NAME=<model>` to the workspace
   `.env` file, preserving existing values for other keys.

This flow is handled by `PostAuthSetupEvent` / `PostAuthSetupResponse` events
between the daemon and the rich client. See `server/__main__.py` for the daemon
side and `jaato-tui/rich_client.py` for the client-side wizard.

## Plugin Protocol

```python
class NIMAuthPlugin:
    provider_name = "nim"
    provider_display_name = "NVIDIA NIM"

    def get_default_models() -> List[Dict[str, str]]   # Curated model list
    def verify_credentials() -> bool                     # Check API key in stored / env
    def get_user_commands() -> List[UserCommand]          # Command declarations
    def get_command_completions(cmd, args) -> List[...]   # Subcommand autocompletion
    def execute_user_command(cmd, args) -> str | HelpLines
```

## File Structure

```
shared/plugins/nim_auth/
├── __init__.py      # PLUGIN_KIND, SESSION_INDEPENDENT, exports
├── plugin.py        # NIMAuthPlugin implementation
└── README.md        # This documentation
```
