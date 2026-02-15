# Installation

This chapter guides you through installing the Jaato Rich Client and its dependencies. We cover system requirements, virtual environment setup, dependency installation, and verification steps to ensure your installation is ready for use.

## System Requirements

### Operating Systems

Jaato runs on all major operating systems:

- **Linux**: Any modern distribution (Ubuntu 20.04+, Debian 11+, Fedora 35+, Arch Linux, etc.)
- **macOS**: 10.15 (Catalina) or later
- **Windows**: Windows 10 or later with Windows Subsystem for Linux (WSL)

> **Note:** Windows native support is limited. We recommend using WSL2 for the best experience. Some plugins (particularly `interactive_shell`) rely on Unix-specific system calls.

### Python Version

Jaato requires **Python 3.10 or later**. Python 3.12 is recommended for optimal performance.

To verify your Python version:

```bash
python3 --version
```

If Python is not installed or the version is older than 3.10:

- **Linux**: Install via your package manager (e.g., `sudo apt install python3.12`)
- **macOS**: Install via Homebrew (`brew install python@3.12`)
- **Windows (WSL)**: Follow the Linux instructions within your WSL environment

### Hardware Requirements

Jaato's hardware requirements are modest:

- **RAM**: 4 GB minimum, 8 GB recommended
- **Disk Space**: 500 MB for core installation, plus additional space for Python virtual environments
- **Network**: Internet connection required for AI provider API calls and dependency installation

### AI Provider Account

Jaato requires an account with at least one supported AI provider:

| Provider | Use Case |
|----------|----------|
| **Google GenAI** | Personal development, quick prototyping (API Key authentication) |
| **Vertex AI** | Organization/enterprise workloads (GCP credentials) |
| **Anthropic Claude** | Alternative LLM provider (API key or OAuth) |
| **GitHub Models** | Access via GitHub (OAuth or Personal Access Token) |
| **Ollama** | Local model execution (requires Ollama installation) |

> **Tip:** For beginners, Google GenAI (AI Studio) provides the quickest setup with an API key. See [docs/gcp-setup.md](../../gcp-setup.md) for detailed provider configuration.

## Virtual Environment Setup

We strongly recommend using a Python virtual environment to isolate Jaato from your system Python and other projects. This prevents dependency conflicts and makes management easier.

### Creating a Virtual Environment

Navigate to your desired project directory and create a virtual environment:

```bash
# Navigate to your project directory
cd ~/projects/jaato-project

# Create virtual environment named .venv
python3 -m venv .venv
```

The `.venv` directory contains an isolated Python installation with its own package registry.

### Activating the Virtual Environment

Activation differs by operating system:

**Linux/macOS:**

```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

When active, your shell prompt will show `(.venv)` as a prefix:

```bash
(.venv) user@hostname:~/projects/jaato-project$
```

> **Warning:** Always activate your virtual environment before running Jaato commands. Without activation, Python will use your system-wide installation, which may lack required dependencies.

### Deactivating the Virtual Environment

When finished working with Jaato, deactivate the virtual environment:

```bash
deactivate
```

The prompt prefix disappears, returning you to your system Python.

## Installing Jaato

Jaato consists of three packages, but most users only need to install the server and TUI client together. The SDK package is for developers building custom clients.

### Installation Methods

#### Method 1: Install from PyPI (Recommended for Users)

When Jaato is published to PyPI, installation is straightforward:

```bash
# Install the server and TUI client
pip install jaato-server jaato-tui

# Or install with optional features
pip install "jaato-server[vision]" jaato-tui  # PNG screenshot support
pip install "jaato-server[all]" jaato-tui     # All optional dependencies
```

#### Method 2: Install from Source (Required for Contributors)

For development or to access the latest unreleased features:

```bash
# Clone the repository
git clone https://github.com/apanoia/jaato.git
cd jaato

# Install all packages in development/editable mode
pip install -e jaato-sdk/ -e . -e rich-client/
```

The `-e` flag installs packages in "editable" mode, allowing code changes to take effect without reinstallation.

#### Method 3: Install with Optional Extras

Jaato supports optional feature groups for extended functionality:

```bash
# Core installation (no extras)
pip install -e jaato-sdk/ -e . -e rich-client/

# Development tools (pytest, testing)
pip install -e ".[dev]"

# Vision capture (PNG screenshots, requires libcairo2-dev on Linux)
pip install -e ".[vision]"

# All optional dependencies
pip install -e ".[all]"
```

**Optional dependency groups:**

| Group | Description |
|-------|-------------|
| `dev` | pytest and development tools |
| `vision` | CairoSVG for PNG screenshot conversion |
| `ast` | AST-based code search (ast-grep-py) |
| `kaggle` | Kaggle dataset integration |
| `mermaid` | Mermaid diagram parsing |
| `pdf` | PDF to markdown conversion |
| `all` | All optional dependencies combined |

> **Tip:** If you only need the client library for building your own interface, install just the SDK:
>
> ```bash
> pip install jaato-sdk/
> ```

### System-Specific Dependencies

Some plugins have platform-specific dependencies:

**Linux (for vision capture):**

```bash
sudo apt-get install libcairo2-dev  # Debian/Ubuntu
sudo dnf install cairo-devel        # Fedora
```

**Windows:**

The interactive shell plugin requires additional packages automatically installed via `pyproject.toml`:

- `pywin32` - Windows API bindings
- `psutil` - Process and system utilities
- `setuptools` - For pkg_resources support

## Environment Configuration

Jaato uses environment variables for configuration. You can set these in your shell profile (`.bashrc`, `.zshrc`) or in a `.env` file in your project directory.

### Minimal Configuration

At minimum, configure your AI provider. For Google GenAI (AI Studio):

```bash
export GOOGLE_GENAI_API_KEY=your-api-key
```

Or for Vertex AI:

```bash
export PROJECT_ID=your-gcp-project-id
export LOCATION=us-central1
export MODEL_NAME=gemini-2.5-flash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### Using a .env File

Create a `.env` file in your project directory:

```bash
cat > .env << 'EOF'
# AI Provider Configuration
GOOGLE_GENAI_API_KEY=your-api-key

# Optional: Specify model
MODEL_NAME=gemini-2.5-flash

# Optional: Function calling
AI_USE_CHAT_FUNCTIONS=1

# Optional: Verbose logging
VERBOSE=1
EOF
```

The `.env` file is automatically loaded by the jaato client on startup.

### Common Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `PROJECT_ID` | GCP project ID | Required for Vertex AI |
| `LOCATION` | Vertex AI region | `us-central1` |
| `MODEL_NAME` | Model identifier | `gemini-2.5-flash` |
| `GOOGLE_GENAI_API_KEY` | Google AI Studio API key | Auto-detected |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `GITHUB_TOKEN` | GitHub Personal Access Token | - |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `AI_USE_CHAT_FUNCTIONS` | Enable function calling | `0` |
| `JAATO_PARALLEL_TOOLS` | Parallel tool execution | `true` |
| `JAATO_DEFERRED_TOOLS` | Deferred tool loading | `true` |
| `VERBOSE` | Enable verbose logging | `1` |

See [CLAUDE.md](../../CLAUDE.md) for a complete list of environment variables and advanced configuration options.

## Verification

After installation, verify that Jaato is correctly installed and configured.

### Check Installation

Verify the jaato client is accessible:

```bash
# Check TUI client
jaato --help

# Or if running from source
python rich-client/rich_client.py --help
```

Expected output includes usage information and available options.

### Check Python Dependencies

Verify all required packages are installed:

```bash
pip list | grep -E "jaato|google-genai|anthropic|mcp|prompt_toolkit|rich"
```

You should see packages like:
- `jaato-sdk`
- `jaato-server`
- `jaato-tui` (or installed in editable mode)
- `google-genai`
- `anthropic`
- `prompt_toolkit`
- `rich`

### Verify AI Provider Access

Test your AI provider credentials:

```bash
# Run the simple connectivity test
python simple-connectivity-test/simple-connectivity-test.py
```

Expected output:

```
Testing connection to Vertex AI...
Project: your-project-id
Location: us-central1
Model: gemini-2.5-flash

✓ Connection successful
✓ Model accessible
```

If this fails, verify your credentials:
- Check that `GOOGLE_GENAI_API_KEY` or `GOOGLE_APPLICATION_CREDENTIALS` is set
- Ensure the API key is valid and not expired
- Verify network connectivity to the provider endpoint

### Run Test Suite

To verify your installation thoroughly, run the included test suite:

```bash
# Run all tests
pytest

# Run specific test directories
pytest shared/tests/
pytest shared/plugins/cli/tests/

# Verbose mode
pytest -v
```

> **Tip:** The test suite requires pytest. If not installed, add it via `pip install -e ".[dev]"`.

All tests should pass with a green status indicator. Failures may indicate missing dependencies or platform incompatibilities.

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'google'`**

```bash
# Solution: Install missing dependencies
pip install google-genai google-api-core google-auth
```

**Issue: `Command 'jaato' not found`**

```bash
# Solution: Ensure bin directory is in PATH, or use full path
export PATH=$HOME/.local/bin:$PATH

# Or run directly with Python
python -m rich_client.rich_client
```

**Issue: `Permission denied` when creating virtual environment**

```bash
# Solution: Ensure directory is writable or choose a different location
python3 -m venv ~/.venv-jaato
source ~/.venv-jaato/bin/activate
```

**Issue: OpenTelemetry import errors**

```bash
# Solution: Install optional telemetry dependencies
pip install -r requirements-telemetry.txt
export JAATO_TELEMETRY_ENABLED=true
```

**Issue: Interactive shell hangs on Windows**

The `interactive_shell` plugin requires WSL. Windows native support is limited. Use WSL2 for full functionality.

### Platform-Specific Notes

**macOS:** If you encounter SSL certificate errors, install certifi:

```bash
pip install certifi
```

**Linux:** Ensure system-level SSL certificates are up-to-date:

```bash
sudo apt-get install ca-certificates  # Debian/Ubuntu
sudo update-ca-certificates
```

**Windows (WSL):** Some Windows paths may not be accessible from WSL. Keep your workspace within the WSL filesystem (`/home/`) rather than `/mnt/c/`.

## Next Steps

With Jaato installed and verified, proceed to:

- **Chapter 03: Quick Start** - Your first interactive session
- **Chapter 04: Configuration** - Advanced customization options
- **Provider Setup Guides** - Detailed configuration for specific AI providers

For additional help:

- **Documentation**: [docs/](../../) for architecture and design details
- **Examples**: [demo-scripts/](../../demo-scripts/) for YAML-driven demos
- **Issues**: [GitHub Issues](https://github.com/apanoia/jaato/issues)