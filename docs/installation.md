# Installing Jaato

Jaato is distributed as three PyPI packages. Install the ones you need based on your use case.

## Packages

| Package | Description | Installs |
|---------|-------------|----------|
| `jaato-sdk` | Lightweight client library for building custom Jaato clients | IPC/WebSocket protocol, `JaatoClient` API |
| `jaato-server` | Runtime daemon with all plugins and AI providers | Everything in `jaato-sdk` plus tool orchestration, model providers, CLI tools |
| `jaato-tui` | Terminal user interface (the rich client) | Everything in `jaato-sdk` plus interactive TUI, themes, keybindings |

## Quick Install (End Users)

Most users want both the server and the TUI client:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install jaato-server jaato-tui
```

After installation, two commands are available:

- `jaato-server` — start/stop/manage the Jaato daemon
- `jaato` — launch the TUI client

## SDK-Only Install (Developers)

For developers building custom clients or embedding Jaato in applications:

```bash
pip install jaato-sdk
```

This gives you `JaatoClient` and the IPC protocol — no server, no TUI, no plugins.

## Optional Extras (jaato-server)

The `jaato-server` package supports optional feature groups:

```bash
# AST-based code search
pip install "jaato-server[ast]"

# Vision support (SVG to PNG conversion for screenshots)
pip install "jaato-server[vision]"

# Mermaid diagram rendering
pip install "jaato-server[mermaid]"

# PDF to markdown conversion (for web_fetch)
pip install "jaato-server[pdf]"

# Kaggle dataset integration
pip install "jaato-server[kaggle]"

# All optional features
pip install "jaato-server[all]"
```

## Optional: OpenTelemetry

For tracing and observability:

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
```

## Developer Install (From Source)

For contributors working on Jaato itself:

```bash
git clone https://github.com/apanoia/jaato.git
cd jaato
python3 -m venv .venv
source .venv/bin/activate
pip install -e jaato-sdk/ -e . -e rich-client/
```

The `-e` flag installs in editable mode so source changes take effect immediately.

## Requirements

- Python 3.10 or later (3.10, 3.11, 3.12 supported)
- A supported terminal emulator (for `jaato-tui`)
- At least one AI provider configured (see provider documentation)

## Verifying Installation

```bash
# Check server
jaato-server --help

# Check TUI
jaato --help

# Start server and connect
jaato-server --ipc-socket /tmp/jaato.sock --daemon
jaato --connect /tmp/jaato.sock
```
