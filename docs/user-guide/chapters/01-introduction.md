# Introduction

## What is Jaato?

**Jaato** ("just another agentic tool orchestrator") is a framework for AI-powered development that enables natural language interaction with AI models while giving them the ability to execute tools, manage files, and assist with complex software engineering tasks.

The **rich client** is an interactive terminal user interface (TUI) that provides a powerful, keyboard-driven environment for working with AI models through Jaato. It combines the flexibility of command-line tools with a modern, feature-rich interface.

## Key Features

The Jaato rich client offers:

- **Multi-Provider Support**: Connect to Google Vertex AI, Anthropic Claude, GitHub Models, Ollama, and more
- **Interactive TUI**: Modern terminal interface with panels, theming, and keyboard shortcuts
- **Tool Orchestration**: AI can execute commands, edit files, and use MCP servers
- **Session Management**: Persistent conversations with context management
- **Permission System**: Fine-grained control over AI tool execution
- **Headless Mode**: Programmatic API for automation and CI/CD
- **Extensibility**: Plugin system for custom tools and providers

## Architecture Overview

Jaato uses a **client-server architecture**:

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│             │   IPC   │             │   API   │             │
│ Rich Client ├────────>│   Server    ├────────>│  AI Model   │
│    (TUI)    │  Socket │   (Daemon)  │  Calls  │  Provider   │
└─────────────┘         └─────────────┘         └─────────────┘
                              │
                              │ Plugins
                              v
                        ┌─────────────┐
                        │    Tools    │
                        │ (CLI, MCP,  │
                        │  File Edit) │
                        └─────────────┘
```

**Benefits of this architecture:**

- **Multi-client support**: Multiple TUI sessions can connect to the same server
- **Resource efficiency**: Server runs as daemon, clients are lightweight
- **Session persistence**: Conversations persist even if client disconnects
- **Isolation**: Server handles API credentials, clients handle presentation

## When to Use the Rich Client

The rich client is ideal for:

- **Interactive development**: Get AI assistance while coding
- **Exploratory tasks**: Iterate on ideas with immediate feedback
- **Learning**: Understand codebases with AI guidance
- **Prototyping**: Quickly test ideas and approaches

For automation and scripting, consider **headless mode** (see Chapter 15).

## What You'll Learn

This guide covers:

1. **Getting Started**: Installation, setup, and first conversation
2. **Core Features**: Commands, permissions, sessions, and UI
3. **Authentication**: Configuring providers and authentication
4. **Customization**: Keybindings, themes, and configuration files
5. **Advanced Usage**: Headless mode, MCP integration, and optimization

By the end of this guide, you'll be able to effectively use the rich client for your AI-assisted development workflow.

## Prerequisites

To use the Jaato rich client, you should have:

- **Basic terminal knowledge**: Comfort with command-line interfaces
- **Python 3.10+**: Required for running Jaato
- **AI provider access**: At least one configured provider (Google, Anthropic, etc.)
- **Development environment**: For file editing and tool execution features

No prior AI or ML experience is required—the rich client handles all the complexity of interacting with AI models.

## Getting Help

Throughout this guide, you'll find:

- **Tip boxes** with helpful suggestions
- **Warning boxes** for important caveats
- **Code examples** demonstrating usage
- **Screenshots** showing the UI in action

If you encounter issues:

1. Check the **Troubleshooting** chapter (Chapter 18)
2. Consult the **Environment Variables** reference (Appendix C)
3. Review the **Glossary** (Appendix E)
4. Report issues at: <https://github.com/apanoia/jaato/issues>

Let's get started!
