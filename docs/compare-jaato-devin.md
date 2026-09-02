# Jaato vs Devin: A Comprehensive Comparison

## Executive Summary

**Jaato** and **Devin** represent two fundamentally different approaches to AI-assisted software engineering. Jaato is an open-source, self-hosted framework for building multi-provider agentic tool orchestrators. Devin is a proprietary, cloud-hosted autonomous AI software engineer. The choice between them depends on whether you need a flexible toolkit to build your own AI agents (Jaato) or a turnkey autonomous coding assistant (Devin).

| Dimension | Jaato | Devin |
|-----------|-------|-------|
| **Philosophy** | Framework & toolkit | Autonomous AI employee |
| **Source** | Open-source (MIT) | Proprietary (closed-source) |
| **Hosting** | Self-hosted | Cloud-only (Cognition Labs) |
| **Model Support** | 8 providers, bring your own | Proprietary models (locked) |
| **Target User** | Teams building AI agents | Teams delegating tasks to AI |
| **Pricing** | Free (you pay for model APIs) | $20–$500+/month + ACU usage |

---

## 1. Architecture & Design Philosophy

### Jaato: Server-First Framework

Jaato is a **framework** — it provides the building blocks for constructing agentic AI applications rather than being an end-user product itself.

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  TUI Client │  │  Web Client │  │   Headless  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │ IPC            │ WebSocket       │ IPC
       └────────┬───────┴─────────┬───────┘
          ┌─────┴─────────────────┴─────┐
          │        Jaato Server          │
          │  ┌───────────────────────┐   │
          │  │    Session Manager    │   │
          │  └───────────────────────┘   │
          │  ┌───────────────────────┐   │
          │  │    Shared Runtime     │   │
          │  │  Providers │ Plugins  │   │
          │  │  Registry  │ Ledger   │   │
          │  └───────────────────────┘   │
          └──────────────────────────────┘
```

**Key design decisions:**
- **Server-first**: Daemon process with multi-client support (TUI, web, headless)
- **Pipeline-presentation split**: Server emits structured events; clients render independently
- **Plugin auto-wiring**: Zero-boilerplate extension mechanism
- **Model-first decision making**: Trusts the model to interpret user intent contextually
- **Resource sharing**: Subagents share runtime (providers, registry, permissions) but isolate conversation state

### Devin: Autonomous Agent Product

Devin is a **product** — a self-contained autonomous AI software engineer that operates in a cloud-hosted sandboxed environment.

```
User → Slack/Web UI → Devin Core (Agent)
                         ├── Shell (command line)
                         ├── Code Editor
                         ├── Web Browser
                         └── Sandboxed Container
                              └── Clone repo → Edit → Test → Push PR
```

**Key design decisions:**
- **Cloud-first**: All compute happens in Cognition's cloud infrastructure
- **Sandboxed environment**: Each task gets an isolated container with shell, editor, browser
- **End-to-end autonomy**: Plans, implements, tests, and deploys with minimal human input
- **Proprietary models**: Custom-trained models optimized for software engineering
- **Task-oriented**: Accepts a task, works autonomously, delivers a PR

---

## 2. Model Provider Support

### Jaato: Multi-Provider, Zero Lock-In

Jaato supports **8 model providers** through a provider-agnostic abstraction layer:

| Provider | Models | Auth Method |
|----------|--------|-------------|
| Google GenAI / Vertex AI | Gemini 2.5 Flash/Pro | Service account / ADC |
| Anthropic Claude | Opus, Sonnet, Haiku | PKCE OAuth or API key |
| Claude CLI | Claude via subscription | `claude login` |
| GitHub Models | Various | Device code OAuth / PAT |
| Google Antigravity | Gemini 3, Claude | Google OAuth |
| Ollama | Qwen, Llama, Mistral, etc. | Local (no auth) |
| ZhipuAI | ZhipuAI models | API key |
| NVIDIA NIM | Llama, DeepSeek-R1, Nemotron | API key / self-hosted |

Switching providers requires changing a single configuration parameter — no code changes. You can even run **cross-provider subagents** (e.g., Claude parent delegating to a Gemini subagent).

### Devin: Proprietary, Single-Provider

Devin uses **proprietary models** trained specifically by Cognition Labs. Users cannot:
- Choose which underlying model powers Devin
- Bring their own API keys for alternative models
- Run Devin against local/self-hosted models
- Inspect or modify model behavior

**Trade-off**: Devin's models are purpose-built for software engineering tasks, potentially offering better out-of-the-box performance for coding. Jaato's multi-provider approach offers flexibility and avoids vendor lock-in.

---

## 3. Tool & Plugin Ecosystem

### Jaato: 55+ Extensible Plugins

Jaato's plugin system covers tool execution, context management, memory, UI, and observability:

| Category | Examples | Count |
|----------|----------|-------|
| **System Tools** | CLI, interactive shell (PTY), environment | 3 |
| **File & Code** | file_edit, filesystem_query, ast_search, LSP, notebook | 6 |
| **Memory & State** | memory (persistent), session, todo, waypoint | 4 |
| **Context Management** | gc_truncate, gc_summarize, gc_hybrid, gc_budget | 4 |
| **Integration** | MCP servers, web_search, web_fetch | 3 |
| **Orchestration** | subagent, background tasks, parallel execution | 3 |
| **User Interaction** | permission, clarification, prompt_library | 3 |
| **Observability** | OpenTelemetry, token accounting, reliability | 3 |
| **Model Providers** | 8 provider plugins | 8 |
| **Other** | calculator, references, multimodal, vision, theming | 15+ |

Plugins support **deferred loading** (on-demand discovery) to minimize token overhead, and **tool traits** for cross-cutting behavior (e.g., `TRAIT_FILE_WRITER` triggers LSP diagnostics automatically).

### Devin: Built-In Toolset

Devin's toolset is fixed and optimized for autonomous coding:

| Tool | Capability |
|------|-----------|
| **Shell** | Execute commands, install packages, run tests |
| **Code Editor** | Read, write, and modify files |
| **Web Browser** | Research documentation, look up errors |
| **Desktop GUI** (v2.2+) | Operate GUI apps (Figma, Photoshop, browsers) |
| **Git/GitHub** | Clone repos, create branches, open PRs |
| **Devin Wiki** | Auto-generated codebase documentation |
| **Devin Review** | Self-reviewing PR system |

**Trade-off**: Devin's tools are tightly integrated and optimized for its autonomous workflow. Jaato's plugin system is far more extensible but requires assembly.

---

## 4. Multi-Agent / Subagent Support

### Jaato

- Subagents share parent's `JaatoRuntime` (providers, registry, permissions, token ledger) but get isolated `JaatoSession` instances
- **Fast spawning**: `create_session()` is lightweight
- **Cross-provider**: Claude parent can delegate to Gemini subagent
- **Predefined profiles**: Auto-discovery from `.jaato/profiles/`
- **Background execution**: Parallel subagent tasks
- **Cancellation propagation**: Parent can cancel child agents

### Devin

- Multi-agent operation since Devin 2.0 (April 2025)
- One Devin instance can dispatch sub-tasks to other Devin instances
- Parallel sessions available on Team plan ($500/month)
- All agents use the same proprietary model stack
- No cross-provider agent support

---

## 5. Context & Memory Management

### Jaato

Jaato provides **pluggable garbage collection** with four strategies:

| Strategy | Approach | Use Case |
|----------|----------|----------|
| **gc_truncate** | Remove oldest turns | Simple, predictable |
| **gc_summarize** | Compress old turns into summary | Preserve semantics |
| **gc_hybrid** | Age-based tiers (truncate/summarize/preserve) | Balanced |
| **gc_budget** | Policy-aware removal with 5-tier priority | Production |

GC is **proactive** — triggers automatically when context reaches a configurable threshold (default 80%). The `memory` plugin provides persistent cross-session knowledge with keyword indexing.

### Devin

- Long-context reasoning with planning capabilities
- Learns over time within a session
- **Devin Wiki**: Machine-generated documentation of your codebase
- Context management is opaque — handled internally by proprietary models
- No user-configurable GC strategies

---

## 6. Deployment & Hosting

### Jaato: Fully Self-Hosted

```bash
# Install
python3 -m venv .venv
.venv/bin/pip install -e jaato-sdk/. -e "jaato-server/.[all]" -e "jaato-tui/.[all]"

# Run as daemon
.venv/bin/python -m server --ipc-socket /tmp/jaato.sock --daemon

# Connect client
.venv/bin/python jaato-tui/rich_client.py --connect /tmp/jaato.sock
```

- Runs on your infrastructure (laptop, server, cloud VM)
- Data never leaves your network (when using local models via Ollama)
- Full control over security, compliance, and auditing
- Supports corporate proxy with Kerberos/SPNEGO authentication
- Configurable SSL verification for intercepting proxies

### Devin: Cloud-Only

- Runs entirely on Cognition Labs' cloud infrastructure
- Enterprise plan offers VPC deployment for data isolation
- Code is sent to Devin's sandboxed environment for processing
- Cannot run on-premises without Enterprise agreement
- Session boot time: ~15 seconds (improved from 45s in v2.2)

---

## 7. Pricing & Cost Model

### Jaato: Free (Open Source, MIT License)

| Cost Component | Amount |
|---------------|--------|
| Jaato framework | **Free** |
| Model API costs | Pay your provider (or $0 with Ollama) |
| Infrastructure | Your own compute |

You can run Jaato with **zero cost** by using Ollama with local models. When using cloud providers, you pay standard API rates.

### Devin: Subscription + Usage

| Plan | Monthly | Included ACUs | Extra ACU Cost |
|------|---------|--------------|----------------|
| **Core** | $20 | Pay-as-you-go | $2.25/ACU |
| **Team** | $500 | 250 ACUs | $2.00/ACU |
| **Enterprise** | Custom | Custom | Custom |

1 ACU ≈ 15 minutes of active Devin work. A typical 4-hour task consumes ~16 ACUs ($32–$36).

API access is only available on Team ($500/month) and Enterprise plans.

---

## 8. Transparency & Debuggability

### Jaato

- **Fully open source** (MIT): Read, modify, and audit every line of code
- **OpenTelemetry tracing**: Span hierarchy (`jaato.turn` → `jaato.tool` → `jaato.permission`)
- **Token accounting**: Detailed JSONL ledger of all token usage
- **Structured events**: 25+ typed events for complete observability
- **Plugin introspection**: Runtime self-inspection tools
- **Reliability policies**: Per-tool thresholds and monitoring

### Devin

- **Closed source**: Cannot inspect internal decision-making
- **Real-time progress**: Reports what it's doing via Slack/web
- **Session replay**: Review what Devin did step-by-step
- **Devin Review**: Self-reviewing PR descriptions
- **Limited debugging**: When Devin makes mistakes, root cause analysis is constrained by the opacity of its reasoning

---

## 9. Integration Ecosystem

### Jaato

| Integration | Method |
|-------------|--------|
| **MCP Servers** | First-class support via `.mcp.json` config |
| **Any CLI tool** | Via `cli` and `interactive_shell` plugins |
| **Language Servers** | LSP plugin for diagnostics |
| **Git** | Via CLI tools |
| **External APIs** | Via service_connector plugin |
| **Custom tools** | Write a plugin (Python) |

### Devin

| Integration | Method |
|-------------|--------|
| **Slack** | Native integration for task dispatch |
| **Linear** | Ticket management |
| **Jira** | Ticket management |
| **GitHub/GitLab** | PR creation, code review |
| **CI/CD** | Pipeline integration via API |
| **API** | REST API (Team/Enterprise only) |

---

## 10. Use Case Alignment

### When Jaato is the Better Choice

| Scenario | Why Jaato |
|----------|-----------|
| **Building custom AI agents** | Framework provides all building blocks |
| **Multi-provider strategy** | 8 providers, zero lock-in |
| **On-premises / air-gapped** | Fully self-hosted, local models via Ollama |
| **Data sovereignty** | Data never leaves your infrastructure |
| **Custom tool orchestration** | 55+ plugins, extensible architecture |
| **Research & experimentation** | Open source, full control, cross-provider agents |
| **Cost-sensitive teams** | Free framework + pay-per-use APIs (or free local models) |
| **Complex permission models** | Granular 8-level permission system |
| **Observability requirements** | OpenTelemetry, token ledger, structured events |

### When Devin is the Better Choice

| Scenario | Why Devin |
|----------|-----------|
| **Autonomous task completion** | "Give it a ticket, get a PR" workflow |
| **Junior engineer replacement** | Handles 4–8 hour tasks independently |
| **Parallelizable bulk work** | Migrations, vulnerability fixes, test writing at scale |
| **Non-technical stakeholders** | Slack-based interface, minimal setup |
| **Enterprise support** | Dedicated support, SSO, VPC deployment |
| **GUI-based tasks** (v2.2+) | Can operate Figma, Photoshop, browsers |
| **Quick start / no setup** | Works out of the box, no infrastructure needed |
| **Codebase documentation** | Devin Wiki auto-generates docs |

---

## 11. Performance & Maturity

### Jaato

- **Stage**: Open-source framework, v0.2.48 (server)
- **Test coverage**: Core tests + plugin tests + provider tests
- **Production use**: Designed for production with daemon mode, session persistence, token accounting
- **Community**: Open-source contributors

### Devin

- **Stage**: Commercial product, v2.2 (February 2026)
- **Track record**: Hundreds of thousands of merged PRs across thousands of companies
- **Enterprise adoption**: Goldman Sachs, Santander, Nubank
- **Benchmark**: 13.86% on SWE-bench (unassisted), 67% PR merge rate (up from 34%)
- **Revenue**: ~$73M ARR, ~$150M combined after Windsurf acquisition
- **Valuation**: ~$10.2B (Series C, 2025)

---

## 12. Summary: Fundamental Differences

| Aspect | Jaato | Devin |
|--------|-------|-------|
| **What it is** | Framework for building AI agents | Autonomous AI software engineer |
| **Analogy** | "Build your own robot" | "Hire a robot employee" |
| **Control** | Full (open source, self-hosted) | Limited (cloud, proprietary) |
| **Flexibility** | Maximum (8 providers, 55+ plugins) | Minimal (fixed toolset, single provider) |
| **Setup effort** | Higher (install, configure, extend) | Lower (sign up, connect repo) |
| **Autonomy level** | Configurable (human-in-the-loop to autonomous) | High (autonomous by default) |
| **Cost floor** | $0 (local models) | $20/month + ACU usage |
| **Cost ceiling** | Your API usage | Unbounded (ACU-based) |
| **Vendor lock-in** | None | Significant |
| **Data control** | Complete | Limited (cloud processing) |
| **Extensibility** | Plugin system, custom tools | API (Team/Enterprise only) |

They are complementary rather than competing: Jaato could orchestrate Devin as one of many tools in a larger agentic workflow, or teams might use Jaato for custom agent development while using Devin for delegated autonomous tasks.
