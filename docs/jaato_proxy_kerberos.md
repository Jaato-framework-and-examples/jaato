# JAATO Proxy & Kerberos Authentication

## Executive Summary

JAATO provides unified proxy support and **Kerberos/SPNEGO authentication** for enterprise environments where outbound traffic must traverse HTTP proxies. The `shared/http/` module abstracts proxy configuration across three HTTP libraries (urllib, requests, httpx) and provides two key improvements over standard proxy handling: **exact host matching** via `JAATO_NO_PROXY` (compared to the standard `NO_PROXY` suffix matching) and **transparent Kerberos token generation** via `pyspnego` for corporate proxies requiring SPNEGO/Negotiate authentication.

---

## Part 1: The Enterprise Proxy Problem

### Why Standard Proxy Support Is Not Enough

Enterprise environments impose strict requirements that standard `HTTP_PROXY`/`NO_PROXY` variables don't fully address:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE NETWORK TOPOLOGY                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Developer Workstation                                               │
│  ┌────────────────┐                                                 │
│  │  JAATO          │                                                 │
│  │  ┌────────────┐ │     ┌──────────────────┐     ┌──────────────┐ │
│  │  │ Anthropic  │─┼────►│  Corporate Proxy  │────►│ api.anthropic│ │
│  │  │ Provider   │ │     │  (Kerberos auth)  │     │ .com         │ │
│  │  └────────────┘ │     └──────────────────┘     └──────────────┘ │
│  │  ┌────────────┐ │     ┌──────────────────┐     ┌──────────────┐ │
│  │  │ Google     │─┼────►│  Corporate Proxy  │────►│ vertex AI    │ │
│  │  │ Provider   │ │     │  (Kerberos auth)  │     │ endpoint     │ │
│  │  └────────────┘ │     └──────────────────┘     └──────────────┘ │
│  │  ┌────────────┐ │                                                │
│  │  │ GitHub     │─┼────►  Direct (no proxy)  ────► github.com     │
│  │  │ Provider   │ │     JAATO_NO_PROXY=github.com                  │
│  │  └────────────┘ │                                                │
│  │  ┌────────────┐ │                                                │
│  │  │ MCP Server │─┼────►  Direct (localhost)  ──► localhost:8080   │
│  │  │            │ │     Standard NO_PROXY=localhost                 │
│  │  └────────────┘ │                                                │
│  └────────────────┘                                                 │
│                                                                      │
│  Challenges:                                                         │
│  1. Proxy requires Kerberos/SPNEGO authentication (not basic auth)  │
│  2. Some hosts must bypass proxy (exact match, not suffix match)     │
│  3. Multiple HTTP libraries used (urllib, requests, httpx)           │
│  4. Both sync and async HTTP clients need proxy support              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Configuration

### Environment Variables

| Variable | Standard | Description |
|----------|----------|-------------|
| `HTTPS_PROXY` | Yes | Proxy URL for HTTPS requests (e.g., `http://proxy:8080`) |
| `HTTP_PROXY` | Yes | Proxy URL for HTTP requests |
| `NO_PROXY` | Yes | Hosts to bypass proxy (suffix matching: `.corp.com` matches `api.corp.com`) |
| `JAATO_NO_PROXY` | JAATO-specific | Hosts to bypass proxy (exact matching: `github.com` only) |
| `JAATO_KERBEROS_PROXY` | JAATO-specific | Enable Kerberos/SPNEGO proxy auth (`true`/`false`) |

### NO_PROXY vs JAATO_NO_PROXY

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HOST MATCHING COMPARISON                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Standard NO_PROXY (suffix matching):                                │
│  ────────────────────────────────────                                │
│  NO_PROXY=".corp.com,localhost"                                      │
│                                                                      │
│  api.corp.com      → MATCH (suffix .corp.com)                       │
│  dev.corp.com      → MATCH (suffix .corp.com)                       │
│  malicious-corp.com → MATCH! (suffix .corp.com — unintended)        │
│  localhost          → MATCH (exact)                                  │
│                                                                      │
│  JAATO_NO_PROXY (exact matching):                                    │
│  ────────────────────────────────                                    │
│  JAATO_NO_PROXY="github.com,api.github.com"                         │
│                                                                      │
│  github.com        → MATCH (exact)                                   │
│  api.github.com    → MATCH (exact)                                   │
│  evil-github.com   → NO MATCH (no suffix matching)                  │
│  gist.github.com   → NO MATCH (not listed)                          │
│                                                                      │
│  JAATO_NO_PROXY provides deterministic, safe bypass behavior.        │
│  Both can be used simultaneously — JAATO_NO_PROXY is checked first.  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Kerberos/SPNEGO Authentication

### How It Works

When `JAATO_KERBEROS_PROXY=true`, JAATO generates SPNEGO tokens using the system's Kerberos credentials (kinit on Linux/Mac, domain login on Windows):

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SPNEGO AUTHENTICATION FLOW                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Initial Request (no auth)                                        │
│  ┌──────────┐                     ┌──────────────┐                  │
│  │  JAATO   │ ── GET /api ──────► │  Corporate   │                  │
│  │  Client  │                     │  Proxy       │                  │
│  │          │ ◄── 407 Proxy Auth ─│  (Negotiate) │                  │
│  └──────────┘    Required         └──────────────┘                  │
│                                                                      │
│  2. SPNEGO Token Generation                                          │
│  ┌──────────┐                     ┌──────────────┐                  │
│  │  JAATO   │ ── call ──────────► │  pyspnego    │                  │
│  │  Client  │                     │  library     │                  │
│  │          │ ◄── SPNEGO token ── │              │                  │
│  └──────────┘                     └──────┬───────┘                  │
│                                          │                           │
│                                   ┌──────┴───────┐                  │
│                                   │  Platform    │                  │
│                                   │  Kerberos    │                  │
│                                   │              │                  │
│                                   │  Windows:SSPI│                  │
│                                   │  macOS:GSS   │                  │
│                                   │  Linux:MIT   │                  │
│                                   └──────────────┘                  │
│                                                                      │
│  3. Authenticated Request                                            │
│  ┌──────────┐                     ┌──────────────┐    ┌──────────┐ │
│  │  JAATO   │ ── GET /api ──────► │  Corporate   │───►│ Target   │ │
│  │  Client  │    + Proxy-Auth:    │  Proxy       │    │ API      │ │
│  │          │    Negotiate <token> │              │    │          │ │
│  │          │ ◄── 200 OK ─────────│              │◄───│          │ │
│  └──────────┘                     └──────────────┘    └──────────┘ │
│                                                                      │
│  pyspnego creates a context for HTTP/<proxy_host> service principal  │
│  The SPNEGO token is base64-encoded and added to Proxy-Authorization│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Platform Support

| Platform | Kerberos Backend | Credential Source |
|----------|-----------------|-------------------|
| **Windows** | SSPI (native) | Domain login (automatic) |
| **macOS** | GSS.framework | `kinit` or Kerberos ticket cache |
| **Linux** | MIT Kerberos | `kinit` (must have valid ticket) |

### Prerequisites

```bash
# Install pyspnego
pip install pyspnego

# Linux: obtain Kerberos ticket
kinit user@CORP.EXAMPLE.COM

# Verify ticket
klist

# Configure proxy
export HTTPS_PROXY=http://proxy.corp.com:8080
export JAATO_KERBEROS_PROXY=true
```

---

## Part 4: Multi-Library Support

The `shared/http/` module provides factory functions for three HTTP libraries, ensuring consistent proxy behavior regardless of which library a provider or plugin uses:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HTTP LIBRARY ADAPTERS                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  shared/http/proxy.py                                                │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Configuration Layer (shared across all adapters)              │   │
│  │                                                               │   │
│  │ get_proxy_url() → Optional[str]                               │   │
│  │ is_kerberos_proxy_enabled() → bool                            │   │
│  │ should_bypass_proxy(url) → bool                               │   │
│  │ generate_spnego_token(host) → Optional[str]                   │   │
│  └──────────────────┬──────────────┬──────────────┬─────────────┘   │
│                      │              │              │                  │
│           ┌──────────┴──┐  ┌───────┴──────┐  ┌───┴──────────────┐  │
│           │ urllib       │  │ requests     │  │ httpx            │  │
│           │              │  │              │  │ (sync + async)   │  │
│           ├──────────────┤  ├──────────────┤  ├──────────────────┤  │
│           │get_url_opener│  │get_requests_ │  │get_httpx_client  │  │
│           │(url)         │  │session()     │  │(**kwargs)        │  │
│           │              │  │              │  │                  │  │
│           │Kerberos-     │  │get_requests_ │  │get_httpx_async_  │  │
│           │ProxyHandler  │  │kwargs(url)   │  │client(**kwargs)  │  │
│           │(407 handler) │  │              │  │                  │  │
│           │              │  │              │  │get_httpx_kwargs   │  │
│           │PreAuth-      │  │              │  │(url)             │  │
│           │ProxyHandler  │  │              │  │                  │  │
│           └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                                      │
│  Used By:                                                            │
│  ─────────                                                           │
│  urllib:    OAuth flows, device code polling                         │
│  requests:  GitHub Copilot client, web fetch plugin                  │
│  httpx:     Anthropic provider, Google Antigravity provider          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Decision Logic (Per Request)

```
should_bypass_proxy(url)?
    │
    ├── YES → No proxy (direct connection)
    │
    └── NO → is_kerberos_proxy_enabled()?
             │
             ├── YES → get_proxy_url()
             │         │
             │         ├── Proxy found → generate_spnego_token(proxy_host)
             │         │                 │
             │         │                 ├── Token OK → Proxy + SPNEGO header
             │         │                 └── Token fail → Proxy + 407 handler
             │         │
             │         └── No proxy → Default (no proxy)
             │
             └── NO → Default (standard proxy env vars)
```

---

## Part 5: urllib Adapter

The urllib adapter provides two handler classes:

### PreAuthProxyHandler

Proactively attaches the SPNEGO token to the initial request, avoiding the 407 round-trip:

```python
class PreAuthProxyHandler(urllib.request.ProxyHandler):
    def proxy_open(self, req, proxy, type):
        req.add_header("Proxy-Authorization", f"Negotiate {self.spnego_token}")
        return super().proxy_open(req, proxy, type)
```

### KerberosProxyHandler

Handles 407 responses reactively if the pre-auth token is stale or missing:

```python
class KerberosProxyHandler(urllib.request.BaseHandler):
    def http_error_407(self, req, fp, code, msg, hdrs):
        # Check Negotiate is supported
        # Generate fresh SPNEGO token
        # Retry with Proxy-Authorization header
```

Both handlers are installed together for defense-in-depth: pre-auth avoids the extra round-trip, while the 407 handler recovers if the pre-auth token expires.

---

## Part 6: requests and httpx Adapters

### requests

```python
from shared.http import get_requests_session, get_requests_kwargs

# Session-level configuration (recommended for multiple requests)
session = get_requests_session()
response = session.get("https://api.anthropic.com/v1/messages")

# Per-request configuration
kwargs = get_requests_kwargs("https://api.anthropic.com/v1/messages")
response = requests.get(url, **kwargs)
```

### httpx (Sync and Async)

```python
from shared.http import get_httpx_client, get_httpx_async_client

# Synchronous
with get_httpx_client() as client:
    response = client.get("https://api.anthropic.com/v1/messages")

# Asynchronous
async with get_httpx_async_client() as client:
    response = await client.get("https://api.anthropic.com/v1/messages")
```

---

## Part 7: Integration with Providers

Each model provider and plugin uses the shared HTTP module for outbound requests:

| Component | HTTP Library | Proxy-Aware Via |
|-----------|-------------|-----------------|
| **Anthropic Provider** | httpx | `get_httpx_client()` |
| **Google GenAI Provider** | google-genai SDK | Standard env vars (SDK handles) |
| **GitHub Models Provider** | azure-ai-inference | `get_httpx_client()` |
| **Antigravity Provider** | httpx | `get_httpx_async_client()` |
| **GitHub OAuth** | httpx | `get_httpx_client()` |
| **Anthropic OAuth** | urllib | `get_url_opener()` |
| **Web Fetch Plugin** | requests | `get_requests_session()` |
| **Notebook Plugin** | requests | `get_requests_kwargs()` |

---

## Part 8: Security Considerations

| Concern | How Addressed |
|---------|--------------|
| **Token freshness** | SPNEGO tokens are generated per-request or per-session; 407 handler refreshes stale tokens |
| **Credential exposure** | Kerberos tickets are managed by the OS; JAATO never sees raw passwords |
| **Proxy bypass safety** | `JAATO_NO_PROXY` uses exact matching to prevent unintended suffix matches |
| **TLS interception** | Proxy operates at the CONNECT tunnel level; end-to-end TLS is maintained |
| **Multi-account** | Kerberos uses the currently active ticket; `kinit` switches identity |

---

## Part 9: Deployment Patterns

### Pattern 1: Corporate Proxy with Kerberos

```bash
export HTTPS_PROXY=http://proxy.corp.example.com:8080
export JAATO_KERBEROS_PROXY=true
export JAATO_NO_PROXY=localhost,127.0.0.1
kinit user@CORP.EXAMPLE.COM
```

### Pattern 2: Proxy with Some Direct Access

```bash
export HTTPS_PROXY=http://proxy.corp.example.com:8080
export JAATO_KERBEROS_PROXY=true
export JAATO_NO_PROXY=github.com,api.github.com
export NO_PROXY=localhost,.internal.corp.com
```

### Pattern 3: No Proxy (Default)

```bash
# No proxy variables set — all connections go direct
# This is the default behavior
```

---

## Part 10: Related Documentation

| Document | Focus |
|----------|-------|
| [jaato_multi_provider.md](jaato_multi_provider.md) | Provider abstraction (each provider uses shared HTTP) |
| [jaato_model_harness.md](jaato_model_harness.md) | Overall harness architecture |
| [architecture.md](architecture.md) | Server-first architecture overview |

---

## Part 11: Color Coding Suggestion for Infographic

- **Blue:** Configuration layer (environment variables, proxy URL resolution)
- **Green:** HTTP library adapters (urllib, requests, httpx)
- **Orange:** Kerberos/SPNEGO token generation (pyspnego, platform backends)
- **Red:** Security boundary (proxy authentication, 407 challenge-response)
- **Purple:** Provider integration (which providers use which adapter)
- **Gray:** Network topology (client → proxy → target)
- **Yellow:** Decision flow arrows (bypass checks, Kerberos checks)
