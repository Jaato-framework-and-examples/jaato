# Application Identity — naming the app, not the framework

## The problem

Open the OpenRouter console for any jaato deployment and every session, from
every harness anyone built on the SDK, reports as one application: **jaato**.

That is not a display quirk. The provider hardcoded the framework's own name
and repository as the app-attribution headers:

```python
# shared/plugins/model_provider/openrouter/env.py (before)
DEFAULT_HTTP_REFERER = "https://github.com/Jaato-framework-and-examples/jaato"
DEFAULT_APP_TITLE = "jaato"
```

Two things follow, and both are wrong:

1. **An integrator cannot see their own product.** Spend, request counts and
   model mix for "Acme Copilot" are indistinguishable from every other
   jaato-based tool sharing that key or that ranking row.
2. **jaato's own app-attribution entry is an aggregate of other people's
   products** — a ranking that says nothing about the framework.

The framework's name is not the application's name. They were the same string.

## The shape of the fix

[`shared/app_identity.py`](../../jaato-server/shared/app_identity.py)
introduces `AppIdentity` — *the application*, as distinct from the framework
it is built with. Both reach the upstream:

```
X-OpenRouter-Title: Acme Copilot (powered by jaato)
HTTP-Referer:       https://acme.example
```

The `(powered by jaato)` suffix is the default rather than an option: an
integrator naming their app should not have to think about whether the
framework keeps the credit it was getting before. A white-labelled product
turns it off explicitly.

| Field | Meaning |
|-------|---------|
| `name` | Display name of the application. Defaults to `jaato`. |
| `url` | The application's own site/repo — what OpenRouter attributes rankings to. Falls back to the framework's repository. |
| `version` | The *application's* version, not the framework's. Used by `user_agent()`. |
| `powered_by` | Whether attribution appends `(powered by jaato)`. Default `True`; ignored when the identity *is* the framework. |

Three derived forms:

- `attribution_title()` → `"Acme Copilot (powered by jaato)"`
- `attribution_url()` → the app's URL, else the framework's
- `user_agent()` → `"Acme-Copilot/1.4.0 (powered by jaato/0.7.0)"`

`user_agent()` has no consumer inside the framework yet. It exists so the next
provider or HTTP client that wants to identify the caller does not re-derive
the convention — and so the framework version in it comes from installed
distribution metadata rather than a literal that can drift.

## How an author sets it

Four surfaces, highest precedence first. The first two already existed and are
unchanged; the mechanism is the last two.

| # | Surface | Scope | Use when |
|---|---------|-------|----------|
| 1 | `plugin_configs.openrouter.app_title` / `http_referer` | one session | one profile must attribute differently from the rest |
| 2 | `JAATO_OPENROUTER_APP_TITLE` / `JAATO_OPENROUTER_HTTP_REFERER` | process / session env | you are tuning OpenRouter specifically |
| 3 | `JaatoRuntime(app_identity=AppIdentity(...))` | the embedding process | your product embeds the framework in-process |
| 4 | `JAATO_APP_NAME` / `JAATO_APP_URL` / `JAATO_APP_VERSION` / `JAATO_APP_POWERED_BY` | deployment | a daemon started by your app, a workspace `.env`, a profile's `env:` map |

```bash
# tier 4 — the deployment surface
export JAATO_APP_NAME="Acme Copilot"
export JAATO_APP_URL="https://acme.example"
export JAATO_APP_VERSION="1.4.0"
```

```python
# tier 3 — the embedding surface
from shared.app_identity import AppIdentity
from shared.jaato_runtime import JaatoRuntime

runtime = JaatoRuntime(
    app_identity=AppIdentity(
        name="Acme Copilot",
        url="https://acme.example",
        version="1.4.0",
    ),
)
```

A client connecting to a daemon it does not start reaches tier 4 through its
workspace `.env` (the daemon loads it into the session env via
`JaatoServer._resolve_session_env`) or through the profile's `env:` map. There
is deliberately no `ClientConfigRequest` field: see *Open question* below.

**With none of the four set, nothing changes.** The resolved identity is the
framework's own, `JaatoRuntime` stamps nothing onto the provider config, and
OpenRouter receives exactly the headers it received before.

## How it reaches the provider

```
AppIdentity (kwarg)  or  JAATO_APP_* (env)
        │
        ▼
JaatoRuntime._inject_session_extras()      ← alongside session_id / workspace_path
        │  extra["app_identity"] = {...}   ← only when an app was actually named
        ▼
ProviderConfig.extra
        │
        ▼
OpenRouterProvider.initialize()
        │  profile knob  >  JAATO_OPENROUTER_*  >  app_identity  >  framework
        ▼
HTTP-Referer / X-OpenRouter-Title
```

Two properties of that path are deliberate:

**The identity is resolved per provider creation, never cached.** The daemon
overlays a session's `env` onto `os.environ` for the duration of a turn, so an
identity frozen at import (or at runtime construction) would attribute every
session to whoever started the process.

**The stamp carries a plain dict, not the dataclass.** Any provider can
consume it without importing the type, and it survives a config that gets
serialized.

## Why the env vars are `host`-scoped

`shared/env_scope.py` tags the four `JAATO_APP_*` vars **`host`**, which means
they carry no typed profile key. That is a claim, so here is the argument:
*which application this is* is a property of the deployment, not of a
conversation, and two sessions in one process disagreeing about who is
spending the money would be a lie rather than a configuration.

Per-session attribution is a real need — it is exactly what tier 1 serves, and
tier 1 outranks everything here.

## Header safety

Every field is sanitised at construction: non-printable characters (CR and LF
above all) are dropped and the value is length-capped, because these strings
are written verbatim into HTTP headers. This is not hypothetical hygiene — an
app name is attacker-influenced in exactly the deployment that most wants this
feature: a hosted product naming itself after the tenant it is serving.

## Scope, and what is not here

- **OpenRouter is the only consumer today** because it is the only provider
  with an app-attribution protocol. Anthropic, Google and the OpenAI-compatible
  providers have no equivalent header; when one gains a `User-Agent` worth
  setting, `AppIdentity.user_agent()` is already the value to send.
- **No typed profile block.** An `app:` key on `SubagentProfile` would have to
  be wired through six profile ingresses and the state snapshot, to express
  something the profile's own `env:` map already expresses. If per-session app
  identity turns out to be a real workflow rather than a theoretical one, that
  is when the typed block earns its cost.

### Open question — a client-declared identity

An SDK author whose app is a *client* of a shared daemon declares itself today
through env (workspace `.env` / profile `env:`). The typed alternative would be
a `ClientConfigRequest.app` field, mirroring `presentation`: the client says
who it is on the handshake, and the daemon applies it to that client's
sessions.

That is the right shape for a multi-tenant daemon serving several distinct
applications, and it is a small protocol addition — but it needs the identity
to reach the runner at *bootstrap* (provider construction), not through the
post-hoc RPC that `presentation` uses. Deferred until a deployment actually
needs one daemon to attribute several applications at once.
