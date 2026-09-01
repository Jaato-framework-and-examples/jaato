# Env vars vs profile keys — which knobs earned a typed key

*Issue [#775](https://github.com/Jaato-framework-and-examples/jaato/issues/775).
The assessment lives in code (`jaato-server/shared/env_scope.py`) and is
enforced by `jaato-server/shared/tests/test_env_scope_catalog.py`; this document
is the argument behind it.*

## The question, and the question it is not

Some env vars have no typed equivalent anywhere in the profile /
`plugin_configs` surface. Which of those earned one, and which are correctly
env-only?

It is worth being precise about what is *not* being asked. `SubagentProfile.env`
is a free-form `Dict[str, str]` with `${VAR}` expansion and secret-URI support,
applied for the session's duration. **Every** env var is therefore already
session-scoped if an author writes it there. The question is not reachability;
it is whether a knob deserves a **typed, validated, discoverable** key instead
of a stringly-typed passthrough.

## What that distinction costs, concretely

`JAATO_PROVIDER_TRACE` is a *path* — the file the provider request/response
trace is written to. Set through a profile's `env:` block as

```yaml
env:
  JAATO_PROVIDER_TRACE: "1"
```

every session wrote its provider trace to a file literally named `1`,
including eval-arm workspaces, contaminating the very trees a comparative judge
was diffing. Nothing rejected it, and nothing was in a position to: `env` is
`Dict[str, str]` and `"1"` is a valid string.

**The obvious diagnosis is wrong, and the correction is the interesting part.**
It is tempting to say the defect was a *relative* path, and to require absolute
paths the way the client handshake does for these same two fields (#742). That
would have broken a documented pattern: `jaato_sdk.trace._resolve_trace_file`
joins a relative trace path onto `JAATO_WORKSPACE_ROOT`, which the runner seeds
per session — so a relative value is precisely how you give each session its own
trace file in its own workspace, and `explain env` teaches it.

The defect is narrower and more human: `"1"` is a **switch written into a path
field**. The author reached for a boolean, and a string-typed map had no way to
notice. The typed `trace:` block shipped with this assessment refuses the
boolean vocabulary and a value naming a directory, and passes every real path —
absolute or relative — through untouched.

That is the whole argument for typed keys, in one incident: not that env vars
are bad, but that a `Dict[str, str]` cannot tell a path from a switch, and a
typed field can.

## Method

Built from scaffold's own introspection: `explain env --json` (186 vars, 35
categories) compared against the profile schema **including its nested blocks**
and against `explain provider <p> --json` knobs for all 18 providers.

The nested comparison is the correction the issue called for. A first pass
compared env names against *top-level* profile fields only and reported ~110
"orphans"; that number is wrong and should not be quoted. `JAATO_GC_THRESHOLD`
/ `_TARGET` / `_PRESSURE` are the clearest false positives — they are the
`default_factory` for `GCConfig.threshold_percent` / `target_percent` /
`pressure_percent`, all settable under the profile's `gc:` block. Counting them
as gaps is how a triage becomes noise.

Every `typed_key` in the catalog is **resolved by the guard** against the real
dataclass field or provider knob it names. "Already covered by a nested block"
is a claim, and an unverified claim is indistinguishable from a wrong one — the
~110 number is what unverified claims cost.

## The answer

186 env vars, all classified:

| scope | count | meaning |
|-------|-------|---------|
| `session` | 130 | a knob two sessions on one host may legitimately differ on — **87 have a typed key, 43 do not** |
| `host` | 30 | process- or host-scoped; a per-session value would be meaningless or a lie |
| `ambient` | 20 | the host environment being **read**, not configured — not a knob at all |
| `internal` | 6 | one framework process handing a value to another |

### The fourth scope

The issue proposed three tags. `internal` came out of doing the classification
and earns its place by what it prevents. Tagging `JAATO_RUNNER_SESSION_ID` as
`host` would be false — it is per session. Tagging it `session` would demand a
profile key for a value the daemon computes and hands to its own subprocess.
Both readings invite someone to "fix" it. `internal` says: this is a
framework-to-framework handoff, and neither an operator nor a profile author
should ever set it. The six are `JAATO_DAEMONIZED`, `JAATO_SESSION_ID`, and the
four `JAATO_RUNNER_*` handoff vars.

### Correctly env-only, stated so nobody "fixes" them

The 30 `host` vars are the issue's tier C, confirmed and extended: the cgroup
root, the pre-warm pool flags, the WS token, the proxy and TLS-trust vars, the
OTel collector wiring, and the developer toggles (`JAATO_DUMP_PROVIDER_REQUEST`,
`AI_TOOL_RUNNER_DEBUG`, `JAATO_BOOTSTRAP_TIMING`) whose whole purpose is to be
global. Each carries a one-line note saying why; the guard requires the note,
because an unexplained tag cannot stop a later reader from removing it.

The 20 `ambient` vars are the issue's tier D — `PATH`, `TERM`, `HOME`, `USER`,
`SHELL`, `TMUX`, `MSYSTEM`, `APPDATA`, `XDG_CONFIG_HOME`, `COLORTERM`,
`PSModulePath`, `ComSpec`, `workspaceRoot` and friends. `env_scope.is_knob()`
excludes them, so tooling can present a knob view without them inflating it.

## The ratchet: 43 session knobs with no typed key

These are declared in `AWAITING_TYPED_KEY`, with a tier. **The set may only
shrink.** The guard derives the same set from the catalog on every run and
fails when the two disagree, so promoting a knob is a deletion and adding an
untyped session knob is an addition — both in a diff, under review, next to the
words "may only shrink". `test_session_env_audit.py`'s ALLOWLIST is the same
mechanism for the orthogonal question of read *route*, and is the repository's
own precedent.

### Tier A — agent-behaviour knobs (14)

Two profiles in one sweep may legitimately want opposite values; today each is
one host-wide setting.

| var | why it wants a key |
|-----|--------------------|
| `JAATO_PARALLEL_TOOLS`, `JAATO_DEFERRED_TOOLS` | tool-loop behaviour: a cheap model may want deferred tools off while an expensive one wants them on, in the same sweep |
| `AI_EXECUTE_TOOLS` | lets unregistered tools run via the generic executor — an agent-behaviour switch |
| `AI_REQUEST_INTERVAL`, `AI_RETRY_ATTEMPTS`, `AI_RETRY_BASE_DELAY`, `AI_RETRY_MAX_DELAY` | retry policy is per-provider in character (an OpenRouter 402 and a local vLLM stall want different budgets) but is set once per process |
| `JAATO_PERMISSION_TIMEOUT`, `JAATO_CLARIFICATION_TIMEOUT` | a headless cascade arm and an interactive TUI session want opposite values, and one daemon serves both |
| `JAATO_TELEMETRY_ENABLED`, `_BACKEND`, `_EXPORTER`, `_FILE`, `_REDACT_CONTENT` | per-session tracing is exactly what you want on for one profile and not the rest |

Note the shape of the retry group: the natural home is not a top-level profile
field but a per-provider one, since the *reason* they differ is the provider.
That is a design question the promotion has to answer, which is why it is
tracked rather than guessed at here.

### Tier B — plugin knobs (11)

Each belongs to exactly one plugin that already has a `plugin_configs.<plugin>`
namespace, so the typed home exists and only the wiring is missing.

| var | home |
|-----|------|
| `JAATO_FILE_BACKUP_COUNT` | `plugin_configs.file_edit` |
| `JAATO_AMBIGUOUS_WIDTH` | `plugin_configs.table_formatter` |
| `JAATO_MERMAID_BACKEND` / `_THEME` / `_SCALE`, `JAATO_KROKI_URL`, `JAATO_VISION_DIR` | `plugin_configs.mermaid_formatter` |
| `JAATO_TOOL_BINDINGS` | `plugin_configs.notebook` |
| `JAATO_SESSION_CONFIG` | `plugin_configs.session` |
| `JAATO_SESSION_LOG_DIR` | the daemon's own session-log root |
| `LEDGER_PATH` | the token ledger |

Three of these — `JAATO_VISION_DIR`, `JAATO_SESSION_LOG_DIR`, `LEDGER_PATH` —
are **paths**, the same shape as the incident. `LEDGER_PATH` is worse than the
others: it is read as `os.environ.get("LEDGER_PATH", filepath)`, so the env var
*outranks* the typed argument the caller passed. An env var that beats a typed
parameter is the inversion this whole issue is about.

`TODO_STORAGE_PATH` was in the issue's tier B and is **not** in the ratchet:
`plugin_configs.todo.storage_path` already exists and already wins, with the env
var as its default. That is the shape every tier-B promotion should copy.

### Tier E — credentials (18)

These need a *policy* decided before a key is added, not a default. The
inconsistency the issue names is real and now measured. Ten of the eighteen
providers expose `api_key` / `api_token` / `oauth_token` as a knob (anthropic,
doubleword, lmstudio, nebius, nim, openrouter, ovhcloud, tensorrt_llm, triton,
vllm). Of the eight that do not, five need no credential at all —
`antigravity` and `claude_cli` authenticate out of band (OAuth, a logged-in
CLI), `ollama` and `chrome_ai` are local and unauthenticated, and
`zhipuai_openai` reads its sibling's key. That leaves **three genuine gaps**:
`github_models` (`GITHUB_TOKEN`), `google_genai` (`GOOGLE_GENAI_API_KEY`,
`GOOGLE_APPLICATION_CREDENTIALS`) and `zhipuai` (`ZHIPUAI_API_KEY`). Alongside them sit the non-provider credentials:
`LANGFUSE_*`, `KAGGLE_*`, `PERMISSION_WEBHOOK_TOKEN`, `TODO_WEBHOOK_TOKEN`.

**Proposed policy.** A provider or plugin that authenticates with a
bearer-style secret should expose the knob, because the knob is what makes a
`pass://` / `vault://` URI usable *per profile* — the framework's rule is that
secrets stay daemon-side and resolve from a URI rather than living as literals,
and a knob is the only place a profile can put that URI. The knob is a
**reference site, not a storage site**: its value should normally be a secret
URI, and a literal in a YAML file is the anti-pattern the knob exists to
replace. Two vars in this tier are not secrets at all and are simpler:
`JAATO_GOOGLE_USE_VERTEX` and the two `*_AUTH_METHOD` selectors pick a backend,
and `PROJECT_ID` / `LOCATION` are connection identity — `antigravity` already
exposes `project_id`, so `google_genai` lacking `project_id` / `location` is
drift rather than a decision.

This policy is proposed here, not applied: applying it means adding knobs to
three providers and four plugins, and it should be adopted deliberately.

## What shipped with this assessment

1. **The tagged catalog** — `shared/env_scope.py`, all 186 vars with scope,
   typed key and reason. Surfaced by `jaato-scaffold explain env`: every row
   carries its scope glyph and its typed key, a summary block heads the
   listing, and `explain env untyped` lists exactly the 43. `explain env host`
   / `ambient` / `internal` filter by scope.
2. **One promotion, the one the incident argues for** — the typed `trace:`
   profile block (`session_log` / `provider_log`), which refuses `"1"` at
   profile-parse time. It seeds the two env vars into the session env above the
   `env:` map and below the post-auth overrides, so the env vars remain the
   lower-precedence default and nothing downstream reads the block. **That
   costlessness is the template**: a promotion is a validated *producer* of an
   env var the framework already reads, not a new reader.
3. **The guard** — `test_env_scope_catalog.py`, in the required `contract-guards`
   CI job, failing on an unclassified var, a stale entry, a `typed_key` that
   names nothing, or a ratchet that disagrees with the catalog. It declares
   `REVERSIONS`, so `test_every_guard_detects_its_own_reversion.py` proves it
   still discriminates.

The 43 entries still in the ratchet are deliberately *not* promoted here. The issue's own
triage is explicit that it is "the case for doing it, not the finished answer",
and two of its items dissolved under the nested comparison (`gc.*`,
`TODO_STORAGE_PATH`) while a third turned out to need a policy rather than a
key. Promoting 43 knobs on the strength of a triage that was wrong about three
of them would repeat the mistake the catalog exists to stop. They are tracked
where a tracker cannot be ignored: in a set that fails CI when it grows.

## Relation to `test_session_env_audit.py`

That audit is about the **read route** — `get_session_env` versus a direct
`os.environ` read, which breaks workspace isolation. This one is about the
**write surface** — a typed key versus a stringly-typed map. A var can pass one
and fail the other; they are orthogonal and both are needed.

## Adding an env var after this

The guard tells you, but for the record: classify it in `CATALOG` with the
scope that is *true* of it. If it is `session`, either give it a `typed_key`
(and check the guard can resolve it) or add it to `AWAITING_TYPED_KEY` with its
tier — which is a request for review, not a formality.
