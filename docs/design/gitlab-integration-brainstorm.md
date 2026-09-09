# Integrating GitLab with jaato — a brainstorm

**Status:** brainstorm, nothing implemented.
**Question:** what does "jaato supports GitLab" actually mean, and which of
the framework's existing seams should carry it?

---

## 1. What the tree contains today

Two incidental mentions, both prompt text, neither a code path:

| Site | What it says |
|------|--------------|
| `jaato-server/shared/plugins/service_connector/plugin.py:720` | `GITLAB_TOKEN` listed among "common env vars by service" in the auth-configuration help |
| `jaato-server/shared/plugins/web_fetch/plugin.py:445` | an example: `web_fetch(url=".../api/v4/projects", headers={"PRIVATE-TOKEN": "${GITLAB_TOKEN}"})` |

So the honest baseline is: **an agent can reach GitLab today the way it can
reach any HTTP API — by hand, one URL at a time, with no schema, no
pagination help, no event ingress, and no credential story.** There is no
GitLab-shaped anything.

The framework's own CI is GitHub Actions (`.github/workflows/`, nine
workflows). There is no `.gitlab-ci.yml`.

---

## 2. Four directions, not four options

"Integrate GitLab" collapses four different questions. They compose; picking
one does not settle the others.

| # | Direction | The agent's relationship to GitLab | Existing seam |
|---|-----------|-----------------------------------|---------------|
| **A** | **Outbound** — read and write GitLab | agent calls the API: read an MR diff, post a note, retry a pipeline, open an issue | `service_connector` / `mcp` / `cli` |
| **B** | **Inbound** — react to GitLab | GitLab calls jaato: MR opened, pipeline failed, comment mentioning the bot | `webhook` (unblocked by #930) |
| **C** | **Host** — jaato runs *inside* GitLab | a `.gitlab-ci.yml` job spawns the daemon and drives a session | runner / `runtime_limits` / AppArmor |
| **D** | **Provider** — GitLab Duo as a model backend | GitLab's AI Gateway serves models | `model_provider/` |

**D is the odd one out and should be parked.** `TRAIT_AUTH_PROVIDER`'s
contract in `shared/plugins/base.py` is explicitly *"provides interactive
authentication for a **model provider**"* — so a `gitlab_auth` plugin modelled
on `github_auth` would be a category error unless we are actually adding Duo
as a provider. Naming it that way early would mislead. If Duo ever lands it is
a normal `model_provider/` plugin and unrelated to A–C.

The interesting work is **A and B**, with **C** as the deployment shape that
makes both matter for GitLab-native shops.

---

## 3. Direction A — how should the agent talk to GitLab?

Four candidate seams already exist. This is the real decision.

### A1. `service_connector` (OpenAPI discovery)

GitLab publishes an OpenAPI document (`/api/v4/openapi/openapi.yaml` on any
instance, self-hosted included). `service_connector` already does the whole
job: `discover_service` parses the spec, `list_endpoints` browses it,
`get_endpoint_schema` hands the model the request shape, `call_service`
executes with validation.

What it gets for free, that a bespoke plugin would have to re-earn:

- **`AuthType.API_KEY`** (`service_connector/types.py:15`) covers GitLab's
  `PRIVATE-TOKEN` header directly; `BEARER` covers OAuth/CI job tokens.
- **`TRAIT_UNTRUSTED_CONTENT` is already declared on `call_service`**
  (`plugin.py:494`, alongside `TRAIT_GREPPABLE_CONTENT`). MR descriptions,
  issue bodies, review comments and job logs are attacker-controlled text —
  this is the single most important property of a GitLab integration and it
  comes for nothing here. A hand-rolled `gitlab` plugin has to *remember* to
  declare it, and the failure mode of forgetting is silent.
- **`TRAIT_GREPPABLE_CONTENT`** routes the full result dict through
  `result_grep`, which matters a lot: a merge-request diff or a pipeline log
  is exactly the "bulk content" that trait exists to shrink.
- **Self-hosted reality.** `_execute_discover_service` already handles SSL
  interception and corporate proxies — `insecure=true` persists as
  `ssl_trusted`, `no_proxy=true` persists as `proxy_bypass`, and a failed
  verified fetch returns a *structured* `ssl_error` with a hint telling the
  agent to ask the user before retrying. GitLab's enterprise deployment shape
  is precisely "self-hosted, behind an SSL-intercepting proxy, custom CA".
  This is the strongest single argument for A1.

Cost: GitLab's OpenAPI document is large and historically incomplete —
coverage of the v4 API is partial and some endpoints are documented only in
prose. So discovery gets you most of the surface, not all of it.

### A2. MCP server

`.mcp.json` + the `mcp` plugin, pointing at GitLab's MCP server. Zero
framework code.

- `mcp/plugin.py:493` already stamps `TRAIT_UNTRUSTED_CONTENT` on every MCP
  tool, so the security property holds here too.
- An MCP server's own `env` block in `.mcp.json` is an **explicit grant** and
  is therefore *not* scrubbed (see §4.2) — so the credential story is cleaner
  here than for `cli`.
- Cost: a third-party process in the loop, its own release cadence, its own
  tool vocabulary, and tool-name hashing (`t_<8 hex>`, `shared/tool_id_map.py`)
  because `mcp.gitlab.create_merge_request` does not pass the upstream
  `^[a-zA-Z0-9_-]{1,128}$` rule. All handled, but it is a moving part.

### A3. `glab` CLI via the `cli` plugin

Cheapest possible. `glab mr list`, `glab ci view`. Permission-gated like any
other command.

- Cost: **breaks by default.** See §4.2 — `GITLAB_TOKEN` matches the framework
  scrub pattern `*_TOKEN` and is stripped from every `cli` subprocess unless
  the profile opts out per-variable.
- Cost: output is human-formatted text, not structured — worse for the model
  than either A1 or A2, and it defeats `result_grep`.

### A4. A bespoke `gitlab` tool plugin

A first-party plugin with hand-written tools (`gitlab_mr_read`,
`gitlab_note_post`, …).

The case for it is real but narrow: **the API is not the workflow.** "Review
this MR" is six API calls (MR, diff, discussions, pipeline, jobs, notes) that
a generic connector makes the model orchestrate turn by turn. A purpose-built
`gitlab_review_context` tool returns all of it in one result. That is the same
argument that justifies any first-party plugin over `web_fetch`.

The case against: it is the only option that starts at zero on untrusted
content, pagination, self-hosted TLS, proxy handling, and schema validation —
every one of which A1 already has.

### Recommendation for A

**Start at A1, and let A4 grow out of it only where a measured workflow
demands it.** Concretely:

1. Ship a `.jaato/` profile and a stored `service_connector` schema set for
   GitLab v4 — no framework code, works against gitlab.com and self-hosted on
   day one.
2. Measure which workflows cost too many turns.
3. Add a thin `gitlab` plugin containing *only* the composite tools that
   measurement justifies (MR review context; pipeline-failure context), with
   `call_service` still available beneath for everything else.

A2 stays a supported alternative for shops that already run the MCP server.
A3 is a documented escape hatch, not a recommendation.

---

## 4. What actually blocks this today

Three concrete findings. The first **was** a defect and is now fixed; the
second is a trap; the third was already solved and is worth knowing.

### 4.1 GitLab webhook ingress — was blocked, fixed in `fa01b47` (#930 / #932)

**Resolved. Kept here because the shape of the fix constrains how direction B
should be configured.**

GitLab does not sign webhook bodies: it sends the configured secret verbatim in
`X-Gitlab-Token` and expects constant-time equality. `RouteConfig.secret_algo`
accepted only `hmac-sha256`, and `routes.py` fails closed on a half-configured
pair — so the only configuration that ingested a GitLab webhook was
`allow_unauthenticated: true`, the flag whose whole purpose is to be
unreachable by omission. A secret sat in the request and was thrown away.

`secret_algo` now names a **verification mode** rather than an algorithm
(`config.py:35`):

```python
SECRET_ALGO_HMAC_SHA256 = 'hmac-sha256'
SECRET_ALGO_TOKEN       = 'token'
SECRET_ALGOS = (SECRET_ALGO_HMAC_SHA256, SECRET_ALGO_TOKEN)
```

so a GitLab route is now expressible directly:

```json
"gitlab": {
  "path": "/webhook/gitlab",
  "secret_header": "X-Gitlab-Token",
  "secret_algo": "token",
  "event_type_header": "X-Gitlab-Event"
}
```

**The two modes are not peers, and direction B has to be built knowing it:**

| Mode | Header carries | Property |
|------|----------------|----------|
| `hmac-sha256` | an HMAC digest over the request **body** | the secret never travels; a captured request cannot be replayed against another payload |
| `token` | the shared secret **verbatim** | **weaker** — readable by any TLS-terminating hop, replayable against any payload |

GitLab leaves no choice of mode, so the compensating controls are the
deployment's: **terminate TLS at the listener** (`tls.enabled`), and prefer
`allowed_ips` where the instance has stable egress. `token` mode WARNs at
listener startup, louder when TLS is off (`http_server.py:200`) — the same
posture as `--ws-unsafe-no-auth`. That warning is a standing signal, not noise
to suppress.

Two properties of the fix worth carrying into any GitLab route review:

- **No cross-mode leniency.** `token` strips no `sha256=` prefix and never
  reads the body, so an HMAC digest does not authenticate a token route and
  vice versa. A route mistyped from one mode to the other fails shut.
- **An unknown algo is still a hard error** — set membership at
  `config.py:444`, and the incomplete-pair 500 is untouched. Widening the
  vocabulary widened what `secret_algo` may *say*, never what it may omit.

### 4.2 `GITLAB_TOKEN` is scrubbed by default (traps A3, and A2 if misconfigured)

`shared/secret_scrub.py`'s framework default set includes the glob `*_TOKEN`,
and scrubbing is **on by default** at three surfaces: `cli`,
`interactive_shell`, `mcp`. So `GITLAB_TOKEN` is removed from the environment
handed to every model-driven subprocess.

This is correct — it is exactly what the default is for — but it means the
naive `glab` setup fails with an authentication error that points nowhere near
the cause. The fix is one profile line, and it must be in whatever docs ship:

```yaml
plugins: [cli, service_connector]
scrub_secret_env: [default, "!GITLAB_TOKEN"]   # glab keeps its token;
                                               # the provider key stays out
```

Note the asymmetry that argues for A2 over A3: an MCP server's `env` block in
`.mcp.json` is an **explicit grant** and is not filtered — only the inherited
`os.environ` is. So the MCP route needs no exemption and never widens the
shell's view of the environment.

`service_connector` (A1) sidesteps this entirely: it runs in-process and
expands `${GITLAB_TOKEN}` itself, so no subprocess boundary is crossed and
nothing is scrubbed.

### 4.3 Self-hosted TLS and proxies — already handled

Worth stating because it is the usual reason a GitLab integration stalls in an
enterprise, and jaato has the answer already: `discover_service`'s structured
`ssl_error` / `proxy_error` returns with agent-facing hints, the persisted
`ssl_trusted` / `proxy_bypass` flags, plus `JAATO_SSL_VERIFY`,
`JAATO_KERBEROS_PROXY` and `JAATO_NO_PROXY` at the process level. A1 inherits
all of it; A4 would have to reimplement it.

---

## 5. Direction C — jaato inside GitLab CI

Separate from A and B, and probably the highest-leverage thing for a
GitLab-native shop: a `.gitlab-ci.yml` job that runs a jaato session against
the MR under review.

Sketch, and the questions it raises:

| Piece | Shape | Open question |
|-------|-------|---------------|
| daemon | `python -m server --ipc-socket /tmp/jaato.sock --daemon` in a job | is a daemon even right for a one-shot job, vs. the in-process facade? |
| credentials | `CI_JOB_TOKEN` is scoped and short-lived — a much better fit than a PAT | does `service_connector`'s `BEARER` auth accept it as-is? (believed yes) |
| confinement | GitLab's docker executor generally cannot load an AppArmor profile | `JAATO_REQUIRE_APPARMOR` must stay unset, so the session degrades to directory-sandbox isolation — this needs saying out loud, not discovering |
| limits | `runtime_limits.pids_max` / `max_parallel_tools` | a shared runner with a small `pids_max` is exactly the case `max_parallel_tools` (#862) was added for — likely wants `2`, not the default `8` |
| output | post findings back as MR notes | closes the loop with direction A |

The AppArmor row is the one that deserves a decision rather than a default:
running unconfined inside a CI runner is defensible (the runner is already
disposable), but it should be an explicit, documented posture.

---

## 6. Suggested staging

| Stage | Deliverable | Framework code? |
|-------|-------------|-----------------|
| 1 | GitLab profile + stored `service_connector` schemas; docs covering §4.2 | none |
| ~~2~~ | ~~`secret_algo: "token"` in the webhook plugin~~ — **landed in `fa01b47`** (#930 / #932), with a `gitlab` route example in CLAUDE.md | done |
| 3 | `.gitlab-ci.yml` reference job (§5) | none; docs + example |
| 4 | Thin `gitlab` plugin with composite workflow tools, *if* stage 1 measurement justifies it | new plugin |
| — | GitLab Duo as a `model_provider` | out of scope; unrelated to 1–4 |

Stage 2 was the only genuine framework defect rather than configuration, and
it has landed — so **stages 1, 3 and 4 are all docs, profiles and examples
until measurement says otherwise.** Nothing else here is blocked on framework
work.

---

## 7. Open questions

1. **Is direction B (webhook ingress) actually wanted**, or is GitLab CI
   (direction C) the trigger mechanism, making the webhook path unnecessary?
   These are alternative ways to be woken by an MR. With #930 landed neither
   needs framework code, so the choice is now purely operational: B means
   running a reachable TLS listener holding a replayable shared secret; C
   means a short-lived `CI_JOB_TOKEN` and no inbound surface at all. That
   argues for C as the default and B for events CI does not raise (a comment
   mentioning the bot, an MR opened without a pipeline).
2. **How complete is GitLab's OpenAPI document** on the endpoints that matter
   (merge request diffs, discussions, pipeline jobs)? This decides how much of
   A4 stage 4 is really needed. Worth measuring against a live instance before
   committing to A1.
3. **Self-hosted vs gitlab.com first?** They differ in auth (PAT vs OAuth vs
   CI job token) and in TLS posture. Picking one to target first keeps stage 1
   honest.
4. **Does the untrusted-content boundary need GitLab-specific wording?** MR
   review is a case where the agent reads attacker-controlled text *and* has
   write access to the same object. The generic boundary may be enough; it is
   worth checking rather than assuming.
