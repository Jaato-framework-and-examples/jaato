"""The env-var scope catalog: which knobs earned a typed profile key.

WHAT THIS ANSWERS.  ``jaato-scaffold explain env`` lists every env var the
installed tree reads -- 186 of them -- by scanning ``os.getenv`` sites.  That
list says what is READ.  It cannot say what a var *is*: whether it is an agent
knob that two sessions on one host may legitimately disagree about, a
process-wide setting where a per-session value would be a lie, or the host
environment being read rather than configured.  Without that distinction a
naive diff of env names against profile fields reports ~110 "orphans", most of
which are neither orphaned nor knobs (issue #775).

This module is the missing half: a scope tag and, where one exists, the typed
profile key that already covers the var.

WHY A TYPED KEY IS NOT THE SAME AS REACHABILITY.  ``SubagentProfile.env`` is a
free-form ``Dict[str, str]`` with ``${VAR}`` expansion and secret-URI support,
so **every** env var is already session-scoped if an author writes it there.
The question this catalog answers is not "can I set it per session" but
"is it typed, validated and discoverable when I do".

The cost of the difference, concretely::

    env:
      JAATO_PROVIDER_TRACE: "1"      # accepted -- it is a valid str

``JAATO_PROVIDER_TRACE`` is a *path*.  Every session then wrote its provider
trace to a file literally named ``1`` -- one per session, in whatever directory
that session resolved a relative trace path against, eval arm workspaces
included, contaminating the trees a comparative judge was diffing.
Nothing rejected it, because nothing was in a position to: ``env`` is
``Dict[str, str]``.  The typed :class:`~shared.plugins.subagent.config.TraceProfileConfig`
block added alongside this catalog refuses that value at profile-parse time --
not because it is relative (a relative trace path is the supported per-session
idiom, resolved against the workspace) but because it is a *switch written into
a path field*, which is what a string-typed map cannot notice.
That incident is the whole argument, and the catalog exists so the next one is
found before it happens rather than after.

THE FOUR SCOPES.  The issue proposed three; the fourth (:data:`INTERNAL`) came
out of doing the classification, and earns its place by what it prevents.
Tagging ``JAATO_RUNNER_SESSION_ID`` as ``host`` would be false (it is per
session), and tagging it ``session`` would demand a profile key for a value the
daemon computes and hands to its own subprocess.  Both readings invite someone
to "fix" it.  ``internal`` says: this is a framework-to-framework handoff, and
neither an operator nor a profile author should ever set it.

  ``session``   a knob two sessions on one host may legitimately differ on.
                Wants a typed key; :data:`AWAITING_TYPED_KEY` lists the ones
                that do not have one yet.
  ``host``      process- or host-scoped.  A per-session value would be
                meaningless or a lie (the cgroup root, the pre-warm pool size,
                the WS token, proxy and TLS trust).
  ``ambient``   the host environment being READ, not configured (``PATH``,
                ``TERM``, ``HOME``, ``MSYSTEM``).  Not a knob at all; these
                inflate any naive diff and are excluded from the knob view.
  ``internal``  set by one framework process and read by another.  Not an
                operator surface in either direction.

HOW THIS STAYS TRUE.  ``shared/tests/test_env_scope_catalog.py`` re-derives the
env-var list from the installed source on every run and fails when the catalog
and the code disagree in either direction -- an unclassified var, a stale
entry, a ``typed_key`` naming a profile field or provider knob that does not
exist, or a session-scoped var missing from the ratchet below.  A catalog that
is not re-derived is a document, and documents drift.

RELATION TO ``test_session_env_audit.py``.  That audit is about the READ ROUTE
(``get_session_env`` versus a direct ``os.environ`` read, which breaks
workspace isolation).  This one is about the WRITE SURFACE (a typed key versus
a stringly-typed map).  A var can pass one and fail the other; they are
orthogonal and both are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# ---------------------------------------------------------------- scopes

SESSION = "session"
HOST = "host"
AMBIENT = "ambient"
INTERNAL = "internal"

#: Every legal :attr:`EnvClass.scope` value.  Iterated by the guard and by
#: ``explain env`` so a new scope cannot be introduced by a typo.
SCOPES = (SESSION, HOST, AMBIENT, INTERNAL)


@dataclass(frozen=True)
class EnvClass:
    """One env var's scope, and the typed key that covers it (if any).

    Attributes:
        scope: One of :data:`SCOPES`.
        typed_key: Dotted path of the typed equivalent, or ``None`` when
            none exists yet.  Three notations, all resolvable and all
            resolved by the guard:

            * a profile field or nested block field -- ``model``,
              ``gc.threshold_percent``, ``trace.provider_log``;
            * a provider or plugin config path --
              ``plugin_configs.openrouter.api_key``,
              ``plugin_configs.anthropic.api_params.thinking_budget``;
            * a client handshake field -- ``client.working_dir``.  Typed and
              validated, but on the wire rather than in a profile: some
              knobs belong to the connection, not the agent.
        note: Why this scope, in one line.  Load-bearing for the ``host`` and
            ``internal`` entries -- their whole job is to stop a later reader
            "fixing" something that is already right.
    """

    scope: str
    typed_key: Optional[str] = None
    note: str = ""


@dataclass(frozen=True)
class Awaiting:
    """A session knob with no typed key yet, and where one should go.

    Attributes:
        tier: ``A`` agent-behaviour knob, ``B`` plugin knob, ``E`` credential.
        proposed_key: Where the typed key SHOULD live, in the same notation
            :attr:`EnvClass.typed_key` uses.  Unlike ``typed_key`` this is a
            proposal, so the guard checks its SHAPE (a known prefix naming a
            real plugin or provider) and deliberately cannot resolve it --
            the thing it names does not exist yet.  That asymmetry is the
            point: a resolvable ``typed_key`` is coverage, a ``proposed_key``
            is an argument.  When one lands, the entry moves to ``typed_key``
            and leaves this set.
        note: Why this key, or what stands in the way -- the telemetry
            entries in particular need a WIRING change, not just a key, and
            an entry that hides that is a promotion someone will under-quote.
    """

    tier: str
    proposed_key: str
    note: str = ""


#: The assessment, machine-readable.  Every env var the installed tree
#: reads, tagged with its scope and the typed key that covers it.  Kept
#: exhaustive by ``test_env_scope_catalog.py``, which re-derives the var
#: list from source and fails on any disagreement in either direction.
CATALOG: Dict[str, EnvClass] = {

    # ---- application identity (app attribution) -----------------------
    # WHICH application is making these requests -- the product built on
    # the SDK, not the framework.  ``host`` rather than ``session``: an
    # application is a property of the deployment, and two sessions in one
    # process disagreeing about who is spending the money would be a lie.
    # Per-session attribution is a real need and is served by the typed
    # provider knobs (``plugin_configs.openrouter.app_title`` /
    # ``http_referer``), which outrank these.  See shared/app_identity.py.
    "JAATO_APP_CATEGORIES": EnvClass(HOST, None,
        "comma-separated marketplace categories the application claims; a "
        "property of the product, not of a conversation"),
    "JAATO_APP_NAME": EnvClass(HOST, None,
        "names the application embedding the framework; one app per "
        "deployment, and the per-session need is served by the provider "
        "knobs that outrank it"),
    "JAATO_APP_POWERED_BY": EnvClass(HOST, None,
        "whether app attribution appends '(powered by jaato)'; a property "
        "of the product, not of a conversation"),
    "JAATO_APP_URL": EnvClass(HOST, None,
        "the embedding application's own site/repo, attributed upstream; "
        "one app per deployment"),
    "JAATO_APP_VERSION": EnvClass(HOST, None,
        "the embedding application's own version; one app per deployment"),

    # ---- daemon / runner lifecycle -----------------------------------
    "JAATO_APPARMOR_COMPLAIN": EnvClass(HOST, None,
        "kernel policy load mode; a host-wide diagnostic posture"),
    "JAATO_BOOTSTRAP_TIMING": EnvClass(HOST, None,
        "prints a bootstrap timing report; a developer toggle, not agent "
        "behaviour"),
    "JAATO_CGROUPS_ROOT": EnvClass(HOST, None,
        "where the host delegated cgroup v2 subtree_control; one per host"),
    "JAATO_CONFIG_ROOT": EnvClass(SESSION, "client.config_root",
        "per-connection config search root; typed on the handshake, not "
        "the profile"),
    "JAATO_DAEMONIZED": EnvClass(INTERNAL, None,
        "the daemon telling its own re-exec that it is the background "
        "copy"),
    "JAATO_EGRESS_NFT_ENFORCE": EnvClass(HOST, None,
        "nftables egress enforcement; a host firewall posture"),
    "JAATO_EPHEMERAL_TIMEOUT_S": EnvClass(HOST, None,
        "the daemon's reaper deadline for relay sessions, applied to all "
        "of them"),
    "JAATO_PROVIDER_TRACE": EnvClass(SESSION, "trace.provider_log",
        "the incident in issue #775; now typed and validated"),
    "JAATO_REVIVE_PERSONA": EnvClass(HOST, None,
        "revive posture of the daemon: reuse the session's persisted "
        "rendered prompt (default) or re-render it from disk, re-running "
        "the persona's prefetch scripts.  Cannot be a profile key -- it "
        "decides whether the persona is read at all (#787)"),
    "JAATO_REVIVE_PROFILE": EnvClass(HOST, None,
        "revive posture of the daemon: reuse the session's persisted "
        "resolved profile (default) or re-resolve the name from disk.  "
        "Self-referential as a profile key: the daemon would have to load "
        "the file to learn whether it may load the file (#787)"),
    "JAATO_RUNNER_DISABLE_CONFINE": EnvClass(HOST, None,
        "disables runner self-confinement host-wide; deliberately NOT per "
        "session"),
    "JAATO_RUNNER_LOG_PATH": EnvClass(INTERNAL, None,
        "daemon tells the runner subprocess where to log"),
    "JAATO_RUNNER_POOL_ENABLED": EnvClass(HOST, None,
        "pre-warm pool routing; a property of the daemon, not of one "
        "session"),
    "JAATO_RUNNER_POOL_SIZE": EnvClass(HOST, None,
        "how many warm slots the daemon keeps; one number per daemon"),
    "JAATO_RUNNER_PROFILE": EnvClass(INTERNAL, None,
        "daemon hands the runner the AppArmor profile name to self- "
        "confine to"),
    "JAATO_RUNNER_SESSION_ID": EnvClass(INTERNAL, None,
        "daemon hands the runner its session id"),
    "JAATO_RUNNER_WORKSPACE": EnvClass(INTERNAL, None,
        "daemon hands the runner its workspace path"),
    "JAATO_SESSION_LOG_DIR": EnvClass(SESSION, None,
        "a per-session path with no typed key -- same shape as "
        "the trace incident"),
    "JAATO_TRACE_LOG": EnvClass(SESSION, "trace.session_log",
        "sibling of JAATO_PROVIDER_TRACE; typed together with it"),
    "JAATO_WORKSPACE_ROOT": EnvClass(SESSION, "client.working_dir",
        "the session's workspace; typed on the handshake and refused when "
        "relative (#742)"),
    "JAATO_WS_TOKEN": EnvClass(HOST, None,
        "the daemon's WS bearer token; one per listener, never per "
        "session"),
    "LOCATION": EnvClass(SESSION, None,
        "the Vertex region -- connection identity with no "
        "google_genai knob"),
    "MODEL_NAME": EnvClass(SESSION, "model",
        "the profile's own `model` field is the typed equivalent"),
    "PATH": EnvClass(AMBIENT, None,
        "the host environment being read"),
    "PROJECT_ID": EnvClass(SESSION, None,
        "the GCP project -- connection identity with no "
        "google_genai knob"),

    # ---- framework core ----------------------------------------------
    "COLORTERM": EnvClass(AMBIENT, None,
        "terminal capability detection"),
    "ENV_VALIDATE_CA": EnvClass(HOST, None,
        "TLS trust configuration; a host posture"),
    "JAATO_DEFERRED_TOOLS": EnvClass(SESSION, None,
        "tool-loop behaviour; a cheap model and an expensive one "
        "want opposite values in one sweep"),
    "JAATO_GRAPHICS_PROTOCOL": EnvClass(HOST, None,
        "forces the terminal graphics protocol; a property of the "
        "terminal, client-side"),
    "JAATO_PARALLEL_TOOLS": EnvClass(SESSION, None,
        "tool-loop behaviour; today one host-wide setting for "
        "every agent"),
    "JAATO_PROVIDER": EnvClass(SESSION, "provider",
        "the profile's own `provider` field is the typed equivalent"),
    "JAATO_SSL_VERIFY": EnvClass(HOST, None,
        "TLS verification escape hatch for intercepting proxies; a host "
        "posture"),
    "LEDGER_PATH": EnvClass(SESSION, None,
        "overrides the caller's ledger path -- an env var that "
        "outranks a typed argument"),
    "MINGW_CHOST": EnvClass(AMBIENT, None,
        "MSYS2 environment detection"),
    "MINGW_PREFIX": EnvClass(AMBIENT, None,
        "MSYS2 environment detection"),
    "MSYSTEM": EnvClass(AMBIENT, None,
        "MSYS2 environment detection"),
    "REQUESTS_CA_BUNDLE": EnvClass(HOST, None,
        "CA bundle for the host's TLS trust"),
    "SSL_CERT_FILE": EnvClass(HOST, None,
        "CA bundle for the host's TLS trust"),
    "STY": EnvClass(AMBIENT, None,
        "screen/tmux detection"),
    "TERM": EnvClass(AMBIENT, None,
        "terminal capability detection"),
    "TERM_PROGRAM": EnvClass(AMBIENT, None,
        "terminal capability detection"),
    "TMUX": EnvClass(AMBIENT, None,
        "tmux detection"),
    "workspaceRoot": EnvClass(AMBIENT, None,
        "a .env convention read as a workspace-root fallback; the host's, "
        "not a knob"),

    # ---- rate limiting / retry ---------------------------------------
    "AI_EXECUTE_TOOLS": EnvClass(SESSION, None,
        "lets unregistered tools run via the generic executor -- "
        "an agent-behaviour switch"),
    "AI_REQUEST_INTERVAL": EnvClass(SESSION, None,
        "retry policy is per-provider in character; set once per "
        "process"),
    "AI_RETRY_ATTEMPTS": EnvClass(SESSION, None,
        "an OpenRouter 402 and a local vLLM stall want different "
        "budgets"),
    "AI_RETRY_BASE_DELAY": EnvClass(SESSION, None,
        "see AI_RETRY_ATTEMPTS"),
    "AI_RETRY_LOG_SILENT": EnvClass(HOST, None,
        "retry log verbosity; a developer toggle"),
    "AI_RETRY_MAX_DELAY": EnvClass(SESSION, None,
        "see AI_RETRY_ATTEMPTS"),
    "AI_TOOL_RUNNER_DEBUG": EnvClass(HOST, None,
        "tool-runner debug logging; a developer toggle"),

    # ---- telemetry ---------------------------------------------------
    "JAATO_TELEMETRY_BACKEND": EnvClass(SESSION, None,
        "per-session tracing is exactly what one profile wants "
        "and the rest do not"),
    "JAATO_TELEMETRY_ENABLED": EnvClass(SESSION, None,
        "see JAATO_TELEMETRY_BACKEND"),
    "JAATO_TELEMETRY_EXPORTER": EnvClass(SESSION, None,
        "see JAATO_TELEMETRY_BACKEND"),
    "JAATO_TELEMETRY_FILE": EnvClass(SESSION, None,
        "a per-session path, same shape as the trace incident"),
    "JAATO_TELEMETRY_REDACT_CONTENT": EnvClass(SESSION, None,
        "redaction posture may legitimately differ per agent"),
    "LANGFUSE_HOST": EnvClass(HOST, None,
        "the Langfuse deployment the host reports to; one per host"),
    "LANGFUSE_PUBLIC_KEY": EnvClass(SESSION, None,
        "a credential with no typed key (see the credential "
        "policy)"),
    "LANGFUSE_SECRET_KEY": EnvClass(SESSION, None,
        "a credential with no typed key (see the credential "
        "policy)"),
    "OTEL_EXPORTER_OTLP_ENDPOINT": EnvClass(HOST, None,
        "the collector this host exports to; deployment wiring"),
    "OTEL_EXPORTER_OTLP_HEADERS": EnvClass(HOST, None,
        "collector auth headers; deployment wiring"),
    "OTEL_EXPORTER_OTLP_PROTOCOL": EnvClass(HOST, None,
        "collector transport; deployment wiring"),
    "OTEL_SERVICE_NAME": EnvClass(HOST, None,
        "the service name this process reports as"),

    # ---- proxy -------------------------------------------------------
    "HTTPS_PROXY": EnvClass(HOST, None,
        "the host's egress proxy"),
    "HTTP_PROXY": EnvClass(HOST, None,
        "the host's egress proxy"),
    "JAATO_KERBEROS_PROXY": EnvClass(HOST, None,
        "SPNEGO proxy auth; a host network posture"),
    "JAATO_NO_PROXY": EnvClass(HOST, None,
        "exact-match no-proxy hosts; a host network posture"),
    "NO_PROXY": EnvClass(HOST, None,
        "the host's no-proxy list"),
    "http_proxy": EnvClass(HOST, None,
        "the host's egress proxy (lowercase convention)"),
    "https_proxy": EnvClass(HOST, None,
        "the host's egress proxy (lowercase convention)"),
    "no_proxy": EnvClass(HOST, None,
        "the host's no-proxy list (lowercase convention)"),

    # ---- plugins -----------------------------------------------------
    "APPDATA": EnvClass(AMBIENT, None,
        "where Windows keeps user config"),
    "ComSpec": EnvClass(AMBIENT, None,
        "Windows shell detection"),
    "EDITOR": EnvClass(AMBIENT, None,
        "the user's editor"),
    "HOME": EnvClass(AMBIENT, None,
        "expanded as a ${HOME} template variable in profiles/personas"),
    "JAATO_AMBIGUOUS_WIDTH": EnvClass(SESSION, None,
        "belongs to plugin_configs.table_formatter"),
    "JAATO_CLARIFICATION_TIMEOUT": EnvClass(SESSION, None,
        "a headless cascade arm and an interactive TUI want "
        "opposite values"),
    "JAATO_FILE_BACKUP_COUNT": EnvClass(SESSION, None,
        "belongs to plugin_configs.file_edit"),
    "JAATO_GC_PRESSURE": EnvClass(SESSION, "gc.pressure_percent",
        "the env var IS the GCConfig field's default_factory"),
    "JAATO_GC_TARGET": EnvClass(SESSION, "gc.target_percent",
        "the env var IS the GCConfig field's default_factory"),
    "JAATO_GC_THRESHOLD": EnvClass(SESSION, "gc.threshold_percent",
        "the env var IS the GCConfig field's default_factory"),
    "JAATO_KROKI_URL": EnvClass(SESSION, None,
        "belongs to plugin_configs.mermaid_formatter"),
    "JAATO_MERMAID_BACKEND": EnvClass(SESSION, None,
        "belongs to plugin_configs.mermaid_formatter"),
    "JAATO_MERMAID_SCALE": EnvClass(SESSION, "plugin_configs.mermaid_formatter.scale",
        "the knob exists; the env read runs after config.get and overrides "
        "it (precedence defect, not a missing key)"),
    "JAATO_MERMAID_THEME": EnvClass(SESSION, "plugin_configs.mermaid_formatter.theme",
        "the knob exists and a profile can set it -- but the env read runs "
        "AFTER config.get and overrides it: a precedence defect, not a "
        "missing key"),
    "JAATO_PERMISSION_TIMEOUT": EnvClass(SESSION, "plugin_configs.permission.channel_config.timeout",
        "the knob exists (channel_config seeds it); channels.py re-reads "
        "the env at request time and overrides it (precedence defect)"),
    "JAATO_SESSION_CONFIG": EnvClass(SESSION, None,
        "a config-file path belonging to plugin_configs.session"),
    "JAATO_SESSION_ID": EnvClass(INTERNAL, None,
        "the framework telling the environment plugin which session it is "
        "in"),
    "JAATO_TOOL_BINDINGS": EnvClass(SESSION, None,
        "belongs to plugin_configs.notebook"),
    "JAATO_VISION_DIR": EnvClass(SESSION, None,
        "a per-session output path belonging to "
        "plugin_configs.mermaid_formatter"),
    "KAGGLE_API_TOKEN": EnvClass(SESSION, None,
        "a credential with no typed key (see the credential "
        "policy)"),
    "KAGGLE_KEY": EnvClass(SESSION, None,
        "a credential with no typed key (see the credential "
        "policy)"),
    "KAGGLE_USERNAME": EnvClass(SESSION, None,
        "a credential with no typed key (see the credential "
        "policy)"),
    "PERMISSION_WEBHOOK_TOKEN": EnvClass(SESSION, "plugin_configs.permission.channel_config.auth_token",
        "config.get(auth_token) or os.environ -- the knob already WINS and "
        "the env var is its fallback"),
    "PSModulePath": EnvClass(AMBIENT, None,
        "PowerShell detection"),
    "PSVersionTable": EnvClass(AMBIENT, None,
        "PowerShell detection"),
    "SHELL": EnvClass(AMBIENT, None,
        "the user's shell"),
    "TODO_STORAGE_PATH": EnvClass(SESSION, "plugin_configs.todo.storage_path",
        "the plugin config already wins; the env var is its default"),
    "TODO_WEBHOOK_TOKEN": EnvClass(SESSION, "plugin_configs.todo.reporter_config.auth_token",
        "config.get(auth_token) or os.environ -- the knob already WINS and "
        "the env var is its fallback"),
    "USER": EnvClass(AMBIENT, None,
        "expanded as a ${USER} template variable in profiles/personas"),
    "VISUAL": EnvClass(AMBIENT, None,
        "the user's editor"),
    "XDG_CONFIG_HOME": EnvClass(AMBIENT, None,
        "where the host keeps user config"),

    # ---- model providers ---------------------------------------------
    "ANTHROPIC_API_KEY": EnvClass(SESSION, "plugin_configs.anthropic.api_key",
        "credential; anthropic exposes the knob, so a profile can carry a "
        "pass:// URI instead of the env var"),
    "ANTHROPIC_AUTH_TOKEN": EnvClass(SESSION, "plugin_configs.anthropic.oauth_token",
        "credential; anthropic exposes the knob, so a profile can carry a "
        "pass:// URI instead of the env var"),
    "CLAUDE_CODE_OAUTH_TOKEN": EnvClass(SESSION, "plugin_configs.anthropic.oauth_token",
        "credential; anthropic exposes the knob, so a profile can carry a "
        "pass:// URI instead of the env var"),
    "GITHUB_TOKEN": EnvClass(SESSION, None,
        "github_models exposes no api_key knob (see the "
        "credential policy)"),
    "GOOGLE_APPLICATION_CREDENTIALS": EnvClass(SESSION, None,
        "google_genai exposes no credential knob (see the "
        "credential policy)"),
    "GOOGLE_GENAI_API_KEY": EnvClass(SESSION, None,
        "google_genai exposes no api_key knob (see the credential "
        "policy)"),
    "JAATO_ANTHROPIC_ENABLE_CACHING": EnvClass(SESSION, "plugin_configs.anthropic.enable_caching",
        "anthropic exposes the knob; the env var is its fallback"),
    "JAATO_ANTHROPIC_ENABLE_THINKING": EnvClass(SESSION, "plugin_configs.anthropic.api_params.enable_thinking",
        "anthropic exposes the knob; the env var is its fallback"),
    "JAATO_ANTHROPIC_THINKING_BUDGET": EnvClass(SESSION, "plugin_configs.anthropic.api_params.thinking_budget",
        "anthropic exposes the knob; the env var is its fallback"),
    "JAATO_ANTIGRAVITY_AUTO_ROTATE": EnvClass(SESSION, "plugin_configs.antigravity.auto_rotate",
        "antigravity exposes the knob; the env var is its fallback"),
    "JAATO_ANTIGRAVITY_ENDPOINT": EnvClass(SESSION, "plugin_configs.antigravity.endpoint",
        "endpoint override; antigravity exposes the knob"),
    "JAATO_ANTIGRAVITY_PROJECT_ID": EnvClass(SESSION, "plugin_configs.antigravity.project_id",
        "antigravity exposes the knob; the env var is its fallback"),
    "JAATO_ANTIGRAVITY_QUOTA": EnvClass(SESSION, "plugin_configs.antigravity.quota_type",
        "antigravity exposes the knob; the env var is its fallback"),
    "JAATO_ANTIGRAVITY_RETRY_EMPTY": EnvClass(SESSION, "plugin_configs.antigravity.retry_empty",
        "antigravity exposes the knob; the env var is its fallback"),
    "JAATO_ANTIGRAVITY_SESSION_RECOVERY": EnvClass(SESSION, "plugin_configs.antigravity.session_recovery",
        "antigravity exposes the knob; the env var is its fallback"),
    "JAATO_ANTIGRAVITY_THINKING_BUDGET": EnvClass(SESSION, "plugin_configs.antigravity.thinking_budget",
        "antigravity exposes the knob; the env var is its fallback"),
    "JAATO_ANTIGRAVITY_THINKING_LEVEL": EnvClass(SESSION, "plugin_configs.antigravity.thinking_level",
        "antigravity exposes the knob; the env var is its fallback"),
    "JAATO_CLAUDE_CLI_MAX_TURNS": EnvClass(SESSION, "plugin_configs.claude_cli.max_turns",
        "claude_cli exposes the knob; the env var is its fallback"),
    "JAATO_CLAUDE_CLI_MODE": EnvClass(SESSION, "plugin_configs.claude_cli.cli_mode",
        "claude_cli exposes the knob; the env var is its fallback"),
    "JAATO_CLAUDE_CLI_PATH": EnvClass(SESSION, "plugin_configs.claude_cli.cli_path",
        "claude_cli exposes the knob; the env var is its fallback"),
    "JAATO_CLAUDE_CLI_PERMISSION_MODE": EnvClass(SESSION, "plugin_configs.claude_cli.permission_mode",
        "claude_cli exposes the knob; the env var is its fallback"),
    "JAATO_DOUBLEWORD_API_KEY": EnvClass(SESSION, "plugin_configs.doubleword.api_key",
        "credential; doubleword exposes the knob, so a profile can carry "
        "a pass:// URI instead of the env var"),
    "JAATO_DOUBLEWORD_BASE_URL": EnvClass(SESSION, "plugin_configs.doubleword.base_url",
        "endpoint override; doubleword exposes the knob"),
    "JAATO_DOUBLEWORD_CONTEXT_LENGTH": EnvClass(SESSION, "plugin_configs.doubleword.context_length",
        "manual context-window override; doubleword exposes the knob"),
    "JAATO_DOUBLEWORD_MODEL": EnvClass(SESSION, "model",
        "the profile's own `model` field selects the model"),
    "JAATO_DOUBLEWORD_SERVICE_TIER": EnvClass(SESSION, "plugin_configs.doubleword.api_params.service_tier",
        "doubleword exposes the knob; the env var is its fallback"),
    "JAATO_DUMP_PROVIDER_REQUEST": EnvClass(HOST, None,
        "dumps every provider request body; a developer toggle, global by "
        "design"),
    "JAATO_GITHUB_AUTH_METHOD": EnvClass(SESSION, None,
        "auth-method selection with no github_models knob"),
    "JAATO_GITHUB_ENDPOINT": EnvClass(SESSION, "plugin_configs.github_models.endpoint",
        "endpoint override; github_models exposes the knob"),
    "JAATO_GITHUB_ENTERPRISE": EnvClass(SESSION, "plugin_configs.github_models.enterprise",
        "github_models exposes the knob; the env var is its fallback"),
    "JAATO_GITHUB_ORGANIZATION": EnvClass(SESSION, "plugin_configs.github_models.organization",
        "github_models exposes the knob; the env var is its fallback"),
    "JAATO_GOOGLE_AUTH_METHOD": EnvClass(SESSION, None,
        "auth-method selection with no google_genai knob"),
    "JAATO_GOOGLE_TARGET_SERVICE_ACCOUNT": EnvClass(SESSION, None,
        "impersonation target with no google_genai knob"),
    "JAATO_GOOGLE_USE_VERTEX": EnvClass(SESSION, None,
        "Vertex-vs-API backend selection with no google_genai "
        "knob"),
    "JAATO_NEBIUS_API_KEY": EnvClass(SESSION, "plugin_configs.nebius.api_key",
        "credential; nebius exposes the knob, so a profile can carry a "
        "pass:// URI instead of the env var"),
    "JAATO_NEBIUS_BASE_URL": EnvClass(SESSION, "plugin_configs.nebius.base_url",
        "endpoint override; nebius exposes the knob"),
    "JAATO_NEBIUS_CONTEXT_LENGTH": EnvClass(SESSION, "plugin_configs.nebius.context_length",
        "manual context-window override; nebius exposes the knob"),
    "JAATO_NEBIUS_MODEL": EnvClass(SESSION, "model",
        "the profile's own `model` field selects the model"),
    "JAATO_NIM_API_KEY": EnvClass(SESSION, "plugin_configs.nim.api_key",
        "credential; nim exposes the knob, so a profile can carry a "
        "pass:// URI instead of the env var"),
    "JAATO_NIM_BASE_URL": EnvClass(SESSION, "plugin_configs.nim.base_url",
        "endpoint override; nim exposes the knob"),
    "JAATO_NIM_CONTEXT_LENGTH": EnvClass(SESSION, "plugin_configs.nim.context_length",
        "manual context-window override; nim exposes the knob"),
    "JAATO_NIM_MODEL": EnvClass(SESSION, "model",
        "the profile's own `model` field selects the model"),
    "JAATO_OPENROUTER_API_KEY": EnvClass(SESSION, "plugin_configs.openrouter.api_key",
        "credential; openrouter exposes the knob, so a profile can carry "
        "a pass:// URI instead of the env var"),
    "JAATO_OPENROUTER_APP_CATEGORIES": EnvClass(SESSION, "plugin_configs.openrouter.app_categories",
        "openrouter exposes the knob; the env var is its fallback"),
    "JAATO_OPENROUTER_APP_TITLE": EnvClass(SESSION, "plugin_configs.openrouter.app_title",
        "openrouter exposes the knob; the env var is its fallback"),
    "JAATO_OPENROUTER_BASE_URL": EnvClass(SESSION, "plugin_configs.openrouter.framework_overrides.base_url",
        "endpoint override; openrouter exposes the knob"),
    "JAATO_OPENROUTER_CONTEXT_LENGTH": EnvClass(SESSION, "plugin_configs.openrouter.framework_overrides.context_length",
        "manual context-window override; openrouter exposes the knob"),
    "JAATO_OPENROUTER_HTTP_REFERER": EnvClass(SESSION, "plugin_configs.openrouter.http_referer",
        "openrouter exposes the knob; the env var is its fallback"),
    "JAATO_OPENROUTER_MODEL": EnvClass(SESSION, "model",
        "the profile's own `model` field selects the model"),
    "JAATO_OVHCLOUD_ALLOW_ANONYMOUS": EnvClass(SESSION, "plugin_configs.ovhcloud.allow_anonymous",
        "ovhcloud exposes the knob; the env var is its fallback"),
    "JAATO_OVHCLOUD_API_KEY": EnvClass(SESSION, "plugin_configs.ovhcloud.api_key",
        "credential; ovhcloud exposes the knob, so a profile can carry a "
        "pass:// URI instead of the env var"),
    "JAATO_OVHCLOUD_BASE_URL": EnvClass(SESSION, "plugin_configs.ovhcloud.base_url",
        "endpoint override; ovhcloud exposes the knob"),
    "JAATO_OVHCLOUD_CONTEXT_LENGTH": EnvClass(SESSION, "plugin_configs.ovhcloud.context_length",
        "manual context-window override; ovhcloud exposes the knob"),
    "JAATO_OVHCLOUD_MODEL": EnvClass(SESSION, "model",
        "the profile's own `model` field selects the model"),
    "JAATO_ZHIPUAI_API_KEY": EnvClass(SESSION, None,
        "zhipuai exposes no api_key knob (see the credential "
        "policy)"),
    "LMSTUDIO_API_TOKEN": EnvClass(SESSION, "plugin_configs.lmstudio.api_token",
        "credential; lmstudio exposes the knob, so a profile can carry a "
        "pass:// URI instead of the env var"),
    "LMSTUDIO_CONTEXT_LENGTH": EnvClass(SESSION, "plugin_configs.lmstudio.context_length",
        "manual context-window override; lmstudio exposes the knob"),
    "LMSTUDIO_HOST": EnvClass(SESSION, "plugin_configs.lmstudio.host",
        "endpoint override; lmstudio exposes the knob"),
    "LMSTUDIO_MODEL": EnvClass(SESSION, "model",
        "the profile's own `model` field selects the model"),
    "NEBIUS_API_KEY": EnvClass(SESSION, "plugin_configs.nebius.api_key",
        "credential; nebius exposes the knob, so a profile can carry a "
        "pass:// URI instead of the env var"),
    "OLLAMA_CONTEXT_LENGTH": EnvClass(SESSION, "plugin_configs.ollama.context_length",
        "manual context-window override; ollama exposes the knob"),
    "OLLAMA_HOST": EnvClass(SESSION, "plugin_configs.ollama.host",
        "endpoint override; ollama exposes the knob"),
    "OLLAMA_MODEL": EnvClass(SESSION, "model",
        "the profile's own `model` field selects the model"),
    "OVH_AI_ENDPOINTS_ACCESS_TOKEN": EnvClass(SESSION, "plugin_configs.ovhcloud.api_key",
        "credential; ovhcloud exposes the knob, so a profile can carry a "
        "pass:// URI instead of the env var"),
    "TENSORRT_LLM_API_TOKEN": EnvClass(SESSION, "plugin_configs.tensorrt_llm.api_token",
        "credential; tensorrt_llm exposes the knob, so a profile can "
        "carry a pass:// URI instead of the env var"),
    "TENSORRT_LLM_CONTEXT_LENGTH": EnvClass(SESSION, "plugin_configs.tensorrt_llm.context_length",
        "manual context-window override; tensorrt_llm exposes the knob"),
    "TENSORRT_LLM_HOST": EnvClass(SESSION, "plugin_configs.tensorrt_llm.host",
        "endpoint override; tensorrt_llm exposes the knob"),
    "TENSORRT_LLM_MODEL": EnvClass(SESSION, "model",
        "the profile's own `model` field selects the model"),
    "TRITON_API_TOKEN": EnvClass(SESSION, "plugin_configs.triton.api_token",
        "credential; triton exposes the knob, so a profile can carry a "
        "pass:// URI instead of the env var"),
    "TRITON_CONTEXT_LENGTH": EnvClass(SESSION, "plugin_configs.triton.context_length",
        "manual context-window override; triton exposes the knob"),
    "TRITON_CONTROL_URL": EnvClass(SESSION, "plugin_configs.triton.control_url",
        "endpoint override; triton exposes the knob"),
    "TRITON_HOST": EnvClass(SESSION, "plugin_configs.triton.host",
        "endpoint override; triton exposes the knob"),
    "TRITON_MODEL": EnvClass(SESSION, "model",
        "the profile's own `model` field selects the model"),
    "TRITON_OPENAI_URL": EnvClass(SESSION, "plugin_configs.triton.openai_url",
        "endpoint override; triton exposes the knob"),
    "VLLM_API_TOKEN": EnvClass(SESSION, "plugin_configs.vllm.api_token",
        "credential; vllm exposes the knob, so a profile can carry a "
        "pass:// URI instead of the env var"),
    "VLLM_CONTEXT_LENGTH": EnvClass(SESSION, "plugin_configs.vllm.context_length",
        "manual context-window override; vllm exposes the knob"),
    "VLLM_HOST": EnvClass(SESSION, "plugin_configs.vllm.host",
        "endpoint override; vllm exposes the knob"),
    "VLLM_MODEL": EnvClass(SESSION, "model",
        "the profile's own `model` field selects the model"),
    "ZHIPUAI_API_KEY": EnvClass(SESSION, None,
        "zhipuai exposes no api_key knob (see the credential "
        "policy)"),
    "ZHIPUAI_BASE_URL": EnvClass(SESSION, "plugin_configs.zhipuai.framework_overrides.base_url",
        "endpoint override; zhipuai exposes the knob"),
    "ZHIPUAI_CONTEXT_LENGTH": EnvClass(SESSION, "plugin_configs.zhipuai.framework_overrides.context_length",
        "manual context-window override; zhipuai exposes the knob"),
    "ZHIPUAI_ENABLE_THINKING": EnvClass(SESSION, "plugin_configs.zhipuai.api_params.enable_thinking",
        "zhipuai exposes the knob; the env var is its fallback"),
    "ZHIPUAI_MODEL": EnvClass(SESSION, "model",
        "the profile's own `model` field selects the model"),
    "ZHIPUAI_OPENAI_BASE_URL": EnvClass(SESSION, "plugin_configs.zhipuai_openai.base_url",
        "endpoint override; zhipuai_openai exposes the knob"),
    "ZHIPUAI_OPENAI_CONTEXT_LENGTH": EnvClass(SESSION, "plugin_configs.zhipuai_openai.context_length",
        "manual context-window override; zhipuai_openai exposes the knob"),
    "ZHIPUAI_OPENAI_MODEL": EnvClass(SESSION, "model",
        "the profile's own `model` field selects the model"),
    "ZHIPUAI_THINKING_BUDGET": EnvClass(SESSION, "plugin_configs.zhipuai.api_params.thinking_budget",
        "zhipuai exposes the knob; the env var is its fallback"),
}

#: THE RATCHET.  Session-scoped vars that have no typed key yet, with the
#: tier from the assessment.  **This set may only shrink.**
#:
#: It is not a backlog anyone has to remember: the guard derives the same
#: set from :data:`CATALOG` on every run and fails when the two disagree,
#: so promoting a knob means deleting its line here, and adding a new
#: untyped session knob means adding one -- in a diff, under review, next
#: to the words "may only shrink".  That is the whole mechanism.  The
#: repository's own precedent is ``test_session_env_audit.py``'s
#: ALLOWLIST, which works the same way for the orthogonal question of
#: read ROUTE.
#:
#: Tiers, from the issue's triage and confirmed by the nested comparison:
#:
#:   A  agent-behaviour knobs -- tool-loop switches, retry budgets,
#:      permission/clarification timeouts, telemetry.  Two profiles in one
#:      sweep may legitimately want opposite values; today each is one
#:      host-wide setting.
#:   B  plugin knobs.  Each belongs to exactly one plugin that already has
#:      a ``plugin_configs.<plugin>`` namespace, so the typed home exists
#:      and only the wiring is missing.  Several are PATHS -- the same
#:      shape as the trace incident.
#:   E  credentials.  These need the policy in
#:      ``docs/design/env-vars-vs-profile-keys.md`` decided before a key is
#:      added, not a default: a knob that accepts a literal secret in a
#:      YAML file is worse than no knob.  The ones listed here are the
#:      providers and plugins whose peers already expose one.
AWAITING_TYPED_KEY: Dict[str, Awaiting] = {

    # ---- tier A: agent-behaviour knobs -----------------------
    "AI_EXECUTE_TOOLS": Awaiting(
        "A", "tools.execute_unregistered",
    ),
    "AI_REQUEST_INTERVAL": Awaiting(
        "A", "retry.request_interval",
        "top-level block overridden by plugin_configs.<provider>.retry, "
        "mirroring how cache: layers with the per-provider knobs",
    ),
    "AI_RETRY_ATTEMPTS": Awaiting(
        "A", "retry.attempts",
    ),
    "AI_RETRY_BASE_DELAY": Awaiting(
        "A", "retry.base_delay",
    ),
    "AI_RETRY_MAX_DELAY": Awaiting(
        "A", "retry.max_delay",
    ),
    "JAATO_CLARIFICATION_TIMEOUT": Awaiting(
        "A", "plugin_configs.clarification.channel_config.timeout",
        "matches the plugin's existing channel_config shape",
    ),
    "JAATO_DEFERRED_TOOLS": Awaiting(
        "A", "tools.deferred",
    ),
    "JAATO_PARALLEL_TOOLS": Awaiting(
        "A", "tools.parallel",
    ),
    "JAATO_TELEMETRY_BACKEND": Awaiting(
        "A", "plugin_configs.telemetry.backend",
    ),
    "JAATO_TELEMETRY_ENABLED": Awaiting(
        "A", "plugin_configs.telemetry.enabled",
        "the key EXISTS; create_plugin() gates construction on the env "
        "var and returns NullTelemetryPlugin, so no profile key can "
        "reach it. Needs the factory to consult the profile -- wiring, "
        "not a key",
    ),
    "JAATO_TELEMETRY_EXPORTER": Awaiting(
        "A", "plugin_configs.telemetry.exporter",
        "the key EXISTS; create_plugin() builds the config dict from "
        "env and passes it to initialize(), so plugin_configs.telemetry "
        "never arrives",
    ),
    "JAATO_TELEMETRY_FILE": Awaiting(
        "A", "plugin_configs.telemetry.file_path",
        "the key EXISTS and already wins (config.get(file_path, env)) "
        "-- but create_plugin() never passes it, so the win is "
        "unreachable",
    ),
    "JAATO_TELEMETRY_REDACT_CONTENT": Awaiting(
        "A", "plugin_configs.telemetry.redact_content",
        "as JAATO_TELEMETRY_EXPORTER -- the key exists, the factory "
        "overwrites it",
    ),

    # ---- tier B: plugin knobs --------------------------------
    "JAATO_AMBIGUOUS_WIDTH": Awaiting(
        "B", "plugin_configs.table_formatter.ambiguous_width",
        "sibling of the existing console_width knob",
    ),
    "JAATO_FILE_BACKUP_COUNT": Awaiting(
        "B", "plugin_configs.file_edit.backup_count",
        "sibling of the existing backup_dir knob",
    ),
    "JAATO_KROKI_URL": Awaiting(
        "B", "plugin_configs.mermaid_formatter.kroki_url",
    ),
    "JAATO_MERMAID_BACKEND": Awaiting(
        "B", "plugin_configs.mermaid_formatter.backend",
    ),
    "JAATO_SESSION_CONFIG": Awaiting(
        "B", "plugin_configs.session.config_path",
        "mirrors the permission plugin's config_path knob",
    ),
    "JAATO_SESSION_LOG_DIR": Awaiting(
        "B", "trace.log_dir",
        "the trace: block already owns per-session diagnostic output "
        "paths",
    ),
    "JAATO_TOOL_BINDINGS": Awaiting(
        "B", "plugin_configs.notebook.tool_bindings",
    ),
    "JAATO_VISION_DIR": Awaiting(
        "B", "plugin_configs.mermaid_formatter.vision_dir",
    ),
    "LEDGER_PATH": Awaiting(
        "B", "trace.ledger",
        "same block; also fixes the inversion where the env var "
        "outranks the filepath argument its caller passed",
    ),

    # ---- tier E: credentials + connection identity -----------
    "GITHUB_TOKEN": Awaiting(
        "E", "plugin_configs.github_models.api_key",
    ),
    "GOOGLE_APPLICATION_CREDENTIALS": Awaiting(
        "E", "plugin_configs.google_genai.credentials_path",
    ),
    "GOOGLE_GENAI_API_KEY": Awaiting(
        "E", "plugin_configs.google_genai.api_key",
    ),
    "JAATO_GITHUB_AUTH_METHOD": Awaiting(
        "E", "plugin_configs.github_models.auth_method",
    ),
    "JAATO_GOOGLE_AUTH_METHOD": Awaiting(
        "E", "plugin_configs.google_genai.auth_method",
    ),
    "JAATO_GOOGLE_TARGET_SERVICE_ACCOUNT": Awaiting(
        "E", "plugin_configs.google_genai.target_service_account",
    ),
    "JAATO_GOOGLE_USE_VERTEX": Awaiting(
        "E", "plugin_configs.google_genai.use_vertex",
        "not a secret -- a backend selector",
    ),
    "JAATO_ZHIPUAI_API_KEY": Awaiting(
        "E", "plugin_configs.zhipuai.api_key",
    ),
    "KAGGLE_API_TOKEN": Awaiting(
        "E", "plugin_configs.notebook.kaggle.api_token",
    ),
    "KAGGLE_KEY": Awaiting(
        "E", "plugin_configs.notebook.kaggle.key",
    ),
    "KAGGLE_USERNAME": Awaiting(
        "E", "plugin_configs.notebook.kaggle.username",
    ),
    "LANGFUSE_PUBLIC_KEY": Awaiting(
        "E", "plugin_configs.telemetry.langfuse.public_key",
    ),
    "LANGFUSE_SECRET_KEY": Awaiting(
        "E", "plugin_configs.telemetry.langfuse.secret_key",
    ),
    "LOCATION": Awaiting(
        "E", "plugin_configs.google_genai.location",
        "not a secret -- connection identity",
    ),
    "PROJECT_ID": Awaiting(
        "E", "plugin_configs.google_genai.project_id",
        "not a secret -- connection identity; antigravity already "
        "exposes project_id",
    ),
    "ZHIPUAI_API_KEY": Awaiting(
        "E", "plugin_configs.zhipuai.api_key",
    ),
}




def classify(name: str) -> Optional[EnvClass]:
    """The catalog entry for *name*, or ``None`` when it is unclassified.

    ``None`` is a real answer, not an error: it means the guard has not
    run since the var was added.  Callers that render the catalog show it
    as ``unclassified`` rather than inventing a scope.
    """
    return CATALOG.get(name)


def is_knob(name: str) -> bool:
    """True when *name* is something an operator or author SETS.

    False for ``ambient`` (the host environment being read) and
    ``internal`` (a framework-to-framework handoff).  This is the filter
    that keeps a naive env-name diff from reporting ``PATH`` and
    ``TERM`` as missing profile keys.  Unclassified vars count as knobs
    -- the conservative answer, since the alternative is hiding one.
    """
    entry = CATALOG.get(name)
    return entry is None or entry.scope in (SESSION, HOST)
