"""The OUTPUT contract of ``jaato-scaffold new`` — one archetype, one doc.

``explain`` documents the framework's *inputs* (plugins, providers, gc
strategies, knobs, transports, paths).  This module documents what ``new``
*produces*: for every archetype it accepts, the file tree it writes, what each
file is for, which parts are placeholders the reader is expected to edit versus
generated-and-correct, and how the flags change that output.

Why it exists as data rather than prose in a README: a generator whose output
is undocumented cannot be trusted sight-unseen, so a careful consumer
reverse-engineers it before building on it — reading ``build.py`` and
``_client_templates.py`` instead of running ``new`` once.  That cost lands on
everyone who is told to prefer the generator over hand-writing (jaato #716).
Having the contract as data means three readers share it:

* ``explain archetypes`` / ``explain archetype <name>`` — "should I use this?"
* ``new --dry-run`` — "what exactly lands in MY workspace with THESE flags?"
  (it annotates each planned path with the :class:`EmittedFile` purpose)
* the guard in ``tests/test_scaffold_archetype_docs.py`` — every archetype
  ``new`` accepts must be documented, and every documented file must actually
  be written by a real run.  That guard is the point: the gap this module
  closes arose because the archetype count was SPELLED rather than counted.
  ``5aa82e1`` (#624) shipped five client templates under a banner reading
  "4 client archetypes" — ``host-tools`` was uncounted on day one — and
  ``ad016d8`` (#649) added ``sweep`` without touching the literal.  Nobody
  ever incremented it; there was nothing behind it to drill into, so nothing
  drew attention to it being wrong.

The one-line summaries are NOT repeated here — client archetypes read theirs
from :data:`_client_templates.TEMPLATES`, the same string ``new`` prints, so
the list cannot disagree with the generator about what an archetype is.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ._client_templates import TEMPLATES

# Archetype names ``new`` treats as "scaffold a profile-set".  ``None`` (no
# archetype at all) is the same thing — profile-set is the default verb.
PROFILE_SET_ALIASES = ("profile-set", "set")

#: The canonical name for the profile-set archetype.
PROFILE_SET = "profile-set"

#: Client archetypes, derived from the template registry so a new template is
#: automatically an accepted archetype (and, via the guard, must be documented).
CLIENT_ARCHETYPES: Tuple[str, ...] = tuple(sorted(TEMPLATES))


@dataclass(frozen=True)
class EmittedFile:
    """One file ``new`` writes into the workspace.

    Attributes:
        path: Workspace-relative path.  May carry the ``{archetype}``,
            ``{set}`` and ``{agent}`` placeholders, which the renderer fills
            from the invocation (and which the guard expands into a glob when
            it checks a real run's output against this declaration).
        what: One line — what the file is.
        status: Who owns the contents afterwards.  One of:
            ``generated`` (correct as emitted; edit only to customise),
            ``fill-in`` (emitted with a blank the reader MUST complete),
            ``edit`` (a worked example the reader is expected to replace),
            ``merged`` (an existing file is appended to, never clobbered).
        detail: Bullet lines describing what is inside.
        when: Condition under which the file is written, or None for always.
            A conditional file is not required to appear in a run's output.
    """

    path: str
    what: str
    status: str
    detail: Tuple[str, ...] = ()
    when: Optional[str] = None

    def render_path(self, **subs) -> str:
        """The path with ``{archetype}`` / ``{set}`` / ``{agent}`` filled in."""
        out = self.path
        for k, v in subs.items():
            out = out.replace("{" + k + "}", str(v))
        return out

    def glob(self) -> str:
        """The path as an fnmatch pattern (placeholders → ``*``)."""
        out = self.path
        for token in ("{archetype}", "{set}", "{agent}"):
            out = out.replace(token, "*")
        return out


@dataclass(frozen=True)
class ArchetypeDoc:
    """The documented output contract of one archetype.

    Attributes:
        name: The archetype as typed after ``new``.
        kind: ``client`` (emits a runnable SDK script) or ``profile-set``
            (emits YAML profiles).
        summary: One line.  For client archetypes this is read from
            ``TEMPLATES`` rather than restated.
        requires: Flags that must be supplied, else ``new`` fails loud.
        writes: The files emitted, in the order they are written.
        flags: ``(flag, effect-on-output)`` pairs — only flags that CHANGE
            what is written; flags with no output effect are omitted.
        edit_before_running: Parts of the output the reader must edit.
        generated_correct: Parts that are correct as emitted — the hard-won
            recipe the archetype exists to carry.  Editing these is how the
            generated client stops working.
        check: The emit-then-check ``new`` runs on its own output.
        next_steps: What to run after scaffolding.
        aliases: Other names ``new`` accepts for this archetype.
    """

    name: str
    kind: str
    requires: Tuple[str, ...]
    writes: Tuple[EmittedFile, ...]
    check: str
    next_steps: Tuple[str, ...]
    summary: str = ""
    flags: Tuple[Tuple[str, str], ...] = ()
    edit_before_running: Tuple[str, ...] = ()
    generated_correct: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()


# --------------------------------------------------------------- client parts
#
# Every client archetype writes the SAME two files with the same flag
# semantics; only the script's body differs.  Declared once so six archetypes
# cannot drift apart on the shared half of their contract.

def _client_script(detail: Tuple[str, ...]) -> EmittedFile:
    return EmittedFile(
        path="run_{archetype}.py",
        what="the runnable SDK client — a complete, executable Python script",
        status="generated",
        detail=detail,
    )


_CLIENT_ENV = EmittedFile(
    path=".env",
    what="workspace env: the active provider/model plus every optional knob, "
         "commented out with its default",
    status="fill-in",
    detail=(
        "JAATO_PROVIDER=<provider> and MODEL_NAME=<model> — active, from the flags",
        "the chosen provider's env vars, commented out (all of them — they are "
        "your provider config)",
        "every OTHER framework knob that has a meaningful default, commented out "
        "and grouped by category — discovered from the installed code, so it "
        "cannot drift from what the daemon reads",
        "the provider credential is NOT written — set it yourself "
        "(`jaato-scaffold explain provider <name>` names the variable)",
    ),
    when="unless it already exists (or --force)",
)

_CLIENT_FLAGS: Tuple[Tuple[str, str], ...] = (
    ("--transport ipc (default)",
     "IPCClient over a Unix socket + SOCKET constant; auto_start=True so the "
     "script cold-starts a daemon"),
    ("--transport ws --url URL [--token T] [--ca BUNDLE]",
     "WSClient + URL/TOKEN (and CA) constants; --ca is threaded as the scoped "
     "ca= knob, never os.environ"),
    ("--transport in_process",
     "InProcessClient — imports from `jaato`, no socket/url constants, runs the "
     "runtime in the script's own process; rejects --recoverable"),
    ("--recoverable",
     "upgrades a daemon transport to its auto-reconnect client "
     "(IPCRecoveryClient / WSRecoveryClient) and adds an on_status_change "
     "callback that prints the connection lifecycle"),
    ("--force", "overwrite an existing run_<archetype>.py / .env"),
    ("--secrets / --secret-path",
     "no effect on the emitted files — only on the credential hint printed "
     "afterwards (they shape a profile-set's YAML, not a client)"),
)

_CLIENT_GENERATED_CORRECT = (
    "client_type=ClientType.API — load-bearing: the daemon keeps "
    "signal_completion for API clients and strips it for TERMINAL/WEB/CHAT",
    "connect(timeout=120.0) — a cold daemon autostart takes ~30-60s; the SDK "
    "default of 5s is too short",
    "env_file is always a real path — env_file=None crashes the IPC handshake "
    "with an opaque os.PathLike TypeError",
    "completion waits on the FIRST of {TURN_COMPLETED, SESSION_TERMINATED} — a "
    "plain turn never self-terminates, so waiting on SESSION_TERMINATED alone "
    "hangs forever",
    "create_session RAISES SessionCreateFailed; it does not return None",
)

_CLIENT_NEXT = (
    "python -m jaato_sdk.doctor --workspace <ws> --env-file <ws>/.env",
    "python <ws>/run_{archetype}.py",
)


def _client(name: str, *, detail: Tuple[str, ...],
            edit: Tuple[str, ...] = ()) -> ArchetypeDoc:
    """One client archetype: the shared contract + this script's specifics."""
    return ArchetypeDoc(
        name=name,
        kind="client",
        summary=TEMPLATES[name][2],
        requires=("--workspace", "--provider", "--model"),
        writes=(_client_script(detail), _CLIENT_ENV),
        flags=_CLIENT_FLAGS,
        edit_before_running=edit,
        generated_correct=_CLIENT_GENERATED_CORRECT,
        check="py_compile of the generated script — the client analogue of "
              "profile-set's emit-then-validate",
        next_steps=_CLIENT_NEXT,
    )


# ------------------------------------------------------------- the registry

ARCHETYPES: Dict[str, ArchetypeDoc] = {

    PROFILE_SET: ArchetypeDoc(
        name=PROFILE_SET,
        kind="profile-set",
        summary="Two-tier profile set — a provider-agnostic base per stage plus "
                "a provider/model binding for each.",
        aliases=("set",),
        requires=("--workspace", "--set", "--provider", "--model", "--agents"),
        writes=(
            EmittedFile(
                path=".jaato/profiles/_base_{agent}.yaml",
                what="tier-1 base profile for one stage — PROVIDER-AGNOSTIC",
                status="edit",
                detail=(
                    "name / description for the stage",
                    "plugins: [] — deliberately empty; choose them yourself "
                    "(`jaato-scaffold explain plugins`).  The generator does not "
                    "guess a plugin set",
                    "holds stage determinism (plugins, schemas, permission "
                    "policy); binding a provider or model here breaks "
                    "set-selection, so it is left out",
                ),
                when="unless it already exists (or --force)",
            ),
            EmittedFile(
                path=".jaato/profiles/{set}/{agent}.yaml",
                what="tier-2 set profile — binds provider + model for one stage",
                status="generated",
                detail=(
                    "inherits: [_base_<agent>] and plugins: [] (empty keeps the "
                    "inherited base surface)",
                    "model + provider from --model / --provider",
                    "plugin_configs.<provider> carrying only knobs THIS provider "
                    "declares — gated on its PROVIDER_KNOBS, so the emit step "
                    "cannot author a key the validate step would reject",
                    "api_key per --secrets (see the flags below)",
                    "temperature: 0.0 when the provider accepts it — the "
                    "determinism knob",
                    "a commented model_tiers block as a worked example "
                    "(`jaato-scaffold explain tiers`)",
                ),
                when="unless it already exists (or --force)",
            ),
            EmittedFile(
                path=".env",
                what="workspace env: selects the set at runtime and holds the "
                     "credential blank",
                status="fill-in",
                detail=(
                    "JAATO_PROFILE_SET=<set> — without it the workspace does not "
                    "run as the set you just generated",
                    "<PROVIDER_KEY_ENV>= — an EMPTY active line to fill in "
                    "(env/none secrets modes only); the name is read from the "
                    "provider's declared auth chain, not guessed",
                    "on a fresh workspace, the same commented knob catalogue a "
                    "client archetype writes",
                    "an EXISTING .env is appended to, never clobbered — a "
                    "JAATO_PROFILE_SET already present is left alone",
                ),
            ),
            EmittedFile(
                path=".gitignore",
                what="ignores .env (keeps .env.example tracked)",
                status="merged",
                detail=(
                    "created, or appended to if it exists and lacks the rule",
                    "the credential now lives in .env; without this rule a live "
                    "key is one `git add` from being published",
                ),
                when="--secrets env (default) or none",
            ),
            EmittedFile(
                path=".jaato/scaffold.json",
                what="records the chosen secrets mode so later `new` calls in "
                     "this workspace stay consistent",
                status="generated",
                when="--secrets was passed explicitly",
            ),
        ),
        flags=(
            ("--agents a,b,c",
             "REQUIRED — one _base_<agent>.yaml and one <set>/<agent>.yaml per "
             "name; this is the only thing that decides how many files land"),
            ("--set NAME", "REQUIRED — the tier-2 directory name and the "
                           "JAATO_PROFILE_SET value written into .env"),
            ("--secrets env (default)",
             'api_key: "${<PROVIDER_KEY_ENV>}" in each set profile, the var '
             "surfaced as a blank in .env, and .env git-ignored — runs on a "
             "public checkout with nothing else installed"),
            ("--secrets none",
             "no api_key line at all; the provider reads its own env var.  Still "
             "surfaces the var in .env and still git-ignores it"),
            ("--secrets pass (or any scheme://)",
             "api_key: <scheme>://<path> secret URIs.  Needs an out-of-tree "
             "resolver (e.g. jaato-premium's `pass`); `new` WARNS at scaffold "
             "time when no resolver for the scheme is installed.  No .env "
             "credential blank and no .gitignore rule — the key is not in the "
             "workspace"),
            ("--secret-path TEMPLATE",
             "the path inside a secret URI (default jaato/{provider}/api-key; "
             "'{provider}' is substituted)"),
            ("--force", "overwrite profiles that already exist"),
        ),
        edit_before_running=(
            "plugins: [] in every _base_<agent>.yaml — the generator will not "
            "guess a plugin set (`jaato-scaffold explain plugins`)",
            "the <PROVIDER_KEY_ENV>= blank in .env (env/none secrets modes)",
            "the commented model_tiers block, if the stage wants per-role models",
        ),
        generated_correct=(
            "the two-tier split itself — base holds stage determinism, the set "
            "profile binds provider/model, and JAATO_PROFILE_SET selects between "
            "sets without editing either file",
            "every emitted plugin_configs key is one the target provider "
            "declares",
            "the api_key reference style matches --secrets and the env var name "
            "comes from the provider's declared auth chain",
        ),
        check="the emitted set is run straight back through the SAME validator "
              "the `validate` verb uses — valid by construction.  Findings from "
              "your USER tier (~/.jaato/profiles) are shown for context but "
              "never blamed on the generator",
        next_steps=(
            "edit plugins: [] in .jaato/profiles/_base_<agent>.yaml",
            "fill the credential blank in .env",
            "jaato-scaffold validate <ws> --set <set>",
        ),
    ),

    "client": _client(
        "client",
        detail=(
            "connect → create_session → send one message → wait for the turn → "
            "print the streamed output → disconnect",
            "an INLINE profile spec ({model, provider}) so it runs before you "
            "have a profile set; swap for profile=\"<name>\", agent=\"<name>\"",
            "SessionCreateFailed is caught and reported, not swallowed",
        ),
        edit=('the prompt ("Who are you? Reply in one sentence.")',
              "the inline profile spec, once you have a profile set"),
    ),

    "fire": _client(
        "fire",
        detail=(
            "connect → create_session → send → disconnect WITHOUT waiting",
            "the session keeps running daemon-side after the script exits; "
            "reattach later with another client or the observer archetype",
            "prints the session id it dispatched to",
        ),
        edit=("the kick-off prompt",
              "the inline profile spec, once you have a profile set"),
    ),

    "cascade": _client(
        "cascade",
        detail=(
            "a linear CHAIN: a WORKLIST of (profile, agent, prompt) stages run "
            "one at a time, each to terminal completion before the next starts",
            "one cascade id (uuid) tenants every stage, so an observer can "
            "attach to the whole run",
            "for INDEPENDENT jobs that do not feed forward, use `sweep` instead",
        ),
        edit=("the WORKLIST — two placeholder stages "
              '("Stage 1: do the first thing.") a real cascade reads from your '
              "orchestration",),
    ),

    "observer": _client(
        "observer",
        detail=(
            "read-only: attaches to a RUNNING cascade by id and live-traces its "
            "events; it never sends a message",
            "reads ev.session_id as a plain attribute — the getattr(…, \"\") "
            "idiom cannot tell an unrouted event from a pre-1.2 server",
        ),
        edit=("CASCADE_ID — emitted as the literal "
              "REPLACE_WITH_THE_CASCADE_DRIVER_ID; pass the SAME id the cascade "
              "driver used",),
    ),

    "sweep": _client(
        "sweep",
        detail=(
            "N INDEPENDENT jobs fanned out concurrently — none feeds another, "
            "each isolated, results collected per job and a failed job does not "
            "stop its siblings",
            "the JOBS matrix carries (name, profile, agent, prompt): profile is "
            "CAPABILITIES, agent is WHO IT IS, and they are orthogonal axes",
            "subscribes before creating each session, so no early event is lost",
        ),
        edit=("the JOBS matrix — the example varies the persona with "
              "capabilities held fixed; vary profile, agent, or both",),
    ),

    "host-tools": _client(
        "host-tools",
        detail=(
            "registers a client-provided (\"host\") tool: the AGENT calls it and "
            "YOUR client executes it locally, returning the result",
            "register_client_tools is called BEFORE create_session — mid-session "
            "registration is not seen by the runner-tier model",
            "ships a worked send_to_user tool with its handler",
        ),
        edit=("the _send_to_user handler and its schema — replace with your own "
              "tool",
              "the prompt that provokes the tool call"),
    ),
}


# ------------------------------------------------------------------ lookup

def accepted() -> Tuple[str, ...]:
    """Every archetype name ``new`` accepts, aliases included, sorted."""
    names = set(ARCHETYPES)
    for doc in ARCHETYPES.values():
        names.update(doc.aliases)
    return tuple(sorted(names))


def resolve(name: Optional[str]) -> Optional[ArchetypeDoc]:
    """The doc for *name*, following aliases.  ``None`` → the default
    (profile-set), matching ``new`` with no archetype argument."""
    if name is None:
        return ARCHETYPES[PROFILE_SET]
    if name in ARCHETYPES:
        return ARCHETYPES[name]
    for doc in ARCHETYPES.values():
        if name in doc.aliases:
            return doc
    return None


def documents(doc: ArchetypeDoc, rel_path: str) -> Optional[EmittedFile]:
    """The :class:`EmittedFile` declaring *rel_path*, or None if undeclared.

    Used by ``new --dry-run`` to annotate a planned path, and by the guard to
    assert a real run writes nothing the docs do not mention.  Matching is by
    fnmatch over the declared path with its placeholders widened to ``*``.
    """
    probe = rel_path.replace("\\", "/")
    for ef in doc.writes:
        if fnmatch.fnmatch(probe, ef.glob()):
            return ef
    return None
