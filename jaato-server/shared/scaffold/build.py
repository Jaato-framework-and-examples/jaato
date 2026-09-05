"""The ``new`` verb — scaffold profile-sets (and SDK clients), then re-validate.

The defining property: whatever ``new`` emits, it runs straight back through
:mod:`validate` (the SAME validator the ``validate`` verb uses).  So scaffolded
output is valid **by construction** — there is no separate "is the generated
profile ok" path, and a generator bug that emits an unknown knob fails loudly
at scaffold time instead of being silently dropped at runtime.

``new`` also consults :mod:`introspect` while emitting — it only writes knobs
the target provider actually declares (e.g. ``api_key`` is emitted only if the
provider has an ``api_key`` top-level knob), so the emit step can't author a
key the validate step would then reject.

Fail-loud, no hardcoded fallbacks: required inputs (workspace / set / provider
/ model / agents) must be supplied; an unknown provider is a hard error, not a
guess.  Emitted base profiles carry ``plugins: []`` + a pointer to
``explain plugins`` rather than a guessed plugin set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from . import archetypes as _archetypes
from . import explain as _explain
from . import introspect
from . import validate as _validate

# --------------------------------------------------------------- secrets mode
#
# How a scaffolded profile REFERENCES its provider credential.  Three styles:
#
#   env  (default) — ``api_key: "${JAATO_OPENROUTER_API_KEY}"``.  Env-var
#                    interpolation, resolved by the core framework.  Runs
#                    against a public checkout with nothing else installed.
#   none           — omit ``api_key`` entirely; the provider reads its own env
#                    var directly.  Also public-safe, minimal.
#   uri:<scheme>   — ``api_key: "pass://jaato/<provider>/api-key"`` (or any
#                    registered scheme).  Nicer — the key never touches the
#                    workspace — but the scheme's resolver is an out-of-tree,
#                    entry-point-only plugin (e.g. jaato-premium's ``pass``);
#                    on a host without it, resolution fails (fail-loud at the
#                    provider boundary, or literal-URI-as-key in the generic
#                    path).  This is why ``env`` is the default: a scaffolded
#                    workspace must run out of the box.
#
# Historically the generator hardcoded ``pass://`` — the root cause of public
# example repos shipping profiles that only work with jaato-premium.

_SECRETS_DEFAULT = "env"
_SECRET_PATH_DEFAULT = "jaato/{provider}/api-key"


def _resolve_secrets_mode(raw: Optional[str]) -> "tuple[str, Optional[str]]":
    """Parse the ``--secrets`` value into ``(kind, scheme)``.

    ``kind`` is one of ``"env"`` / ``"none"`` / ``"uri"``.  ``scheme`` is the
    URI scheme (``"pass"``, ``"vault"``, …) when ``kind == "uri"``, else None.
    Accepts ``pass`` or ``pass://`` for a scheme.
    """
    mode = (raw or _SECRETS_DEFAULT).strip()
    if mode in ("env", "none"):
        return mode, None
    scheme = mode[:-3] if mode.endswith("://") else mode
    return "uri", scheme


def _primary_key_env_var(info, provider: str) -> str:
    """The env var a scaffolded profile should reference for the provider key.

    Read from the provider's declared ``AuthSource`` chain (``info.auth``) so
    the name is CORRECT per provider — ``ZHIPUAI_API_KEY``, ``ANTHROPIC_API_KEY``,
    ``JAATO_DOUBLEWORD_API_KEY``, ``JAATO_OPENROUTER_API_KEY`` — rather than a
    guessed ``JAATO_<PROVIDER>_API_KEY`` template (which is wrong for several
    providers).  Prefers an explicit ``*_API_KEY`` env source over OAuth-token
    vars; falls back to the ``JAATO_<PROVIDER>_API_KEY`` convention only when
    the provider declares no env source at all.
    """
    env_names = [s.name for s in (getattr(info, "auth", ()) or ())
                 if getattr(s, "kind", "") == "env" and getattr(s, "name", "")]
    for n in env_names:
        if n.upper().endswith("API_KEY"):
            return n
    if env_names:
        return env_names[0]
    return f"JAATO_{provider.upper()}_API_KEY"


def _api_key_line(provider: str, info, kind: str, scheme: Optional[str],
                  secret_path: str) -> Optional[str]:
    """The YAML ``api_key:`` line for a set profile, per secrets mode.

    Returns None for ``none`` mode (no line emitted).
    """
    if kind == "none":
        return None
    if kind == "uri":
        path = secret_path.format(provider=provider)
        return f"    api_key: {scheme}://{path}"
    env_var = _primary_key_env_var(info, provider)
    return f'    api_key: "${{{env_var}}}"'


def _resolver_registered(scheme: str) -> bool:
    """True if a resolver for *scheme* is discoverable (e.g. jaato-premium's
    ``pass``).  Used to WARN at scaffold time when ``--secrets uri:<scheme>``
    is chosen but nothing can resolve it — the same failure the runtime hits at
    the provider credential boundary, surfaced early."""
    try:
        from shared.plugins.subagent.config import _discover_secret_resolvers
        return scheme in _discover_secret_resolvers()
    except Exception:
        return False


def _ensure_env_gitignore(ws: Path, plan: "_Plan") -> None:
    """Ensure the workspace ``.gitignore`` ignores ``.env`` (keeps
    ``.env.example`` tracked).  Converting to env-var credentials means the
    user now puts a LIVE key in ``.env``; an absent ignore rule turns that into
    a leak.  Creates or appends as needed, idempotently.

    Writes go through *plan* rather than the filesystem directly, so
    ``--dry-run`` reports this file without creating it.
    """
    gi = ws / ".gitignore"
    block = ("# Local env holds a LIVE provider credential — never commit it.\n"
             ".env\n"
             "!.env.example\n")
    if not gi.exists():
        plan.write(gi, block)
        return
    text = gi.read_text(encoding="utf-8")
    lines = {ln.strip() for ln in text.splitlines()}
    if ".env" in lines:
        return  # already ignored
    prefix = text if text.endswith("\n") else text + "\n"
    plan.write(gi, prefix + "\n" + block, action="update")


def _ws_secrets_marker(ws: Path) -> Path:
    return ws / ".jaato" / "scaffold.json"


def _read_ws_secrets(ws: Path) -> Optional[str]:
    """The secrets mode recorded for this workspace by a prior ``new`` (so a
    later ``new`` inherits the same style), or None."""
    import json
    marker = _ws_secrets_marker(ws)
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("secrets")
    except Exception:
        return None


def _write_ws_secrets(ws: Path, raw: str, plan: "_Plan") -> None:
    """Record the chosen secrets mode so subsequent ``new`` calls default to
    it — keeps a workspace's credential-reference style consistent.

    Writes go through *plan*, so ``--dry-run`` reports the marker without
    writing it.
    """
    import json
    marker = _ws_secrets_marker(ws)
    existing = {}
    had = marker.exists()
    if had:
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    if existing.get("secrets") == raw:
        return
    existing["secrets"] = raw
    plan.write(marker, json.dumps(existing, indent=2) + "\n",
               action="update" if had else "create")


def _compose_env(provider: str, active: list) -> str:
    """Build a workspace .env: active settings + commented optional knobs.

    Lists, commented-out, the chosen provider's env vars (all — they're user
    provider config) and the framework knobs that have a MEANINGFUL literal
    default (the tunable ones; empty-default flags + framework-internal vars
    are left to `explain env`).  Discovered from installed code, so the list
    can't drift from what the daemon actually reads.
    """
    info = introspect.resolve_provider(provider)
    pdir = info.dir_name if info else provider
    EV = introspect.env_vars()
    # names already set in the active block — never re-emit them as knobs
    active_names = {ln.split("=", 1)[0].strip()
                    for ln in active if "=" in ln and not ln.startswith("#")}

    lines = ["# Generated by `jaato-scaffold new`.  Active settings below;",
             "# commented lines are optional knobs the daemon/plugins read",
             "# (see `jaato-scaffold explain env`).", ""]
    lines += active + [""]

    # the chosen provider's vars (all — they're user provider config)
    pvars = sorted(n for n, v in EV.items()
                   if v.category == f"provider:{pdir}" and n not in active_names)
    if pvars:
        lines.append(f"# ---- provider: {pdir} (uncomment + set as needed) ----")
        for n in pvars:
            lines.append(f"# {n}={EV[n].default or ''}")
        lines.append("")

    # all OTHER meaningfully-defaulted knobs (daemon + plugins), grouped by
    # category; OTHER providers excluded.  Empty-default flags + internal
    # vars (no meaningful default) are left to `explain env`.
    by_cat: Dict[str, list] = {}
    for n, v in EV.items():
        if v.category.startswith("provider:") or n in active_names:
            continue
        if v.default in (None, ""):
            continue
        by_cat.setdefault(v.category, []).append(n)
    for cat in sorted(by_cat):
        lines.append(f"# ---- {cat} knobs (defaults shown) ----")
        for n in sorted(by_cat[cat]):
            lines.append(f"# {n}={EV[n].default}")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------- the plan


class _Plan:
    """The set of files one ``new`` invocation writes — applied, or rehearsed.

    Every write in this module goes through a plan.  With ``dry_run=False``
    (the normal path) it writes the file and records the label ``new`` prints;
    with ``dry_run=True`` it records the same entry and writes NOTHING, which
    is what makes ``new --dry-run`` answer "what exactly lands in MY workspace
    with THESE flags?" without a throwaway directory.

    Existence checks still read the REAL workspace either way, so a rehearsal
    distinguishes a created file from an appended-to one exactly as the real
    run would.

    Each entry is annotated from :mod:`archetypes` — the same registry
    ``explain archetype`` renders — so the rehearsed tree says what each file
    is FOR, not just that it appears.

    Attributes:
        ws: The workspace root; entries are recorded relative to it.
        doc: The :class:`archetypes.ArchetypeDoc` being built, used to annotate
            entries.  ``None`` disables annotation.
        dry_run: True to rehearse (record, never write).
        entries: ``(relative_path, action)`` in write order, where *action* is
            ``"create"`` or ``"update"``.
    """

    def __init__(self, ws: Path, doc=None, *, dry_run: bool = False):
        self.ws = ws
        self.doc = doc
        self.dry_run = dry_run
        self.entries: List[tuple] = []

    def write(self, path: Path, text: str, action: str = "create") -> None:
        """Record (and unless rehearsing, perform) one write."""
        if not self.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.entries.append((str(path.relative_to(self.ws)), action))

    @property
    def labels(self) -> List[str]:
        """The ``+ <path>`` lines ``new`` prints after a real run."""
        return [rel + (" (updated)" if act == "update" else "")
                for rel, act in self.entries]

    def render(self) -> str:
        """The rehearsed tree, annotated with each file's purpose + status."""
        if not self.entries:
            return "  (nothing — every file already exists; pass --force to overwrite)"
        width = max(len(rel) for rel, _ in self.entries)
        out = []
        for rel, act in self.entries:
            ef = _archetypes.documents(self.doc, rel) if self.doc else None
            glyph = "+" if act == "create" else "~"
            status = f"[{ef.status}]" if ef else "[undocumented]"
            out.append(f"  {glyph} {rel.ljust(width)}  {status:<13} "
                       f"{ef.what if ef else ''}".rstrip())
            if act == "update":
                out.append(f"    {' ' * width}  (appended to — the existing file "
                           f"is not clobbered)")
        return "\n".join(out)


def _dry_run_footer(doc, skipped: str) -> None:
    """Close a rehearsal: say nothing was written and where to look next."""
    print(f"\n(dry run — nothing was written; {skipped} was skipped)")
    print(f"what is IN each file:  jaato-scaffold explain archetype {doc.name}")
    print("write it for real:     re-run without --dry-run")


def run(args) -> int:
    """Dispatch to the archetype builder.

    The accepted names come from :mod:`archetypes` rather than a literal list
    here, so adding a client template makes it an accepted archetype AND makes
    the documentation guard demand a doc entry for it — the drift that left
    ``new`` accepting six archetypes while ``explain`` advertised four.
    """
    archetype = args.archetype
    if archetype is None or archetype in _archetypes.PROFILE_SET_ALIASES:
        return _new_profile_set(args)
    if archetype in _archetypes.CLIENT_ARCHETYPES:
        return _new_client_archetype(args, archetype)
    print(f"unknown archetype {archetype!r} — one of: "
          + ", ".join(_archetypes.accepted()))
    return 2


# --------------------------------------------------------- client archetypes

def _apply_transport(args, subs: Dict[str, str], socket: str) -> Optional[int]:
    """Fill the transport-dependent substitutions; return an error code or None.

    ``--transport`` decides three placeholders at once — the client import, the
    connection constants, and the ``_new_client()`` construction — and
    ``--recoverable`` swaps the class inside two of them.  Extracted from
    :func:`_new_client_archetype` so the emit path reads as "resolve the
    transport, then write the files"; a caller propagates a non-None return as
    its own exit code.
    """
    # --transport selects the client. ipc (default) + ws are daemon clients that
    # share the low-level template (WSClient is IPCClient with the transport
    # swapped, same facade-client API); the connection constants + the
    # _new_client() construction differ. in_process (embedded) is facade-native
    # — `jaato.session(mode="in_process")` — and is documented by
    # `jaato-scaffold explain transports`; scaffold it from the README facade
    # snippet rather than this low-level client template.
    transport = getattr(args, "transport", None) or "ipc"
    if transport == "ws" and not getattr(args, "url", None):
        print("new --transport ws requires --url (ws:// or wss://)")
        return 2
    subs["__ON_STATUS_DEF__"] = ""
    on_status_arg = ""
    if transport == "ws":
        url = getattr(args, "url", None)
        token = getattr(args, "token", None) or ""
        # --recoverable: emit WSRecoveryClient (auto-reconnect over WS, survives
        # daemon restarts / dropped WebSockets) instead of the plain WSClient.
        # Mirrors the IPC branch — WS now has a recovery client at parity
        # (reattaches via the same transport-agnostic server replay).
        if getattr(args, "recoverable", False):
            subs["__CLIENT_CLASS__"] = "WSRecoveryClient"
            subs["__ON_STATUS_DEF__"] = (
                "def _on_status(status):\n"
                "    # Reconnection lifecycle — WSRecoveryClient auto-reconnects and\n"
                "    # survives daemon restarts / dropped WebSockets;\n"
                "    # IncompatibleServerError is treated as permanent.\n"
                "    print(f\"[connection] {getattr(status, 'state', status)}\")\n\n\n"
            )
            on_status_arg = "\n        on_status_change=_on_status,"
        else:
            subs["__CLIENT_CLASS__"] = "WSClient"
        client_class = subs["__CLIENT_CLASS__"]
        subs["__CLIENT_IMPORT__"] = f"from jaato_sdk import {client_class}, ClientType, EventType"
        # --ca: CA-bundle path for wss:// with a self-signed / dev cert, threaded
        # as the SCOPED ca= knob — loaded into a per-connection SSLContext, NEVER
        # os.environ (unlike an SSL_CERT_FILE env hack, which leaks into a
        # subprocess-restarted daemon's OUTBOUND HTTPS and breaks it).
        ca = getattr(args, "ca", None)
        ca_const = f'\nCA = "{ca}"' if ca else ""
        ca_arg = "\n        ca=CA," if ca else ""
        subs["__CONN_CONSTANTS__"] = f'URL = "{url}"\nTOKEN = "{token}"{ca_const}'
        subs["__NEW_CLIENT_CALL__"] = (
            f"{client_class}(\n"
            "        URL,\n"
            "        token=TOKEN or None,\n"
            "        client_type=ClientType.API,   # load-bearing: keeps signal_completion\n"
            "        env_file=ENV_FILE,            # never None (handshake crashes on None)\n"
            f"        workspace_path=WORKSPACE,{ca_arg}{on_status_arg}\n"
            "    )"
        )
    elif transport == "in_process":
        # in_process (embedded) is facade-native — the InProcessClient runs the
        # runtime + session IN-PROCESS, no daemon / socket / url. It shares the
        # low-level client contract (connect/create_session/send_message/
        # subscribe/disconnect), so it rides the same template — only the import
        # (from `jaato`, not `jaato_sdk`) and the constructor differ. There is no
        # recovery client: nothing to reconnect to (the session lives in THIS
        # process), so --recoverable is rejected, mirroring the facade's
        # session(mode="in_process", recovery=True) -> ValueError.
        if getattr(args, "recoverable", False):
            print("new --transport in_process does not support --recoverable "
                  "(no daemon to reconnect to — the session is embedded)")
            return 2
        subs["__CLIENT_CLASS__"] = "InProcessClient"
        subs["__CLIENT_IMPORT__"] = (
            "from jaato import InProcessClient\n"
            "from jaato_sdk import EventType"
        )
        subs["__CONN_CONSTANTS__"] = ""  # no socket/url — model/provider/workspace/env are in the header
        subs["__NEW_CLIENT_CALL__"] = (
            "InProcessClient(\n"
            "        model=MODEL,\n"
            "        provider=PROVIDER,\n"
            "        env_file=ENV_FILE,            # embedded runtime reads the workspace .env\n"
            "        workspace_path=WORKSPACE,     # embedded session workspace (no daemon, no socket)\n"
            "    )"
        )
    else:  # ipc (default)
        # --recoverable: emit IPCRecoveryClient (auto-reconnect, survives daemon
        # restarts) instead of the plain IPCClient.
        if getattr(args, "recoverable", False):
            subs["__CLIENT_CLASS__"] = "IPCRecoveryClient"
            subs["__ON_STATUS_DEF__"] = (
                "def _on_status(status):\n"
                "    # Reconnection lifecycle — IPCRecoveryClient auto-reconnects and\n"
                "    # survives daemon restarts (a per-run jaato-server --stop +\n"
                "    # autostart); IncompatibleServerError is treated as permanent.\n"
                "    print(f\"[connection] {getattr(status, 'state', status)}\")\n\n\n"
            )
            on_status_arg = "\n        on_status_change=_on_status,"
        else:
            subs["__CLIENT_CLASS__"] = "IPCClient"
        client_class = subs["__CLIENT_CLASS__"]
        subs["__CLIENT_IMPORT__"] = f"from jaato_sdk import {client_class}, ClientType, EventType"
        subs["__CONN_CONSTANTS__"] = f'SOCKET = "{socket}"'
        subs["__NEW_CLIENT_CALL__"] = (
            f"{client_class}(\n"
            "        SOCKET,\n"
            "        client_type=ClientType.API,   # load-bearing: keeps signal_completion\n"
            "        auto_start=True,\n"
            "        env_file=ENV_FILE,            # never None (handshake crashes on None)\n"
            f"        workspace_path=WORKSPACE,{on_status_arg}\n"
            "    )"
        )
    return None


def _new_client_archetype(args, archetype: str) -> int:
    """Emit a runnable SDK client (+ .env), then py_compile it (emit-then-check).

    The client templates bake in the known-good recipe; we can't fully *run*
    them here (needs a live daemon + provider auth) so the build-time check is
    a syntax compile — the client analog of profile-set's emit-then-validate.
    Next step for the user is the doctor, which checks the runtime env.
    """
    import py_compile
    from ._client_templates import TEMPLATES

    dry_run = bool(getattr(args, "dry_run", False))
    doc = _archetypes.resolve(archetype)
    missing = [f for f in ("workspace", "provider", "model")
               if not getattr(args, f, None)]
    if missing:
        print(f"new {archetype}: missing required --{' / --'.join(missing)}")
        return 2
    provider = args.provider
    if introspect.resolve_provider(provider) is None:
        known = ", ".join(sorted(introspect.providers()))
        print(f"new {archetype}: unknown provider '{provider}' (have: {known})")
        return 2

    ws = Path(args.workspace).resolve()
    if not dry_run:
        ws.mkdir(parents=True, exist_ok=True)
    env_file = ws / ".env"
    py_file = ws / f"run_{archetype}.py"
    socket = "/tmp/jaato.sock"
    _, template, title = TEMPLATES[archetype]

    subs = {
        "__SOCKET__": socket,
        "__ENV_FILE__": str(env_file),
        "__WORKSPACE__": str(ws),
        "__MODEL__": args.model,
        "__PROVIDER__": provider,
        "__TITLE__": title,
        "__ARCHETYPE__": archetype,
        # Correct per-provider key var from the declared AuthSource chain
        # (ZHIPUAI_API_KEY, ANTHROPIC_API_KEY, …), not a guessed template.
        "__KEY_ENV__": _primary_key_env_var(
            introspect.resolve_provider(provider), provider),
        "__CASCADE_ID__": "REPLACE_WITH_THE_CASCADE_DRIVER_ID",
    }
    # --transport decides the client class, its import, the connection
    # constants and the _new_client() construction (see _apply_transport).
    transport = getattr(args, "transport", None) or "ipc"
    rc = _apply_transport(args, subs, socket)
    if rc is not None:
        return rc

    # Provenance: the FULL resolved invocation that produced this file, stamped
    # into the docstring so it's copy-paste reproducible (not just the bare
    # archetype). Resolved flags only; --token is omitted (it's a secret).
    prov = [f"jaato-scaffold new {archetype}",
            f"--workspace {args.workspace}",
            f"--provider {provider}",
            f"--model {args.model}",
            f"--transport {transport}"]
    if getattr(args, "recoverable", False):
        prov.append("--recoverable")
    if getattr(args, "url", None):
        prov.append(f"--url {args.url}")
    if getattr(args, "ca", None):
        prov.append(f"--ca {args.ca}")
    if getattr(args, "set", None):
        prov.append(f"--set {args.set}")
    if getattr(args, "agents", None):
        prov.append(f"--agents {args.agents}")
    subs["__PROVENANCE__"] = " ".join(prov)

    def _fill(text: str) -> str:
        for k, v in subs.items():
            text = text.replace(k, v)
        return text

    plan = _Plan(ws, doc, dry_run=dry_run)
    if py_file.exists() and not args.force:
        print(f"new {archetype}: {py_file} exists (use --force to overwrite)")
        return 2
    plan.write(py_file, _fill(template))
    if not env_file.exists() or args.force:
        plan.write(env_file, _compose_env(provider, [
            f"JAATO_PROVIDER={provider}", f"MODEL_NAME={args.model}"]),
            action="update" if env_file.exists() else "create")

    if dry_run:
        print(f"`jaato-scaffold new {archetype}` would write into {ws}:\n")
        print(plan.render())
        _dry_run_footer(doc, "the compile check")
        return 0

    print(f"scaffolded {archetype} client in {ws}:")
    for w in plan.labels:
        print(f"  + {w}")

    # emit-then-check: the generated client must at least compile.
    print("\ncompile-checking the generated client …")
    try:
        py_compile.compile(str(py_file), doraise=True)
    except py_compile.PyCompileError as e:
        print(f"✘ generated client does not compile — generator bug:\n{e}")
        return 1
    print("✓ generated client compiles.")
    # Next-steps hint, matched to how the credential is referenced.
    ckind, cscheme = _resolve_secrets_mode(getattr(args, "secrets", None)
                                           or _read_ws_secrets(ws))
    csecret_path = getattr(args, "secret_path", None) or _SECRET_PATH_DEFAULT
    key_env_var = _primary_key_env_var(
        introspect.resolve_provider(provider), provider)
    if ckind == "uri":
        secret_hint = (f" --secret {cscheme}://"
                       f"{csecret_path.format(provider=provider)}")
        cred_note = ""
    else:
        secret_hint = ""
        cred_note = f"  # first set {key_env_var}=... in {env_file}\n"
    print(f"\nnext:\n{cred_note}"
          f"  python -m jaato_sdk.doctor --workspace {ws} "
          f"--env-file {env_file}{secret_hint}\n"
          f"  python {py_file}")
    return 0


# ----------------------------------------------------------- profile-set

def _base_profile_yaml(agent: str) -> str:
    """Tier-1 provider-agnostic base profile — plugins left for the author."""
    return (
        f"# Tier-1 base for the '{agent}' stage — PROVIDER-AGNOSTIC.\n"
        f"# Holds stage determinism (plugins, schemas, permission policy).\n"
        f"# Active model + provider live in profiles/<set>/{agent}.yaml,\n"
        f"# selected by JAATO_PROFILE_SET.  This base MUST stay inherit-able\n"
        f"# (do not bind a provider/model here — that breaks set-selection).\n"
        f"name: _base_{agent}\n"
        f"description: {agent} stage (base; bind provider/model in a set).\n"
        f"plugins: []  # choose plugins — see `jaato-scaffold explain plugins`\n"
    )


def _set_profile_yaml(agent: str, provider: str, model: str,
                      kind: str = _SECRETS_DEFAULT, scheme: Optional[str] = None,
                      secret_path: str = _SECRET_PATH_DEFAULT) -> str:
    """Tier-2 set profile — binds provider+model + only valid knobs.

    Emitted knobs are gated on the provider's declared PROVIDER_KNOBS, so the
    emit step cannot author a key the validate step would reject.

    The ``api_key`` reference style is chosen by *kind* / *scheme* (see the
    secrets-mode section above): ``env`` interpolates ``${<PROVIDER_KEY_ENV>}``
    (default, public-checkout friendly), ``none`` omits it, ``uri`` emits a
    ``<scheme>://<path>`` secret URI.

    Two knobs are emitted COMMENTED OUT, in the same shape: a worked example
    plus an ``explain`` pointer.  Both earn the space by being undiscoverable
    from the authoring surface -- ``model_tiers`` because nothing in a profile
    hints that a stage can span models, ``env`` because the tool whose job is
    env discoverability named only the lower-precedence route (jaato #752).
    The bar is that high on purpose: a generated profile where every knob has
    a commented example is noise nobody reads.  The ``env:`` facts come from
    :data:`explain.PROFILE_ENV_FACTS` rather than being restated here, so this
    half and ``explain env`` cannot drift apart; the worked example's value
    comes from :data:`explain.ENV_EXAMPLE_VALUE` for the same reason, and is
    relative rather than absolute on purpose (see that constant).
    """
    info = introspect.resolve_provider(provider)
    lines = [
        f"# {agent} — {provider} set: {model}.",
        f"name: {agent}",
        f"inherits: [_base_{agent}]",
        "plugins: []  # empty keeps the inherited _base surface",
        f"model: {model}",
        f"provider: {provider}",
        "# Optional multi-model tiers — cognitive roles + a 'vision' modality",
        "# tier.  V2 allows a DIFFERENT provider per tier (e.g. this cheap text",
        "# executor + a vision model elsewhere).  See `jaato-scaffold explain tiers`.",
        "# model_tiers:",
        f"#   executor: {{model: {model}, provider: {provider}}}",
        "#   vision:   {model: google/gemini-2.5-flash-lite, provider: openrouter,",
        "#              modalities: {image: inbound},",
        "#              description: 'view screenshots and diagrams; switch back after'}",
        "#   initial: executor",
        "#   fallback: executor",
        "# 'description' is what the MODEL reads as that tier's bullet in the",
        "# enter_tier tool; omit it to keep the framework's own wording.",
        "# 'modalities' declares which non-text roles a tier fills and in which",
        "# direction (inbound | outbound | bidirectional), so the content gate",
        "# knows where to send an image.  The list form [image] is sugar for",
        "# inbound.  A tier named 'vision' implies image inbound; any other tier",
        "# must say so.  OUTBOUND roles parse but are inert today — see",
        "# docs/design/binary-media-chunks.md.",
        "# Optional per-SESSION env vars.  This block:",
    ]
    lines += [f"#   - {fact}" for fact in _explain.PROFILE_ENV_FACTS]
    lines += [
        "#     — the trace vars resolve theirs against the session workspace,",
        "#     so the RELATIVE form below writes one file per session, in its",
        "#     own workspace, where an absolute path would be fixed at this",
        "#     profile and shared by every session using it.",
        "#     See `jaato-scaffold explain env`.",
        "# env:",
        f"#   {_explain.ENV_EXAMPLE_VAR}: {_explain.ENV_EXAMPLE_VALUE}",
    ]
    knobs = info.knobs if info else None
    if knobs is not None:
        cfg = [f"plugin_configs:", f"  {provider}:"]
        if knobs.accepts("top_level", "api_key"):
            key_line = _api_key_line(provider, info, kind, scheme, secret_path)
            if key_line is not None:
                cfg.append(key_line)
        if knobs.accepts("api_params", "temperature"):
            cfg.append("    api_params:")
            cfg.append("      temperature: 0.0  # determinism knob")
        if len(cfg) > 2:
            lines.extend(cfg)
    return "\n".join(lines) + "\n"


def _report_revalidation(diags) -> int:
    """Print post-generation findings and decide the generator's verdict.

    ``validate_workspace`` reports over the MERGED profile tree — the workspace
    set just generated AND the inherited user tier (``~/.jaato/profiles``).
    Only the workspace tier is ours, so only it can convict the generator.

    Counting user-tier findings here accused the scaffold of a bug it did not
    commit, emphatically ("this is a generator bug; please report"), on a CLEAN
    generation — sending the reader into the scaffold templates hunting for a
    plugin reference that lives in their home directory.  ``validate`` already
    labels findings ``[workspace]`` / ``[user]``; this never used it.
    Reported from the cascade-coordination example, 2026-08-24.

    Extracted from ``_new_profile_set`` so it can be tested by CALLING it: the
    alternative is asserting on the source of a function that needs a real
    filesystem, which survives the very edit it is meant to catch.

    Returns:
        Process exit code — non-zero only when the GENERATED set has errors.
    """
    ours = [d for d in diags if getattr(d, "tier", None) != "user"]
    theirs = [d for d in diags if getattr(d, "tier", None) == "user"]
    errs = [d for d in ours if d.severity == "error"]
    for d in diags:
        loc = f" @ {d.where}" if d.where else ""
        tier = f"[{d.tier}] " if getattr(d, "tier", None) else ""
        print(f"  [{d.severity}] {tier}{d.profile}: {d.code}: {d.message}{loc}")
    if theirs:
        print(f"\nnote: {len(theirs)} finding(s) above are in your USER tier "
              "(~/.jaato/profiles), not in the generated set — shown for "
              "context, not attributed to the scaffold.")
    if errs:
        print(f"\n✘ scaffold emitted {len(errs)} error(s) in the generated set "
              "— this is a generator bug; please report.")
        return 1
    print("✓ scaffolded set is valid by construction.")
    return 0


def _emit_set_env(ws: Path, plan: "_Plan", provider: str, active: List[str],
                  set_name: str, kind: str, key_env_var: str) -> None:
    """Write or extend the workspace ``.env`` for a scaffolded profile-set.

    A FRESH workspace gets the full composed file (the set selector plus the
    commented knob catalogue).  An EXISTING one is only appended to, and only
    with lines it lacks: a ``JAATO_PROFILE_SET`` already there points at the
    set the user is running and must not be retargeted behind their back, and
    a credential already filled in must not be blanked.

    Args:
        active: The active (uncommented) block for a fresh file — the set
            selector and, in env/none secrets modes, the credential blank.
        kind: The resolved secrets mode; only ``env`` / ``none`` put the
            credential in the environment at all.
        key_env_var: The provider's declared key variable.
    """
    envf = ws / ".env"
    if not envf.exists():
        plan.write(envf, _compose_env(provider, active))
        return
    existing = envf.read_text(encoding="utf-8")
    add: List[str] = []
    if "JAATO_PROFILE_SET" not in existing:
        add.append(f"JAATO_PROFILE_SET={set_name}")
    if kind in ("env", "none") and f"{key_env_var}=" not in existing \
            and f"{key_env_var} =" not in existing:
        add.append(f"{key_env_var}=")
    if add:
        prefix = existing if existing.endswith("\n") else existing + "\n"
        plan.write(envf, prefix + "\n".join(add) + "\n", action="update")


def _new_profile_set(args) -> int:
    # --- fail-loud required inputs --------------------------------------
    missing = [f for f in ("workspace", "set", "provider", "model")
               if not getattr(args, f, None)]
    if missing:
        print(f"new profile-set: missing required --{' / --'.join(missing)}")
        return 2
    if not args.agents:
        print("new profile-set: --agents a,b,c is required (the stage names)")
        return 2

    provider = args.provider
    if introspect.resolve_provider(provider) is None:
        known = ", ".join(sorted(introspect.providers()))
        print(f"new profile-set: unknown provider '{provider}' (have: {known})")
        return 2

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    dry_run = bool(getattr(args, "dry_run", False))
    doc = _archetypes.resolve(_archetypes.PROFILE_SET)
    ws = Path(args.workspace).resolve()
    pdir = ws / ".jaato" / "profiles"
    setdir = pdir / args.set
    if not dry_run:
        setdir.mkdir(parents=True, exist_ok=True)

    # How profiles REFERENCE the provider credential (env / none / uri:<scheme>).
    # Explicit --secrets wins; else the workspace's recorded choice; else the
    # public-checkout-friendly default (env-var interpolation).
    raw_secrets = getattr(args, "secrets", None) or _read_ws_secrets(ws) \
        or _SECRETS_DEFAULT
    kind, scheme = _resolve_secrets_mode(raw_secrets)
    secret_path = getattr(args, "secret_path", None) or _SECRET_PATH_DEFAULT
    if kind == "uri" and not _resolver_registered(scheme):
        print(f"  [warning] --secrets {scheme}://: no resolver for the "
              f"'{scheme}' scheme is installed, so every profile in this set "
              f"will FAIL to resolve its key at connect (is the plugin that "
              f"provides it, e.g. jaato-premium, installed?). Use --secrets env "
              f"for a public checkout.")
    key_env_var = _primary_key_env_var(introspect.resolve_provider(provider),
                                       provider)

    plan = _Plan(ws, doc, dry_run=dry_run)
    for agent in agents:
        base = pdir / f"_base_{agent}.yaml"
        if not base.exists() or args.force:
            plan.write(base, _base_profile_yaml(agent),
                       action="update" if base.exists() else "create")
        setf = setdir / f"{agent}.yaml"
        if not setf.exists() or args.force:
            plan.write(setf,
                       _set_profile_yaml(agent, provider, args.model,
                                         kind, scheme, secret_path),
                       action="update" if setf.exists() else "create")

    # emit/merge the workspace .env so the set is SELECTED at runtime
    # (JAATO_PROFILE_SET) — without it the workspace isn't runnable as the
    # intended set.  Never clobber an existing JAATO_PROFILE_SET line.
    #
    # In env/none mode the credential lives in the env, so surface the provider
    # key var as an ACTIVE, empty fill-in (and git-ignore .env so the live key
    # can't be committed).  In uri mode the key is in the secret store, not the
    # env, so neither applies.
    active = ["# select this profile-set at runtime (tier-2 overlay)",
              f"JAATO_PROFILE_SET={args.set}"]
    if kind in ("env", "none"):
        active += ["",
                   f"# provider credential — fill in ({provider}); referenced by",
                   f"# the set profiles as ${{{key_env_var}}}.",
                   f"{key_env_var}="]
    _emit_set_env(ws, plan, provider, active, args.set, kind, key_env_var)

    if kind in ("env", "none"):
        _ensure_env_gitignore(ws, plan)
    if getattr(args, "secrets", None):
        _write_ws_secrets(ws, raw_secrets, plan)

    if dry_run:
        print(f"`jaato-scaffold new profile-set` would write into {ws} "
              f"(set '{args.set}', {provider}/{args.model}, "
              f"secrets={raw_secrets}):\n")
        print(plan.render())
        _dry_run_footer(doc, "the re-validation")
        return 0

    print(f"scaffolded profile-set '{args.set}' ({provider}/{args.model}, "
          f"secrets={raw_secrets}):")
    for w in plan.labels:
        print(f"  + {w}")

    # --- emit-then-validate: the same validator the `validate` verb runs -
    print("\nre-validating scaffolded set …")
    return _report_revalidation(
        _validate.validate_workspace(str(ws), profile_set=args.set))
