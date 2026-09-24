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

import json
import logging
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
        from jaato_server.shared.plugins.subagent.config import _discover_secret_resolvers
        return scheme in _discover_secret_resolvers()
    except Exception:
        return False


_ENV_RULE_BLOCK = ("# Local env holds a LIVE provider credential — never commit it.\n"
                   ".env\n"
                   "!.env.example\n")


def _with_env_rule(text: Optional[str]) -> Optional[str]:
    """*text* with the ``.env`` rule appended, or ``None`` if it has one.

    Converting to env-var credentials means the user now puts a LIVE key in
    ``.env``; an absent ignore rule turns that into a leak.  Keeps
    ``.env.example`` tracked.  Pure — the caller decides whether and how the
    result reaches disk — so it composes with the ``.jaato/`` block in
    :func:`_ensure_gitignore` instead of racing it for the same file.

    Args:
        text: The current ``.gitignore`` text, or ``None`` for no file.
    """
    if text is None:
        return _ENV_RULE_BLOCK
    if ".env" in {ln.strip() for ln in text.splitlines()}:
        return None  # already ignored
    prefix = text if text.endswith("\n") else text + "\n"
    return prefix + "\n" + _ENV_RULE_BLOCK


def _ensure_gitignore(ws: Path, plan: "_Plan", *, env_rule: bool) -> None:
    """Bring the workspace ``.gitignore`` up to date, in ONE write.

    Two blocks share the file.  The ``.jaato/`` block (:mod:`gitignore`)
    is needed by every archetype that writes under ``.jaato/``: without it
    a workspace either commits its session records, logs and stored
    credentials, or ignores the directory wholesale and loses the profiles
    its sessions ran under.  The ``.env`` rule is needed only by the
    secrets modes that put a live key in ``.env``.  They are composed here
    rather than written by two helpers because a plan may be a rehearsal:
    the second helper would read the file the first never wrote, and both
    would report ``create``.

    Idempotent — a file already carrying what it needs is not touched, so a
    re-run of ``new`` leaves no trace, and an existing line is never
    rewritten (a wholesale ``.jaato/`` rule is neutralised by the block's
    leading ``!.jaato/`` instead).  Writes go through *plan*, so
    ``--dry-run`` reports this file without creating it.

    Args:
        ws: The workspace root.
        plan: The invocation's plan; the write is recorded on it.
        env_rule: Also ensure the ``.env`` rule (``--secrets env`` / ``none``).
    """
    from . import gitignore as _gitignore
    gi = ws / ".gitignore"
    original = gi.read_text(encoding="utf-8") if gi.exists() else None
    text = original
    if env_rule:
        text = _with_env_rule(text) or text
    merged = _gitignore.merge(text)
    text = merged if merged is not None else text
    if text is None or text == original:
        return
    plan.write(gi, text, action="update" if original is not None else "create")


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


def _compose_env(provider: Optional[str], active: list) -> str:
    """Build a workspace .env: active settings + commented optional knobs.

    Lists, commented-out, the chosen provider's env vars (all — they're user
    provider config) and the framework knobs that have a MEANINGFUL literal
    default (the tunable ones; empty-default flags + framework-internal vars
    are left to `explain env`).  Discovered from installed code, so the list
    can't drift from what the daemon actually reads.

    ``provider`` may be None for the archetypes that do not bind one (see
    ``_client_templates.PROVIDER_OPTIONAL``).  The provider stanza is then
    simply absent — every OTHER knob is still catalogued, because the
    framework knobs are the same whoever ends up serving the model, and
    inventing a provider section for a provider nobody chose would write a
    misleading default into the one file a reader treats as configuration
    (jaato #820).
    """
    info = introspect.resolve_provider(provider) if provider else None
    pdir = info.dir_name if info else provider
    EV = introspect.env_vars()
    # names already set in the active block — never re-emit them as knobs
    active_names = {ln.split("=", 1)[0].strip()
                    for ln in active if "=" in ln and not ln.startswith("#")}

    lines = ["# Generated by `jaato-scaffold new`.  Active settings below;",
             "# commented lines are optional knobs the daemon/plugins read",
             "# (see `jaato-scaffold explain env`).", ""]
    if active:
        lines += active + [""]
    lines += _provider_env_stanza(EV, pdir, active_names)
    lines += _knob_env_stanzas(EV, active_names)
    return "\n".join(lines)


def _provider_env_stanza(EV, pdir: Optional[str], active_names: set) -> list:
    """The chosen provider's env vars, commented out — ALL of them, since
    they are the reader's provider config.  Empty when no provider is bound
    (see :func:`_compose_env`)."""
    pvars = sorted(n for n, v in EV.items()
                   if pdir and v.category == f"provider:{pdir}"
                   and n not in active_names)
    if not pvars:
        return []
    out = [f"# ---- provider: {pdir} (uncomment + set as needed) ----"]
    out += [f"# {n}={EV[n].default or ''}" for n in pvars]
    return out + [""]


def _knob_env_stanzas(EV, active_names: set) -> list:
    """Every OTHER meaningfully-defaulted knob (daemon + plugins), grouped by
    category.  Other providers are excluded; empty-default flags and
    framework-internal vars (no meaningful default) are left to
    ``explain env``."""
    by_cat: Dict[str, list] = {}
    for n, v in EV.items():
        if v.category.startswith("provider:") or n in active_names:
            continue
        if v.default in (None, ""):
            continue
        by_cat.setdefault(v.category, []).append(n)
    out: list = []
    for cat in sorted(by_cat):
        out.append(f"# ---- {cat} knobs (defaults shown) ----")
        out += [f"# {n}={EV[n].default}" for n in sorted(by_cat[cat])]
        out.append("")
    return out


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

    def write(self, path: Path, text: str, action: str = "create", *,
              executable: bool = False) -> None:
        """Record (and unless rehearsing, perform) one write.

        Args:
            path: Absolute target, under :attr:`ws`.
            text: File contents.
            action: ``"create"`` or ``"update"`` — what the label says.
            executable: Set the owner/group/other execute bits.  Needed by
                ``acceptance.sh``: the gate invokes it as ``./acceptance.sh``,
                so a checks script emitted non-executable is an environment
                fault on every job of the sweep — and one that reports as the
                gate being broken rather than as the generator being wrong.
        """
        if not self.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            if executable:
                path.chmod(path.stat().st_mode | 0o111)
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
    if archetype == _archetypes.PROCESSOR:
        return _new_processor(args)
    if archetype == _archetypes.GITIGNORE:
        return _new_gitignore(args)
    if archetype == _archetypes.DOSSIER:
        return _new_dossier(args)
    print(f"unknown archetype {archetype!r} — one of: "
          + ", ".join(_archetypes.accepted()))
    return 2


# --------------------------------------------------------------- dossier

def _dossier_flag_refusal(profile, component: bool, eval_results) -> Optional[int]:
    """The three ways the two flags contradict each other, or ``None``.

    Split out of :func:`_new_dossier` so the builder reads as "decide which
    document, render it, check it" -- validation and work are the two halves
    this function used to do at once.
    """
    if component and profile:
        print("--component describes the framework and --profile describes a "
              "system built on it: two documents for two readers. Pick one.")
        return 2
    if not component and not profile:
        print("`new dossier` needs --profile NAME (the Annex IV dossier for "
              "that system) or --component (the Article 25(4) pack for jaato "
              "itself).")
        return 2
    if component and eval_results:
        print("--eval-results fills the accuracy section of an Annex IV "
              "dossier; the component pack has none. Pass --profile NAME.")
        return 2
    return None


def _dossier_document(args, ws: Path):
    """``(relative_path, markdown)`` for the document this invocation asks for.

    Returns ``(None, exit_code)`` when the request cannot be served -- a
    profile that does not resolve, or a renderer that raised.  Nothing is
    written on either path: a dossier generated for the wrong system is worse
    than none.
    """
    from . import dossier as _dossier

    if getattr(args, "component", False):
        try:
            return "docs/jaato-component-pack.md", _dossier.render_component_pack()
        except Exception as exc:  # noqa: BLE001 -- name it, never half-write
            print(f"could not render the component pack: {exc}")
            return None, 1

    profile = args.profile
    # The same resolver, and the same refusal wording, `new client --profile`
    # uses: a profile that exists only under an unselected set is reported as
    # that, never as missing.
    refusal = _check_named_profile(args, _archetypes.DOSSIER, profile)
    if refusal is not None:
        return None, refusal
    try:
        text = _dossier.render_dossier(
            profile, str(ws),
            eval_results=getattr(args, "eval_results", None),
            profile_set=getattr(args, "set", None) or _env_profile_set(ws))
    except KeyError:
        print(f"new dossier: profile {profile!r} could not be resolved in "
              f"{ws}. A dossier generated for the wrong system is worse than "
              f"none, so nothing was written.")
        return None, 2
    return f"docs/annex-iv-{_slug(profile)}.md", text


def _new_dossier(args) -> int:
    """``new dossier``: the EU AI Act paperwork this tree can compute (#1121).

    Two documents with two different readers, selected by flag rather than
    both emitted: ``--profile`` writes the Annex IV technical documentation
    for the system that profile defines, ``--component`` writes the Article
    25(4) pack for jaato as somebody else's component.  Neither is a
    compliance document -- every section the framework cannot fill carries a
    ``TODO`` naming the Article that asks for it.

    Emit-then-check, like every archetype: the rendered markdown is read back
    and every Annex IV heading must be present.  A section quietly dropped
    because the framework had nothing to say for it is the exact failure this
    archetype exists not to commit, so it is checked rather than trusted.
    """
    from . import dossier as _dossier

    ws = Path(args.workspace).resolve()
    dry_run = bool(getattr(args, "dry_run", False))
    component = bool(getattr(args, "component", False))

    refusal = _dossier_flag_refusal(getattr(args, "profile", None), component,
                                    getattr(args, "eval_results", None))
    if refusal is not None:
        return refusal

    rel, text = _dossier_document(args, ws)
    if rel is None:
        return text            # the exit code _dossier_document chose

    target = ws / rel
    if target.exists() and not getattr(args, "force", False):
        print(f"{rel} already exists — pass --force to regenerate it. "
              f"Regenerating is the intended way to keep the computed "
              f"sections true; editing them in place makes the document a "
              f"second source of truth about the framework.")
        return 1

    doc = _archetypes.resolve(_archetypes.DOSSIER)
    plan = _Plan(ws, doc, dry_run=dry_run)
    plan.write(target, text, action="update" if target.exists() else "create")

    if dry_run:
        print(f"`jaato-scaffold new dossier` would write into {ws}:\n")
        print(plan.render())
        _dry_run_footer(doc, "the heading read-back")
        return 0

    print(f"scaffolded into {ws}:")
    for label in plan.labels:
        print(f"  + {label}")

    print("\nreading it back …")
    missing = _dossier.missing_sections(text) if not component else ()
    if missing:
        print("✘ headings absent from the rendered document — generator bug: "
              + ", ".join(missing))
        return 1
    print("✓ every declared heading is present.")
    print("\nnext:")
    for step in doc.next_steps:
        print(f"  {step}")
    return 0


def _slug(name: str) -> str:
    """A filename-safe stem for a profile name, preserving what it says."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in name)
    return safe.strip("-") or "profile"


# ------------------------------------------------------------- gitignore

def _new_gitignore(args) -> int:
    """``new gitignore``: the ``.jaato/`` block on its own.

    For a workspace whose assets were written by hand, or scaffolded before
    every archetype merged the block — it is the fix ``validate`` names in
    its ``gitignore_*`` findings.  Idempotent: on a workspace that already
    carries the block it writes nothing and says so, exit 0, so a script can
    run it unconditionally.

    Emit-then-check, like every archetype: the file is read back through
    the daemon's own parser — the same assessment ``validate`` runs — and
    the run fails if an authored entry is still ignored or a state probe is
    not, which is what a wholesale rule this block failed to neutralise
    would look like.
    """
    from . import gitignore as _gitignore

    ws = Path(args.workspace).resolve()
    dry_run = bool(getattr(args, "dry_run", False))
    doc = _archetypes.resolve(_archetypes.GITIGNORE)
    plan = _Plan(ws, doc, dry_run=dry_run)
    _ensure_gitignore(ws, plan, env_rule=False)

    if dry_run:
        print(f"`jaato-scaffold new gitignore` would write into {ws}:\n")
        if plan.entries:
            print(plan.render())
        else:
            print("  (nothing — .gitignore already carries the .jaato/ block)")
        _dry_run_footer(doc, "the read-back check")
        return 0

    if not plan.entries:
        print(f".gitignore in {ws} already carries the .jaato/ block — "
              f"nothing written")
        return 0
    print(f"scaffolded .gitignore in {ws}:")
    for w in plan.labels:
        print(f"  + {w}")

    print("\nreading it back through the daemon's gitignore parser …")
    verdict = _gitignore.assess(ws)
    if not verdict.clean:
        hidden = ", ".join(verdict.hidden_authored) or "-"
        leaked = ", ".join(p for p, _ in verdict.unignored_state) or "-"
        print(f"✘ the block did not take — generator bug: still ignored "
              f"[{hidden}]; still committable [{leaked}]")
        return 1
    print("✓ authored .jaato/ entries committable, runtime state ignored.")
    print("\nnext:\n  git status   # profiles/, agents/, ... show as untracked;"
          " sessions/ and logs/ do not")
    return 0


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
        # ``WSClient.session`` / ``WSRecoveryClient.session`` are their OWN
        # overrides (they wire url / token / ssl / ca and reuse the
        # transport-agnostic _SessionContext), so the facade is available here
        # at parity with IPC — the URL is positional, everything else is a
        # keyword, and that is the whole per-transport difference.
        subs["__OPEN_SESSION_CALL__"] = (
            f"{client_class}.session(\n"
            "        URL,\n"
            "        token=TOKEN or None,\n"
            "        client_type=ClientType.API,   # load-bearing: keeps signal_completion\n"
            "        env_file=ENV_FILE,            # never None (handshake crashes on None)\n"
            f"        workspace_path=WORKSPACE,{ca_arg}{on_status_arg}\n"
            "        connect_timeout=120.0,        # cold daemon autostart ~30-60s\n"
            "        **spec,\n"
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
        subs["__CONN_CONSTANTS__"] = ""  # no socket/url — model/provider/workspace/env are in the header
        subs["__NEW_CLIENT_CALL__"] = (
            "InProcessClient(\n"
            "        model=MODEL,\n"
            "        provider=PROVIDER,\n"
            "        env_file=ENV_FILE,            # embedded runtime reads the workspace .env\n"
            "        workspace_path=WORKSPACE,     # embedded session workspace (no daemon, no socket)\n"
            "    )"
        )
        # ``InProcessClient.session`` yields the SAME facade ``Session`` the
        # daemon transports do, so the archetype bodies are byte-identical
        # across transports.  No client_type / connect_timeout: there is no
        # daemon to declare a role to and nothing to connect to.
        subs["__OPEN_SESSION_CALL__"] = (
            "InProcessClient.session(\n"
            "        model=MODEL,\n"
            "        provider=PROVIDER,\n"
            "        env_file=ENV_FILE,            # embedded runtime reads the workspace .env\n"
            "        workspace_path=WORKSPACE,     # embedded session workspace (no daemon, no socket)\n"
            "        **spec,\n"
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
        # ``IPCClient.session`` / ``IPCRecoveryClient.session`` route through
        # ``open_session``, whose socket knob is the KEYWORD ``socket_path=``
        # (the constructor takes it positionally — the one shape difference
        # between the two calls above).
        subs["__OPEN_SESSION_CALL__"] = (
            f"{client_class}.session(\n"
            "        socket_path=SOCKET,\n"
            "        client_type=ClientType.API,   # load-bearing: keeps signal_completion\n"
            "        auto_start=True,\n"
            "        env_file=ENV_FILE,            # never None (handshake crashes on None)\n"
            f"        workspace_path=WORKSPACE,{on_status_arg}\n"
            "        connect_timeout=120.0,        # cold daemon autostart ~30-60s\n"
            "        **spec,\n"
            "    )"
        )
    return None


#: SDK names a generated client may reference, and where each comes from.
#:
#: Only names the rendered body actually uses are imported.  An unused import
#: in generated code is not cosmetic: a scaffold is read as a statement of what
#: the script needs, so importing ``EventType`` into a client that never
#: subscribes teaches that the low-level event API is part of the recipe.
_SDK_IMPORT_CANDIDATES = (
    # name                  # why it appears in a body
    "ClientType",           # client_type=ClientType.API on the connection
    "EventType",            # low-level subscribe() alongside the facade
    "SessionCreateFailed",  # create_session raises; it does not return None
    "AgentError",           # the facade re-raises an error terminal typed
    "SessionEnded",         # a terminal cut the turn short (#1007/#1044)
    "TurnTimeout",          # a per-job wall clock expired
    "truncation_reason",    # "did this session end where it meant to?"
)


def _referenced_names(body: str) -> "set":
    """Every bare NAME the rendered body evaluates.

    Read from the parse tree, not by substring: the templates discuss these
    names in prose — the observer's comment says in so many words that a
    filter must NOT be written in ``EventType`` wire values — and a substring
    match imports a symbol because the script explains why it does not use it.

    Falls back to substring membership if the body does not parse; the
    emit-then-compile check that follows will fail on the real defect rather
    than on a confusing import line.
    """
    import ast

    try:
        tree = ast.parse(body)
    except SyntaxError:
        return {n for n in _SDK_IMPORT_CANDIDATES if n in body}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def _import_block(body: str, transport: str, client_class: str) -> str:
    """The import line(s) for a rendered client body.

    ``body`` is the script with every placeholder filled EXCEPT the import
    itself, so membership is decided by what the finished code references
    rather than by a per-archetype table that has to be kept in step with the
    templates.

    The client class is always imported (every archetype constructs or opens
    from it); ``in_process`` takes it from ``jaato`` rather than ``jaato_sdk``
    because the embedded client is the runtime, not a transport to it.
    """
    used = [n for n in _SDK_IMPORT_CANDIDATES if n in _referenced_names(body)]
    if transport == "in_process":
        lines = [f"from jaato import {client_class}"]
        if used:
            lines.append("from jaato_sdk import " + ", ".join(used))
        return "\n".join(lines)
    return "from jaato_sdk import " + ", ".join([client_class] + used)


def _client_factory_block(template: str) -> str:
    """The factory definition(s) a template's body calls, and only those.

    ``_open_session`` (the facade context manager) and ``_new_client`` (a raw
    client) are both legitimate — an observer attaches to someone else's
    cascade and opens no session of its own, and a sweep's owner connection
    declares the budget pool without one — and ``sweep`` genuinely uses both.
    Reading the requirement off the template keeps a generated script free of
    a factory nothing calls, which is the same defect class as the dead
    ``MODEL`` / ``PROVIDER`` constants #820 removed from ``observer``.
    """
    from ._client_templates import RAW_CLIENT_FACTORY, SESSION_FACTORY

    blocks = []
    if "_open_session(" in template:
        blocks.append(SESSION_FACTORY)
    if "_new_client(" in template:
        blocks.append(RAW_CLIENT_FACTORY)
    if not blocks:
        raise AssertionError(
            "template calls neither _open_session() nor _new_client(); a "
            "generated client with no way to reach the daemon is a generator "
            "bug, not a valid archetype"
        )
    return "\n\n".join(blocks)


def workspace_profile_names(workspace, set_name=None) -> Optional[List[str]]:
    """Profile names declared in a workspace, sorted — or ``None``.

    Read through the framework's own resolver rather than by globbing stems,
    so a profile whose ``name:`` differs from its filename is named the way a
    caller would have to write it, and one that exists only through
    ``inherits`` still counts.

    ``[]`` and ``None`` are deliberately different answers.  ``[]`` is "this
    workspace declares no profiles", which is a fact ``--profile`` can be
    refused against; ``None`` is "I could not look" — no workspace given, or
    the resolver raised — and turning that into "your profile does not exist"
    would block a legitimate generate against a workspace this process cannot
    read.  A generator that raises while composing an error message is worse
    than one that offers no suggestion, so every failure lands on ``None``.
    """
    if not workspace:
        return None
    try:
        from jaato_server.shared.plugins.subagent.config import discover_profiles

        ws = Path(workspace).resolve()
        result = discover_profiles(profiles_dir=".jaato/profiles",
                                   base_path=str(ws),
                                   config_root=str(ws / ".jaato"),
                                   force_profile_set=set_name
                                   or _env_profile_set(ws))
        return sorted(result.profiles)
    except Exception:           # pragma: no cover - suggestion is best-effort
        return None


def _env_profile_set(ws: Path) -> Optional[str]:
    """``JAATO_PROFILE_SET`` from the workspace ``.env``, if it declares one.

    A profile inside ``profiles/<set>/`` is only in the effective set when
    that set is selected, and the selector a generated workspace runs under
    lives in its own ``.env`` — written there by ``new profile-set``.  So
    reading it is what makes ``--profile <name>`` resolve against the SAME
    set the generated client will run under, rather than against a
    set-less view in which every scaffolded agent profile is invisible.
    """
    # One definition, in ``explain``: the generator and the explain pages
    # must not disagree about which set a workspace is on, and the pages
    # need the same answer to resolve a profile the daemon would load.
    return _explain.workspace_profile_set(str(ws))


def _profile_sets(ws: Path) -> List[str]:
    """Profile-set directory names under ``<ws>/.jaato/profiles/``."""
    root = ws / ".jaato" / "profiles"
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def _check_named_profile(args, archetype: str, name: str):
    """``None`` if *name* resolves in the workspace, else an exit code.

    Refusing here is the point of the flag: a generated client naming a
    profile that does not exist fails at ``session.new``, in a daemon, with a
    message about a profile rather than about the command that wrote it.  The
    check is the same resolver the daemon uses, so a profile that resolves
    only through ``inherits`` counts as present.

    A workspace whose profiles cannot be enumerated at all is ACCEPTED rather
    than refused — see :func:`workspace_profile_names` on why ``None`` must
    not be read as "no profiles".  An enumerated EMPTY workspace is refused,
    because there the absence is measured.
    """
    ws_arg = getattr(args, "workspace", None)
    set_name = getattr(args, "set", None)
    names = workspace_profile_names(ws_arg, set_name)
    if names is None or name in names:
        return None
    elsewhere = _sets_declaring(Path(ws_arg).resolve(), name) if ws_arg else []
    if elsewhere:
        # Found, but only under a set this workspace is not running.  Saying
        # "does not exist" about a file the author can see would send them
        # looking for a typo they did not make.
        print(f"new {archetype}: profile '{name}' exists only in profile-set "
              f"{' / '.join(elsewhere)}, which this workspace does not "
              f"select — pass --set {elsewhere[0]}, or set JAATO_PROFILE_SET "
              f"in the workspace .env")
        return 2
    have = ", ".join(names) if names else "this workspace declares none"
    print(f"new {archetype}: no profile '{name}' in {ws_arg} ({have}) — "
          f"`jaato-scaffold new profile-set` creates one")
    return 2


def _sets_declaring(ws: Path, name: str) -> List[str]:
    """Profile-set names under which *name* resolves, for a not-found report."""
    return [s for s in _profile_sets(ws)
            if name in (workspace_profile_names(str(ws), s) or ())]


def _binding_flags_missing(args, archetype: str, missing: List[str]) -> int:
    """Report an absent session binding, naming ``--profile`` when it applies.

    The generator knows which situation it is in and used to say only
    "missing required --provider / --model", which is the inline-spec answer
    — so an author with a workspace full of profiles was steered, by the only
    message they saw, at the one form that cannot carry plugins, a persona, a
    GC strategy or a completion schema.  When the workspace already declares
    profiles, they are named here and ``--profile`` is offered first.
    """
    names = workspace_profile_names(getattr(args, "workspace", None),
                                    getattr(args, "set", None)) or []
    print(f"new {archetype}: missing required --{' / --'.join(missing)}")
    if names:
        preview = ", ".join(names[:8]) + (" …" if len(names) > 8 else "")
        print(f"  this workspace already declares profiles ({preview}) — "
              f"prefer --profile <name>, which carries the plugins, persona, "
              f"ceilings and completion schema an inline --provider/--model "
              f"spec cannot")
    return 2


def _resolve_client_binding(args, archetype: str, transport: str):
    """Validate ``--profile`` / ``--provider`` / ``--model`` for this archetype.

    Returns ``(exit_code_or_None, provider, model)`` — the caller propagates a
    non-None code as its own.

    WHO OWNS THE BINDING DECIDES WHO MUST SUPPLY IT.  ``--provider`` /
    ``--model`` name what CREATES a session, and a profile is what carries
    them, so the archetypes whose stages/jobs name a profile (``cascade``,
    ``sweep``) and the one with no provider relationship at all (``observer``,
    read-only) accept the flags and do not require them.  Requiring them
    forced an arbitrary choice the profile then overrode, and baked a
    misleading default into ``.env`` and the placeholder (jaato #820).

    ``--profile`` satisfies the binding on any archetype.  It is the
    form a real client uses — a profile carries plugins, a persona, GC, the
    ceilings and the completion schema, none of which an inline spec can
    express — and until it existed the ONLY way to generate a client was the
    inline one, whatever the workspace already had in it.  It is mutually
    exclusive with ``--provider`` / ``--model``: emitting both would put two
    bindings in one call, and the profile wins at runtime, so the flags the
    author passed would silently decide nothing.

    ``in_process`` is the exception on EVERY archetype: the embedded client IS
    the binding — there is no daemon to resolve a profile against — so the
    flags stay required there whatever the archetype.
    """
    from ._client_templates import PROVIDER_OPTIONAL

    profile_name = getattr(args, "profile", None)
    provider = getattr(args, "provider", None)
    model = getattr(args, "model", None)
    if profile_name:
        return _resolve_profile_binding(args, archetype, transport,
                                        profile_name, provider, model)


    binding_optional = (archetype in PROVIDER_OPTIONAL
                        and transport != "in_process")
    required = (["workspace"] if binding_optional
                else ["workspace", "provider", "model"])
    missing = [f for f in required if not getattr(args, f, None)]
    if missing:
        return _binding_flags_missing(args, archetype, missing), None, None
    # Half a binding is not a binding: a --model with no --provider (or the
    # reverse) would emit one live constant and one placeholder, which reads
    # as a working spec and is not one.
    if bool(provider) != bool(model):
        print(f"new {archetype}: --provider and --model go together "
              f"(got only --{'provider' if provider else 'model'}); "
              f"supply both, or neither and point the stages at a profile")
        return 2, None, None
    if provider and introspect.resolve_provider(provider) is None:
        known = ", ".join(sorted(introspect.providers()))
        print(f"new {archetype}: unknown provider '{provider}' (have: {known})")
        return 2, None, None
    return None, provider, model


def _profile_set_env_lines(args) -> List[str]:
    """``JAATO_PROFILE_SET=<set>`` when a set is what made ``--profile`` resolve.

    ``--profile X --set Y`` resolved X only because Y was forced, and the
    GENERATED client resolves its profile from the workspace ``.env`` — so
    writing that file without the selector produces a client naming a
    profile it cannot find, after the generator accepted the name.  The
    set is honoured at generation and lost at run time, which is the shape
    ``--profile`` exists to remove.

    Only when both are given: ``--set`` alone is the profile-set
    generator's flag and this archetype has no business acting on it, and
    ``--profile`` alone resolved without a set and needs none.
    """
    set_name = getattr(args, "set", None)
    if not (set_name and getattr(args, "profile", None)):
        return []
    return [f"JAATO_PROFILE_SET={set_name}"]


def _append_profile_set(plan: "_Plan", env_file: Path, args) -> None:
    """Add the selector to an EXISTING ``.env`` that lacks it.

    The file is otherwise left alone (:func:`_should_write_client_env`), and
    an existing ``JAATO_PROFILE_SET`` is never retargeted behind the
    author's back — the rule ``_emit_set_env`` already applies to a
    scaffolded profile-set's own ``.env``.
    """
    lines = _profile_set_env_lines(args)
    if not lines or not env_file.exists():
        return
    existing = env_file.read_text(encoding="utf-8")
    if "JAATO_PROFILE_SET" in existing:
        return
    prefix = existing if existing.endswith("\n") else existing + "\n"
    plan.write(env_file, prefix + "\n".join(lines) + "\n", action="update")


def _should_write_client_env(args, env_file: Path) -> bool:
    """Whether ``new <client-archetype>`` may write the workspace ``.env``.

    A missing file is always written.  An EXISTING one is replaced only under
    ``--force``, and never for a ``--profile`` client: that workspace's
    ``.env`` is where ``JAATO_PROFILE_SET`` lives, and this archetype's
    template carries a provider/model pair the profile supersedes — so
    rewriting it would drop the selector the named profile needs in order to
    resolve at all, in the name of writing two lines that decide nothing.
    """
    if not env_file.exists():
        return True
    return bool(getattr(args, "force", False)) and not getattr(
        args, "profile", None)


def _archetype_takes_a_session_binding(archetype: str) -> bool:
    """Whether this archetype's template has somewhere to put ``--profile``.

    DERIVED from the template text rather than tabulated: the placeholder
    is what actually receives the binding, so a template that gains or
    loses one is answered correctly with no edit here.  A hardcoded list is
    how the flag came to be accepted where it could not be honoured —
    ``new cascade --profile worker`` resolved the name, accepted it, and
    emitted the ``"<profile-name>"`` placeholder, byte-identical to passing
    nothing.

    ``cascade`` / ``sweep`` are the archetypes that legitimately have no
    single binding: each stage or job names its OWN profile, which is the
    point of them, and ``observer`` opens no session at all.
    """
    from ._client_templates import TEMPLATES

    entry = TEMPLATES.get(archetype)
    if not entry:
        return True         # unknown archetype: let the normal path refuse it
    return "__SESSION_BINDING__" in entry[1]


def _resolve_profile_binding(args, archetype: str, transport: str,
                             profile_name: str, provider, model):
    """The ``--profile`` half of :func:`_resolve_client_binding`.

    Split out to keep its caller under the cyclomatic ceiling, and because
    the two halves answer different questions: this one asks whether a NAME
    resolves in a workspace, the other whether a provider/model PAIR is
    complete and installed.

    Returns the same ``(exit_code_or_None, provider, model)`` triple, with
    both binding values ``None`` on success — a profile carries them, and
    emitting a ``MODEL`` / ``PROVIDER`` constant beside it would put a second
    binding in the generated file that decides nothing.
    """
    if provider or model:
        print(f"new {archetype}: --profile and --provider/--model are two "
              f"bindings for one session; the profile wins at runtime, so "
              f"pass one or the other")
        return 2, None, None
    if transport == "in_process":
        print(f"new {archetype}: --profile needs a daemon to resolve it "
              f"against; --transport in_process IS the binding, so pass "
              f"--provider/--model there")
        return 2, None, None
    if not _archetype_takes_a_session_binding(archetype):
        print(f"new {archetype}: --profile binds ONE session, and this "
              f"archetype does not open one it can bind — its stages/jobs "
              f"carry their own profile names, which you edit in the "
              f"generated file.  Drop --profile")
        return 2, None, None
    code = _check_named_profile(args, archetype, profile_name)
    return (code, None, None) if code else (None, None, None)


def _binding_substitutions(archetype: str, provider, model,
                           profile_name=None) -> Dict[str, str]:
    """The placeholders that depend on whether a provider/model was bound.

    ``MODEL`` / ``PROVIDER`` are emitted only where the body READS them.
    ``observer`` never does — it neither creates a session nor sends a message
    — so those constants were two dead variables in a script with no provider
    relationship (jaato #820); and with no binding supplied there is nothing
    to put in them either.

    The stage/job placeholder follows the same fact: an inline SPEC when a
    binding was given (so the generated cascade runs before any profile
    exists), a profile NAME when it was not, which is the shape a
    profile-driven cascade actually uses.
    """
    from ._client_templates import NO_MODEL_CONSTANTS

    bind = bool(provider) and archetype not in NO_MODEL_CONSTANTS
    spec = '{"model": MODEL, "provider": PROVIDER}'
    return {
        "__MODEL__": model or "",
        "__PROVIDER__": provider or "",
        "__MODEL_CONSTANTS__": (f'MODEL = "{model}"\nPROVIDER = "{provider}"'
                                if bind else ""),
        # What the generated body passes as ``profile=``.  A named profile
        # when one was given, else the inline spec — so a client generated
        # against a workspace that has profiles uses them, and one generated
        # against an empty workspace still runs.
        "__SESSION_BINDING__": (f'"{profile_name}"' if profile_name
                                else spec),
        "__STAGE_PROFILE__": spec if bind else '"<profile-name>"',
        "__INLINE_SPEC__": (spec if bind else
                            '{"model": "<model>", "provider": "<provider>"}'),
        # Correct per-provider key var from the declared AuthSource chain
        # (ZHIPUAI_API_KEY, ANTHROPIC_API_KEY, …), not a guessed template.
        "__KEY_ENV__": (_primary_key_env_var(
            introspect.resolve_provider(provider), provider)
            if provider else ""),
    }


def _provenance(args, archetype: str, transport: str, provider, model) -> str:
    """The FULL resolved invocation that produced a generated file.

    Stamped into the docstring so it is copy-paste reproducible, rather than
    the bare archetype.  Resolved flags only; ``--token`` is omitted because
    it is a secret, and an absent binding contributes no flag rather than an
    empty one.
    """
    prov = [f"jaato-scaffold new {archetype}",
            f"--workspace {args.workspace}"]
    if provider:
        prov.append(f"--provider {provider}")
    if model:
        prov.append(f"--model {model}")
    # ``--profile`` IS the binding when it is given, and a banner that
    # claims to be copy-paste reproducible must carry it: without it the
    # printed command re-runs to `missing required --provider / --model`,
    # so the one line asserting reproducibility reproduced a failure.
    if getattr(args, "profile", None):
        prov.append(f"--profile {args.profile}")
    prov.append(f"--transport {transport}")
    for flag, value in (("--recoverable", getattr(args, "recoverable", False)),
                        ("--url", getattr(args, "url", None)),
                        ("--ca", getattr(args, "ca", None)),
                        ("--set", getattr(args, "set", None)),
                        ("--agents", getattr(args, "agents", None))):
        if value is True:
            prov.append(flag)
        elif value:
            prov.append(f"{flag} {value}")
    return " ".join(prov)


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
    transport = getattr(args, "transport", None) or "ipc"
    rc, provider, model = _resolve_client_binding(args, archetype, transport)
    if rc is not None:
        return rc

    ws = Path(args.workspace).resolve()
    if not dry_run:
        ws.mkdir(parents=True, exist_ok=True)
    env_file = ws / ".env"
    py_file = ws / f"run_{archetype}.py"
    socket = "/tmp/jaato.sock"
    _, template, title = TEMPLATES[archetype]

    subs = _binding_substitutions(archetype, provider, model,
                                  getattr(args, "profile", None))
    subs.update({
        "__SOCKET__": socket,
        "__ENV_FILE__": str(env_file),
        "__WORKSPACE__": str(ws),
        "__TITLE__": title,
        "__ARCHETYPE__": archetype,
        "__CASCADE_ID__": "REPLACE_WITH_THE_CASCADE_DRIVER_ID",
    })
    # --transport decides the client class, its import, the connection
    # constants and the _new_client() / _open_session() calls (_apply_transport).
    rc = _apply_transport(args, subs, socket)
    if rc is not None:
        return rc
    # Emit the factory the BODY calls, and only that one.  Derived from the
    # template rather than tabulated, so a template that changes which factory
    # it uses cannot leave a dead definition behind (or lose a live one).
    subs["__CLIENT_FACTORY__"] = _client_factory_block(template)
    # One constants block, joined from the halves that are actually present —
    # a transport with no socket/url (in_process) and an archetype with no
    # binding (observer) each contribute nothing, and concatenating raw would
    # leave a stray blank line where the absent half used to be.
    subs["__CONSTANTS__"] = "\n".join(
        part for part in (subs["__MODEL_CONSTANTS__"],
                          subs["__CONN_CONSTANTS__"]) if part)
    subs["__PROVENANCE__"] = _provenance(args, archetype, transport,
                                         provider, model)
    # The JOBS matrix names the gate profile we are about to write, rather
    # than the "your-profile" placeholder, so the emitted client and the
    # emitted profile refer to each other on the first run.  With --no-gate
    # there is no profile to name and the placeholder stands.
    gate_name = _gate_name(args)
    gated = _gate_wanted(args, archetype)
    subs.update(_gate_substitutions(gated, gate_name))

    def _fill(text: str) -> str:
        """Substitute to a FIXED POINT, not once.

        A substitution can introduce another placeholder — the factory block
        is chosen per archetype and itself carries ``__OPEN_SESSION_CALL__`` /
        ``__NEW_CLIENT_CALL__`` — and a single pass leaves whichever token was
        already visited unresolved, depending on dict order.  That failure is
        silent in the emitted file and only shows up as a NameError when the
        reader runs it.
        """
        for _ in range(len(subs) + 1):
            filled = text
            for k, v in subs.items():
                filled = filled.replace(k, v)
            if filled == text:
                return text
            text = filled
        raise AssertionError(
            "placeholder substitution did not converge — a template token "
            "expands to itself"
        )

    # Two passes, because the IMPORT LINE is a fact about the finished body.
    # An unused import in generated code reads as a capability the script
    # needs, so the names are computed from what the rendered script actually
    # references rather than tabulated per archetype and left to drift.
    subs["__CLIENT_IMPORT__"] = _import_block(
        _fill(template), transport, subs["__CLIENT_CLASS__"])

    plan = _Plan(ws, doc, dry_run=dry_run)
    if py_file.exists() and not args.force:
        print(f"new {archetype}: {py_file} exists (use --force to overwrite)")
        return 2
    plan.write(py_file, _fill(template))
    if _should_write_client_env(args, env_file):
        active = ([f"JAATO_PROVIDER={provider}", f"MODEL_NAME={model}"]
                  if provider else [])
        active += _profile_set_env_lines(args)
        plan.write(env_file, _compose_env(provider, active),
                   action="update" if env_file.exists() else "create")
    else:
        _append_profile_set(plan, env_file, args)

    # The completion gate, for the archetypes whose jobs are graded.  Written
    # in the same pass as the client so the two agree about the profile name
    # on the first run (jaato #772).
    gate_skipped: List[Path] = []
    if gated:
        gate_skipped = _emit_sweep_gate(plan, ws, gate_name,
                                        subs["__PROVENANCE__"],
                                        provider, model, bool(args.force))
        # The gate lands under .jaato/ (processor, schema, profile), so the
        # workspace needs the block that keeps it committable.
        _ensure_gitignore(ws, plan, env_rule=False)

    if dry_run:
        print(f"`jaato-scaffold new {archetype}` would write into {ws}:\n")
        print(plan.render())
        _dry_run_footer(doc, _rehearsal_skips(gated))
        return 0

    print(f"scaffolded {archetype} client in {ws}:")
    for w in plan.labels:
        print(f"  + {w}")
    _report_kept(ws, gate_skipped)

    # emit-then-check: the generated client must at least compile.
    print("\ncompile-checking the generated client …")
    try:
        py_compile.compile(str(py_file), doraise=True)
    except py_compile.PyCompileError as e:
        print(f"✘ generated client does not compile — generator bug:\n{e}")
        return 1
    print("✓ generated client compiles.")

    rc = _check_generated_gate(ws, gate_name, gated, gate_skipped)
    if rc is not None:
        return rc

    _print_next_steps(args, ws, env_file, py_file, provider, gated, gate_name)
    return 0


def _rehearsal_skips(gated: bool) -> str:
    """What ``--dry-run`` did NOT run, named so the rehearsal is not oversold."""
    return ("the compile check and the gate probe" if gated
            else "the compile check")


def _report_kept(ws: Path, skipped: List[Path]) -> None:
    """Name every gate file left as it was.

    A silent skip is indistinguishable from a rewrite, and these are files the
    author is expected to have edited — so the one thing worse than not
    overwriting them is not saying so.
    """
    for path in skipped:
        print(f"  · {path.relative_to(ws)} (kept — pass --force to overwrite)")


def _check_generated_gate(ws: Path, gate_name: str, gated: bool,
                          skipped: List[Path]) -> Optional[int]:
    """Emit-then-check for the gate; an exit code on failure, else ``None``.

    Checked far harder than the client, which only has to compile: the gate is
    loaded through the framework's own loader and DRIVEN, because the failure
    that matters here is not a syntax error but a gate that ACCEPTS what it
    should refuse — and that failure is invisible until a graded run reports a
    clean board it never earned.

    Skipped entirely when files were kept: what is on disk is then the
    author's, and vouching for it would be vouching for something this
    generator did not write.
    """
    if not gated or skipped:
        return None
    print("\ndriving the generated completion gate …")
    reason = _probe_generated_gate(ws, gate_name)
    if reason:
        print(f"✘ generated gate is not usable — generator bug: {reason}")
        return 1
    print("✓ it loads, and refuses a completion while acceptance.sh has no "
          "checks configured.")
    return None


def _print_next_steps(args, ws: Path, env_file: Path, py_file: Path,
                      provider, gated: bool, gate_name: str) -> None:
    """The closing hint, matched to how the credential is referenced.

    With no provider bound there is no credential to name — the profile each
    stage points at owns it — so the hint says nothing rather than naming a
    variable this workspace does not use.
    """
    ckind, cscheme = _resolve_secrets_mode(getattr(args, "secrets", None)
                                           or _read_ws_secrets(ws))
    csecret_path = getattr(args, "secret_path", None) or _SECRET_PATH_DEFAULT
    secret_hint = ""
    cred_note = ""
    if provider:
        key_env_var = _primary_key_env_var(
            introspect.resolve_provider(provider), provider)
        if ckind == "uri":
            secret_hint = (f" --secret {cscheme}://"
                           f"{csecret_path.format(provider=provider)}")
        else:
            cred_note = f"  # first set {key_env_var}=... in {env_file}\n"
    print(f"\nnext:\n{_gate_next_step(ws, gated, gate_name)}{cred_note}"
          f"  python -m jaato_sdk.doctor --workspace {ws} "
          f"--env-file {env_file}{secret_hint}\n"
          f"  python {py_file}")


def _gate_next_step(ws: Path, gated: bool, gate_name: str) -> str:
    """The gate's line in the next-steps hint, first because its omission is
    the one that fails silently: the sweep runs, every job is refused by an
    unconfigured gate, and the run reads as a model failure rather than as a
    missing edit."""
    if not gated:
        return ""
    profile = ws / ".jaato" / "profiles" / f"{gate_name}.yaml"
    return (f"  # put this sweep's acceptance criteria in "
            f"{ws / 'acceptance.sh'}\n"
            f"  #   (run_checks is empty as emitted, so every job is refused "
            f"until you fill it)\n"
            f"  # then choose plugins: [] in {profile}\n")


# ------------------------------------------------------------ the sweep gate

#: The archetype whose emitted set includes a completion gate.
#:
#: One archetype rather than a general flag because the gate is not a generic
#: nicety: a sweep's jobs are GRADED, so "did this job meet the criteria" is
#: the measurement itself.  A `client` or `fire` script has no scoreboard for a
#: gate to agree with (jaato #772).
GATED_ARCHETYPES = ("sweep",)


def _gate_wanted(args, archetype: str) -> bool:
    """Whether this invocation emits a completion gate.

    On by default for :data:`GATED_ARCHETYPES` — the point of #772 is that the
    gate ARRIVES wired, and an opt-in flag reproduces one level up the very
    discovery problem it exists to remove (the author who does not know a gate
    is the missing piece does not know to ask for one).  ``--no-gate`` is the
    escape hatch for a sweep that genuinely grades nothing.
    """
    return archetype in GATED_ARCHETYPES and not getattr(args, "no_gate", False)


def _gate_name(args) -> str:
    """The stem shared by all four files of the gate set.

    One name across the set — module, schema, profile and the entry's
    ``name:`` — so the four files are visibly one thing rather than four
    that happen to be related.
    """
    from . import _gate_templates as _gate

    return str(getattr(args, "gate_name", None)
               or _gate.DEFAULT_GATE_NAME).strip()


def _gate_substitutions(gated: bool, name: str) -> Dict[str, str]:
    """Fill the sweep template's gate-dependent tokens.

    Both are emitted for every client archetype (only ``sweep`` contains
    them), because ``_fill`` substitutes whatever it is given and a token
    with no entry survives into the generated file as literal
    ``__JOBS_PROFILE__``.
    """
    if not gated:
        return {"__JOBS_PROFILE__": '"your-profile"', "__GATE_NOTE__": ""}
    return {
        "__JOBS_PROFILE__": f'"{name}"',
        "__GATE_NOTE__": (
            "# THE JOBS ARE GATED.  The profile named below was written beside\n"
            "# this script and carries a completion gate: a job cannot signal\n"
            f"# completion until ./acceptance.sh passes, which is where you put\n"
            f"# this sweep's acceptance criteria.  As emitted that script has no\n"
            "# checks in it and every job is refused, deliberately — a gate with\n"
            "# nothing configured must not read as a gate that passed.\n"
            "#\n"
            "# The same script is what should grade the sweep afterwards, so the\n"
            "# gate and the scoreboard cannot end up measuring different things.\n"
            f"#   the checks       ./acceptance.sh\n"
            f"#   the gate         .jaato/scripts/processors/{name}.py\n"
            f"#   the wiring       .jaato/profiles/{name}.yaml\n"
            "#   what it all does  jaato-scaffold explain completion\n"
        ),
    }


def _gate_paths(ws: Path, name: str) -> Dict[str, Path]:
    """Where each file of the gate set lands.

    The two ``.jaato/`` paths are not free choices: ``script:`` and
    ``completion_payload_schema:`` in the emitted profile are resolved by
    ``script_loader.resolve_script_path`` and
    ``completion_schema_loader._resolve_schema_path``, whose workspace tier is
    ``<ws>/.jaato/<path>``.  A file written anywhere else is a profile that
    parses and a gate that never loads.
    """
    return {
        "checks": ws / "acceptance.sh",
        "processor": ws / ".jaato" / "scripts" / "processors" / f"{name}.py",
        "schema": ws / ".jaato" / "completion_schemas" / f"{name}.json",
        "profile": ws / ".jaato" / "profiles" / f"{name}.yaml",
    }


def _emit_sweep_gate(plan: "_Plan", ws: Path, name: str, provenance: str,
                     provider, model, force: bool) -> List[Path]:
    """Write the gate set; return the paths that were skipped as existing.

    The four files are written as ONE unit deliberately.  Each is inert alone:
    a processor with no checks script faults on every job, a checks script no
    processor runs grades nothing, and the profile's two keys are what make
    ``signal_completion`` exist for the processor to gate at all.  Emitting
    them separately is what left an author holding three files and the
    relationship between them (jaato #772).

    An existing file is never clobbered without ``--force``: these are files an
    author is expected to EDIT (the checks above all), and a re-run of ``new``
    that silently reverted them to the template would be worse than one that
    refuses.  Skips are reported by the caller rather than being silent.
    """
    from . import _gate_templates as _gate
    from . import _processor_template as _tpl

    paths = _gate_paths(ws, name)
    skipped: List[Path] = []

    wiring = _gate_profile_wiring(name)
    contents = {
        "checks": _gate.render_acceptance_sh(name, provenance),
        "processor": _tpl.render(name, provenance,
                                 checks_command=_gate.CHECKS_COMMAND,
                                 wiring=wiring),
        "schema": _gate.render_schema(name),
        "profile": _gate.render_profile(name, provenance, provider, model),
    }
    for key in ("checks", "processor", "schema", "profile"):
        target = paths[key]
        if target.exists() and not force:
            skipped.append(target)
            continue
        plan.write(target, contents[key],
                   "update" if target.exists() else "create",
                   executable=(key == "checks"))
    return skipped


def _gate_profile_wiring(name: str) -> str:
    """The ``completion_processors:`` block as the emitted PROFILE carries it.

    Read back out of the rendered profile rather than re-templated, so the
    block reproduced in the processor's docstring is the wiring that was
    actually written and not a second copy free to drift from it.  Comment
    lines are dropped: the docstring wants the shape, and the profile's
    commentary is already there for anyone reading the profile.
    """
    from . import _gate_templates as _gate

    lines = _gate.render_profile(name, "").splitlines()
    start = lines.index("completion_processors:")
    kept = [ln for ln in lines[start:] if not ln.lstrip().startswith("#")]
    return "\n".join(kept).rstrip() + "\n"


def _probe_generated_gate(ws: Path, name: str) -> Optional[str]:
    """Drive the emitted gate the way the daemon will; report why it is unusable.

    Stronger than the clients' ``py_compile`` and than ``new processor``'s
    probe, because the thing under test here is a SET: the profile has to parse
    into a processor entry whose ``script:`` resolves to the module that was
    written, the module has to load, and running it has to invoke the
    ``acceptance.sh`` that was written beside it.

    The assertion that matters is the LAST one.  Fresh from the generator,
    ``acceptance.sh`` has no checks configured, and the tempting behaviour —
    a script with nothing to check exiting 0 — would have the gate report "no
    failures" and wave every job through.  So the probe requires the
    unconfigured gate to BLOCK, and to block as a ``faults[]`` entry rather
    than an ``errors[]`` one: it is an environment fault the author must clear,
    not a wrong answer costing the agent a retry.  A generated set that would
    accept a completion on checks that never ran fails here, at scaffold time,
    rather than silently in a graded run.

    Args:
        ws: The workspace the set was written into.
        name: The gate name (the shared stem of all four files).

    Returns:
        A one-line reason the set is not usable, or ``None`` when it is.
    """
    import yaml

    from jaato_server.shared.completion_processors import invoke_processors, load_processors
    from jaato_server.shared.plugins.subagent.config import build_inline_profile

    paths = _gate_paths(ws, name)

    # 1. The profile parses, and its processor entry is the one we wrote.
    try:
        raw = yaml.safe_load(paths["profile"].read_text(encoding="utf-8"))
        profile = build_inline_profile(raw, name=name)
    except Exception as exc:                        # noqa: BLE001
        return f"the emitted profile does not parse: {type(exc).__name__}: {exc}"
    if not profile.completion_processors:
        return "the emitted profile declares no completion_processors"
    entry = profile.completion_processors[0]
    if entry.max_refusals is None:
        return ("the emitted processor entry carries no max_refusals — an "
                "unbounded gate does not terminate on its own (jaato #768)")

    # 2. The schema the profile points at exists and is loadable JSON.  Without
    #    it signal_completion is hidden outright, so there is nothing to gate.
    if not paths["schema"].is_file():
        return f"completion_payload_schema points at a missing {paths['schema']}"
    try:
        json.loads(paths["schema"].read_text(encoding="utf-8"))
    except Exception as exc:                        # noqa: BLE001
        return f"the emitted completion schema is not valid JSON: {exc}"

    # 3. The module loads through the framework's own loader, addressed exactly
    #    as the profile addresses it.
    loaded = load_processors([entry], workspace_path=str(ws), config_root=None)
    if loaded and loaded[0].load_error:
        return f"the emitted processor does not load: {loaded[0].load_error}"

    # 4. And driving it runs the emitted acceptance.sh, which is unconfigured,
    #    which must BLOCK as a fault.
    class _Ctx:
        tool_calls: list = []
        agent_params: dict = {}
        workspace_path = str(ws)
        config_root = None
        env: dict = {}
        session_id = "scaffold-probe"
        logger = logging.getLogger(__name__)

    honest = {"summary": "done", "errors": [], "warnings": []}
    first = invoke_processors(loaded, payload=honest, context=_Ctx(),
                              phase_filter="finalization")
    if not first.has_fatal:
        return ("the emitted gate ACCEPTED a completion although "
                "acceptance.sh has no checks configured — a gate that is not "
                "running must never read as a gate that passed (jaato #768 "
                "rule 5)")

    # It blocked.  WHICH CHANNEL it blocked on is not readable off the result
    # — a fault and an error both land in ``failed`` for the round-trip they
    # block — so the discrimination is behavioural, which is the stronger
    # test anyway: a fault costs no refusal and blocks exactly once, an error
    # costs one and blocks every time.  Invoking a second time separates them
    # without depending on how a message happens to be worded.
    second = invoke_processors(loaded, payload=honest, context=_Ctx(),
                               phase_filter="finalization")
    if loaded[0].refusals:
        return (f"the emitted gate spent {loaded[0].refusals} refusal(s) on "
                f"an unconfigured acceptance.sh — an environment fault the "
                f"agent cannot clear must not consume its retry budget "
                f"(jaato #768 rule 6)")
    if second.has_fatal:
        return ("the emitted gate blocked twice on an unconfigured "
                "acceptance.sh — a fault blocks for the one round-trip the "
                "agent needs to record it, and blocking on a condition no "
                "retry can clear is the loop the budget exists to prevent")
    return None


# ----------------------------------------------------- completion processor

def _probe_generated_processor(path: Path) -> Optional[str]:
    """Drive the emitted module through the framework; return a reason to fail.

    The emit-then-check for this archetype, and deliberately stronger than
    the clients' ``py_compile``: a processor that compiles can still be
    unloadable (the framework probes for ``render`` / ``validate`` by name)
    or, worse, can accept a completion it should have gated — the failure
    mode that does not announce itself.  So the check is the framework's own
    loader and invoker, run against a payload that a correct processor must
    refuse.

    Args:
        path: The emitted module.

    Returns:
        A one-line reason the generated processor is not usable, or ``None``
        when it loads and gates correctly.
    """
    from jaato_server.shared.completion_processors import invoke_processors, load_processors
    from jaato_server.shared.plugins.subagent.config import CompletionProcessor

    loaded = load_processors(
        [CompletionProcessor(script=str(path), max_refusals=2)],
        workspace_path=str(path.parent), config_root=None,
    )
    if loaded[0].load_error:
        return loaded[0].load_error
    if loaded[0].validate_fn is None:
        return "the module exposes no top-level `validate` callable"

    class _Ctx:
        workspace_path = str(path.parent)
        config_root = None
        agent_params: Dict[str, object] = {}
        # A session in which a tool call failed and the payload says nothing
        # about it: the emitted ledger check must refuse this.
        tool_calls = [{"name": "cli", "success": False,
                       "result": {"error": "boom"}, "turn_index": 0}]

    clean = invoke_processors(loaded, payload={}, context=_Ctx())
    if not clean.has_fatal:
        return ("it accepted a payload claiming a clean run over a failed "
                "tool call — a gate that does not gate")
    honest = invoke_processors(
        loaded, payload={"errors": ["the cli call failed"]}, context=_Ctx())
    if honest.has_fatal:
        return "it refused an honest payload that already reported the failure"
    return None


def _new_processor(args) -> int:
    """Emit a completion processor, then load it through the framework.

    The output-side sibling of :func:`_new_client_archetype`.  What it emits
    is documented in :mod:`_processor_template`; what makes it worth
    generating rather than describing is that the contract around the check
    — the refusal ceiling being declared rather than hand-rolled, the
    environment-fault split, the broken-gate discrimination — is the part a
    hand-written processor gets wrong, not the check itself (jaato #768).
    """
    from . import _processor_template as _tpl

    dry_run = bool(getattr(args, "dry_run", False))
    doc = _archetypes.resolve(_archetypes.PROCESSOR)
    missing = [f for f in ("workspace", "name") if not getattr(args, f, None)]
    if missing:
        print(f"new processor: missing required --{' / --'.join(missing)}")
        return 2
    name = str(args.name).strip()
    if not name.isidentifier():
        print(f"new processor: --name {name!r} must be a valid Python "
              f"identifier — it becomes the module stem, which the loader "
              f"imports by name")
        return 2

    ws = Path(args.workspace).resolve()
    target = ws / ".jaato" / "scripts" / "processors" / f"{name}.py"
    if target.exists() and not args.force:
        print(f"new processor: {target} already exists — pass --force to "
              f"overwrite")
        return 2
    if not dry_run:
        ws.mkdir(parents=True, exist_ok=True)

    plan = _Plan(ws, doc, dry_run=dry_run)
    provenance = (f"jaato-scaffold new processor --name {name} "
                  f"--workspace {ws}")
    plan.write(target, _tpl.render(name, provenance),
               "update" if target.exists() else "create")
    # The module lands under .jaato/scripts/, so the workspace needs the
    # block that keeps it committable while sessions/ and logs/ stay ignored.
    _ensure_gitignore(ws, plan, env_rule=False)

    if dry_run:
        print(f"`new processor --name {name}` would write into {ws}:\n")
        print(plan.render())
        _dry_run_footer(doc, "the load check")
        return 0

    print(f"scaffolded processor '{name}' in {ws}:")
    for w in plan.labels:
        print(f"  + {w}")

    # emit-then-check: load it the way the daemon will, and prove it gates.
    print("\nloading the generated processor through the framework …")
    reason = _probe_generated_processor(target)
    if reason:
        print(f"✘ generated processor is not usable — generator bug: {reason}")
        return 1
    print("✓ it loads, refuses a dishonest payload, and passes an honest one.")

    print("\nnext:\n  wire it into the profile it should gate —\n")
    for line in _tpl.wiring_for(name).splitlines():
        print(f"    {line}")
    print("\n  then set CHECKS_COMMAND in the module, and read:\n"
          "    jaato-scaffold explain completion")
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
        + "\n".join(_compliance_example()) + "\n"
    )


def _compliance_example() -> List[str]:
    """The EU AI Act keys, emitted COMMENTED OUT in the tier-1 base — and why.

    A workspace scaffolded the documented way used to start with no
    ``regulatory:`` declaration, no ledger and no retention, and ``validate``
    had nothing to say about it: ``disclosure_absent`` fires on a
    persona-bound profile and ``high_risk_*`` under a declared class, and a
    fresh set is neither.  The only surface that would have told the author
    the keys exist was ``explain profile``, which they have to think to run
    -- the same discoverability gap ``api_params`` and ``model_tiers`` earn
    their commented examples for.

    Commented rather than live, and it has to be: a live ``regulatory:``
    with no fields is a determination nobody made (Art. 6(4) makes it the
    provider's, and the framework never infers one), and a live
    ``record_keeping:`` changes what DELETE means for the workspace.  The
    tier-1 base is the right home because the block describes the
    APPLICATION, not the model binding a set profile carries.
    """
    return [
        "# EU AI Act (Regulation (EU) 2024/1689) — declare, never inferred.",
        "# Uncomment and fill in; `jaato-scaffold validate` then checks it and",
        "# `explain oversight <profile>` / `explain audit <profile>` say what it armed.",
        "# regulatory:",
        "#   intended_purpose: <one sentence: what this system is for>",
        "#   risk_class: minimal          # minimal | limited | high (Art. 6(4), yours to determine)",
        "#   interacts_with_persons: true # true: the AI announces itself (Art. 50(1))",
        "#   provider: {name: <your organisation>, contact: <email>}",
        "# trace:",
        "#   ledger: .jaato/logs/ledger.jsonl            # every model round trip + verdict (Art. 12)",
        "#   session_log: .jaato/logs/session_trace.jsonl",
        "# record_keeping:",
        "#   retention_days: 180                   # Art. 19(1): keep the logs; 0 = until deleted",
        "#   conversation_retention_days: 30       # the session record may go sooner",
        "#   integrity: sha256-chain               # tamper evidence; `jaato-doctor --audit-verify`",
    ]


def _temperature_example(provider: str) -> List[str]:
    """The determinism knob, emitted COMMENTED OUT — and why.

    ``temperature`` is a valid knob of every provider that declares it, so
    ``validate`` passes it: the allow-list it checks belongs to the PROVIDER,
    and which values a given MODEL accepts is not declared anywhere in this
    tree (see ``explain provider <name>``, "scope of the check").

    The template used to emit ``temperature: 0.0`` live, as a determinism
    knob.  Reasoning models — the o-series, the thinking GPT-5.x variants and
    their peers — accept only their DEFAULT temperature and answer an explicit
    value with ``400 BadRequest``, so a generated profile bound to one of them
    could not make a single request, and the validator that just approved it
    had nothing to say about why.

    Commented out, the knob is still discoverable (that is what earns it the
    space) and costs nothing when it does not apply.  Omitting an ``api_params``
    key is never the cause of a 400; setting one can be.

    The ``api_params:`` HEADER is commented out too, and has to be: a live
    header over nothing but comments parses as a YAML null, which ``validate``
    correctly reports as ``unknown_knob`` (``api_params`` is a LAYER, and a
    layer that is not a dict falls through to the top-level check).  The
    generator would have emitted a set it then failed itself on.
    """
    return [
        "    # api_params:",
        "    #   temperature: 0.0  # determinism knob — UNCOMMENT BOTH LINES ONLY IF",
        "    #                     # this MODEL takes it.  Reasoning models commonly",
        "    #                     # accept only the DEFAULT and answer an explicit",
        "    #                     # value with 400; `validate` checks the PROVIDER's",
        "    #                     # allow-list, not the model's.  See",
        f"    #                     # `explain provider {provider}`.",
    ]


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
        # REQUIRED, and NOT rescued by `inherits`: the merge takes
        # description=child.description, so omitting it here overrides the
        # base's with "" -- which is the prose the subagent tool advertises to
        # the model as what a delegate is chosen from.  Every generated set
        # profile used to ship in that state (`validate` now reports it as
        # `missing_description`).
        f"description: {agent} stage on {provider} ({model}) — "
        f"replace with what this stage is FOR.",
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
        "# must say so.  An OUTBOUND role is delivered only by a provider that",
        "# declares `output_media`; `validate` warns when yours does not.",
        "# Declaring ANY role also bounds what that tier is HANDED: it then",
        "# receives only the kinds it declared inbound, which is how",
        "# {audio: outbound} stops inbound audio reaching a tier that only",
        "# speaks.  A tier declaring nothing is unchanged — the model's own",
        "# catalog decides.  See `jaato-scaffold explain tiers`.",
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
        live = 0                       # uncommented keys under the provider
        if knobs.accepts("top_level", "api_key"):
            key_line = _api_key_line(provider, info, kind, scheme, secret_path)
            if key_line is not None:
                cfg.append(key_line)
                live += 1
        if knobs.accepts("api_params", "temperature"):
            cfg.extend(_temperature_example(provider))
        if live:
            lines.extend(cfg)
        elif len(cfg) > 2:
            # Nothing in the section is actually SET (e.g. --secrets none), so
            # a live `plugin_configs:` header would map the provider to a YAML
            # null.  Comment the headers too rather than dropping the worked
            # example: the knob it documents is exactly as hard to discover
            # under --secrets none as anywhere else.
            lines.extend([f"# {line}" for line in cfg[:2]] + cfg[2:])
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
        who = f"{d.profile}: " if getattr(d, "profile", None) else ""
        print(f"  [{d.severity}] {tier}{who}{d.code}: {d.message}{loc}")
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

    # Every profile-set puts authored assets under .jaato/, so the block that
    # keeps them committable while sessions/ and logs/ stay ignored is
    # unconditional; the .env rule only when the credential lives there.
    _ensure_gitignore(ws, plan, env_rule=kind in ("env", "none"))
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
