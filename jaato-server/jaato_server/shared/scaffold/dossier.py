"""The Annex IV dossier and the Article 25(4) component pack (#1121).

Two deliverables `docs/design/eu-ai-act.md` §4.6 names.

**The dossier** is the technical documentation Articles 11 and 13 ask a
high-risk system's provider to draw up, in Annex IV's shape.  Most of its
sections are facts the installed framework already holds -- plugins with
provenance, providers and models per tier, the tool surface and the
permission policy, ``runtime_limits``, the ``scrub_secret_env`` posture,
the confinement tier and whether it is ENFORCED, disclosure, record
keeping and the audit schema.  A deployer would otherwise assemble that
by hand from ``explain`` pages and source.

**The component pack** is the other half, and it binds NOW rather than in
2027.  Both distributions are ``BUSL-1.1``, so the free-and-open-source
carve-out of Article 25(4) does not apply and its first sentence does: a
third party supplying a component to a high-risk system must provide, *by
written agreement*, "the necessary information, capabilities, technical
access and other assistance" the provider needs.  That agreement needs a
versioned document to point at.

Four properties, each attached to a way a generated legal document goes
wrong:

* **Computed, never asserted.**  Every fact in a computed section comes
  from the same helper an ``explain`` page reads, so the dossier cannot
  disagree with ``explain``.  A dossier that restated the framework would
  be a second source of truth about it, and the one that goes stale is
  always the document.
* **Side-effect free**, like ``validate``: files are located, never
  rendered -- rendering a persona runs its ``{{!py:...}}`` prefetch.
* **A ``TODO`` section is never silently omitted.**  An absent section
  reads as "nothing to declare", which is the wrong default for a legal
  document.  Every Annex IV heading is present; the ones the framework
  cannot fill say so, in the Article's own words.
* **Every computed section is dated and stamped.**  A fact about a tree
  is a fact about a COMMIT, and a dossier read six months later must say
  which one.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: The Annex IV headings, in the Regulation's own order and wording
#: (points 1-9 of Annex IV, Regulation (EU) 2024/1689).  A tuple, because
#: the guard checks that every one appears in the output: a heading
#: dropped because the framework had nothing to say for it is exactly the
#: silent omission this must not do.
ANNEX_IV_SECTIONS: Tuple[Tuple[str, str], ...] = (
    ("1", "General description of the AI system"),
    ("2", "Detailed description of the elements of the system and of the "
          "process for its development"),
    ("3", "Detailed information about the monitoring, functioning and "
          "control of the system"),
    ("4", "Description of the appropriateness of the performance metrics"),
    ("5", "Risk management system (Article 9)"),
    ("6", "Changes made through the lifecycle"),
    ("7", "Harmonised standards applied"),
    ("8", "EU declaration of conformity"),
    ("9", "Post-market monitoring plan (Article 72)"),
)

#: What the framework can fill, per Annex IV section.  A section absent
#: from this map is all-TODO -- and says why, rather than appearing
#: empty.
_COMPUTED_BY_SECTION: Dict[str, str] = {
    "1": "system description, versions, model bindings, disclosure",
    "2": "plugins with provenance, tool surface, permission policy",
    "3": "oversight measures, logging arrangements, confinement",
    "4": "eval results, when a results file is named",
    "9": "the incident register's kinds and where it reads from",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _framework_checkout() -> Optional[Path]:
    """The framework's own git checkout, or ``None`` for an installed wheel.

    The discriminator is ``check_checkout_skew``'s (#823): a CHECKOUT has
    the distribution's ``pyproject.toml`` beside the package directory,
    an install in ``site-packages`` does not.  Asked of the package this
    module lives in, so the answer is about the framework rather than
    about whatever directory the operator happened to run from.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / ".git").exists():
            return parent
        if (parent.parent / ".git").exists() and (parent / "pyproject.toml").is_file():
            return parent.parent
    return None


@lru_cache(maxsize=1)
def _commit() -> str:
    """What version of the framework the computed sections describe.

    A fact about a tree is a fact about a COMMIT -- and about the RIGHT
    tree.  Run in the process CWD, ``git rev-parse`` answered with the
    commit of whatever repository the operator was standing in: an
    operator generating a dossier from their own project, with jaato
    installed from a wheel, got THEIR commit stamped as the framework's,
    in the one field that exists to make the document auditable six
    months later.

    So the checkout is resolved from this module's own location, and an
    installed wheel -- which has no commit and must not borrow one --
    reports its distribution versions instead.  That is the fact a wheel
    install can actually state.

    Memoised: ``_stamp()`` is called once per section, and a subprocess
    per section of a nine-section document is nine subprocesses to answer
    one question that cannot change during a run.
    """
    checkout = _framework_checkout()
    if checkout is not None:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(checkout),
                capture_output=True, text=True, timeout=5, check=False)
            if out.stdout.strip():
                return out.stdout.strip()
        except Exception:  # noqa: BLE001 -- a diagnostic must not raise
            pass
    try:
        from jaato_sdk.release_channels import installed_distributions
        installed = installed_distributions()
        if installed:
            return "installed " + ", ".join(
                f"{name} {version}"
                for name, version in sorted(installed.items()))
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _stamp() -> str:
    """The disclaimer every computed section carries."""
    return (f"> *Computed from the installed framework at commit "
            f"`{_commit()}` on {_now()}. Regenerate after any release or "
            f"profile change; a stale computed section is worse than an "
            f"absent one, because it reads as current.*")


def _todo(what: str) -> str:
    """A section the framework cannot fill, said out loud."""
    return (f"> **TODO — the framework cannot supply this.** {what}\n>\n"
            f"> Left in rather than omitted: an absent section in a legal "
            f"document reads as *nothing to declare*.")


# --------------------------------------------------------------- sections

def _distributions() -> List[str]:
    """Every installed ``jaato-*`` distribution and its version (#966)."""
    try:
        from jaato_sdk.release_channels import installed_distributions
        return [f"- `{name}` {version}"
                for name, version in sorted(installed_distributions().items())]
    except Exception:  # noqa: BLE001
        return ["- (could not read installed distribution metadata)"]


def _profile_facts(name: str, workspace: str,
                   profile_set: Optional[str] = None) -> Dict[str, Any]:
    """What a named profile declares, read through the framework's own resolver.

    Goes through ``explain._resolve_workspace_profile`` rather than calling
    ``discover_profiles`` directly, so the dossier resolves EXACTLY the
    profile ``explain oversight`` and ``explain audit`` resolve -- including
    the workspace's selected profile set.  A dossier whose §3 quoted a
    different profile from its §1 would be the second-source-of-truth
    failure this module exists to avoid, one page in.

    Side-effect free: profiles are located and parsed; nothing here renders
    a persona.
    """
    from . import explain

    profile, ws = explain._resolve_workspace_profile(name, workspace,
                                                     profile_set)
    if profile is None:
        raise KeyError(name)
    return {"profile": profile, "workspace": ws, "profile_set": profile_set}


def _oversight_section(profile_name: str, workspace: str,
                       profile_set: Optional[str] = None) -> str:
    """Annex IV §3, quoting ``explain oversight`` rather than restating it."""
    from . import explain

    try:
        _data, text = explain.oversight_profile(profile_name, workspace,
                                                profile_set)
    except Exception as exc:  # noqa: BLE001
        return f"(oversight could not be computed: {exc})"
    return "```\n" + text.strip() + "\n```"


def _audit_section(profile_name: str, workspace: str,
                   profile_set: Optional[str] = None) -> str:
    """Annex IV §3's logging arrangements -- Art. 13(3)(f)."""
    from . import explain

    try:
        _data, text = explain.audit_profile(profile_name, workspace,
                                            profile_set)
    except Exception as exc:  # noqa: BLE001
        return f"(the audit record could not be computed: {exc})"
    return "```\n" + text.strip() + "\n```"


def _regulatory_block(profile: Any) -> str:
    reg = getattr(profile, "regulatory", None)
    if reg is None:
        return _todo(
            "This profile declares no `regulatory:` block, so its intended "
            "purpose and risk class are UNDECLARED — not `minimal`. "
            "Article 6(4) makes the risk determination the provider's own, "
            "and a framework that printed a class for it would be making "
            "that determination on your behalf.")
    lines = ["| Field | Declared |", "|---|---|"]
    for key, value in sorted(reg.to_dict().items()):
        lines.append(f"| `{key}` | {value} |")
    if reg.risk_class is None:
        lines.append("")
        lines.append(_todo("`risk_class` is undeclared (Art. 6(4))."))
    return "\n".join(lines)


def _bindings(profile: Any) -> str:
    """Every (provider, model) pair the profile can run on."""
    rows = ["| Tier | Provider | Model |", "|---|---|---|"]
    provider = getattr(profile, "provider", None) or "(runtime default)"
    model = getattr(profile, "model", None) or "(runtime default)"
    rows.append(f"| (initial) | {provider} | {model} |")
    tiers = getattr(profile, "model_tiers", None) or {}
    for tier_name in sorted(tiers):
        entry = tiers[tier_name]
        rows.append(f"| {tier_name} | "
                    f"{getattr(entry, 'provider', None) or '(inherits)'} | "
                    f"{getattr(entry, 'model', None) or '(inherits)'} |")
    return "\n".join(rows)


def _plugins_with_provenance(profile: Any) -> str:
    """Annex IV §2: the components, and who supplied each (#684)."""
    from . import introspect

    declared = list(getattr(profile, "plugins", None) or [])
    if not declared:
        return ("This profile declares no plugins, so the session runs the "
                "minimal framework set (permission, reliability, lifecycle).")
    try:
        known = introspect.plugins()
    except Exception:  # noqa: BLE001
        known = {}
    rows = ["| Plugin | Supplied by | Tools |", "|---|---|---|"]
    for name in declared:
        info = known.get(name)
        if info is None:
            rows.append(f"| `{name}` | **not installed here** | — |")
            continue
        rows.append(f"| `{name}` | {info.source or 'built-in'} | "
                    f"{len(info.tools)} |")
    return "\n".join(rows)


def _incident_kinds() -> str:
    """Annex IV §9: what post-market monitoring records."""
    from jaato_sdk.incidents import INCIDENT_KINDS, KIND_DESCRIPTIONS

    rows = ["| Kind | What it means |", "|---|---|"]
    for kind in INCIDENT_KINDS:
        rows.append(f"| `{kind}` | {KIND_DESCRIPTIONS.get(kind, '')} |")
    return "\n".join(rows)


# ---------------------------------------------------------- the dossier

def render_dossier(
    profile_name: str,
    workspace: str,
    eval_results: Optional[str] = None,
    profile_set: Optional[str] = None,
) -> str:
    """The Annex IV skeleton for one profile.

    Raises:
        KeyError: when ``profile_name`` resolves to no profile.  Loud,
            because a dossier generated for the wrong system is worse
            than none.
    """
    facts = _profile_facts(profile_name, workspace, profile_set)
    profile = facts["profile"]
    stamp = _stamp()

    parts: List[str] = [
        f"# Technical documentation — {profile_name}",
        "",
        "Annex IV of Regulation (EU) 2024/1689, for the AI system this "
        f"profile defines. Generated by `jaato-scaffold new dossier "
        f"--profile {profile_name}`.",
        "",
        "> **This is a skeleton, not a compliance document.** The sections "
        "marked *Computed* are facts about the installed framework, read "
        "from the same helpers `jaato-scaffold explain` reads. Everything "
        "else is the provider's to write, and the framework leaves it "
        "marked rather than blank.",
        "",
        stamp,
        "",
    ]

    for number, title in ANNEX_IV_SECTIONS:
        parts.append(section_heading(number, title))
        parts.append("")
        parts.extend(_section_body(number, profile, profile_name, workspace,
                                   eval_results, profile_set))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _section_body(
    number: str,
    profile: Any,
    profile_name: str,
    workspace: str,
    eval_results: Optional[str],
    profile_set: Optional[str] = None,
) -> List[str]:
    """One Annex IV section: what the framework knows, then what it does not."""
    if number == "1":
        return [
            "### Declared under `regulatory:`", "",
            _regulatory_block(profile), "",
            "### Model bindings", "",
            _bindings(profile), "",
            "### Framework versions", "",
            *_distributions(), "",
            _todo("Annex IV(1) also asks for the system's intended purpose "
                  "in prose, the persons or groups it is intended to be "
                  "used on, the hardware it runs on, and how it is placed "
                  "on the market."),
        ]
    if number == "2":
        return [
            "### Components and who supplied each", "",
            _plugins_with_provenance(profile), "",
            "Both jaato distributions are licensed `BUSL-1.1`, which is "
            "source-available and **not** free and open-source — so "
            "Article 25(4)'s carve-out does not apply and its first "
            "sentence does. The component information pack is "
            "`jaato-scaffold new dossier --component`.", "",
            _todo("Annex IV(2) also asks for the development process, "
                  "design choices and their rationale, the training "
                  "methodology and data where the provider trains a model "
                  "(jaato trains none), and the human-oversight measures' "
                  "design rationale."),
        ]
    if number == "3":
        return [
            "### Human oversight, as armed by this profile", "",
            _oversight_section(profile_name, workspace, profile_set), "",
            "### Logging arrangements (Art. 13(3)(f))", "",
            _audit_section(profile_name, workspace, profile_set), "",
            _todo("Annex IV(3) also asks for the system's expected "
                  "lifetime, foreseeable unintended outcomes, and the "
                  "measures for the deployer's own monitoring."),
        ]
    if number == "4":
        return _accuracy_body(eval_results)
    if number == "9":
        return [
            "### What the framework records automatically", "",
            _incident_kinds(), "",
            "`jaato-doctor --incidents <trace> --since 15d` lists them with "
            "the Article 73 reporting windows beside each. It does **not** "
            "classify: whether an entry is a serious incident under "
            "Art. 3(49) is a determination about consequences the "
            "framework cannot see.", "",
            _todo("Article 72 asks for a post-market monitoring PLAN — who "
                  "reads the register, how often, and what they do about "
                  "each kind. That is organisational and yours."),
        ]
    return [_todo(_SECTION_TODOS[number])]


#: Why the framework cannot fill the sections it cannot fill.  Named per
#: section rather than one generic sentence, so a reader knows whether
#: they are looking at a gap or at something genuinely outside a
#: framework's knowledge.
_SECTION_TODOS: Dict[str, str] = {
    "5": "Article 9's risk management system is a continuous, iterative "
         "process over the system's whole lifecycle — identifying risks, "
         "estimating them, evaluating post-market data, adopting measures. "
         "No property of a framework determines any of it.",
    "6": "The change history of THIS system. The framework can date its "
         "own versions (§1) and knows nothing about when you changed the "
         "profile, the persona or the intended purpose.",
    "7": "Which harmonised standards you applied, in full or in part, and "
         "where you departed from them. A citation, not a measurement.",
    "8": "The EU declaration of conformity (Article 47) is a document the "
         "provider signs. Nothing generates it.",
}


def _accuracy_body(eval_results: Optional[str]) -> List[str]:
    """Annex IV §4 -- the accuracy section (#1124 fills it)."""
    if not eval_results:
        return [
            _todo("Article 15(3) asks that accuracy levels and the relevant "
                  "accuracy metrics be declared in the instructions for "
                  "use, and 9(8) asks for testing against prior-defined "
                  "metrics. Pass `--eval-results <file>` to render a "
                  "`jaato-eval` run's metrics here."),
        ]
    from . import eval_results as _eval

    try:
        return _eval.render_section(eval_results)
    except _eval.EvalResultsError as exc:
        return [f"> **Refused: {exc}**", "",
                _todo("The named results file could not be rendered, so "
                      "this section is empty rather than wrong.")]


# ---------------------------------------------------- the component pack

#: What the framework GUARANTEES -- each a contract the tree already
#: holds, named with the thing that enforces it.  A guarantee with no
#: enforcer is a claim, and a claim in a 25(4) pack is the thing that
#: gets relied on.
GUARANTEES: Tuple[Tuple[str, str], ...] = (
    ("Provider capability declarations",
     "`PROVIDER_CAPABILITIES` per provider plugin. A capability a "
     "provider does not declare is not used: `jaato-scaffold validate` "
     "flags an outbound modality role on a provider that does not "
     "declare `output_media`, rather than letting it fail at runtime."),
    ("History pairing across every provider",
     "`shared/history_invariant.py` repairs the per-request copy at the "
     "one seam every `provider.complete()` call reads, so a cancelled "
     "batch, a GC pass, a rewind or a wire that streamed a tool call "
     "with no id cannot produce a request the upstream rejects."),
    ("One permission verdict, one exit",
     "`check_permission` is a single-exit wrapper: every terminal "
     "decision is traced with `asked=`, and an AST guard fails the "
     "build if a branch returns a verdict without recording one."),
    ("Per-thread confinement verification",
     "`verify_thread_confinement` walks `/proc/self/task/*/attr/current` "
     "and REFUSES the bootstrap on divergence. It fails closed, and "
     "only positive evidence counts — a label that could not be read "
     "proves nothing and does not refuse."),
    ("Enforcement mode, not merely attachment",
     "`shared/apparmor_label.py` is the one parser; `sandbox_mode` "
     "distinguishes `apparmor` from `apparmor-complain`, so a profile "
     "attached in complain mode is never recorded as a boundary."),
    ("Secret scrubbing on every model-driven subprocess",
     "ON by default since #863. `cli`, `interactive_shell` and `mcp` "
     "strip the framework secret set from the inherited environment; "
     "opting out is announced at WARNING."),
    ("The guards are live",
     "`test_every_guard_detects_its_own_reversion.py` puts each "
     "contract guard's defect back and fails the build if the guard "
     "still passes. It is the evidence that the rows above are "
     "enforced rather than described."),
)

#: What the framework does NOT guarantee.  Longer than the list above on
#: purpose: a component pack whose limitations section is short is a
#: pack nobody can rely on, because the reader cannot tell what was
#: considered from what was forgotten.
NON_GUARANTEES: Tuple[Tuple[str, str], ...] = (
    ("The prompt-injection boundary is defence in depth",
     "`TRAIT_UNTRUSTED_CONTENT` wraps third-party content so injected "
     "instructions read as data. It is not a proof: a sufficiently "
     "persuasive payload inside the boundary can still steer a model."),
    ("A notebook cell's in-process reach",
     "On the audit tier the containment is a PEP 578 hook in the same "
     "interpreter. It closes the accidental and opportunistic case; a "
     "cell that sets out to escape can attack the hook. AppArmor is "
     "the kernel answer, and the layering says so."),
    ("`finalize` and `escalate` are advice",
     "Of `budget_control`'s terminal actions only `abort` stops a run. "
     "A looping model can decline the other two, and has."),
    ("Text output is not marked",
     "Art. 50(2) marking covers model MEDIA. `AgentOutputEvent.source` "
     "attributes text at the event layer and stops at the client; no "
     "text watermark ships until the Art. 50(7) code of practice names "
     "one."),
    ("A chain proves no-edit-in-place, not authorship",
     "`record_keeping.integrity: sha256-chain` detects an in-place "
     "edit. A writer holding the file can re-chain from any point; "
     "signing is not built."),
    ("The session runs as the daemon's uid",
     "Peer entitlement (IPC) and connect tickets (WS) segregate "
     "IDENTITY and the paths a caller may name. They do not separate "
     "the OS principal."),
    ("`limits` without an `abort` rung enforce nothing",
     "`budget_control.limits` are observed; the degrade ladder is the "
     "only consumer of the usage fraction. `validate` warns."),
    ("Risk classification is the provider's",
     "The framework never infers `risk_class`. An absent declaration "
     "is UNDECLARED, not `minimal`."),
)

#: The versioned surfaces a written agreement can cite.  Values are read
#: at render time, not written down: a pack quoting a version it is not
#: running is the failure mode of every generated document.
def _versioned_surfaces() -> List[str]:
    rows = ["| Surface | Value | What it versions |", "|---|---|---|"]
    try:
        from jaato_sdk.events import PROTOCOL_VERSION
        rows.append(f"| Event protocol | `{PROTOCOL_VERSION}` | the "
                    f"client/daemon wire |")
    except Exception:  # noqa: BLE001
        pass
    try:
        from jaato_sdk.audit import AUDIT_SCHEMA_VERSION
        rows.append(f"| Audit schema | `{AUDIT_SCHEMA_VERSION}` | what is "
                    f"guaranteed to be recorded |")
    except Exception:  # noqa: BLE001
        pass
    try:
        from jaato_server.shared.plugins.subagent.config import PROFILE_FILE_KEYS
        rows.append(f"| Profile keys | {len(PROFILE_FILE_KEYS)} accepted | "
                    f"what a profile file may declare |")
    except Exception:  # noqa: BLE001
        pass
    return rows


def render_component_pack() -> str:
    """The Article 25(4) information pack for jaato itself."""
    parts: List[str] = [
        "# jaato as a component of a high-risk AI system",
        "",
        "Article 25(4) of Regulation (EU) 2024/1689, information pack.",
        "",
        "Both jaato distributions are licensed **`BUSL-1.1`**, which is "
        "source-available and not free and open-source. The carve-out in "
        "Article 25(4) — for third parties making components available "
        "under a free and open-source licence — therefore does not apply, "
        "and the Article's first sentence does: a supplier of tools, "
        "services, components or processes used in a high-risk AI system "
        "must specify, **by written agreement**, the information, "
        "capabilities, technical access and other assistance the provider "
        "needs to comply.",
        "",
        "This document is what such an agreement can point at. It is "
        "generated from the installed tree, so it describes the version "
        "you are running and not a release note.",
        "",
        _stamp(),
        "",
        "## Installed",
        "",
        *_distributions(),
        "",
        "## What the framework guarantees",
        "",
        "Each row names the thing that ENFORCES it. A guarantee with no "
        "enforcer is a claim, and a claim in this document is what gets "
        "relied on.",
        "",
    ]
    for title, detail in GUARANTEES:
        parts += [f"**{title}.** {detail}", ""]

    parts += [
        "## What the framework does NOT guarantee",
        "",
        "Longer than the list above, on purpose: a limitations section a "
        "reader can finish quickly is one that leaves them unable to tell "
        "what was considered from what was forgotten.",
        "",
    ]
    for title, detail in NON_GUARANTEES:
        parts += [f"**{title}.** {detail}", ""]

    parts += [
        "## Versioned surfaces an agreement can cite",
        "",
        *_versioned_surfaces(),
        "",
        "## What the provider still owes",
        "",
        "The framework is a component. The AI system is your profile, "
        "persona, tool surface and model binding, and every obligation in "
        "Chapter III attaches to you as its provider — including the risk "
        "determination (Art. 6(4)), which nothing here infers.",
        "",
        "`jaato-scaffold new dossier --profile <name>` generates the "
        "Annex IV skeleton for one of your systems.",
    ]
    return "\n".join(parts).rstrip() + "\n"


# ------------------------------------------------------- the read-back

def section_heading(number: str, title: str) -> str:
    """The one spelling of an Annex IV heading.

    Read by the renderer and by :func:`missing_sections`, so the emit and
    the check cannot disagree about what a heading looks like -- a check
    that matches a different string from the one the generator writes
    reports every document as broken, or none.
    """
    return f"## {number}. {title}"


def missing_sections(text: str) -> Tuple[str, ...]:
    """Annex IV headings absent from a rendered dossier.

    The emit-then-check ``new dossier`` runs on its own output.  An empty
    tuple is the only acceptable answer: a heading dropped because the
    framework had nothing to say for it is the silent omission this module
    exists not to commit, and it is checked rather than trusted because the
    failure is invisible in the document itself -- an absent section reads
    as *nothing to declare*.
    """
    return tuple(f"{number}. {title}"
                 for number, title in ANNEX_IV_SECTIONS
                 if section_heading(number, title) not in text)
