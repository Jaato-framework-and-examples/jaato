"""The workspace ``.gitignore`` block that keeps ``.jaato/`` committable.

``<workspace>/.jaato/`` mixes two kinds of content that a version-control
rule must treat oppositely:

* **authored assets** — profiles, personas, instructions, the two payload
  schemas, prefetch scripts and completion processors, service specs, the
  template catalog.  These ARE the workspace's configuration; a checkout
  without them does not run as the author intended.
* **runtime state** — session records, logs, memories, caches, language
  server data, tool venvs, and the ``<provider>_auth.json`` files a stored
  credential lands in.  None of it belongs in a repository, and the last
  kind must never reach one.

The split was already written down — once, as the ``audit deny`` rules in
``server/apparmor.py`` that name the "user-authored config subpaths" a
confined runner may not rewrite — and stated in prose by ``explain paths``.
It was expressed nowhere a repository could read, so a workspace either
committed its ``.jaato/`` wholesale (sessions, logs and stored keys
included) or ignored it wholesale (and lost the profiles the sessions were
run under).  ``jaato-tui/.jaato.example/README.md`` even documented the
second posture as the default, with a hand-typed re-include list as the
remedy.

This module is the one place the split is declared as data.
:data:`AUTHORED` is read by three consumers, which is what stops them
disagreeing:

* :func:`render_block` — what ``jaato-scaffold new`` writes (every archetype
  that puts a file under ``.jaato/``, and the ``gitignore`` archetype on its
  own);
* :func:`assess` — what ``jaato-scaffold validate`` judges an existing
  ``.gitignore`` by.  It judges by EFFECT, through the daemon's own
  :class:`shared.utils.gitignore.GitignoreParser`, never by spelling: a
  workspace that reached the same result with different lines is not
  reported;
* ``shared/tests/test_gitignore_authored_set_tracks_apparmor.py`` — the
  guard that every subpath the AppArmor template write-denies under
  ``.jaato/`` is in :data:`AUTHORED`, so the two declarations of
  "authored" cannot drift.

Two properties the block holds to:

* **Everything under ``.jaato/`` is ignored unless named.**  The block is
  ``.jaato/*`` followed by one ``!`` re-include per authored entry.  A
  runtime-state directory a later release adds is therefore ignored by
  default, and a credential file nobody anticipated is never one
  ``git add -A`` from a remote.  The failure direction of an incomplete
  list is an asset that has to be added by hand, not a leak.
* **It repairs a wholesale ``.jaato/`` rule without editing it.**  Git
  cannot re-include a file whose PARENT directory is excluded, so an
  existing ``.jaato/`` (or a ``.*`` dotfile rule) would make every
  ``!.jaato/<x>/`` line inert.  The block therefore opens with ``!.jaato/``,
  which un-excludes the directory itself before ``.jaato/*`` excludes its
  children — last match wins, so the author's line stays where it is and
  stops mattering.  The same shape this repository's own ``.gitignore``
  uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from jaato_server.shared.utils.gitignore import GitignoreParser


@dataclass(frozen=True)
class AuthoredEntry:
    """One committable entry directly under ``.jaato/``.

    Attributes:
        path: The entry relative to ``.jaato/``.  A directory carries its
            trailing ``/`` so the rendered rule is directory-only, the way
            git reads it.
        why: What the entry holds — rendered nowhere in the block (a
            ``.gitignore`` is not the place for a manual) but read by
            ``explain archetype gitignore`` and by the guard's failure
            message, so an entry cannot be added without saying what it is.
        confined: True when ``server/apparmor.py`` write-denies the entry
            for a confined runner — the framework's own statement that the
            content is user-authored.  The guard asserts every such subpath
            in the template appears here; an entry with ``confined=False``
            is authored config the template has no reason to name (a file
            only the daemon reads, a TUI setting).
    """

    path: str
    why: str
    confined: bool = False


#: The committable half of ``.jaato/``, in the order the block lists it.
#:
#: The first group is the AppArmor template's own "user-authored config
#: subpaths" (``audit deny {workspace_path}/.jaato/<x> wlk``).  ``prompts/``
#: is the one authored directory that template deliberately leaves
#: writable — the prompt library's ``savePrompt`` runs in the confined
#: runner — and it is authored content all the same.  The rest are the
#: configuration FILES a session or a client reads from the workspace tier,
#: none of which holds a credential: the provider key lives in ``.env`` (a
#: rule ``new profile-set`` writes beside this block) or behind a secret
#: URI, and ``webhook.json``'s documented form is ``"secret":
#: "${WEBHOOK_SECRET}"``.
AUTHORED: Tuple[AuthoredEntry, ...] = (
    AuthoredEntry("profiles/", "the profile tier: _base_<agent>.yaml and "
                  "<set>/<agent>.yaml", confined=True),
    AuthoredEntry("agents/", "personas (.md), the PERSONA layer", confined=True),
    AuthoredEntry("instructions/", "the base instruction layer every persona "
                  "sits on", confined=True),
    AuthoredEntry("completion_schemas/", "completion_payload_schema JSON — the "
                  "OUTPUT boundary", confined=True),
    AuthoredEntry("spawn_schemas/", "spawn_payload_schema JSON — the INPUT "
                  "boundary", confined=True),
    AuthoredEntry("scripts/", "prefetch scripts and completion processors "
                  "(scripts/processors/)", confined=True),
    AuthoredEntry("services/", "service_connector specs: <alias>/_service.yaml",
                  confined=True),
    AuthoredEntry("references/", "the references catalog", confined=True),
    AuthoredEntry("templates/", "the template catalog (renders become code)",
                  confined=True),
    AuthoredEntry("template_routing.yaml", "where a rendered template lands",
                  confined=True),
    AuthoredEntry("apparmor-fragments/", "workspace-tier confinement rules",
                  confined=True),
    AuthoredEntry("plans/", "predefined plans a profile's "
                  "plugin_configs.todo.initial_plan_name loads", confined=True),
    AuthoredEntry("reactors.json", "reactor rules", confined=True),
    AuthoredEntry("prompts/", "the prompt library (authored; write-allowed "
                  "under confinement so savePrompt works)"),
    AuthoredEntry("gc.json", "the GC strategy and thresholds"),
    AuthoredEntry("pricing.json", "the operator pricing table"),
    AuthoredEntry("permissions.json", "permission policy and channel"),
    AuthoredEntry("reliability-policies.json", "per-tool reliability policies "
                  "(reliability.json, the plugin's STATE, stays ignored)"),
    AuthoredEntry("webhook.json", "webhook listener + routes (secret as "
                  "${WEBHOOK_SECRET}, never literal)"),
    AuthoredEntry("sandbox.json", "sandbox_manager configuration"),
    AuthoredEntry("filesystem_query.json", "filesystem_query configuration"),
    AuthoredEntry("thinking.json", "thinking plugin configuration"),
    AuthoredEntry("formatters.json", "formatter pipeline configuration"),
    AuthoredEntry("scaffold.json", "the --secrets mode `new` recorded for "
                  "this workspace"),
    AuthoredEntry("keybindings.json", "TUI keybindings"),
    AuthoredEntry("keybindings/", "TUI keybinding presets"),
    AuthoredEntry("openers.json", "TUI per-extension openers"),
    AuthoredEntry("theme.json", "TUI theme"),
    AuthoredEntry("themes/", "TUI theme definitions"),
)

#: Runtime state ``validate`` probes to prove the rule bites.  Each is a
#: FILE a session, a plugin or a client writes under ``.jaato/``; the block
#: names none of them, because ``.jaato/*`` ignores them by construction.
#: Listed here so the finding can say which of them a hand-written
#: ``.gitignore`` leaves committable, and so the last entry — a stored
#: credential — is checked by name.
STATE_PROBES: Tuple[Tuple[str, str], ...] = (
    ("sessions/20260101_000000/session.json", "persisted session records"),
    ("logs/session.log", "per-session logs"),
    ("memories/notes.md", "the memory plugin's store"),
    ("template_extracts/index.json", "the template plugin's extracts"),
    ("cache/entry", "caches"),
    ("jdtls-data/workspace", "language-server state"),
    ("tool-venv/bin/python", "tool virtualenvs"),
    ("reliability.json", "the reliability plugin's state"),
    ("openrouter_auth.json", "a STORED PROVIDER CREDENTIAL"),
)

#: The line that un-excludes ``.jaato/`` itself (see the module docstring).
REINCLUDE_DIR = "!.jaato/"

#: The line that ignores everything beneath it.
IGNORE_CHILDREN = ".jaato/*"

#: Byte-compiled prefetch scripts and processors: ``!.jaato/scripts/``
#: re-includes the directory, and the interpreter drops ``__pycache__``
#: beside every module it imports from it.
IGNORE_PYCACHE = ".jaato/**/__pycache__/"

_COMMENT = (
    "# .jaato/ mixes AUTHORED assets (profiles, agents, schemas, processors,",
    "# ...) with RUNTIME STATE (sessions, logs, memories, caches, stored",
    "# credentials).  Only the first is committable, so everything under it",
    "# is ignored unless re-included below.  Maintained by",
    "# `jaato-scaffold new gitignore`; `jaato-scaffold validate` checks it.",
)


def rule_lines() -> List[str]:
    """The block's RULE lines, in order — no comments.

    ``!.jaato/`` first, so a wholesale ``.jaato/`` or ``.*`` rule earlier in
    the file stops excluding the directory; then ``.jaato/*``; then one
    re-include per authored entry; then the ``__pycache__`` rule, which
    must come AFTER ``!.jaato/scripts/`` re-included its parent.
    """
    return ([REINCLUDE_DIR, IGNORE_CHILDREN]
            + [f"!.jaato/{e.path}" for e in AUTHORED]
            + [IGNORE_PYCACHE])


def render_block() -> str:
    """The whole block as written into a fresh ``.gitignore``."""
    return "\n".join(_COMMENT) + "\n" + "\n".join(rule_lines()) + "\n"


def _present(text: str) -> set:
    """The rule lines *text* already carries, stripped of whitespace."""
    return {ln.strip() for ln in text.splitlines()}


def merge(existing: Optional[str]) -> Optional[str]:
    """The ``.gitignore`` text after adding the block, or ``None`` if nothing
    is missing.

    Idempotent, and never rewrites a line the author wrote:

    * no file, or neither anchor line present → the WHOLE block is appended.
      A re-include already in the file is appended again in that case
      rather than skipped: it has to come AFTER the ``.jaato/*`` being
      added, or the new rule would silence it.
    * both anchors present → only the missing lines are appended, under a
      one-line comment, so a file that was current under an older list
      gains the new entries and nothing else.

    Args:
        existing: The current file text, or ``None`` when there is no file.
    """
    if existing is None:
        return render_block()
    have = _present(existing)
    if REINCLUDE_DIR in have and IGNORE_CHILDREN in have:
        missing = [ln for ln in rule_lines() if ln not in have]
        if not missing:
            return None
        addition = ("# .jaato/ entries added by `jaato-scaffold new gitignore`\n"
                    + "\n".join(missing) + "\n")
    else:
        addition = render_block()
    prefix = existing if existing.endswith("\n") or not existing else existing + "\n"
    return prefix + ("\n" if prefix else "") + addition


@dataclass(frozen=True)
class Assessment:
    """What ``validate`` found in a workspace's ``.gitignore``.

    Attributes:
        exists: There is a ``.gitignore`` in the workspace.
        dir_excluded: A rule excludes ``.jaato/`` ITSELF.  Under git no
            later ``!`` line can re-include anything beneath an excluded
            directory, so the authored half is uncommittable whatever else
            the file says.
        hidden_authored: Authored entries (relative to ``.jaato/``) the file
            ignores.  Every entry when ``dir_excluded``.
        unignored_state: ``(probe, what)`` pairs from :data:`STATE_PROBES`
            the file does NOT ignore.  Every probe when there is no file.
    """

    exists: bool
    dir_excluded: bool
    hidden_authored: Tuple[str, ...]
    unignored_state: Tuple[Tuple[str, str], ...]

    @property
    def clean(self) -> bool:
        """Nothing to report: assets committable, state ignored."""
        return not self.hidden_authored and not self.unignored_state


def _probe_file(entry: AuthoredEntry) -> str:
    """A representative FILE under an authored entry, relative to ``.jaato/``.

    Probing a file rather than the directory sidesteps the parser's
    ``is_dir()`` check, which reads the real filesystem — the entry need not
    exist for the question "would a file here be ignored" to be answerable.
    """
    return entry.path + "x" if entry.path.endswith("/") else entry.path


def assess(workspace: Path) -> Assessment:
    """Judge *workspace*'s ``.gitignore`` by what it DOES to ``.jaato/``.

    Read-only.  Uses the daemon's own parser (defaults off, so ``.git/`` is
    the file's business and nothing is assumed) rather than a second
    matcher, and asks it the two questions the block exists to settle: is
    each authored entry committable, and is each state probe ignored.
    """
    ws = Path(workspace)
    exists = (ws / ".gitignore").is_file()
    if not exists:
        return Assessment(False, False, (), STATE_PROBES)
    parser = GitignoreParser(ws, include_defaults=False)
    jaato = ws / ".jaato"
    dir_excluded = parser.is_ignored(jaato)
    if dir_excluded:
        hidden = tuple(e.path for e in AUTHORED)
    else:
        hidden = tuple(e.path for e in AUTHORED
                       if parser.is_ignored(jaato / _probe_file(e)))
    unignored = tuple((probe, what) for probe, what in STATE_PROBES
                      if not parser.is_ignored(jaato / probe))
    return Assessment(True, dir_excluded, hidden, unignored)
