"""``jaato-scaffold integration`` — wire jaato into the tool you work in.

An integration is not a jaato asset; it is jaato's side of a contract with
ANOTHER tool.  Today there is one — ``claude-code``, which installs the
``jaato-sdk`` skill where Claude Code looks for skills — and the shape
generalises to whatever comes next (an editor plugin, shell completion, a CI
action), because each is defined by the tool it integrates WITH.

That is also why the target path lives in each integration's
``integration.json`` rather than in this module: a Cursor integration would not
write into ``.claude/skills``, and hardcoding one tool's convention into
generic code is the mistake this layout exists to avoid.

WHY THIS VERB EXISTS.  The skill used to be a file in a git repo, so the only
way to get it was to copy it by hand — and hand-copies drift.  A survey of one
org found the same skill living in four repos at four different lengths, and
two large skills whose user-global installs were 2.5 months behind the repo
originals they were copied from.  Nothing detected any of it.

The cure is structural rather than procedural: the skill ships as package data
of the distribution it documents, so an installed copy cannot describe a
different framework than the one running, and every copy this verb writes
carries a stamp naming the version it came from.  ``jaato-doctor`` compares
that stamp against the installed framework and says so when they part company.

Scope is explicit because it decides who sees it:

    --user       ~/<target>          every repo on this machine (default)
    --workspace  DIR/<target>        that project only

where ``<target>`` is the path the integration declares.

``--user`` is the default because a skill about the framework is not a property
of any one project, and per-project copies are the pattern that produced the
drift above.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STAMP = ".jaato-integration"
"""Filename of the provenance stamp written beside an installed integration.

Read by ``jaato-doctor``.  Its presence is what makes a stale copy detectable
instead of merely wrong.
"""


def payload_digest(root: Path) -> str:
    """A content digest of every file under ``root``, excluding the stamp.

    Exists because "same version, files differ" cannot say WHICH side moved,
    and the two causes want opposite advice: a payload edited locally must not
    be overwritten, while a payload that changed upstream at the same version
    should be.  Recording this at apply time makes the question answerable
    instead of guessed.

    Walks recursively.  The predecessor compared with ``filecmp.dircmp``, which
    only inspects the TOP level — so a change confined to ``references/`` was
    invisible, and an installed copy could drift arbitrarily far in the one
    place most of the prose lives.
    """
    h = hashlib.sha256()
    if not root.is_dir():
        return ""
    for f in sorted(p for p in root.rglob("*") if p.is_file() and p.name != STAMP):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def framework_version() -> str:
    """The version of the distribution this asset ships with."""
    try:
        from importlib.metadata import version
        return version("jaato-server")
    except Exception:      # noqa: BLE001 — a source checkout may not be installed
        return "unknown"


def _source_root() -> Path:
    return Path(__file__).resolve().parent / "integrations"


def manifest(name: str) -> Dict[str, Any]:
    """What an integration declares about itself, or ``{}`` if it has none."""
    f = _source_root() / name / "integration.json"
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def available() -> List[str]:
    """Integrations this build ships."""
    root = _source_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "integration.json").is_file())


def payload_dir(name: str) -> Path:
    """Payload for ``name``, optionally shared with another integration."""
    source = manifest(name).get("payload_from", name)
    return _source_root() / source / "payload"


class IntegrationManifestError(ValueError):
    """An integration's ``integration.json`` cannot be acted on.

    Raised rather than defaulted.  The previous code resolved a missing
    ``target`` to ``.jaato-integration-<name>`` and said the caller would
    report it; no caller did, and the string appeared exactly once in the
    tree — at the site that built it.  So a manifest that forgot the key
    installed a real payload to a plausible-looking wrong path, silently.
    """


def resolve_targets(name: str) -> Tuple[str, str]:
    """``(user_target, workspace_target)`` as declared by ``name``.

    ``target`` is a string when a harness uses one relative path at both
    scopes, or ``{"user": ..., "workspace": ...}`` when they differ::

        "target": ".claude/skills/jaato-sdk"

        "target": {"user":      ".pi/agent/skills/jaato-sdk",
                   "workspace": ".pi/skills/jaato-sdk"}

    ONE key with two self-describing forms, rather than a ``target`` key
    plus ``user_target``/``workspace_target`` companions: with three keys a
    reader has to know which are alternatives and which are siblings, every
    row of ``listing()`` carries two nulls, and an author who declares one
    scope and forgets the other gets a silently wrong path for the missing
    one instead of an error.

    Raises `IntegrationManifestError` for anything it cannot act on — a
    missing ``target``, an object missing a scope, a non-string path, or an
    absolute one (targets are joined onto `$HOME` or the workspace, so an
    absolute path would escape the scope it was asked for).
    """
    target = manifest(name).get("target")
    if target is None:
        raise IntegrationManifestError(
            f"integration '{name}' declares no 'target' in integration.json")

    if isinstance(target, str):
        pair = (target, target)
    elif isinstance(target, dict):
        missing = [k for k in ("user", "workspace") if not target.get(k)]
        if missing:
            raise IntegrationManifestError(
                f"integration '{name}' declares target.{' and target.'.join(missing)}"
                f" nowhere; an object target must name both scopes")
        pair = (target["user"], target["workspace"])
    else:
        raise IntegrationManifestError(
            f"integration '{name}' declares a target of type "
            f"{type(target).__name__}; expected a string or "
            f"{{'user': ..., 'workspace': ...}}")

    for scope, value in zip(("user", "workspace"), pair):
        if not isinstance(value, str):
            raise IntegrationManifestError(
                f"integration '{name}' declares a non-string {scope} target")
        if Path(value).is_absolute():
            raise IntegrationManifestError(
                f"integration '{name}' declares an absolute {scope} target "
                f"({value!r}); targets are relative to $HOME or the workspace")
    return pair


def target_dir(name: str, *, user: bool, workspace: Optional[str]) -> Path:
    """Where ``name`` installs, per its own manifest.

    Relative to `$HOME` for user scope, to the workspace otherwise.  Raises
    `IntegrationManifestError` when the manifest cannot say — see
    `resolve_targets`.
    """
    base = Path.home() if user else Path(workspace or ".").resolve()
    user_target, workspace_target = resolve_targets(name)
    return base / (user_target if user else workspace_target)


def harness_present(name: str) -> Optional[bool]:
    """Is the tool ``name`` integrates WITH actually on this machine?

    ``None`` means the manifest declares no ``detect`` and we therefore do
    not know — which is NOT ``False``.  Only an author who is certain
    declares it, so a caller may act on ``False`` and must not act on
    ``None``; an integration that says nothing behaves exactly as it did
    before this key existed.

    ``detect`` is a fact about the OTHER tool, so only its author can write
    it::

        "detect": {"commands": ["claude"],
                   "paths": ["~/.claude/projects"],
                   "why": "Claude Code writes ~/.claude/projects on first
                           run; jaato never creates it."}

    The trap, and why ``why`` is part of the contract: a path that is an
    ANCESTOR of this integration's own target is created by our installer,
    so it answers "the harness is here" on a machine that has never had it.
    Measured: on a host with neither harness, installing only our skills
    brings ``~/.claude``, ``~/.claude/skills``, ``~/.pi`` and ``~/.pi/agent``
    into existence.  `manifest_detect_problems` refuses that shape; nothing
    can check whether a path is *truly* harness-owned, which is what the
    author is asserting and a reviewer reads ``why`` to judge.

    ``commands`` cannot be contaminated that way — we never put a binary on
    ``PATH`` — but it is not absolute either: a harness installed outside
    this process's ``PATH`` reads as absent.  That direction loses a nudge
    rather than inventing noise, which is the safer way to be wrong.
    """
    detect = manifest(name).get("detect")
    if not isinstance(detect, dict):
        return None
    paths = detect.get("paths") or []
    commands = detect.get("commands") or []
    if not paths and not commands:
        return None
    if any(Path(p).expanduser().exists() for p in paths if isinstance(p, str)):
        return True
    return any(shutil.which(c) for c in commands if isinstance(c, str))


def manifest_detect_problems(name: str) -> List[str]:
    """Why ``name``'s ``detect`` block cannot mean what it says, if so.

    The one error in a `detect` that is mechanically checkable: a path our
    own installer creates cannot be evidence of the harness.  Everything
    else about a detect signal rests on the author's knowledge of their own
    tool and is reviewed by reading ``why``, not by running code.
    """
    detect = manifest(name).get("detect")
    if not isinstance(detect, dict):
        return []
    try:
        targets = resolve_targets(name)
    except IntegrationManifestError:
        return []      # the target itself is broken; that is reported already

    problems = []
    owned = {Path("~", t).expanduser().resolve() for t in targets}
    for raw in detect.get("paths") or []:
        if not isinstance(raw, str):
            problems.append(f"detect.paths entry is not a string: {raw!r}")
            continue
        p = Path(raw).expanduser().resolve()
        for target in owned:
            if p == target or p in target.parents:
                problems.append(
                    f"detect.paths entry {raw!r} is this integration's own "
                    f"install location or an ancestor of it, so jaato creates "
                    f"it — it cannot be evidence that the harness is present")
                break
    return problems


def read_stamp(installed: Path) -> Dict[str, str]:
    """The provenance of an installed copy, or ``{}`` when it has none.

    A copy with no stamp predates this verb — it was hand-copied — which is
    worth reporting rather than treating as absent.
    """
    f = installed / STAMP
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:      # noqa: BLE001 — a corrupt stamp is a missing stamp
        return {}


def compare(name: str, installed: Path) -> Tuple[str, str]:
    """``(state, detail)`` for an installed copy against what this build ships.

    States:

        absent      never applied
        current     stamp and content both match
        stale       a different framework version
        outdated    same version, but the PAYLOAD moved upstream — re-apply,
                    nothing of yours is lost
        edited      same version, but the INSTALLED copy was changed — do not
                    overwrite it; upstream the change first
        diverged    both moved; re-applying discards the local side
        unstamped   hand-applied, provenance unknown

    `outdated` and `edited` were one state (`modified`) until this told an
    operator "local edits will be lost, upstream them first" about a payload
    that had simply changed upstream at the same version — confidently wrong
    advice, and the reverse of the truth.  The digest recorded at apply time
    is what separates them: it says what the payload looked like when it was
    applied, so either side can be compared against that fixed point.
    """
    src = payload_dir(name)
    if not installed.is_dir():
        return "absent", str(installed)
    stamp = read_stamp(installed)
    if not stamp:
        return "unstamped", "installed by hand — provenance unknown"
    got, want = stamp.get("version", "?"), framework_version()
    if got != want:
        return "stale", f"installed from {got}, framework is {want}"
    if not src.is_dir():
        return "current", got

    applied = stamp.get("digest")
    here, there = payload_digest(installed), payload_digest(src)
    if not applied:
        # A stamp from before digests existed: the version matches and there is
        # no fixed point to compare against, so say exactly that rather than
        # pick one of the two answers and sound sure.
        if here == there:
            return "current", got
        return "diverged", (f"content differs from {got} and the stamp predates "
                            f"content tracking, so which side moved is unknown "
                            f"— re-apply to resync, losing any local change")
    if here == applied and there == applied:
        return "current", got
    if here == applied:
        return "outdated", (f"the payload changed upstream at {got}; "
                            f"re-applying is safe, nothing local is lost")
    if there == applied:
        return "edited", (f"the installed copy was changed since it was applied "
                          f"from {got}; upstream it before re-applying")
    return "diverged", (f"both the installed copy and the payload changed since "
                        f"{got}; re-applying discards the local side")


def install(name: str, dest: Path, *, force: bool = False,
            dry_run: bool = False) -> Tuple[bool, List[str]]:
    """Copy ``name`` to ``dest``; return ``(changed, lines)``.

    Refuses to overwrite an existing copy without ``--force``, and says which
    state it found — a local edit and a stale version want different answers
    from the operator, so the message names which one it is.
    """
    src = payload_dir(name)
    if not src.is_dir():
        if name not in available():
            return False, [f"unknown integration '{name}' — this build ships: "
                           f"{', '.join(available()) or '(none)'}"]
        source = manifest(name).get("payload_from", name)
        return False, [f"integration '{name}' payload '{source}' is missing "
                       "from this build"]

    state, detail = compare(name, dest)
    if state != "absent" and not force:
        return False, [f"{dest} already exists ({state}: {detail})",
                       "pass --force to overwrite, or --dry-run to see what would change"]

    files = sorted(p.relative_to(src).as_posix()
                   for p in src.rglob("*") if p.is_file() and p.name != STAMP)
    if dry_run:
        return False, [f"would write {dest}/"] + [f"  + {f}" for f in files]

    if dest.exists():
        try:
            shutil.rmtree(dest)
        except OSError as exc:
            return False, [f"could not replace {dest}: {exc}"]
    shutil.copytree(src, dest)
    (dest / STAMP).write_text(json.dumps(
        {"integration": name, "tool": manifest(name).get("tool", name),
         "version": framework_version(), "source": str(src),
         "digest": payload_digest(dest)},
        indent=2) + "\n", encoding="utf-8")
    return True, [f"integrated {manifest(name).get('tool', name)}: {dest}  (from jaato-server {framework_version()})"] \
        + [f"  + {f}" for f in files]


def listing() -> tuple[dict[str, Any], str]:
    """What this build can integrate with, and where each one currently stands.

    Backs both the bare ``integration`` verb and ``explain integrations`` — one
    source, so the two can never disagree about what exists.
    """
    rows = []
    for name in available():
        m = manifest(name)
        row = {"name": name, "tool": m.get("tool", name),
               "summary": m.get("summary", ""), "why": m.get("why", "")}
        try:
            user_target, workspace_target = resolve_targets(name)
        except IntegrationManifestError as exc:
            # One unusable manifest must not take down the listing: this verb
            # is how an operator finds out something is wrong.
            rows.append({**row, "user_target": None, "workspace_target": None,
                         "user_path": None, "state": "invalid",
                         "detail": str(exc)})
            continue
        user = target_dir(name, user=True, workspace=None)
        state, detail = compare(name, user)
        rows.append({**row,
                     "user_target": user_target,
                     "workspace_target": workspace_target,
                     "user_path": str(user), "state": state, "detail": detail})
    data = {"integrations": rows, "framework": framework_version()}
    if not rows:
        return data, "this build ships no integrations"

    lines = ["integrations — jaato's side of a contract with another tool", ""]
    for r in rows:
        mark = {"current": "✔", "absent": "·", "stale": "!", "outdated": "!",
                "edited": "~", "diverged": "~", "unstamped": "?",
                "invalid": "✖"}.get(r["state"], "?")
        lines.append(f"  {mark} {r['name']:14} {r['tool']}")
        if r["summary"]:
            lines.append(f"    {'':14} {r['summary']}")
        if r["user_path"]:
            lines.append(f"    {'':14} user scope: {r['user_path']}")
        lines.append(f"    {'':14} state: {r['state']}"
                     + (f" — {r['detail']}" if r["detail"] else ""))
        lines.append("")
    lines += ["  apply one:   jaato-scaffold integration <name> [--user | --workspace DIR]",
              "  refresh:     jaato-scaffold integration <name> --force",
              "  jaato-doctor reports these states without being asked."]
    return data, "\n".join(lines)
