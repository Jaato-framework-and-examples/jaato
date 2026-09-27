"""What a confined session's ``//child`` profile grants, read runner-side (#1348).

A command the model runs through ``cli`` executes in the session's
``//child`` AppArmor sub-profile.  When the kernel refuses it, the command
sees ``Permission denied`` and nothing else: the same words a missing
execute bit produces.  A model reading them retries, tries another path to
the same file, or concludes the file is broken.  Only ``journalctl -k``
says which profile refused what, and nothing a session can reach reads it.

The daemon records what each profile was provisioned with (#1326).  This
module is the runner's copy of the part that answers "would the kernel
allow this": the exec scope and the file rules of the ``//child`` body,
carried on ``SessionInitEnvelope.confinement_grants`` and installed once
per session by the runner bootstrap (:func:`set_confinement_grants`).

:func:`explain_denial` turns a failed command into a hint only on
**positive evidence**:

- the session is confined and has a record (otherwise ``None``);
- the path the failure names resolves, through the same PATH the command
  ran with, to a file that exists;
- ordinary permissions allow the access (``os.access``), so the refusal
  is not a missing mode bit;
- the recorded rules do not grant it, or deny it outright.

A rule this module cannot read makes the verdict uncertain, and an
uncertain verdict gives no hint.  A read is judged only when the file is
one the profile lets the session EXECUTE (the script case: the program
runs, its interpreter cannot open it), and never for a path a reference
selection authorized, because those grants are added after provisioning
and are not in the record.

Stdlib-only, like :mod:`jaato_server.shared.apparmor_label`, because the
runner imports it before plugin discovery and ``shared`` cannot import
``server``.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Pattern, Tuple

# ---------------------------------------------------------------------------
# The //child body of a rendered profile
# ---------------------------------------------------------------------------


def profile_body_rules(profile_text: str, body: str = "child") -> List[str]:
    """The rule lines of the ``profile <body> {`` block in *profile_text*.

    The block ends at a line holding only ``}``.  Braces are not counted,
    because a rule may contain an alternation (``/usr/{bin,sbin}/** ix,``).
    Comments and blank lines are dropped.  An absent block gives ``[]``.
    """
    header = f"profile {body} {{"
    rules: List[str] = []
    inside = False
    for raw in profile_text.splitlines():
        line = raw.strip()
        if not inside:
            inside = line == header
            continue
        if line == "}":
            break
        if line and not line.startswith("#"):
            rules.append(line)
    return rules


# ---------------------------------------------------------------------------
# AppArmor globs
# ---------------------------------------------------------------------------


def glob_to_regex(glob: str) -> Optional[Pattern[str]]:
    """Compile an AppArmor path glob, or ``None`` when it cannot be read.

    ``*`` is any run without ``/``, ``**`` any run, ``?`` one character
    other than ``/``, ``[...]`` a class, ``{a,b}`` an alternation.  A
    variable (``@{HOME}``, ``@{PROC}``) matches ANY run, ``/`` included:
    its value is set by the host's tunables, which the runner does not
    read, and matching more than the kernel would only ever withhold a
    hint, never give a wrong one.
    """
    out: List[str] = []
    depth = 0
    i, n = 0, len(glob)
    while i < n:
        ch = glob[i]
        if ch == "@" and glob.startswith("@{", i):
            end = glob.find("}", i)
            if end < 0:
                return None
            out.append(".*")
            i = end + 1
            continue
        if ch == "*":
            if glob.startswith("**", i):
                out.append(".*")
                i += 2
            else:
                out.append("[^/]*")
                i += 1
            continue
        if ch == "?":
            out.append("[^/]")
        elif ch == "[":
            end = glob.find("]", i + 1)
            if end < 0:
                return None
            body = glob[i + 1:end]
            if body.startswith("^"):
                body = "^" + body[1:].replace("\\", "\\\\")
            else:
                body = body.replace("\\", "\\\\")
            out.append(f"[{body}]")
            i = end + 1
            continue
        elif ch == "{":
            depth += 1
            out.append("(?:")
        elif ch == "}" and depth:
            depth -= 1
            out.append(")")
        elif ch == "," and depth:
            out.append("|")
        elif ch == "\\" and i + 1 < n:
            out.append(re.escape(glob[i + 1]))
            i += 2
            continue
        else:
            out.append(re.escape(ch))
        i += 1
    if depth:
        return None
    try:
        return re.compile("".join(out), re.DOTALL)
    except re.error:
        return None


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

_QUALIFIERS = frozenset({"audit", "allow", "deny", "owner", "quiet"})

#: First words of the rule kinds that grant or deny nothing on a FILE.
_NON_FILE_KINDS = frozenset({
    "network", "capability", "ptrace", "signal", "unix", "dbus", "mount",
    "umount", "remount", "pivot_root", "change_profile", "rlimit", "set",
    "userns", "io_uring", "mqueue", "link",
})

_FILE_RULE = re.compile(
    r'^(?:file\s+)?(?P<path>"[^"]*"|[/@]\S*)\s+(?P<modes>[a-zA-Z]+)'
    r'(?:\s*->\s*\S+)?\s*,$'
)


@dataclass(frozen=True)
class FileRule:
    """One file rule of the ``//child`` body."""

    glob: str
    modes: str
    deny: bool
    owner: bool
    pattern: Optional[Pattern[str]]

    def grants(self, need: str) -> bool:
        """Whether the rule's modes cover *need* (``"x"`` or ``"r"``)."""
        if need == "x":
            return "x" in self.modes.lower()
        return need in self.modes


@dataclass
class _Parsed:
    """The rules of one body, and whether any line could not be read."""

    rules: List[FileRule] = field(default_factory=list)
    unreadable: List[str] = field(default_factory=list)
    reference_include: bool = False
    grants_everything: bool = False


def _split_qualifiers(line: str) -> Tuple[set, str]:
    words = line.split()
    quals = set()
    while words and words[0] in _QUALIFIERS:
        quals.add(words.pop(0))
    return quals, " ".join(words)


def parse_rules(lines: Iterable[str]) -> _Parsed:
    """Classify each rule line.  A line not understood is recorded, not dropped."""
    parsed = _Parsed()
    for line in lines:
        quals, rest = _split_qualifiers(line.strip())
        head = rest.split(None, 1)[0].rstrip(",") if rest else ""
        if head in _NON_FILE_KINDS:
            continue
        if head == "include":
            # The one include in a //child body is the per-session refs
            # directory: read-only reference grants (``_fragment_content``).
            if ".refs.d/" in rest:
                parsed.reference_include = True
            else:
                parsed.unreadable.append(line)
            continue
        if rest == "file," and "deny" not in quals:
            parsed.grants_everything = True
            continue
        match = _FILE_RULE.match(rest)
        if not match:
            parsed.unreadable.append(line)
            continue
        glob = match.group("path").strip('"')
        parsed.rules.append(FileRule(
            glob=glob,
            modes=match.group("modes"),
            deny="deny" in quals,
            owner="owner" in quals,
            pattern=glob_to_regex(glob),
        ))
    return parsed


# ---------------------------------------------------------------------------
# The grants, and the verdict on one path
# ---------------------------------------------------------------------------


@dataclass
class ConfinementGrants:
    """The ``//child`` grants of one confined session.

    Built from the envelope's wire dict by :meth:`from_wire`.  Immutable in
    use: the runner replaces it per session and never edits it.
    """

    profile_name: str
    exec_scope: Optional[str]
    rules: List[str]
    _parsed: _Parsed = field(default_factory=_Parsed, repr=False)

    def __post_init__(self) -> None:
        self._parsed = parse_rules(self.rules)

    @classmethod
    def from_wire(cls, wire: Any) -> Optional["ConfinementGrants"]:
        """Build from ``SessionInitEnvelope.confinement_grants``; ``None`` if malformed."""
        if not isinstance(wire, dict):
            return None
        name = wire.get("profile_name")
        rules = wire.get("rules")
        if not isinstance(name, str) or not name or not isinstance(rules, list):
            return None
        if not rules or not all(isinstance(r, str) for r in rules):
            return None
        scope = wire.get("exec_scope")
        return cls(
            profile_name=name,
            exec_scope=scope if isinstance(scope, str) else None,
            rules=list(rules),
        )

    def verdict(self, path: str, need: str) -> Optional[bool]:
        """Whether the rules give *need* on *path*: granted, not granted, or unknown.

        ``False`` when an unconditional deny matches, or when no allow rule
        does and every rule was read.  An ``owner`` deny depends on who owns
        the file, so a matching one makes the answer unknown, allow or not.
        """
        parsed = self._parsed
        if parsed.grants_everything:
            return True
        if parsed.unreadable:
            return None
        allowed = uncertain = False
        for rule in parsed.rules:
            if not rule.grants(need):
                continue
            if rule.pattern is None:
                uncertain = True
                continue
            if not rule.pattern.fullmatch(path):
                continue
            if not rule.deny:
                allowed = True
            elif rule.owner:
                uncertain = True
            else:
                return False
        if uncertain:
            return None
        return allowed

    def may_add_reads(self) -> bool:
        """Whether reference selections can widen reads after provisioning."""
        return self._parsed.reference_include


_LOCK = threading.Lock()
_CURRENT: Optional[ConfinementGrants] = None


def set_confinement_grants(wire: Any) -> Optional[ConfinementGrants]:
    """Install this session's grants; ``None`` or a malformed dict clears them.

    Called by the runner bootstrap for every session, so a pool slot never
    judges a session by the grants of the one before it.
    """
    global _CURRENT
    grants = ConfinementGrants.from_wire(wire)
    with _LOCK:
        _CURRENT = grants
    return grants


def confinement_grants() -> Optional[ConfinementGrants]:
    """The installed grants, or ``None`` for an unconfined or unrecorded session."""
    with _LOCK:
        return _CURRENT


# ---------------------------------------------------------------------------
# From a failed command to a hint
# ---------------------------------------------------------------------------

#: ``cannot open /usr/bin/which: Permission denied`` (dash),
#: ``can't open file '/x': [Errno 13] Permission denied`` (python): an
#: interpreter that could not open the script it was started for.
_READ_DENIED = re.compile(
    r"(?:cannot open|can't open file)\s+'?(?P<path>[^':\n]+?)'?:?\s+"
    r"(?:\[Errno 13\] )?Permission denied"
)
#: ``Permission denied: '/usr/bin/ls'``: a PermissionError from exec.
_ERRNO_DENIED = re.compile(r"Permission denied: '(?P<path>[^']+)'")
#: ``bash: line 1: /usr/lib/cargo/bin/coreutils/ls: Permission denied`` or
#: ``sh: 1: ls: Permission denied``: a SHELL that could not run a command.
#: Only a shell's own line counts; ``cat: /x: Permission denied`` is a
#: program failing to read its data, which is not judged here.
_SHELL_DENIED = re.compile(
    r"^(?:\S*/)?(?:bash|sh|dash|zsh|ksh|mksh)(?:: line \d+|: \d+)?: "
    r"(?P<path>.+?): Permission denied$"
)

#: What a candidate may be refused: exec, read (a script its interpreter
#: could not open), or either (a shell names the script it tried to run,
#: which the kernel refuses for exec, or its interpreter for read).
EXEC, READ, EXEC_OR_READ = "x", "r", "x|r"


def _denied_candidates(
    command: str, text: str, returncode: Optional[int],
) -> List[Tuple[str, str]]:
    """``(path as named, what was refused)`` pairs a failure's own words name."""
    found: List[Tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if "Permission denied" not in line:
            continue
        read = _READ_DENIED.search(line)
        if read:
            found.append((read.group("path").strip(), READ))
            continue
        errno = _ERRNO_DENIED.search(line)
        if errno:
            found.append((errno.group("path"), EXEC))
            continue
        shell = _SHELL_DENIED.match(line)
        if shell:
            found.append((shell.group("path"), EXEC_OR_READ))
    if returncode == 126 and not found:
        first = _first_word(command)
        if first:
            found.append((first, EXEC_OR_READ))
    return found


def _first_word(command: str) -> Optional[str]:
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    for word in words:
        if "=" in word and not word.startswith(("/", ".")):
            continue  # a VAR=value prefix
        return word
    return None


def _resolve(named: str, search_path: Optional[str], cwd: Optional[str]) -> Optional[str]:
    """The real path *named* refers to, or ``None`` when it does not resolve."""
    if "/" in named:
        path = named if os.path.isabs(named) else os.path.join(cwd or os.getcwd(), named)
    else:
        path = shutil.which(named, path=search_path)
        if path is None:
            return None
    try:
        real = os.path.realpath(path)
    except (OSError, ValueError):
        return None
    return real if os.path.isfile(real) else None


def _advice(grants: ConfinementGrants, refused: str) -> str:
    if refused == READ:
        return (
            "The profile lets the session execute it but not read it: it is a "
            "script, and its interpreter has to open it. An operator can grant "
            "read on it with an AppArmor fragment."
        )
    if grants.exec_scope == "scoped":
        return (
            "This session's profile limits exec to what its apparmor_fragments "
            "grant. An operator can add a fragment granting it and list it in "
            "the profile's apparmor_fragments."
        )
    return (
        "An operator can grant it with an AppArmor fragment (a .rules file in "
        "~/.jaato/apparmor-fragments/ or the workspace's "
        ".jaato/apparmor-fragments/)."
    )


def _read_refused(
    grants: ConfinementGrants,
    resolved: str,
    maybe_authorized: Optional[Callable[[str], bool]],
) -> bool:
    """The script case: exec granted, ordinary read allowed, the rules' read not.

    A read is judged only for a file the profile lets the session execute.
    Reference selections add read grants after provisioning, so a path one
    of them may have authorized is never judged, and without a way to ask,
    no read is judged at all where references can widen reads.
    """
    if grants.verdict(resolved, EXEC) is not True:
        return False
    if not os.access(resolved, os.R_OK):
        return False  # ordinary permissions refuse it; not the profile
    if grants.may_add_reads():
        if maybe_authorized is None:
            return False
        try:
            if maybe_authorized(resolved):
                return False
        except Exception:  # noqa: BLE001 -- unknown means no hint
            return False
    return grants.verdict(resolved, READ) is False


def _refusal(
    grants: ConfinementGrants,
    resolved: str,
    kind: str,
    maybe_authorized: Optional[Callable[[str], bool]],
) -> Optional[str]:
    """``EXEC`` or ``READ`` when the rules positively refuse it, else ``None``."""
    if kind in (EXEC, EXEC_OR_READ) and os.access(resolved, os.X_OK):
        if grants.verdict(resolved, EXEC) is False:
            return EXEC
    if kind in (READ, EXEC_OR_READ) and _read_refused(grants, resolved, maybe_authorized):
        return READ
    return None


def _hint(grants: ConfinementGrants, named: str, resolved: str, refused: str) -> str:
    verb = "exec" if refused == EXEC else "read"
    shown = resolved if resolved == named else f"{resolved} (what {named} resolves to)"
    return (
        f"AppArmor refused this: the session's confinement profile "
        f"{grants.profile_name} does not grant {verb} on {shown}. The kernel "
        f"will refuse it again however it is invoked, so do not retry it or "
        f"look for another route to the same file; tell the user what was "
        f"refused. {_advice(grants, refused)}"
    )


def explain_denial(
    *,
    command: str,
    output: str,
    returncode: Optional[int],
    search_path: Optional[str],
    cwd: Optional[str],
    maybe_authorized: Optional[Callable[[str], bool]] = None,
) -> Optional[str]:
    """A hint naming the refused path and the profile, or ``None``.

    Args:
        command: The command as the model wrote it.
        output: Its stderr, plus any error text the runner produced.
        returncode: Its exit status, or ``None`` when it never ran.
        search_path: The PATH it ran with, to resolve a bare name.
        cwd: The directory it ran in, to resolve a relative name.
        maybe_authorized: Answers whether a path was authorized for
            reading after provisioning (a reference selection).  Without
            it, no read is judged when references could have widened reads.
    """
    grants = confinement_grants()
    if grants is None:
        return None
    for named, kind in _denied_candidates(command, output, returncode):
        resolved = _resolve(named, search_path, cwd)
        if resolved is None:
            continue
        refused = _refusal(grants, resolved, kind, maybe_authorized)
        if refused:
            return _hint(grants, named, resolved, refused)
    return None
