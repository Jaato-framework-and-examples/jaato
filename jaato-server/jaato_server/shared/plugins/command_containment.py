"""Workspace containment for a *command string*, shared by the exec plugins.

Every tool that hands a model-authored command to the operating system owes
the same answer to the same question: *do the paths in this string stay
inside the session workspace?*  ``cli`` answered it; ``interactive_shell``
did not, so the model could route around ``cli``'s refusal by spawning a PTY
and reading anything the daemon user could (jaato issues #722, #503).  The
answer now lives in one module rather than in whichever plugin happened to
implement it, because a boundary enforced in one of two doors is not a
boundary.

The logic here is ``cli``'s, moved verbatim -- the write/read classification
heuristics, the ``path_like`` filter, and the
``check_path_with_jaato_containment`` call that decides each path.
``CLIToolPlugin`` keeps its result-shaping methods (it refuses with an
explicit ``cli containment (workspace boundary): ...`` message -- until
#1202 it mimicked "No such file or directory", which sessions read as
missing binaries) and delegates the analysis; ``InteractiveShellPlugin``
refuses in its own vocabulary.

**The executable position (#1202).**  With ``mark_executables=True`` the
word a segment directly executes is classified ``"exec"`` rather than as
data, and :func:`executable_on_search_path` allows it iff its directory is
an entry of the PATH the bare name would resolve against: ``git`` and
``/usr/bin/git`` are one program and get one verdict.  Only ``cli`` asks for
this, because only ``cli`` holds the subprocess PATH to judge against;
``interactive_shell`` keeps the pre-#1202 classification.

**Fail-closed is the caller's decision, not this module's.**
:func:`classify_command_paths` raises
:class:`~shared.command_analysis.UnanalyzableCommand` when the string cannot
be modelled the way ``/bin/sh`` would parse it.  For a ``cli`` command --
which *is* a shell command by construction -- the only safe answer is to
refuse.  For text typed into a live PTY it is not: that text may be Python,
SQL, or a password, and refusing everything the shell grammar cannot parse
would refuse most legitimate input.  :func:`first_denied_path` therefore
takes an explicit ``on_parse_error`` and forces each caller to say which it
means.

**What this cannot do.**  It reads a string.  A live PTY accepts strings
turn after turn, its working directory drifts under ``cd``, and a program
running inside it can name a path this module never sees (a heredoc, an
encoded argument, a child shell's own input).  One blind spot is worth
naming because it is reachable in a single call and predates this module:
``analyze_command`` does not descend into the quoted argument of
``sh -c`` / ``bash -c``, so ``sh -c 'cat /etc/shadow'`` classifies as *no
paths at all*.  ``cli`` has always had that gap -- this is its analyzer,
moved, not a new one -- and it is the concrete reason the interactive
shell announces when kernel confinement is absent instead of presenting
this check as a boundary.  Against all of it the only real boundary is
kernel confinement: see ``server/apparmor.py`` and
``InteractiveShellPlugin.set_apparmor_child_transition_callback``.  This is
the userspace half, and it is worth having for the same reason ``cli``'s is:
it refuses the direct attempt, in the vocabulary the model can act on.
"""

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from jaato_server.shared.command_analysis import (
    Segment,
    WRAPPER_COMMANDS,
    analyze_command,
    executable_word_index,
)
from jaato_server.shared.path_utils import msys2_to_windows_path
from .sandbox_utils import check_path_with_jaato_containment


# Commands whose path arguments are write targets.
# For commands with mixed semantics (cp, mv, install), the *last* path argument
# is treated as write; earlier ones are read.  For single-target commands (rm,
# touch, mkdir, etc.) all path arguments are write targets.
WRITE_ALL_CMDS = frozenset({
    'rm', 'rmdir', 'touch', 'mkdir', 'mkfifo', 'mknod',
    'truncate', 'shred',
})
WRITE_LAST_CMDS = frozenset({
    'cp', 'mv', 'install', 'rsync', 'scp',
})
WRITE_OUTPUT_CMDS = frozenset({
    'tee',
})

# Every command name whose presence in a segment implies a write somewhere.
ALL_WRITE_CMDS = WRITE_ALL_CMDS | WRITE_LAST_CMDS | WRITE_OUTPUT_CMDS


def path_like(token: str) -> bool:
    """True if a command word should be treated as a filesystem path.

    Absolute paths, ``..`` traversal, explicit ``./`` and ``~`` prefixes
    count; option flags, URLs and npm-style ``@scope/package`` names do not.

    Args:
        token: One word from a command, quoting already removed.

    Returns:
        True when the token should be run through the workspace check.
    """
    if not token or token.startswith('-'):
        return False
    if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', token):
        return False
    if token.startswith('@') and '/' in token and not token.startswith('@/'):
        return False
    return (token.startswith('/') or '..' in token or
            token.startswith('./') or token.startswith('~'))


def arg_path_like(arg: str) -> bool:
    """True if an explicit ``args`` entry should be workspace-checked.

    Looser than :func:`path_like` on purpose: entries in the separate
    ``args`` list are never shell-parsed, so flag/URL exclusions (which
    exist to avoid mis-reading shell words) must not weaken the check.
    """
    return (arg.startswith('/') or '..' in arg or
            arg.startswith('./') or arg.startswith('~'))


def effective_command_name(segment: Segment) -> str:
    """Pick the command name that governs a segment's path semantics.

    A segment can name more than one command: ``sudo rm -rf x`` resolves to
    ``['sudo', 'rm']``.  Write semantics win, so the first resolved name in
    :data:`ALL_WRITE_CMDS` is returned.  When the segment is headed by a
    wrapper (``sudo``, ``env``, ``xargs``, ...) whose argument layout this
    module does not model precisely, every word is scanned for a write
    command -- deliberately over-classifying as write rather than risking a
    write that reads as ``read``.

    Args:
        segment: One analyzed shell segment.

    Returns:
        The governing command basename, or ``''`` when the segment names
        no command (assignments only).
    """
    names = segment.command_names
    for name in names:
        if name in ALL_WRITE_CMDS:
            return name
    if any(name in WRAPPER_COMMANDS for name in names):
        for word in segment.words:
            base = os.path.basename(word)
            if base in ALL_WRITE_CMDS:
                return base
    return names[-1] if names else ''


def classify_word_paths(
    cmd_name: str, paths: List[str]
) -> List[Tuple[str, str]]:
    """Apply the command-name write heuristics to a segment's path words.

    Args:
        cmd_name: The governing command name for the segment.
        paths: Path-looking words, in source order.

    Returns:
        List of ``(path, mode)`` tuples where mode is "read" or "write".
    """
    if cmd_name in WRITE_ALL_CMDS or cmd_name in WRITE_OUTPUT_CMDS:
        return [(path, 'write') for path in paths]
    result: List[Tuple[str, str]] = [(path, 'read') for path in paths]
    if cmd_name in WRITE_LAST_CMDS and result:
        result[-1] = (result[-1][0], 'write')
    return result


#: Mode given to a path in the direct executable position when the caller
#: asks for it (``mark_executables=True``).  It is the weakest mode: the same
#: path appearing anywhere else in the command as data upgrades it to
#: ``read`` / ``write``, so ``/usr/bin/cat /usr/bin/cat`` is judged as a
#: READ of ``/usr/bin/cat`` and an executable exemption can never cover a
#: data argument that happens to share its spelling.
EXEC_MODE = 'exec'

_MODE_RANK = {EXEC_MODE: 0, 'read': 1, 'write': 2}


def classify_segment(
    segment: Segment, mark_executables: bool = False
) -> List[Tuple[str, str]]:
    """Classify the paths of a single shell segment.

    Each segment is judged on its own command name and its own
    redirections, which is what makes compound commands safe to reason
    about: in ``cat README.md && rm -rf notes/`` the ``rm`` segment
    classifies ``notes/`` as write even though the string starts with
    ``cat``.

    Args:
        segment: One segment from :func:`analyze_command`.
        mark_executables: When True, a path-like word in the segment's
            DIRECT executable position (see
            :func:`shared.command_analysis.executable_word_index`) is
            reported with mode :data:`EXEC_MODE` instead of being run
            through the read/write heuristics.  Only the word at that
            position is affected -- the same path as an argument is still
            data.  When False (the default) the output is exactly what it
            was before #1202: the executable is a ``read``.

    Returns:
        List of ``(path, mode)`` tuples where mode is "read", "write" or
        (only with *mark_executables*) :data:`EXEC_MODE`.
    """
    cmd_name = effective_command_name(segment)
    exe_index = (
        executable_word_index(segment.words) if mark_executables else None
    )
    word_paths = [
        word for index, word in enumerate(segment.words)
        if index != exe_index and path_like(word)
    ]
    result = classify_word_paths(cmd_name, word_paths)
    if exe_index is not None and path_like(segment.words[exe_index]):
        result.insert(0, (segment.words[exe_index], EXEC_MODE))

    # Redirection targets carry the mode the operator grants, regardless
    # of what the command itself does.
    for redirect in segment.redirects:
        if redirect.mode == 'none' or not path_like(redirect.target):
            continue
        result.append((redirect.target, redirect.mode))
    return result


def classify_arg_list(
    segments: List[Segment],
    arg_list: Sequence[str],
) -> List[Tuple[str, str]]:
    """Classify paths supplied through a separate ``args`` list.

    The ``args`` form is never shell-parsed, so its entries are checked
    with the looser :func:`arg_path_like` filter but classified with the
    same command-name heuristics as inline words.

    Args:
        segments: Segments parsed from the ``command`` string (used only
            to resolve the command name).
        arg_list: The explicit argument list.

    Returns:
        List of ``(path, mode)`` tuples.
    """
    head_words = list(segments[0].words) if segments else []
    args = [str(arg) for arg in arg_list]
    synthetic = Segment(words=head_words + args)
    cmd_name = effective_command_name(synthetic)
    return classify_word_paths(
        cmd_name, [arg for arg in args if arg_path_like(arg)]
    )


def classify_command_paths(
    command: str,
    arg_list: Optional[Sequence[str]] = None,
    mark_executables: bool = False,
) -> List[Tuple[str, str]]:
    """Classify each path token in a command as "read" or "write".

    The command is first segmented into the simple commands the shell
    would actually run (see
    :func:`shared.command_analysis.analyze_command`), including the bodies
    of command substitutions.  Each segment is then classified
    independently and the results are unioned, with "write" winning over
    "read" for a path that appears in both roles.

    Per-segment heuristics (in order of priority):
    1. Redirection targets take the mode the operator grants -- the full
       file-descriptor grammar, not just ``>``/``>>`` (so ``2>f``,
       ``&>f``, ``>&f``, ``<>f``, ``>|f`` are all writes, and heredoc
       delimiters are not paths at all).
    2. All path args of commands in :data:`WRITE_ALL_CMDS` are "write".
    3. The last path arg of commands in :data:`WRITE_LAST_CMDS` is "write".
    4. All path args of commands in :data:`WRITE_OUTPUT_CMDS` are "write".
    5. Everything else defaults to "read".

    With *mark_executables*, the word in each segment's direct executable
    position is reported as :data:`EXEC_MODE` (jaato #1202) -- see
    :func:`classify_segment`.  A path seen in several roles keeps the
    strongest (write > read > exec), so the executable marking can never
    survive a data use of the same path.

    Args:
        command: The shell command string.
        arg_list: Optional separate argument list.
        mark_executables: Report executable-position words as
            :data:`EXEC_MODE` instead of ``read``.  Off by default so a
            caller that has no PATH to judge them against
            (``interactive_shell``) sees the pre-#1202 classification.

    Returns:
        List of ``(path, mode)`` tuples, first-seen order, where mode is
        "read", "write" or (only with *mark_executables*) "exec".

    Raises:
        UnanalyzableCommand: When the command cannot be modelled the way
            the shell would parse it.  Callers decide what that means --
            see :func:`first_denied_path`.
    """
    segments = analyze_command(command)

    modes: Dict[str, str] = {}
    order: List[str] = []

    def record(pairs: List[Tuple[str, str]]) -> None:
        for path, mode in pairs:
            if path not in modes:
                modes[path] = mode
                order.append(path)
            elif _MODE_RANK[mode] > _MODE_RANK[modes[path]]:
                modes[path] = mode

    for segment in segments:
        record(classify_segment(segment, mark_executables))

    if arg_list:
        record(classify_arg_list(segments, arg_list))

    return [(path, modes[path]) for path in order]


def path_within_workspace(
    path: str,
    workspace_root: Optional[str],
    plugin_registry: Any = None,
    mode: str = "write",
) -> bool:
    """Check whether one path token is allowed for access.

    A path is allowed if:
    1. No *workspace_root* is configured (sandboxing disabled)
    2. The path is within *workspace_root*
    3. The path is under ``.jaato`` and within the ``.jaato`` containment
       boundary (see :mod:`shared.plugins.sandbox_utils`)
    4. The path is authorized via the plugin registry (respecting access
       mode)

    Handles absolute paths, relative paths (resolved against
    *workspace_root*), ``..`` traversal, symlinks (resolved to the real
    path, with ``.jaato`` handled specially) and ``~`` expansion.

    Args:
        path: The path token to check.
        workspace_root: The session workspace root, or ``None``/empty when
            no sandboxing is configured.
        plugin_registry: Optional registry consulted for explicitly
            authorized (``sandbox add``) and denied paths.
        mode: Access mode -- "read" or "write" (default: "write").
            :func:`classify_command_paths` infers this per token.

    Returns:
        True if the path is allowed, False otherwise.  A path that cannot
        be resolved at all is treated as outside the workspace.
    """
    if not workspace_root:
        # No sandboxing configured
        return True

    try:
        # Convert MSYS2 drive paths (/c/...) to Windows (C:/...) for Python
        candidate = msys2_to_windows_path(path)

        # Expand ~ to home directory
        expanded = os.path.expanduser(candidate)

        # Make absolute relative to workspace_root
        if not os.path.isabs(expanded):
            expanded = os.path.join(workspace_root, expanded)

        return check_path_with_jaato_containment(
            expanded,
            workspace_root,
            plugin_registry,
            mode=mode,
        )
    except (OSError, ValueError):
        # If path resolution fails, treat as outside workspace for safety
        return False


def executable_on_search_path(
    executable: str,
    search_path: Optional[str],
    cwd: Optional[str],
) -> bool:
    """True when *executable* sits directly in a directory of *search_path*.

    The executable-position rule of jaato #1202: **an absolute (or
    otherwise path-shaped) executable is allowed iff its directory is an
    entry of the PATH the bare name would be resolved against.**  Anything
    in such a directory already runs by bare name, so reaching it by path
    widens nothing -- ``git`` and ``/usr/bin/git`` are one program and must
    get one verdict.

    Semantics chosen to match ``shutil.which`` / ``execvp`` rather than to
    be generous:

    - **Exact directory equality**, never a prefix: ``/usr/bin/sub/tool``
      is not in ``/usr/bin``.  Both sides are ``normpath``-ed (and
      ``normcase``-d, a no-op on POSIX) so spelling differences like
      ``/usr//bin`` or a trailing slash do not matter.
    - **The executable is NOT realpath-ed.**  The bare-name lookup does not
      follow symlinks when deciding which directory a name is in either;
      matching its semantics is the point.
    - **A ``..`` component disqualifies the word outright.**  ``normpath``
      collapses ``..`` lexically while the kernel resolves it through
      symlinks, so ``<bin>/link/../tool`` normalises into ``<bin>`` and may
      execute somewhere else entirely (``link -> /outside/sub`` runs
      ``/outside/tool``).  Such a word is left to the ordinary data check.
    - **Relative and empty PATH entries** are resolved against *cwd* (the
      directory the command runs in), an empty entry meaning *cwd* itself,
      as POSIX ``execvp`` does.

    Args:
        executable: The word in the executable position, as written
            (``~`` is expanded, as the shell would).
        search_path: The PATH value the command will run under -- the ONE
            the caller built for the subprocess, never one parsed out of the
            command string.  ``None`` authorizes nothing.
        cwd: The subprocess working directory, for relative entries and a
            relative *executable*.

    Returns:
        True when the executable's directory equals a PATH entry.
    """
    if search_path is None or not cwd:
        return False
    candidate = os.path.expanduser(msys2_to_windows_path(executable))
    if not os.path.basename(candidate):
        return False
    if '..' in re.split(r'[\\/]', candidate):
        return False
    exe_dir = _normalized_dir(os.path.dirname(candidate), cwd)
    return any(
        _normalized_dir(entry, cwd) == exe_dir
        for entry in search_path.split(os.pathsep)
    )


def _normalized_dir(directory: str, cwd: str) -> str:
    """Absolute, lexically normalised form of a PATH entry or exe directory.

    An empty *directory* is the working directory (POSIX ``execvp``'s
    reading of an empty PATH entry, and ``os.path.dirname`` of a bare
    ``tool``).  No symlink resolution: see
    :func:`executable_on_search_path`.
    """
    directory = directory or os.curdir
    if not os.path.isabs(directory):
        directory = os.path.join(cwd, directory)
    return os.path.normcase(os.path.normpath(directory))


def first_denied_path(
    command: str,
    workspace_root: Optional[str],
    plugin_registry: Any = None,
    arg_list: Optional[Sequence[str]] = None,
    on_parse_error: str = "deny",
) -> Optional[Tuple[str, str]]:
    """Return the first path in *command* that escapes the workspace.

    The single entry point for a plugin that has a whole command string
    and needs a verdict.

    Args:
        command: The command string to analyse.
        workspace_root: Session workspace root; falsy disables the check.
        plugin_registry: Optional registry for authorized/denied paths.
        arg_list: Optional separate argument list (the ``cli`` ``args``
            form).
        on_parse_error: What an unparseable command means.  ``"deny"`` --
            the fail-closed answer for a string that *is* a shell command
            (``cli``'s ``command``, ``interactive_shell``'s spawn line) --
            re-raises
            :class:`~shared.command_analysis.UnanalyzableCommand` so the
            caller can render its own refusal.  ``"allow"`` returns
            ``None``, which is the right answer only for text whose
            grammar is not the shell's (a line typed into a live REPL).

    Returns:
        ``(path, mode)`` for the first token that fails containment, or
        ``None`` when every token is allowed.

    Raises:
        UnanalyzableCommand: When the command cannot be parsed and
            *on_parse_error* is ``"deny"``.
        ValueError: When *on_parse_error* is neither ``"deny"`` nor
            ``"allow"`` -- a typo there would silently pick a posture.
    """
    if on_parse_error not in ("deny", "allow"):
        raise ValueError(
            f"on_parse_error must be 'deny' or 'allow', got {on_parse_error!r}"
        )
    if not workspace_root:
        return None

    try:
        classified = classify_command_paths(command, arg_list)
    except Exception:
        if on_parse_error == "deny":
            raise
        return None

    for path, mode in classified:
        if not path_within_workspace(
            path, workspace_root, plugin_registry, mode=mode
        ):
            return (path, mode)
    return None
