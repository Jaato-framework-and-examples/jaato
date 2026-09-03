"""Resolve the working-tree line ending git itself would produce for a path.

When a workspace is a git repository, the repository's own configuration is
the authority on line endings — an editor that imposes its own convention
fights the checkout on every write.  This module reproduces the subset of
git's ``eol`` decision that a write tool needs, reading the same files git
reads, so ``file_edit`` can honour it (jaato #805).

The decision, highest precedence first:

1. ``.gitattributes`` for the path — ``-text`` / ``binary`` means *never*
   convert; ``eol=crlf`` / ``eol=lf`` forces that ending outright.
2. ``core.autocrlf`` — ``true`` means CRLF in the working tree, ``input``
   means LF.
3. ``core.eol`` (default ``native``), but only when a ``text`` attribute is
   actually in force: with git's default configuration (``autocrlf=false``
   and no ``text`` attribute) git converts nothing at all.
4. No opinion — the caller falls back to whatever the file already uses.

Everything is best-effort and read-only: an unreadable config, a malformed
attributes line, or a repository that isn't there resolves to "no opinion"
rather than an error, because a line-ending preference must never be the
reason a file edit fails.

Why this reads the files rather than asking git.  ``git check-attr text eol
-- <path>`` would answer the same question authoritatively in one call, and
is the obvious first suggestion — but ``file_edit`` is a ``runner``-tier
plugin (``PLUGIN_TIER`` in ``file_edit/__init__.py``), so it runs under
AppArmor confinement, and its ``get_apparmor_rules`` contributes backup-path
``rw`` rules only.  It has no exec grant: ``cli``, which does spawn things,
has to ask for ``ix`` explicitly (``pip_apparmor_rules`` in
``shared/plugins/workspace_venv.py``).  A subprocess to ``git`` from here
would simply be denied in the deployment this plugin actually runs in.  The
pure-Python path is not a preference — it is the only one that works.  Per
write it is also the cheaper of the two, which is why the caches below exist.

Not implemented (deliberately): ``core.attributesFile``, the system-level
gitconfig, ``include``/``includeIf`` directives, and attribute macros other
than the built-in ``binary``.  Each would widen the surface well past what
a write tool needs; a repository that uses them still gets rule 4, which
preserves the file's existing endings.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LF = "\n"
CRLF = "\r\n"

#: Ending that ``core.eol=native`` resolves to on this platform.
NATIVE_EOL = CRLF if os.name == "nt" else LF

#: Sentinel for "the attribute was never mentioned", distinct from ``!attr``
#: (mentioned, explicitly unspecified) which git treats the same way here.
_ABSENT = object()


def _translate_class(pattern: str, index: int) -> Tuple[str, int]:
    """Translate a ``[...]`` character class starting at *index* into regex.

    Args:
        pattern: The full glob pattern.
        index: Offset of the character *after* the opening bracket.

    Returns:
        ``(regex_fragment, next_index)``.  An unterminated class is emitted
        as a literal ``[``, matching git's own tolerance for malformed
        patterns.
    """
    scan = index
    if scan < len(pattern) and pattern[scan] in "!^":
        scan += 1
    if scan < len(pattern) and pattern[scan] == "]":
        scan += 1
    while scan < len(pattern) and pattern[scan] != "]":
        scan += 1
    if scan >= len(pattern):
        return re.escape("["), index
    body = pattern[index:scan]
    if body.startswith("!"):
        body = "^" + body[1:]
    return f"[{body}]", scan + 1


def _translate_glob(pattern: str) -> str:
    """Translate a gitignore-style glob into a regex body.

    ``*`` and ``?`` stop at a ``/``; ``**`` crosses directory boundaries.
    """
    out: List[str] = []
    i, n = 0, len(pattern)
    while i < n:
        char = pattern[i]
        i += 1
        if char == "*":
            if i < n and pattern[i] == "*":
                i += 1
                if i < n and pattern[i] == "/":
                    i += 1
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
            else:
                out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            fragment, i = _translate_class(pattern, i)
            out.append(fragment)
        else:
            out.append(re.escape(char))
    return "".join(out)


def _compile_pattern(pattern: str) -> Tuple[re.Pattern, bool]:
    """Compile one attributes pattern.

    Returns:
        ``(regex, match_basename)``.  A pattern with no ``/`` in it matches
        the path's basename at any depth; anything else is anchored to the
        directory holding the ``.gitattributes`` file.
    """
    stripped = pattern.rstrip("/")
    match_basename = "/" not in stripped
    if stripped.startswith("/"):
        stripped = stripped[1:]
    return re.compile(f"(?s:{_translate_glob(stripped)})\\Z"), match_basename


def _parse_attribute_tokens(tokens: List[str]) -> Dict[str, object]:
    """Turn the attribute words of one line into a ``{name: value}`` map.

    Values are ``True`` (``attr``), ``False`` (``-attr``), ``None``
    (``!attr``, i.e. unspecified) or the string after ``=``.
    """
    attrs: Dict[str, object] = {}
    for token in tokens:
        if token.startswith("-"):
            attrs[token[1:]] = False
        elif token.startswith("!"):
            attrs[token[1:]] = None
        elif "=" in token:
            name, _, value = token.partition("=")
            attrs[name] = value
        else:
            attrs[token] = True
    return attrs


def _parse_attributes_file(text: str) -> List[Tuple[re.Pattern, bool, Dict[str, object]]]:
    """Parse a ``.gitattributes`` file into ordered match rules.

    Rules are returned in file order; the caller applies later matches over
    earlier ones, which is git's "last match wins" behaviour.
    """
    rules: List[Tuple[re.Pattern, bool, Dict[str, object]]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        pattern = parts[0]
        if pattern.startswith('"') and pattern.endswith('"') and len(pattern) > 1:
            pattern = pattern[1:-1]
        try:
            regex, match_basename = _compile_pattern(pattern)
        except re.error:
            logger.debug("Skipping unparseable gitattributes pattern: %r", pattern)
            continue
        rules.append((regex, match_basename, _parse_attribute_tokens(parts[1:])))
    return rules


def _strip_config_value(value: str) -> str:
    """Strip a git config value of its inline comment and surrounding quotes."""
    if not value.startswith('"'):
        for marker in ("#", ";"):
            cut = value.find(marker)
            if cut != -1:
                value = value[:cut]
    value = value.strip()
    if len(value) > 1 and value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


def _section_header(line: str) -> Tuple[str, str]:
    """Split a ``[section]`` header into its name and any trailing text.

    ``[core] autocrlf = true`` is legal git config, so the remainder of the
    header line is handed back rather than discarded.
    """
    close = line.find("]")
    if close == -1:
        return "", ""
    inner = line[1:close].strip()
    name = inner.split()[0].strip('"').lower() if inner else ""
    return name, line[close + 1:].strip()


def _parse_core_config(text: str) -> Dict[str, str]:
    """Extract the ``[core]`` keys from a git config file.

    Only ``[core]`` is read — it is the only section holding line-ending
    settings — and keys are lowercased, as git compares them
    case-insensitively.
    """
    core: Dict[str, str] = {}
    in_core = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("["):
            name, line = _section_header(line)
            in_core = name == "core"
            if not line:
                continue
        if not in_core or "=" not in line:
            continue
        key, _, value = line.partition("=")
        core[key.strip().lower()] = _strip_config_value(value)
    return core


def _read_text(path: Path) -> Optional[str]:
    """Read a config/attributes file, or ``None`` if it cannot be read."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def _global_config_paths() -> List[Path]:
    """Paths git would consult for the user's global configuration."""
    paths: List[Path] = []
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        paths.append(Path(xdg) / "git" / "config")
    try:
        home = Path.home()
    except (RuntimeError, OSError):
        return paths
    if not xdg:
        paths.append(home / ".config" / "git" / "config")
    paths.append(home / ".gitconfig")
    return paths


class GitEolResolver:
    """Answers "what line ending would git put in the working tree here?".

    One instance caches its discoveries — repository roots, ``[core]``
    settings and parsed ``.gitattributes`` rules — for the life of the
    object.  Attributes files are re-read when their mtime or size changes,
    so a long-lived daemon picks up an edited ``.gitattributes`` without a
    restart; ``[core]`` settings are read once per repository, since they
    change far more rarely than the cost of re-statting them on every write
    would justify.

    The resolver never writes and never raises: every failure path returns
    "no opinion" so a caller can fall back to preserving the file.
    """

    def __init__(self) -> None:
        self._repo_roots: Dict[str, Optional[Path]] = {}
        self._core: Dict[str, Dict[str, str]] = {}
        self._attr_cache: Dict[str, Tuple[Optional[Tuple[int, int]], List]] = {}

    # -- public API ------------------------------------------------------

    def ending_for(self, path: Path) -> Optional[str]:
        """Return the ending git would impose on *path*, or ``None``.

        ``None`` means git has no opinion — either the path is outside any
        repository, or it is marked non-text, or the repository is using
        git's default configuration, which converts nothing.  The caller
        should then preserve whatever the file already uses.
        """
        try:
            return self._ending_for(Path(os.path.abspath(path)))
        except Exception:  # pragma: no cover - defensive; never fail a write
            logger.debug("git eol resolution failed for %s", path, exc_info=True)
            return None

    # -- resolution ------------------------------------------------------

    def _ending_for(self, path: Path) -> Optional[str]:
        repo_root = self._find_repo_root(path.parent)
        if repo_root is None:
            return None

        attrs = self._attributes_for(repo_root, path)

        text_attr = attrs.get("text", _ABSENT)
        if attrs.get("binary") is True or text_attr is False:
            return None

        eol_attr = attrs.get("eol")
        if eol_attr == "crlf":
            return CRLF
        if eol_attr == "lf":
            return LF

        core = self._core_config(repo_root)
        autocrlf = core.get("autocrlf", "").lower()
        if autocrlf == "true":
            return CRLF
        if autocrlf == "input":
            return LF

        if text_attr is _ABSENT or text_attr is None:
            # Default git configuration converts nothing.
            return None
        return self._config_eol(core)

    @staticmethod
    def _config_eol(core: Dict[str, str]) -> Optional[str]:
        """Map ``core.eol`` (default ``native``) onto a concrete ending."""
        value = core.get("eol", "native").lower()
        if value == "crlf":
            return CRLF
        if value == "lf":
            return LF
        return NATIVE_EOL

    # -- repository discovery --------------------------------------------

    def _find_repo_root(self, start: Path) -> Optional[Path]:
        """Walk up from *start* to the nearest directory holding ``.git``."""
        key = str(start)
        if key in self._repo_roots:
            return self._repo_roots[key]

        found: Optional[Path] = None
        for candidate in (start, *start.parents):
            marker = candidate / ".git"
            if marker.is_dir() or marker.is_file():
                found = candidate
                break

        self._repo_roots[key] = found
        return found

    @staticmethod
    def _git_dir(repo_root: Path) -> Optional[Path]:
        """Resolve the repository's git directory.

        Handles the ``.git`` *file* form used by worktrees and submodules,
        following ``commondir`` so a linked worktree reads the shared
        ``config`` rather than its own (which has none).
        """
        marker = repo_root / ".git"
        if marker.is_dir():
            return marker
        text = _read_text(marker)
        if not text or not text.startswith("gitdir:"):
            return None
        git_dir = Path(text.split(":", 1)[1].strip())
        if not git_dir.is_absolute():
            git_dir = (repo_root / git_dir).resolve()
        common = _read_text(git_dir / "commondir")
        if common:
            shared = Path(common.strip())
            git_dir = shared if shared.is_absolute() else (git_dir / shared).resolve()
        return git_dir

    # -- configuration ----------------------------------------------------

    def _core_config(self, repo_root: Path) -> Dict[str, str]:
        """``[core]`` settings for a repository, global config underneath."""
        key = str(repo_root)
        cached = self._core.get(key)
        if cached is not None:
            return cached

        core: Dict[str, str] = {}
        for config_path in _global_config_paths():
            text = _read_text(config_path)
            if text:
                core.update(_parse_core_config(text))

        git_dir = self._git_dir(repo_root)
        if git_dir is not None:
            text = _read_text(git_dir / "config")
            if text:
                core.update(_parse_core_config(text))

        self._core[key] = core
        return core

    # -- attributes -------------------------------------------------------

    def _attributes_for(self, repo_root: Path, path: Path) -> Dict[str, object]:
        """Collect the attributes in force for *path*, lowest source first.

        Sources are read from the repository root downwards so that a deeper
        ``.gitattributes`` overrides a shallower one, with
        ``.git/info/attributes`` applied last — git's own precedence order.
        """
        attrs: Dict[str, object] = {}
        try:
            relative = path.relative_to(repo_root)
        except ValueError:
            return attrs

        directories = [repo_root, *(repo_root / part for part in relative.parts[:-1])]
        for directory in directories:
            rules = self._rules_for(directory / ".gitattributes")
            self._apply_rules(rules, directory, path, attrs)

        git_dir = self._git_dir(repo_root)
        if git_dir is not None:
            rules = self._rules_for(git_dir / "info" / "attributes")
            self._apply_rules(rules, repo_root, path, attrs)
        return attrs

    @staticmethod
    def _apply_rules(
        rules: List[Tuple[re.Pattern, bool, Dict[str, object]]],
        base: Path,
        path: Path,
        attrs: Dict[str, object],
    ) -> None:
        """Merge every rule in *rules* that matches *path* into *attrs*.

        Rules are applied in file order, so the last matching line wins.
        """
        if not rules:
            return
        try:
            relative = path.relative_to(base).as_posix()
        except ValueError:
            return
        name = path.name
        for regex, match_basename, rule_attrs in rules:
            subject = name if match_basename else relative
            if regex.match(subject):
                attrs.update(rule_attrs)

    def _rules_for(self, attributes_path: Path) -> List[Tuple[re.Pattern, bool, Dict[str, object]]]:
        """Parsed rules for one attributes file, re-read when it changes."""
        key = str(attributes_path)
        try:
            info = attributes_path.stat()
            stamp: Optional[Tuple[int, int]] = (info.st_mtime_ns, info.st_size)
        except OSError:
            stamp = None

        cached = self._attr_cache.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]

        rules: List[Tuple[re.Pattern, bool, Dict[str, object]]] = []
        if stamp is not None:
            text = _read_text(attributes_path)
            if text:
                rules = _parse_attributes_file(text)

        self._attr_cache[key] = (stamp, rules)
        return rules
