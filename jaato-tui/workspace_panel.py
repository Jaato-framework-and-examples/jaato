"""Workspace file panel for the rich client.

Displays a compact directory tree of files created or modified during the
current session.  Toggled via a configurable keybinding (default Ctrl+W).
The panel is a non-blocking floating overlay, following the same pattern as
the plan and budget panels.

Data flow:
    Server emits ``WorkspaceFilesChangedEvent`` (incremental) and
    ``WorkspaceFilesSnapshotEvent`` (full state on reconnect).  The client
    calls ``apply_changes()`` or ``apply_snapshot()`` on this panel, which
    updates its internal set.  The ``render_popup()`` method builds a Rich
    Panel with a compact directory tree.

Tree building:
    Files are organised into a nested dict keyed by path component.
    Directories are sorted before files at each level.  Nodes can be
    collapsed/expanded with Left/Right arrow keys.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from rich.panel import Panel
from rich.text import Text
from rich.console import Group

from keybindings import KeyBinding, format_key_for_display

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from theme import ThemeConfig

# How long (seconds) a newly added/removed entry stays highlighted.
_HIGHLIGHT_DURATION = 8.0


class _TreeNode:
    """Internal node for building the directory tree.

    Each node represents either a directory (children non-empty) or a
    file (leaf).  The ``status`` field is only meaningful for leaf nodes.
    """

    __slots__ = ("name", "children", "status", "highlight_until", "collapsed")

    def __init__(self, name: str):
        self.name = name
        self.children: Dict[str, "_TreeNode"] = {}
        self.status: Optional[str] = None  # "created" | "modified" | "deleted"
        self.highlight_until: float = 0.0  # time.monotonic() deadline
        self.collapsed: bool = False

    @property
    def is_dir(self) -> bool:
        return bool(self.children)


class WorkspacePanel:
    """Renders workspace file changes as a compact tree popup overlay.

    Lifecycle:
        1. Created once by ``pt_display.py`` at startup.
        2. ``apply_snapshot(files)`` replaces state on reconnect.
        3. ``apply_changes(changes)`` applies incremental deltas.
        4. ``toggle_popup()`` shows/hides the overlay.
        5. ``render_popup(width)`` produces a Rich Panel for display.

    Navigation (when popup is visible and input is empty):
        Up/Down    – scroll through the file list.
        Left/Right – collapse/expand directory at cursor.
        Delete     – clear file list (only future changes will appear).
        Ctrl+W     – close the panel.
        Escape     – close the panel.

    Pasting an ``@`` reference (works regardless of input state, as long as
    the panel is visible) is handled in ``pt_display.py`` via the
    ``workspace_paste_ref`` keybinding (default Alt+P).

    Hide / show-hidden:
        ``hide_at_cursor()`` adds the entry under the cursor to a
        per-session ``_hidden_paths`` set.  ``_get_flat_entries`` then
        skips entries whose path equals or descends from any hidden
        directory.  ``toggle_show_hidden()`` flips ``_show_hidden``: while
        true, hidden entries reappear (with a dim ``H`` marker) so the
        user can navigate to one and call ``hide_at_cursor()`` again to
        unhide it.  The hidden set is purely client-side state — it
        survives panel close/reopen but is dropped on TUI restart.
    """

    def __init__(
        self,
        toggle_key: Optional[KeyBinding] = None,
        open_file_key: Optional[KeyBinding] = None,
        diff_key: Optional[KeyBinding] = None,
        clear_key: Optional[KeyBinding] = None,
        paste_ref_key: Optional[KeyBinding] = None,
        hide_key: Optional[KeyBinding] = None,
        show_hidden_key: Optional[KeyBinding] = None,
        gitignore_key: Optional[KeyBinding] = None,
    ):
        """Initialize the workspace panel.

        Args:
            toggle_key: Keybinding used to toggle the panel.  Shown in the
                        footer hint.
            open_file_key: Keybinding used to open the selected file in an
                          external editor (the ``raw`` opener action).
                          Shown in the footer hint.
            diff_key: Keybinding used to open the selected file in an
                     external diff viewer (the ``diff`` opener action).
                     Shown in the footer hint.
            clear_key: Keybinding used to clear the file list.  Shown in the
                       footer hint.
            paste_ref_key: Keybinding used to paste the selected file or
                           directory as an ``@`` reference into the input
                           line.  Shown in the footer hint.
            hide_key: Keybinding used to hide / unhide the entry under the
                      cursor (per-session, not persisted).  Shown in the
                      footer hint.
            show_hidden_key: Keybinding used to toggle visibility of the
                             hidden set (so they can be unhidden).  Shown
                             in the footer hint.
            gitignore_key: Keybinding used to add / remove the cursor
                           entry from the workspace's ``.gitignore``.
                           Shown in the footer hint.
        """
        # File state: {relative_path: status_string}
        self._files: Dict[str, str] = {}

        # Tree built from _files for rendering
        self._root = _TreeNode("")

        # Popup state
        self._visible: bool = False
        self._scroll_offset: int = 0
        self._cursor_index: int = 0  # Index into the flattened visible list
        self._max_visible_lines: int = 20

        # Collapse state keyed by directory path (e.g., "server/")
        self._collapsed_dirs: Set[str] = set()

        # Highlight tracking: path → monotonic deadline
        self._highlights: Dict[str, float] = {}

        # Per-session hide state.  ``_hidden_paths`` stores the *exact*
        # entry identifier — directory paths end with ``/``, files do not.
        # ``_show_hidden`` is a render-only toggle: when False (default)
        # hidden entries are filtered out of ``_get_flat_entries``; when
        # True they reappear with a dim "H" marker so the user can
        # navigate to one and unhide it.
        self._hidden_paths: Set[str] = set()
        self._show_hidden: bool = False

        self._toggle_key = toggle_key or "c-w"
        self._open_file_key = open_file_key or "enter"
        self._diff_key = diff_key or "d"
        self._clear_key = clear_key or "delete"
        self._paste_ref_key = paste_ref_key or ["escape", "p"]
        self._hide_key = hide_key or "h"
        self._show_hidden_key = show_hidden_key or "H"
        self._gitignore_key = gitignore_key or "i"
        self._theme: Optional["ThemeConfig"] = None
        self._tree_dirty: bool = True  # Rebuild tree before next render

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def set_theme(self, theme: "ThemeConfig") -> None:
        """Set the theme configuration for styling.

        Args:
            theme: ThemeConfig instance for Rich style lookups.
        """
        self._theme = theme

    def _style(self, name: str) -> str:
        """Look up a semantic style from the theme, falling back to empty."""
        if self._theme:
            s = self._theme.get_rich_style(name)
            if s:
                return s
        return ""

    # ------------------------------------------------------------------
    # Data API (called from rich_client.py event handler)
    # ------------------------------------------------------------------

    def apply_snapshot(self, files: List[Dict[str, str]]) -> None:
        """Replace the entire file state from a snapshot event.

        Args:
            files: List of ``{"path": str, "status": str}`` dicts.
        """
        self._files.clear()
        self._highlights.clear()
        for entry in files:
            self._files[entry["path"]] = entry["status"]
        self._tree_dirty = True

    def apply_changes(self, changes: List[Dict[str, str]]) -> None:
        """Apply incremental changes from a changed event.

        Args:
            changes: List of ``{"path": str, "status": str}`` dicts.
        """
        now = time.monotonic()
        for entry in changes:
            path = entry["path"]
            status = entry["status"]
            if status == "deleted":
                # Keep deleted entries briefly for visual feedback, then remove.
                if path in self._files:
                    self._files[path] = "deleted"
                    self._highlights[path] = now + _HIGHLIGHT_DURATION
                else:
                    # Was not in our set (e.g., created-then-deleted) – skip.
                    continue
            else:
                self._files[path] = status
                self._highlights[path] = now + _HIGHLIGHT_DURATION
        self._tree_dirty = True

    def clear(self) -> None:
        """Clear all tracked files, resetting the panel to its initial state.

        After clearing, only new filesystem changes will appear.
        """
        self._files.clear()
        self._highlights.clear()
        self._collapsed_dirs.clear()
        self._cursor_index = 0
        self._scroll_offset = 0
        self._tree_dirty = True

    @property
    def file_count(self) -> int:
        """Number of created/modified files (excludes deleted)."""
        return sum(1 for s in self._files.values() if s != "deleted")

    @property
    def has_files(self) -> bool:
        """Whether there are any tracked files."""
        return bool(self._files)

    # ------------------------------------------------------------------
    # Popup visibility
    # ------------------------------------------------------------------

    @property
    def is_visible(self) -> bool:
        return self._visible

    def toggle_popup(self) -> None:
        """Toggle popup visibility."""
        self._visible = not self._visible
        if not self._visible:
            self._scroll_offset = 0
            self._cursor_index = 0

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def scroll_up(self) -> bool:
        """Move cursor up one line.

        Returns:
            True if the cursor moved.
        """
        if self._cursor_index > 0:
            self._cursor_index -= 1
            # Adjust scroll window if cursor is above visible area
            if self._cursor_index < self._scroll_offset:
                self._scroll_offset = self._cursor_index
            return True
        return False

    def scroll_down(self) -> bool:
        """Move cursor down one line.

        Returns:
            True if the cursor moved.
        """
        flat = self._get_flat_entries()
        if self._cursor_index < len(flat) - 1:
            self._cursor_index += 1
            # Adjust scroll window if cursor is below visible area
            if self._cursor_index >= self._scroll_offset + self._max_visible_lines:
                self._scroll_offset = self._cursor_index - self._max_visible_lines + 1
            return True
        return False

    def scroll_page_up(self) -> bool:
        """Move cursor up by a page.

        Returns:
            True if the cursor moved.
        """
        if self._cursor_index > 0:
            self._cursor_index = max(0, self._cursor_index - self._max_visible_lines)
            self._scroll_offset = max(0, self._scroll_offset - self._max_visible_lines)
            return True
        return False

    def scroll_page_down(self) -> bool:
        """Move cursor down by a page.

        Returns:
            True if the cursor moved.
        """
        flat = self._get_flat_entries()
        max_idx = len(flat) - 1
        if self._cursor_index < max_idx:
            self._cursor_index = min(max_idx, self._cursor_index + self._max_visible_lines)
            self._scroll_offset = min(
                max(0, len(flat) - self._max_visible_lines),
                self._scroll_offset + self._max_visible_lines,
            )
            return True
        return False

    def collapse_at_cursor(self) -> bool:
        """Collapse the directory at the current cursor position.

        If the cursor is on a file, collapses its parent directory.

        Returns:
            True if a directory was collapsed.
        """
        flat = self._get_flat_entries()
        if not flat or self._cursor_index >= len(flat):
            return False

        entry = flat[self._cursor_index]
        dir_path = entry.get("dir_path")

        if entry["is_dir"] and dir_path and dir_path not in self._collapsed_dirs:
            self._collapsed_dirs.add(dir_path)
            self._tree_dirty = True
            return True
        elif not entry["is_dir"] and entry.get("parent_dir"):
            # Collapse parent directory
            parent = entry["parent_dir"]
            if parent not in self._collapsed_dirs:
                self._collapsed_dirs.add(parent)
                self._tree_dirty = True
                return True
        return False

    def expand_at_cursor(self) -> bool:
        """Expand the directory at the current cursor position.

        Returns:
            True if a directory was expanded.
        """
        flat = self._get_flat_entries()
        if not flat or self._cursor_index >= len(flat):
            return False

        entry = flat[self._cursor_index]
        dir_path = entry.get("dir_path")

        if entry["is_dir"] and dir_path and dir_path in self._collapsed_dirs:
            self._collapsed_dirs.discard(dir_path)
            self._tree_dirty = True
            return True
        return False

    def get_selected_file_path(self) -> Optional[str]:
        """Return the path of the file at the current cursor position.

        Reconstructs the full path by combining the parent directory prefix
        with the file name.  Returns ``None`` if the cursor is on a directory
        or the list is empty.

        For workspace files, returns a relative path (e.g. ``"server/core.py"``).
        For sandbox-monitored files (tracked with absolute paths), returns
        an absolute path (e.g. ``"/home/user/data/config.json"``).

        Returns:
            File path (relative for workspace, absolute for sandbox), or None.
        """
        flat = self._get_flat_entries()
        if not flat or self._cursor_index >= len(flat):
            return None

        entry = flat[self._cursor_index]
        if entry["is_dir"]:
            return None

        parent = entry.get("parent_dir", "")
        return f"{parent}{entry['name']}"

    def get_selected_path(self) -> Optional[str]:
        """Return the path of the file *or directory* at the cursor.

        Like :meth:`get_selected_file_path` but also returns directory paths
        (with a trailing ``/``) so callers that don't care whether the entry
        is a file can use a single API.

        Returns:
            File path (relative for workspace, absolute for sandbox) or
            directory path with trailing ``/``, or ``None`` if the list is
            empty.
        """
        flat = self._get_flat_entries()
        if not flat or self._cursor_index >= len(flat):
            return None

        entry = flat[self._cursor_index]
        if entry["is_dir"]:
            return entry.get("dir_path")

        parent = entry.get("parent_dir", "")
        return f"{parent}{entry['name']}"

    # ------------------------------------------------------------------
    # Hide / show-hidden (per-session, client-side filter)
    # ------------------------------------------------------------------

    def _selected_entry_id(self) -> Optional[str]:
        """Return the canonical hide-set identifier for the cursor entry.

        Directories use their ``dir_path`` (with trailing ``/``); files use
        their relative path.  ``None`` if no valid entry is selected.
        """
        flat = self._get_flat_entries()
        if not flat or self._cursor_index >= len(flat):
            return None
        entry = flat[self._cursor_index]
        if entry["is_dir"]:
            return entry.get("dir_path")
        parent = entry.get("parent_dir", "")
        return f"{parent}{entry['name']}"

    def hide_at_cursor(self) -> Optional[str]:
        """Toggle the hidden state of the entry under the cursor.

        Returns the entry id (path) that was toggled, or ``None`` if the
        list is empty.  When ``_show_hidden`` is True the cursor may be on
        an already-hidden entry — calling this method unhides it.
        """
        entry_id = self._selected_entry_id()
        if not entry_id:
            return None
        if entry_id in self._hidden_paths:
            self._hidden_paths.discard(entry_id)
        else:
            self._hidden_paths.add(entry_id)
        # Keep the cursor in range after the visible set shrinks/grows.
        flat = self._get_flat_entries()
        if flat:
            self._cursor_index = min(self._cursor_index, len(flat) - 1)
        else:
            self._cursor_index = 0
        return entry_id

    def toggle_show_hidden(self) -> bool:
        """Toggle whether hidden entries are shown.

        Returns the new value.  When True, hidden entries appear with a
        dim ``H`` marker so the user can navigate to them and unhide.
        """
        self._show_hidden = not self._show_hidden
        return self._show_hidden

    @property
    def show_hidden(self) -> bool:
        return self._show_hidden

    @property
    def hidden_paths(self) -> Set[str]:
        return frozenset(self._hidden_paths)  # read-only view

    def _is_entry_hidden(self, entry: Dict[str, Any]) -> bool:
        """True if ``entry`` is hidden directly or under a hidden directory."""
        if not self._hidden_paths:
            return False
        if entry["is_dir"]:
            entry_id = entry.get("dir_path") or ""
        else:
            entry_id = f"{entry.get('parent_dir', '')}{entry['name']}"
        if entry_id in self._hidden_paths:
            return True
        # Descendant of a hidden directory.
        for hidden in self._hidden_paths:
            if hidden.endswith("/") and entry_id.startswith(hidden):
                return True
        return False

    def set_max_visible_lines(self, n: int) -> None:
        """Set maximum visible lines in the popup.

        Args:
            n: Number of lines (clamped to at least 5).
        """
        self._max_visible_lines = max(5, n)

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------

    def _rebuild_tree(self) -> None:
        """Rebuild the internal tree from ``self._files``."""
        now = time.monotonic()

        # Prune expired deleted entries
        expired = [
            p for p, s in self._files.items()
            if s == "deleted" and self._highlights.get(p, 0) < now
        ]
        for p in expired:
            del self._files[p]
            self._highlights.pop(p, None)

        # Prune expired highlights
        self._highlights = {
            p: t for p, t in self._highlights.items() if t > now
        }

        root = _TreeNode("")
        for path, status in sorted(self._files.items()):
            parts = path.replace("\\", "/").split("/")
            node = root
            for i, part in enumerate(parts):
                if part not in node.children:
                    node.children[part] = _TreeNode(part)
                node = node.children[part]
            node.status = status
            hl = self._highlights.get(path, 0)
            node.highlight_until = hl

        self._root = root
        self._tree_dirty = False

    def _get_flat_entries(self) -> List[Dict[str, Any]]:
        """Flatten the tree into a list of render entries.

        Each entry is a dict with:
            ``name``, ``depth``, ``is_dir``, ``status``, ``highlighted``,
            ``collapsed``, ``dir_path``, ``parent_dir``.

        Returns:
            Ordered list of entries for rendering.
        """
        if self._tree_dirty:
            self._rebuild_tree()

        entries: List[Dict[str, Any]] = []
        now = time.monotonic()

        def walk(node: _TreeNode, depth: int, path_prefix: str, parent_dir: str) -> None:
            # Sort: directories first, then files, alphabetical within each.
            dirs = []
            files = []
            for name, child in sorted(node.children.items()):
                if child.is_dir:
                    dirs.append((name, child))
                else:
                    files.append((name, child))

            for name, child in dirs:
                dir_path = f"{path_prefix}{name}/" if path_prefix else f"{name}/"
                is_collapsed = dir_path in self._collapsed_dirs
                dir_entry = {
                    "name": name,
                    "depth": depth,
                    "is_dir": True,
                    "status": None,
                    "highlighted": False,
                    "collapsed": is_collapsed,
                    "dir_path": dir_path,
                    "parent_dir": parent_dir,
                    "hidden": False,
                }
                hidden = self._is_entry_hidden(dir_entry)
                dir_entry["hidden"] = hidden
                if not hidden or self._show_hidden:
                    entries.append(dir_entry)
                # Skip walking into a hidden directory regardless of
                # _show_hidden — its presence in the list is enough for
                # the user to navigate to it and unhide.
                if not is_collapsed and not hidden:
                    walk(child, depth + 1, dir_path, dir_path)

            for name, child in files:
                file_entry = {
                    "name": name,
                    "depth": depth,
                    "is_dir": False,
                    "status": child.status,
                    "highlighted": child.highlight_until > now,
                    "collapsed": False,
                    "dir_path": None,
                    "parent_dir": parent_dir,
                    "hidden": False,
                }
                hidden = self._is_entry_hidden(file_entry)
                file_entry["hidden"] = hidden
                if not hidden or self._show_hidden:
                    entries.append(file_entry)

        walk(self._root, 0, "", "")
        return entries

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_popup(self, width: int = 50) -> Panel:
        """Render the workspace popup overlay.

        Args:
            width: Width of the popup panel.

        Returns:
            Rich Panel containing the directory tree.
        """
        count = self.file_count
        title = f"Workspace ({count} file{'s' if count != 1 else ''} created/modified)"

        border = self._style("workspace_popup_border") or "blue"
        bg = self._style("workspace_popup_background")

        if not self._files:
            return Panel(
                Text("No file changes in this session", style="dim"),
                title=f"[bold]{title}[/bold]",
                border_style=border,
                style=bg,
                width=width,
            )

        flat = self._get_flat_entries()
        total = len(flat)

        # Clamp cursor and scroll
        if total == 0:
            self._cursor_index = 0
            self._scroll_offset = 0
        else:
            self._cursor_index = min(self._cursor_index, total - 1)
            max_offset = max(0, total - self._max_visible_lines)
            self._scroll_offset = min(self._scroll_offset, max_offset)

        elements: List[Any] = []

        # "More above" indicator
        if self._scroll_offset > 0:
            above = Text()
            above.append(f" ↑ {self._scroll_offset} more above", style="dim")
            elements.append(above)

        # Visible entries
        start = self._scroll_offset
        end = min(start + self._max_visible_lines, total)
        visible = flat[start:end]

        # Calculate available width for filename (panel width - borders - indent - status label)
        inner_width = width - 4  # panel borders

        for i, entry in enumerate(visible):
            global_idx = start + i
            is_cursor = (global_idx == self._cursor_index)

            line = Text()
            indent = "  " * entry["depth"]
            is_hidden = entry.get("hidden", False)

            if entry["is_dir"]:
                # Directory line
                marker = "▸ " if entry["collapsed"] else "▾ " if entry.get("dir_path") else "  "
                dir_style = "reverse bold" if is_cursor else ""
                if is_hidden:
                    dir_style = (dir_style + " dim").strip()
                line.append(f" {indent}{marker}{entry['name']}/", style=dir_style)
                if is_hidden:
                    line.append(" H", style="dim")
            else:
                # File line
                status = entry.get("status", "")
                highlighted = entry.get("highlighted", False)

                # Pick style based on status
                if status == "created":
                    file_style = "bold green" if highlighted else "green"
                    label = "new"
                    label_style = "green"
                elif status == "modified":
                    file_style = "bold yellow" if highlighted else "yellow"
                    label = "mod"
                    label_style = "yellow"
                elif status == "deleted":
                    file_style = "strike dim red"
                    label = "del"
                    label_style = "red dim"
                else:
                    file_style = ""
                    label = ""
                    label_style = "dim"

                name_text = entry["name"]

                # Hidden entries override style with a dim wash so they
                # read as "filtered out" even when show_hidden is on.
                if is_hidden:
                    file_style = "dim"
                    label = "H"
                    label_style = "dim"

                # Cursor highlight: reverse video
                if is_cursor:
                    file_style = f"reverse {file_style}"

                prefix = f" {indent}    "
                # Right-align the status label
                name_width = inner_width - len(prefix) - len(label) - 2
                padded_name = name_text[:name_width].ljust(name_width)

                line.append(prefix)
                line.append(padded_name, style=file_style)
                if label:
                    line.append(f" {label}", style=label_style)

            elements.append(line)

        # "More below" indicator
        remaining = total - end
        if remaining > 0:
            below = Text()
            below.append(f" ↓ {remaining} more below", style="dim")
            elements.append(below)

        # Separator
        elements.append(Text("─" * (width - 4), style="dim"))

        # Footer is two lines:
        #   1. file count (+ hidden tally) on the left, primary action
        #      hints on the right ending with the close shortcut.
        #   2. secondary action hints (hide / show-hidden / gitignore /
        #      paste-as-@reference).  Splitting the hints over two rows
        #      keeps each line readable instead of overflowing on narrow
        #      panel widths.
        hidden_count = sum(1 for e in flat if e.get("hidden"))
        if self._show_hidden and hidden_count:
            tally = f" {count} file{'s' if count != 1 else ''} · {hidden_count} hidden shown"
        elif self._hidden_paths:
            # _hidden_paths can outlive the current snapshot, so the
            # length there doesn't necessarily match what's renderable.
            tally = f" {count} file{'s' if count != 1 else ''} ({len(self._hidden_paths)} hidden)"
        else:
            tally = f" {count} file{'s' if count != 1 else ''}"

        close_hint = f"[{format_key_for_display(self._toggle_key)} close]"
        open_hint = f"{format_key_for_display(self._open_file_key)} open"
        diff_hint = f"{format_key_for_display(self._diff_key)} diff"
        clear_hint = f"{format_key_for_display(self._clear_key)} clear"
        primary_hint = f"↑↓ · ◂▸ · {open_hint} · {diff_hint} · {clear_hint} "

        footer1 = Text()
        footer1.append(tally, style="dim")
        right1 = primary_hint + close_hint
        pad1 = max(1, width - 4 - len(tally) - len(right1))
        footer1.append(" " * pad1)
        footer1.append(primary_hint, style="dim")
        footer1.append(close_hint, style="dim")
        elements.append(footer1)

        hide_hint = f"{format_key_for_display(self._hide_key)} hide"
        show_hint = f"{format_key_for_display(self._show_hidden_key)} show"
        ignore_hint = f"{format_key_for_display(self._gitignore_key)} ignore"
        paste_hint = f"{format_key_for_display(self._paste_ref_key)} @ref"
        secondary_hint = (
            f"{hide_hint} · {show_hint} · {ignore_hint} · {paste_hint}"
        )
        footer2 = Text()
        pad2 = max(1, width - 4 - len(secondary_hint))
        footer2.append(" " * pad2)
        footer2.append(secondary_hint, style="dim")
        elements.append(footer2)

        return Panel(
            Group(*elements),
            title=f"[bold]{title}[/bold]",
            border_style=border,
            style=bg,
            width=width,
        )
