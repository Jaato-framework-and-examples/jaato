"""File editing plugin implementation.

Provides tools for reading, modifying, and managing files with
integrated permission approval (showing diffs) and automatic backups.
"""

import logging
import os
from shared.session_context import get_workspace_root, get_config_root
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

from shared.ui_utils import ellipsize_path, ellipsize_path_pair
from jaato_sdk.plugins.base import UserCommand, PermissionDisplayInfo
from jaato_sdk.plugins.model_provider.types import EditableContent, ToolSchema, TRAIT_FILE_WRITER, TRAIT_REPLAY_SAFE, DISCOVERABILITY_EAGER, DISCOVERABILITY_DEFERRED
from ..sandbox_utils import check_path_with_jaato_containment, detect_jaato_symlink
from ..path_safety import (
    move_verified,
    read_bytes_verified,
    read_text_verified,
    unlink_verified,
    write_text_verified,
)
from .backup import BackupManager
from .diff_utils import (
    generate_unified_diff,
    generate_new_file_diff,
    generate_delete_file_diff,
    generate_move_file_diff,
    summarize_diff,
    DEFAULT_MAX_LINES,
    DEFAULT_MAX_PATH_WIDTH,
)
from .line_endings import LineEndingPolicy, restore
from .multi_file import (
    MultiFileExecutor,
    MultiFileResult,
    generate_multi_file_diff_preview,
)
from .find_replace import (
    FindReplaceExecutor,
    FindReplaceResult,
    generate_find_replace_preview,
)
from .edit_core import apply_edit, EditNotFoundError, AmbiguousEditError
from shared.path_utils import msys2_to_windows_path, normalize_result_path
from shared.plugins.runner_forwarding import RunnerForwardingMixin
from shared.trace import trace as _trace_write


def _detect_workspace_root() -> Optional[str]:
    """Auto-detect workspace root from environment variables.

    Checks JAATO_WORKSPACE_ROOT first, then workspaceRoot.

    Returns:
        Absolute path to workspace root, or None if not configured.
    """
    workspace = get_workspace_root()
    if workspace:
        return os.path.realpath(os.path.abspath(workspace))
    workspace = os.environ.get('workspaceRoot')
    if workspace:
        return os.path.realpath(os.path.abspath(workspace))
    return None


def _detect_config_root() -> Optional[str]:
    """Auto-detect config root from per-task ContextVar / env.

    Mirrors :func:`_detect_workspace_root` but for the framework
    config root — the directory that holds ``profiles/`` /
    ``agents/`` / ``scripts/`` / ``logs/`` / ``sessions/`` / etc.
    Typically ``<workspace>/.jaato`` in interactive sessions, but
    can be a separate location (e.g. the kb's ``.jaato`` at repo
    root while the workspace is a sandbox/ subdir).

    Returns:
        Absolute path to config root, or None if not configured.
    """
    config_root = get_config_root()
    if config_root:
        return os.path.realpath(os.path.abspath(config_root))
    return None


class FileEditPlugin(RunnerForwardingMixin):
    """Plugin for file reading and editing operations.

    Tools provided:
    - readFile: Read file contents
    - updateFile: Modify existing file (shows diff preview, creates backup)
    - writeNewFile: Create new file
    - removeFile: Delete file (creates backup)
    - moveFile: Move/rename file (creates backup)
    - renameFile: Alias for moveFile (for discoverability)
    - undoFileChange: Restore from most recent backup

    Integrates with the permission system to show formatted diffs
    for file modifications.

    Path Sandboxing:
        When workspace_root is configured, file operations are restricted to
        paths within the workspace. Paths outside the workspace are only
        allowed if they've been authorized via the plugin registry (e.g.,
        by the references plugin for external documentation).
    """

    def __init__(self):
        self._backup_manager: Optional[BackupManager] = None
        self._initialized = False
        # Line-ending policy, shared by every write path so they cannot
        # drift apart.  Holds the git lookup caches, so it must outlive a
        # single tool call to be worth anything (jaato #805).
        self._line_endings = LineEndingPolicy()
        # Agent context for trace logging
        self._agent_name: Optional[str] = None
        # Workspace root for path sandboxing — drives which paths the
        # plugin's tools may read/write.  Set via config / detection at
        # initialize time; refreshed on ``set_workspace_path`` broadcast.
        self._workspace_root: Optional[str] = None
        # Config root for meta-state placement (backups, future per-
        # plugin caches).  Mirrors the established ``set_config_root``
        # hook other plugins (references, subagent, prompt_library,
        # template, service_connector) already implement.  Set via
        # config / detection at initialize time; refreshed on
        # ``set_config_root`` broadcast.
        self._config_root: Optional[str] = None
        # Session ID — cached so ``_reinit_backup_manager`` can rebuild
        # the session-scoped path when either anchor (config_root or
        # workspace_root fallback) changes via broadcast.
        self._session_id: Optional[str] = None
        # Explicit ``backup_dir`` from operator config (takes precedence
        # over derived paths).  Cached so broadcasts respect it.
        self._explicit_backup_dir: Optional[str] = None
        # Per-profile cap on the character span of a single targeted edit's
        # ``old``/``new`` arguments (``updateFile`` targeted mode).  ``None``
        # = unlimited (default).  Set via
        # ``plugin_configs.file_edit.max_edit_span_chars`` and read in
        # ``initialize``.  Enforced symmetrically (``old`` AND ``new``) by
        # ``_check_edit_span`` at the ``updateFile`` executor sites, and
        # advertised to the model through the dynamic tool description built
        # in ``get_tool_schemas`` (``_edit_span_hint``).  Full-file rewrites
        # are unaffected — they go through ``new_content`` (full-replacement
        # mode), which the cap intentionally does not touch.
        self._max_edit_span_chars: Optional[int] = None
        # Per-profile gate on ``updateFile``'s full-replacement mode
        # (``new_content`` — the old-absent branch).  ``True`` (default) keeps
        # whole-file replacement available; ``False`` removes it entirely for
        # this session — the ``new_content`` property is dropped from the
        # ``updateFile`` schema (model never sees it) AND a runtime attempt is
        # rejected with a guiding error steering to targeted edits.  Set via
        # ``plugin_configs.file_edit.allow_full_replace``.  Orthogonal to
        # ``max_edit_span_chars``: a constrained profile typically sets the cap
        # AND this gate together so a weak model has no whole-file path that
        # bypasses the cap (the clobber path — each step regenerating the file
        # and dropping prior edits).  See ``_check_edit_span`` /
        # ``_edit_span_hint`` for how the cap-rejection wording becomes
        # conditional on this gate.
        self._allow_full_replace: bool = True
        # Plugin registry for checking external path authorization
        self._plugin_registry = None

    @property
    def name(self) -> str:
        return "file_edit"

    @classmethod
    def get_apparmor_rules(
        cls,
        *,
        workspace_path: str,
        session_id: str,
        config_root: Optional[str],
        plugin_config: Dict[str, Any],
    ) -> List[str]:
        """Contribute file_edit's backup-path apparmor fragment.

        Server 0.6.130+ (PR-147): grants ``rw`` on the full
        ``<config_root>/sessions/`` subtree.  Two rules suffice via
        AppArmor specificity:

        - ``<config_root>/sessions/    rw,`` — the dir entry itself
          (covers mkdir of ``sessions/`` from the framework template's
          read-only ``<config_root>/** r,`` baseline).
        - ``<config_root>/sessions/**  rw,`` — all descendants
          (``<session_id>/``, ``<session_id>/backups/``, all files
          inside).

        Both rules are more specific than ``<config_root>/** r,`` from
        the framework template, so they win via AppArmor's most-
        specific-match resolution.  Together they cover the entire
        mkdir chain for ``<config_root>/sessions/<session_id>/backups/``
        plus all writes inside the backup directory.

        **History.**  PR-145 emitted leaf-only rules
        (``<config_root>/sessions/<session_id>/backups/ rw,``) which
        failed because the rule didn't grant mkdir of the parent
        ``sessions/`` directory.  v126 evidence: PermissionError on
        ``<config_root>/sessions`` — the parent was never grantable.
        PR-147 fixes the fragment shape to grant the full subtree.

        **Workspace branch dropped** (server 0.6.130+).  Per Daniel's
        framing: framework + plugins are confined to config_root;
        workspace is tenant territory.  file_edit's framework-side
        backup writes go in config_root, never in workspace.  The
        ``workspace_path`` parameter is kept on the signature for
        compatibility with the apparmor composer's contract, but
        emitted rules don't reference it.

        Operator-explicit ``plugin_config["backup_dir"]`` override
        is honored — emits a grant for whatever path the operator
        chose (their responsibility to ensure the path is reachable
        under the active apparmor confinement).

        See ``memory/plugin.py:101-132`` for the canonical pattern
        these rules follow.  See
        ``project_backlog_plugin_apparmor_rules_audit`` memory for
        the broader plugin-fragment audit context.
        """
        rules: List[str] = []

        def _add_subtree(base: str) -> None:
            base = base.rstrip("/")
            rules.append(f"{base}/    rw,")
            rules.append(f"{base}/**  rw,")

        # Operator override wins.  Grant whatever the operator
        # supplied; they own ensuring the apparmor template covers
        # paths outside the framework's standard config_root tree.
        explicit_dir = plugin_config.get("backup_dir") if plugin_config else None
        if isinstance(explicit_dir, str) and explicit_dir.strip():
            _add_subtree(explicit_dir.strip())

        # Framework-side backups go under config_root/sessions/.  Grant
        # the full subtree (NOT just the leaf) so the mkdir chain
        # resolves: <config_root>/sessions/ → <session_id>/ → backups/
        # → files inside.
        if config_root:
            _add_subtree(f"{config_root}/sessions")

        return rules

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        _trace_write("FILE_EDIT", msg)

    @staticmethod
    def _parse_edit_span_cap(raw: Any) -> Optional[int]:
        """Validate the ``max_edit_span_chars`` config value.

        Returns the cap as a positive ``int`` when *raw* is a valid positive
        integer (accepting ``bool``-free numeric strings too, since profile
        loaders may pass strings).  Returns ``None`` (unlimited) when *raw* is
        absent.  For any malformed value (non-numeric, ``<= 0``, ``bool``) it
        logs a WARNING and returns ``None`` — the knob disables itself rather
        than clamping to a hardcoded default, keeping behaviour deterministic
        and predictable for the operator who set it.
        """
        if raw is None:
            return None
        # Reject bool explicitly — bool is an int subclass and ``True``/``False``
        # as a span cap is always an operator mistake.
        if isinstance(raw, bool):
            logger.warning(
                "file_edit: ignoring max_edit_span_chars=%r (bool is not a valid "
                "span cap); treating as unlimited.", raw
            )
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "file_edit: ignoring max_edit_span_chars=%r (not an integer); "
                "treating as unlimited.", raw
            )
            return None
        if value <= 0:
            logger.warning(
                "file_edit: ignoring max_edit_span_chars=%d (must be a positive "
                "integer); treating as unlimited.", value
            )
            return None
        return value

    @staticmethod
    def _parse_allow_full_replace(raw: Any) -> bool:
        """Validate the ``allow_full_replace`` config value (default ``True``).

        Accepts a real ``bool`` directly, or the common YAML/string spellings
        (``"true"``/``"false"``/``"1"``/``"0"``/``"yes"``/``"no"``/``"on"``/
        ``"off"``, case-insensitive).  An absent value defaults to ``True``
        (back-compatible: full replacement stays available).  Any unrecognised
        value logs a WARNING and falls back to ``True`` (the safe, non-
        restrictive default — a typo must not silently strip a tool mode).
        """
        if raw is None:
            return True
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            token = raw.strip().lower()
            if token in ("true", "1", "yes", "on"):
                return True
            if token in ("false", "0", "no", "off"):
                return False
        logger.warning(
            "file_edit: ignoring allow_full_replace=%r (expected a boolean); "
            "defaulting to True (full replacement available).", raw
        )
        return True

    def _full_replace_gate_error(self) -> str:
        """Guiding error returned when full replacement is attempted but gated.

        Steers the model to targeted edits — the only edit path available on a
        profile that has set ``allow_full_replace: false``.
        """
        return (
            "Full-file replacement ('new_content') is disabled for this profile. "
            "Make a targeted edit instead: 'old' (the exact current text, minimal) "
            "→ 'new'. Try a unique 'old' alone first; add 'prologue'/'epilogue' "
            "(literal adjacent lines, copied verbatim) only if the tool reports "
            "the match ambiguous."
        )

    @staticmethod
    def _full_replacement_content(args: Dict[str, Any]) -> Optional[str]:
        """``new_content``, then the ``content`` alias, or ``None`` if absent.

        A method rather than two inline checks because ``_execute_update_file``
        sits at the complexity ceiling and is frozen in the audit baseline —
        the same reason the caller reads as one branch.  ``None`` means NOT
        SUPPLIED; an empty string is a deliberate truncation and must survive.
        """
        content = args.get("new_content")
        return args.get("content") if content is None else content

    def _missing_content_error(self) -> str:
        """Guiding error for a full-replacement call that names no content.

        'new_content' absent entirely is a malformed call, not a request for
        an empty file: absent and empty must not share a representation, or
        a call that merely forgot its payload silently truncates the file.
        An explicit ``new_content: ""`` remains the way to truncate
        deliberately.  Names both modes so the model can self-correct.
        """
        return (
            "updateFile needs content to apply: provide 'old' (the exact "
            "current text, minimal) → 'new' for a targeted edit, or "
            "'new_content' for full-file replacement. To truncate the file "
            "deliberately, pass 'new_content' as an empty string (\"\")."
        )

    def _check_edit_span(self, old: Optional[str], new: Optional[str]) -> Optional[str]:
        """Enforce the targeted-edit span cap on ``old`` and ``new``.

        Applied to each argument independently (capping only one leaves the
        other as an escape hatch).  Returns a model-facing guiding error string
        when either exceeds ``self._max_edit_span_chars``, else ``None``.  When
        the cap is unset (``None``) this is always a no-op.

        The error names the offending side(s), then teaches the correct
        disambiguation primitive — minimal ``old``/``new`` pinned by
        ``prologue``/``epilogue`` context, NOT a widened ``old`` — so the model
        self-corrects via the tool contract, not persona prose.  The whole-file
        remedy (omit ``old``, use ``new_content``) is appended ONLY when
        ``self._allow_full_replace`` is true; on a gated profile that remedy is
        both unavailable and the clobber path, so it is suppressed.
        """
        cap = self._max_edit_span_chars
        if cap is None:
            return None
        old_len = len(old or "")
        new_len = len(new or "")
        over: List[str] = []
        if old_len > cap:
            over.append(f"'old' is {old_len} chars")
        if new_len > cap:
            over.append(f"'new' is {new_len} chars")
        if not over:
            return None
        msg = (
            f"Targeted edit rejected: {' and '.join(over)}; 'old' and 'new' must "
            f"each be ≤ {cap} characters. Shrink them to the changed text only — "
            f"'old' is the minimal current text you are replacing, usually a "
            f"single already-unique line (a signature/import/package line). Try "
            f"'old' ALONE; add 'prologue'/'epilogue' only if the tool then reports "
            f"the match ambiguous, copying the literal adjacent lines verbatim. "
            f"Never widen 'old' itself to disambiguate — locator size and edit "
            f"size are independent axes."
        )
        if self._allow_full_replace:
            msg += (
                " To rewrite the entire file, OMIT 'old' and provide only "
                "'new_content' (it is ignored whenever 'old' is present)."
            )
        return msg

    def _edit_span_hint(self) -> str:
        """Return the dynamic schema suffix advertising the span cap.

        Empty string when the cap is unset, so the static tool description is
        unchanged.  When set, the returned sentence is appended to the
        ``updateFile`` description (and a shorter variant to the ``old``/``new``
        property descriptions) in :meth:`get_tool_schemas`, so the constraint
        reaches the model through the tool contract — never the persona.
        Computed at schema-build time from the current cap value.
        """
        cap = self._max_edit_span_chars
        if cap is None:
            return ""
        hint = (
            f" IMPORTANT: in targeted-edit mode, 'old' and 'new' must each be "
            f"≤ {cap} characters — the changed text only. Use a unique 'old' "
            f"alone; add 'prologue'/'epilogue' (literal adjacent lines, verbatim) "
            f"only when the match is reported ambiguous, never to widen 'old'."
        )
        if self._allow_full_replace:
            hint += (
                " To replace the entire file, OMIT 'old' and provide only "
                "'new_content'."
            )
        return hint

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the file edit plugin.

        Creates backup directory and ensures .jaato is in .gitignore.

        Args:
            config: Optional configuration dict with:
                - backup_dir: Custom backup directory (default: .jaato/backups
                              or .jaato/sessions/{session_id}/backups if session_id provided)
                - session_id: Session identifier for session-scoped backup storage.
                              When provided, backups are stored per-session to align
                              with waypoint session scoping.
                - workspace_root: Path to workspace root for sandboxing.
                                  Auto-detected from JAATO_WORKSPACE_ROOT or
                                  workspaceRoot env vars if not specified.
                - max_edit_span_chars: Per-profile cap on the character length
                                  of a single targeted edit's ``old`` and
                                  ``new`` arguments (``updateFile`` targeted
                                  mode), applied to each independently.  Must
                                  be a positive int; ``None``/absent = unlimited
                                  (default).  Drives both the runtime rejection
                                  (``_check_edit_span``) and the dynamic tool
                                  description (``_edit_span_hint``).  Does NOT
                                  cap ``new_content`` (full-replacement mode) —
                                  whole-file rewrites stay legal through that
                                  argument (see ``allow_full_replace`` to gate
                                  that path).
                - allow_full_replace: Bool (default ``True``).  When ``False``,
                                  ``updateFile``'s whole-file replacement mode
                                  (``new_content``) is removed for this session —
                                  dropped from the tool schema and rejected at
                                  runtime — leaving targeted ``old``/``new`` edits
                                  as the only path.  A constrained profile sets
                                  this alongside ``max_edit_span_chars`` so a weak
                                  model has no whole-file path that bypasses the
                                  cap (the clobber path).
        """
        config = config or {}

        # Extract agent name for trace logging
        self._agent_name = config.get("agent_name")

        # Configure workspace root for path sandboxing
        workspace_root = config.get("workspace_root")
        if workspace_root:
            self._workspace_root = os.path.realpath(os.path.abspath(workspace_root))
        else:
            self._workspace_root = _detect_workspace_root()

        # Configure config root for backup-path anchoring.  PR-143
        # mistakenly anchored backups on ``workspace_root`` —
        # backups are jaato-meta-state and belong with other
        # meta-state (``.jaato/logs/``, ``.jaato/cache/`` etc.)
        # under ``config_root``, NOT polluting the workspace with
        # a second ``.jaato/`` tree.  In typical interactive use
        # ``config_root == <workspace>/.jaato`` so the bug was
        # invisible; in the handoff_test pattern (config_root at
        # repo root, sandbox/ inside as workspace) the misdirection
        # surfaces.  See PR-144 commit message + v122 retrospective.
        config_root = config.get("config_root")
        if config_root:
            self._config_root = os.path.realpath(os.path.abspath(config_root))
        else:
            self._config_root = _detect_config_root()

        # Cache derivation inputs so subsequent broadcasts
        # (set_config_root / set_workspace_path) can rebuild the
        # backup-manager path without losing operator-supplied state.
        self._explicit_backup_dir = config.get("backup_dir")
        self._session_id = config.get("session_id")

        # Per-profile targeted-edit span cap.  Validated to a positive int;
        # any other value (incl. negative/zero/non-numeric) is rejected with a
        # WARNING and treated as unset (unlimited) — deterministic, no silent
        # clamping to an arbitrary default.
        self._max_edit_span_chars = self._parse_edit_span_cap(
            config.get("max_edit_span_chars")
        )

        # Full-replacement gate.  Defaults True (back-compatible — every
        # existing profile keeps new_content).  A profile opts out with
        # ``allow_full_replace: false`` to forbid whole-file replacement.
        self._allow_full_replace = self._parse_allow_full_replace(
            config.get("allow_full_replace")
        )

        self._reinit_backup_manager()

        # Ensure .jaato is in .gitignore
        self._ensure_gitignore()

        self._initialized = True
        backup_dir_str = str(self._backup_manager._base_dir) if self._backup_manager else "none"
        workspace_str = self._workspace_root or "none"
        config_root_str = self._config_root or "none"
        self._trace(
            f"initialize: backup_dir={backup_dir_str}, "
            f"workspace_root={workspace_str}, "
            f"config_root={config_root_str}"
        )

    def _resolve_backup_base_dir(self) -> Optional[Path]:
        """Resolve the backup base directory from current anchor state.

        Server 0.6.130+ (PR-147): backups go under ``config_root``
        only.  Per the framework architectural rule (Daniel,
        2026-05-19): framework + plugins are confined to
        ``config_root``; ``workspace`` is tenant territory and must
        not receive framework-side writes.

        Priority:
          1. ``self._explicit_backup_dir`` (operator override —
             operator owns ensuring the apparmor profile covers it).
          2. ``<config_root>/sessions/<session_id>/backups``
             (session-scoped — the typical path).
          3. ``<config_root>/backups`` (when no session_id).
          4. ``None`` — config_root unset AND no operator override.
             Caller (``_reinit_backup_manager``) treats this as a
             configuration error and raises.

        The pre-PR-147 workspace-fallback branches were dropped: per
        the framework rule, falling back to ``<workspace>/.jaato/...``
        would write tenant territory for a framework-side concern.
        Instead, the framework requires config_root to be set
        (PluginRegistry pre-init injection ensures this in normal
        operation; cf. PR-146 layer fix).
        """
        if self._explicit_backup_dir:
            return Path(self._explicit_backup_dir)
        if self._config_root:
            base = Path(self._config_root)
            if self._session_id:
                return base / "sessions" / self._session_id / "backups"
            return base / "backups"
        return None

    def _reinit_backup_manager(self) -> None:
        """Rebuild ``self._backup_manager`` from current anchor state.

        Called from :meth:`initialize` and from the
        :meth:`set_config_root` / :meth:`set_workspace_path`
        broadcast handlers so a config-root change after plugin init
        actually moves the backup root — not just the in-memory copy
        of the value.

        Raises:
            RuntimeError: when ``_resolve_backup_base_dir`` returns
                ``None`` (no config_root resolved AND no operator
                override).  This is a configuration error — the
                operator must set ``config_root`` in plugin config
                or via :func:`shared.session_context.set_config_root`.
                Pre-PR-147 a workspace-fallback masked this; per
                Daniel's "framework confined to config_root, workspace
                belongs to tenant" rule, falling back to workspace
                is incorrect.
        """
        base = self._resolve_backup_base_dir()
        if base is None:
            raise RuntimeError(
                "file_edit: cannot resolve backup base directory — "
                "no config_root set AND no operator override "
                "(plugin_config['backup_dir']) supplied.  The framework "
                "writes backups under <config_root>/sessions/<id>/backups/; "
                "set config_root in plugin config or ensure "
                "PluginRegistry.set_config_root has fired before "
                "expose_all (cf. PR-146).",
            )
        self._backup_manager = BackupManager(base)

        # Log .jaato symlink detection for visibility
        if self._workspace_root:
            is_symlink, target = detect_jaato_symlink(self._workspace_root)
            if is_symlink:
                self._trace(f"initialize: .jaato is symlink -> {target} (contained escape enabled)")

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self._trace("shutdown: cleaning up")
        self._backup_manager = None
        self._initialized = False
        self._workspace_root = None
        self._config_root = None
        self._session_id = None
        self._explicit_backup_dir = None
        self._plugin_registry = None

    def reset_for_next_session(self) -> None:
        """Cascade-sharing reset — NO-OP for this plugin.

        Phase 1 hotfix (server 0.6.148+): added to satisfy the
        ``ToolPlugin`` / ``EnrichmentPlugin`` protocol's runtime
        ``isinstance`` check.  Per Daniel's litmus test (see
        ``docs/design/runner-cascade-sharing.md`` §4.3), this
        plugin holds no per-session state that the next cascade
        session would benefit from having cleared.  Override in
        future PRs if the litmus test changes.
        """
        pass


    def set_plugin_registry(self, registry) -> None:
        """Set the plugin registry for checking external path authorization.

        Args:
            registry: The PluginRegistry instance.
        """
        self._plugin_registry = registry
        registry.register_category("filesystem", "Read, write, search, and navigate files and directories")
        self._trace("set_plugin_registry: registry set")

    @property
    def backup_manager(self) -> Optional[BackupManager]:
        """Get the backup manager instance.

        This provides access to the backup infrastructure for other plugins
        that need to integrate with file backup functionality (e.g., waypoint).

        Returns:
            The BackupManager instance, or None if not initialized.
        """
        return self._backup_manager

    def set_workspace_path(self, path: Optional[str]) -> None:
        """Update the workspace root path.

        Called when a client connects with a different working directory.
        Refreshes the backup manager so the new path actually takes
        effect (broadcast updates the in-memory value AND the
        backup-root anchor; pre-fix update only touched the value).

        Args:
            path: The new workspace root path, or None to disable sandboxing.
        """
        if path:
            self._workspace_root = os.path.realpath(os.path.abspath(path))
        else:
            self._workspace_root = None
        self._trace(f"set_workspace_path: workspace_root={self._workspace_root}")
        if self._initialized:
            self._reinit_backup_manager()

    def set_config_root(self, path: Optional[str]) -> None:
        """Update the config root path.

        Broadcast by :meth:`shared.plugins.registry.PluginRegistry.set_config_root`
        after a session resolves its config_root (workspace ``.jaato/``
        in typical use, or an explicit override from
        :class:`ClientConfigRequest.config_root`).  Mirrors the
        established hook on the 5 sibling plugins (references,
        subagent, prompt_library, template, service_connector).
        Refreshes the backup manager so backups follow the new
        anchor — backup root is meta-state and belongs with the
        rest of the framework's meta-state under config_root, NOT
        in the workspace.

        Args:
            path: The new config root path, or None to clear.
        """
        if path:
            self._config_root = os.path.realpath(os.path.abspath(path))
        else:
            self._config_root = None
        self._trace(f"set_config_root: config_root={self._config_root}")
        if self._initialized:
            self._reinit_backup_manager()

    def _is_path_allowed(self, path: str, mode: str = "read") -> bool:
        """Check if a path is allowed for access.

        A path is allowed if:
        1. No workspace_root is configured (sandboxing disabled)
        2. The path is within the workspace_root
        3. The path is under .jaato and within the .jaato containment boundary
           (see sandbox_utils.py for .jaato contained symlink escape rules)
        4. The path is authorized via the plugin registry (respecting access mode)

        Args:
            path: Path to check.
            mode: Access mode - "read" or "write" (default: "read").
                 Used to check if externally authorized paths have sufficient
                 access (e.g., "readonly" paths block write operations).

        Returns:
            True if access is allowed, False otherwise.
        """
        # If no workspace_root, allow all paths
        if not self._workspace_root:
            return True

        # Resolve path relative to workspace first
        resolved = self._resolve_path(path)
        abs_path = str(resolved.absolute())

        # Use shared sandbox utility with .jaato containment support
        allowed = check_path_with_jaato_containment(
            abs_path,
            self._workspace_root,
            self._plugin_registry,
            mode=mode
        )

        if not allowed:
            self._trace(f"_is_path_allowed: {path} blocked (outside sandbox, mode={mode})")
        return allowed

    def _sandbox_validator(self, mode: str) -> Callable[[str], bool]:
        """Bind :meth:`_is_path_allowed` for the ``path_safety`` primitives.

        The helpers in ``shared.plugins.path_safety`` resolve a path once,
        hand the canonical form to this callback, and then prove the object
        they opened is the one the callback approved.  That ordering is what
        closes the check-then-open window: calling ``_is_path_allowed(path)``
        and then opening ``path`` separately validates one object and acts on
        another whenever a symlink is swapped in between (jaato issue #669,
        shape A).

        Args:
            mode: ``"read"`` or ``"write"`` — the access being requested,
                so ``readonly`` authorized paths still block writes.

        Returns:
            A predicate over canonical paths, suitable as the ``validate``
            argument of the ``*_verified`` helpers.
        """
        return lambda resolved: self._is_path_allowed(resolved, mode=mode)

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path, making relative paths relative to workspace_root.

        This ensures that when a client specifies a relative path like
        'customer-domain-api/pom.xml', it resolves to the client's workspace
        directory rather than the server's current working directory.

        Under MSYS2, also converts /c/Users/... paths to C:/Users/... so
        that Python can resolve them via Windows APIs.

        Args:
            path: Path string (absolute or relative, Windows or MSYS2 format).

        Returns:
            Resolved Path object. Relative paths are resolved against
            workspace_root if configured, otherwise against CWD.
        """
        # Convert MSYS2 drive paths (/c/...) to Windows (C:/...) for Python
        path = msys2_to_windows_path(path)
        p = Path(path)
        if p.is_absolute():
            return p
        if self._workspace_root:
            resolved = Path(self._workspace_root) / p
            self._trace(f"_resolve_path: {path} -> {resolved} (relative to workspace)")
            return resolved
        return p

    @staticmethod
    def _is_jaato_rule(line: str) -> bool:
        """True if a .gitignore line is ANY rule anchored at ``.jaato`` — bare,
        ``/``-anchored, ``/*`` / ``/**`` globbed, a ``.jaato/<sub>`` path, or a
        ``!``-negation thereof.  Used to detect that the user is already managing
        ``.jaato`` so we never auto-append a redundant (or negation-defeating)
        rule on top of a deliberate setup."""
        s = line.strip()
        if not s or s.startswith("#"):
            return False
        s = s.lstrip("!").lstrip("/")            # drop negation marker + anchor
        if s.startswith(".jaato/"):
            return True
        return s.rstrip("*").rstrip("/") == ".jaato"

    def _ensure_gitignore(self) -> None:
        """Add a ``.jaato/`` ignore to the WORKSPACE .gitignore — but only if it
        exists and has NO existing ``.jaato``-anchored rule.

        - Anchored to the session workspace (``get_workspace_root``), not the
          process cwd, so it never edits an unrelated repo's .gitignore when the
          daemon was launched outside the workspace.
        - Skips entirely when ANY ``.jaato`` rule is already present, so a
          deliberate setup (e.g. ``.jaato/*`` + ``!.jaato/profiles/``) is not
          fought every session (dani's principle: no auto-mods over a
          deterministic user setup).
        - Appends the directory form ``.jaato/``; a bare ``.jaato`` placed after a
          ``!``-negation re-excludes the whole dir and silently un-tracks future
          profile/agent files.
        """
        workspace = get_workspace_root()
        if not workspace:
            return
        gitignore = Path(workspace) / ".gitignore"
        if not gitignore.exists():
            return

        try:
            # No sandbox validator here — this is the framework reading its own
            # workspace's .gitignore, not a model-supplied path.  The verified
            # reader is still used for its special-file guard: a hostile repo
            # shipping .gitignore as a FIFO would otherwise hang session start.
            content = read_text_verified(gitignore)
            lines = content.splitlines()

            # Any existing .jaato-anchored rule → the user is managing it; leave
            # it alone (the narrow exact-match check used to miss `.jaato/*` and
            # re-append a bare `.jaato` every session).
            if any(self._is_jaato_rule(l) for l in lines):
                return

            with gitignore.open("a", encoding="utf-8") as f:
                # Add newline if file doesn't end with one
                if content and not content.endswith("\n"):
                    f.write("\n")
                f.write(".jaato/\n")
        except OSError as exc:
            # If we can't read/write gitignore, just log and skip
            logger.debug(f"Failed to update .gitignore: {exc}")

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for file editing tools.

        The ``updateFile`` description and its ``old``/``new`` property
        descriptions are computed at build time:
        - when a per-profile ``max_edit_span_chars`` cap is set, the span
          constraint is appended so the model learns the limit through the
          tool contract (see :meth:`_edit_span_hint`);
        - when ``allow_full_replace`` is false, the ``new_content`` property
          and its description clause are dropped entirely, so the model never
          sees the whole-file-replacement mode on a gated profile.
        When neither knob is set, the schema is unchanged from the static form.
        """
        # Dynamic span-cap advertisement (empty string when no cap is set).
        span_hint = self._edit_span_hint()
        span_prop_hint = (
            f" Must be ≤ {self._max_edit_span_chars} characters."
            if self._max_edit_span_chars is not None
            else ""
        )
        # updateFile description is computed at build time: the full-replacement
        # mode is advertised only when ``allow_full_replace`` is true, so a gated
        # profile's model never sees 'new_content' (the property is also dropped
        # from ``parameters`` below).
        # Canonical targeted-edit guidance — the tool contract is the home for
        # these anchoring rules (every updateFile caller, not just personas).
        # Evidence (transform-stage diff, 36 calls): a unique 'old' alone
        # matched 15/15; supplying prologue/epilogue failed ~9/10 because models
        # build them from semantic landmarks that are NOT verbatim-contiguous
        # with 'old', so prologue+old+epilogue isn't a real substring.  The two
        # sizing axes are independent and must not be conflated: the EDIT shrinks
        # to the changed text; the LOCATOR grows only as needed for uniqueness.
        targeted_clause = (
            "Targeted edit: 'old' (the exact current text) → 'new' (its "
            "replacement). Two INDEPENDENT rules, do not conflate them: the EDIT "
            "('old'/'new') is the changed text only, minimal; the LOCATOR "
            "('prologue'+'old'+'epilogue') is sized for UNIQUENESS. Try 'old' "
            "ALONE first — a method signature, import, or package line is almost "
            "always already unique. Add 'prologue'/'epilogue' ONLY if the tool "
            "reports the match is ambiguous: copy the LITERAL adjacent lines "
            "verbatim (blank lines included), extending outward until the match "
            "is unique. Never add them pre-emptively or from memory — "
            "'prologue'+'old'+'epilogue' must be an exact substring of the file."
        )
        if self._allow_full_replace:
            update_desc = (
                "Update an existing file. Two modes. (1) " + targeted_clause
                + " (2) Full replacement: provide 'new_content' to replace the "
                "entire file."
                + span_hint
            )
        else:
            update_desc = (
                "Update an existing file (targeted edit only). " + targeted_clause
                + " (Whole-file replacement via 'new_content' is disabled for "
                "this profile.)"
                + span_hint
            )
        return [
            ToolSchema(
                name="readFile",
                description="Read the contents of a file. ALWAYS use this instead of `cat`, `head`, "
                           "`tail`, or `less` CLI commands. Returns file content as text with proper "
                           "encoding handling and structured metadata. Supports chunked reading with "
                           "offset and limit parameters for large files. For image files (PNG, JPG, GIF, "
                           "WebP, BMP, ICO, SVG), returns the image as multimodal content that you can view.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Line number to start reading from (1-indexed). Default is 1 (start of file)."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of lines to read. If not specified, reads the entire file."
                        }
                    },
                    "required": ["path"]
                },
                category="filesystem",
                discoverability=DISCOVERABILITY_EAGER,
                traits=frozenset({TRAIT_REPLAY_SAFE}),
            ),
            ToolSchema(
                name="updateFile",
                description=update_desc,
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to update"
                        },
                        "old": {
                            "type": "string",
                            "description": (
                                "The exact current text to replace — the changed "
                                "text only, minimal. Copied verbatim from the file. "
                                "Try it ALONE first (a signature/import/package line "
                                "is usually already unique); only if the tool reports "
                                "the match ambiguous, add prologue/epilogue."
                                + span_prop_hint
                            )
                        },
                        "new": {
                            "type": "string",
                            "description": (
                                "Replacement for 'old' — the changed text only. "
                                "Replaces just the matched 'old' text."
                                + span_prop_hint
                            )
                        },
                        "prologue": {
                            "type": "string",
                            "description": (
                                "Optional disambiguator — add ONLY when 'old' alone "
                                "matched ambiguously. The LITERAL line(s) immediately "
                                "before 'old', copied verbatim from the file (blank "
                                "lines included), extended outward until "
                                "prologue+old+epilogue is unique. Never invent from "
                                "memory. Not modified in the output."
                            )
                        },
                        "epilogue": {
                            "type": "string",
                            "description": (
                                "Optional disambiguator — add ONLY when 'old' alone "
                                "matched ambiguously. The LITERAL line(s) immediately "
                                "after 'old', copied verbatim from the file (blank "
                                "lines included), extended outward until unique. "
                                "Never invent from memory. Not modified in the output."
                            )
                        },
                        # Full-replacement arg — present only when the profile
                        # allows whole-file replacement.  Dropped from the schema
                        # entirely on a gated profile so the model never attempts
                        # it (the runtime gate in _execute_update_file is the
                        # backstop for models that send it anyway).
                        **({"new_content": {
                            "type": "string",
                            "description": (
                                "The complete new content to write to the file "
                                "(for full replacement mode). "
                                "Provide the raw file content directly - do NOT wrap in quotes, "
                                "triple-quotes, or treat as a string literal. The content is "
                                "written verbatim to the file."
                            )
                        }} if self._allow_full_replace else {})
                    },
                    "required": ["path"]
                },
                category="filesystem",
                discoverability=DISCOVERABILITY_DEFERRED,
                editable=EditableContent(
                    parameters=["old", "new", "new_content"],
                    format="diff",
                ),
                traits=frozenset({TRAIT_FILE_WRITER, TRAIT_REPLAY_SAFE}),
            ),
            ToolSchema(
                name="writeNewFile",
                description="Create a new file with the specified content. "
                           "Fails if the file already exists. "
                           "NOTE: This tool is for NON-TEMPLATED files only. "
                           "If a matching template exists (check listAvailableTemplates), "
                           "you MUST use renderTemplateToFile instead — do NOT read "
                           "a .tpl file and pass its content here.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path where the new file should be created"
                        },
                        "content": {
                            "type": "string",
                            "description": (
                                "The content to write to the new file. "
                                "Provide the raw file content directly - do NOT wrap in quotes, "
                                "triple-quotes, or treat as a string literal. For example, for a "
                                "Python file, start directly with 'import ...' or code, not with "
                                "'''...''' or quotes around the code. The content is written "
                                "verbatim to the file."
                            )
                        }
                    },
                    "required": ["path", "content"]
                },
                category="filesystem",
                discoverability=DISCOVERABILITY_DEFERRED,
                editable=EditableContent(
                    parameters=["content"],
                    format="text",
                    template="# Edit the file content below. Save and exit to continue.\n",
                ),
                traits=frozenset({TRAIT_FILE_WRITER, TRAIT_REPLAY_SAFE}),
            ),
            ToolSchema(
                name="removeFile",
                description="Delete a file. Creates a backup before deletion so it can be restored "
                           "with undoFileChange if needed.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to delete"
                        }
                    },
                    "required": ["path"]
                },
                category="filesystem",
                discoverability=DISCOVERABILITY_DEFERRED,
                traits=frozenset({TRAIT_REPLAY_SAFE}),
            ),
            ToolSchema(
                name="moveFile",
                description="Move or rename a file. ALWAYS use this instead of `mv` CLI command. "
                           "Creates destination directories if needed and creates a backup before "
                           "moving. Fails if destination already exists unless overwrite=True.",
                parameters={
                    "type": "object",
                    "properties": {
                        "source_path": {
                            "type": "string",
                            "description": "Path to the source file to move"
                        },
                        "destination_path": {
                            "type": "string",
                            "description": "Path where the file should be moved to"
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": "If True, overwrite destination if it exists. Default is False."
                        }
                    },
                    "required": ["source_path", "destination_path"]
                },
                category="filesystem",
                discoverability=DISCOVERABILITY_DEFERRED,
                traits=frozenset({TRAIT_REPLAY_SAFE}),
            ),
            ToolSchema(
                name="renameFile",
                description="Rename a file (alias for moveFile). ALWAYS use this instead of `mv` CLI "
                           "command. Creates destination directories if needed and creates a backup "
                           "before renaming. Fails if destination already exists unless overwrite=True.",
                parameters={
                    "type": "object",
                    "properties": {
                        "source_path": {
                            "type": "string",
                            "description": "Path to the source file to rename"
                        },
                        "destination_path": {
                            "type": "string",
                            "description": "New path/name for the file"
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": "If True, overwrite destination if it exists. Default is False."
                        }
                    },
                    "required": ["source_path", "destination_path"]
                },
                category="filesystem",
                discoverability=DISCOVERABILITY_DEFERRED,
                traits=frozenset({TRAIT_REPLAY_SAFE}),
            ),
            ToolSchema(
                name="undoFileChange",
                description="Restore a file from its most recent backup. Use this to undo a previous "
                           "updateFile or removeFile operation.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to restore"
                        }
                    },
                    "required": ["path"]
                },
                category="filesystem",
                discoverability=DISCOVERABILITY_DEFERRED,
                traits=frozenset({TRAIT_REPLAY_SAFE}),
            ),
            ToolSchema(
                name="multiFileEdit",
                description="Execute multiple file operations atomically. Either ALL changes succeed "
                           "or NONE are applied (automatic rollback on failure). Supports mixed operations: "
                           "edit, create, delete, rename. Use this for refactoring operations that must "
                           "succeed together (e.g., renaming a symbol across multiple files).",
                parameters={
                    "type": "object",
                    "properties": {
                        "operations": {
                            "type": "array",
                            "description": "Array of file operations to execute atomically",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "action": {
                                        "type": "string",
                                        "enum": ["edit", "create", "delete", "rename"],
                                        "description": "Type of operation"
                                    },
                                    "path": {
                                        "type": "string",
                                        "description": "File path (for edit, create, delete)"
                                    },
                                    "old": {
                                        "type": "string",
                                        "description": "Text fragment to find and replace (for edit). Must appear exactly once, or use prologue/epilogue to disambiguate."
                                    },
                                    "new": {
                                        "type": "string",
                                        "description": "Replacement text for the matched fragment (for edit)"
                                    },
                                    "prologue": {
                                        "type": "string",
                                        "description": "Context anchor: text immediately before 'old', copied verbatim from the file. Use a short, distinctive fragment. Not modified."
                                    },
                                    "epilogue": {
                                        "type": "string",
                                        "description": "Context anchor: text immediately after 'old', copied verbatim from the file. Use a short, distinctive fragment. Not modified."
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Content for new file (for create)"
                                    },
                                    "from": {
                                        "type": "string",
                                        "description": "Source path (for rename)"
                                    },
                                    "to": {
                                        "type": "string",
                                        "description": "Destination path (for rename)"
                                    }
                                },
                                "required": ["action"]
                            }
                        }
                    },
                    "required": ["operations"]
                },
                category="filesystem",
                discoverability=DISCOVERABILITY_DEFERRED,
                traits=frozenset({TRAIT_REPLAY_SAFE}),
            ),
            ToolSchema(
                name="findAndReplace",
                description="Find and replace text across multiple files using regex patterns. "
                           "Supports glob patterns for file selection and respects .gitignore by default. "
                           "Use dry_run=true to preview changes without applying them.",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for"
                        },
                        "replacement": {
                            "type": "string",
                            "description": "Replacement string (supports backreferences like \\1, \\2)"
                        },
                        "paths": {
                            "type": "string",
                            "description": "Glob pattern for files to search (e.g., 'src/**/*.py')"
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "If true, preview changes without applying. Default is false."
                        },
                        "include_ignored": {
                            "type": "boolean",
                            "description": "If true, include files normally ignored by .gitignore. Default is false."
                        }
                    },
                    "required": ["pattern", "replacement", "paths"]
                },
                category="filesystem",
                discoverability=DISCOVERABILITY_DEFERRED,
                traits=frozenset({TRAIT_REPLAY_SAFE}),
            ),
            ToolSchema(
                name="restoreFile",
                description="Restore a file from a specific backup or the most recent backup. "
                           "Lists available backups if no backup_path is specified.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to restore"
                        },
                        "backup_path": {
                            "type": "string",
                            "description": "Specific backup file to restore from (optional, uses latest if not specified)"
                        }
                    },
                    "required": ["path"]
                },
                category="filesystem",
                discoverability=DISCOVERABILITY_DEFERRED,
                traits=frozenset({TRAIT_REPLAY_SAFE}),
            ),
            ToolSchema(
                name="listBackups",
                description="List all available file backups. Can filter by file path or list all backups.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Optional file path to list backups for. If not specified, lists all backups."
                        }
                    },
                    "required": []
                },
                category="filesystem",
                discoverability=DISCOVERABILITY_DEFERRED,
                traits=frozenset({TRAIT_REPLAY_SAFE}),
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor functions for each tool.

        Phase 3 §3.4 wave 1: forwards via runner-RPC when a runner
        is attached; falls through to in-process otherwise.
        """
        return self.wrap_executors_for_runner_forwarding({
            "readFile": self._execute_read_file,
            "updateFile": self._execute_update_file,
            "writeNewFile": self._execute_write_new_file,
            "removeFile": self._execute_remove_file,
            "moveFile": self._execute_move_file,
            "renameFile": self._execute_move_file,  # Alias for moveFile
            "undoFileChange": self._execute_undo_file_change,
            "multiFileEdit": self._execute_multi_file_edit,
            "findAndReplace": self._execute_find_and_replace,
            "restoreFile": self._execute_restore_file,
            "listBackups": self._execute_list_backups,
        })

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for file editing tools."""
        return """You have access to file editing tools.

IMPORTANT: Always prefer these tools over CLI commands:
- Use `readFile` instead of `cat`, `head`, `tail`, or `less`
- Use `updateFile`/`writeNewFile` instead of `echo >`, `sed`, or heredocs
- Use `removeFile` instead of `rm`
- Use `moveFile`/`renameFile` instead of `mv`
- Use `multiFileEdit` for atomic multi-file operations (refactoring)
- Use `findAndReplace` instead of `sed` for multi-file replacements

These tools provide structured output, automatic backups, and proper encoding handling.

Tools available:
- `readFile(path, offset=None, limit=None)`: Read file contents.
  - For large files, use `offset` (1-indexed line number) and `limit` (max lines) for chunked reading.
  - Example: `readFile(path="large.txt", offset=1, limit=100)` reads lines 1-100.
  - Example: `readFile(path="large.txt", offset=101, limit=100)` reads lines 101-200.
  - Chunked responses include: `total_lines`, `start_line`, `end_line`, and `has_more` (boolean).
  - **Image support**: For image files (PNG, JPG, GIF, WebP, BMP, ICO, SVG), returns multimodal content you can view directly.
- `updateFile`: Update an existing file. Creates backup. Two modes:
  - **Targeted edit** (preferred): `updateFile(path, old="text to find", new="replacement text")`
    Finds `old` in the file and replaces it with `new`. The `old` text must appear exactly once.
    If `old` matches multiple locations, use `prologue` and/or `epilogue` to disambiguate.
    Matching is **exact** — copy text verbatim from the file, preserving whitespace and newlines.
    Tips for prologue/epilogue:
    - Prefer a single distinctive line (e.g., a function signature, class declaration, or unique comment)
    - Copy the anchor text exactly as it appears in the file — do not reformat or re-indent
    - Shorter, more unique anchors are better than long multi-line blocks
    - Example: `updateFile(path, old="x = 1", new="x = 2", prologue="def setup():\\n")`
  - **Full replacement**: `updateFile(path, new_content="entire file content")`
    Replaces the entire file. Use only when the targeted mode is impractical.
- `writeNewFile(path, content)`: Create a new file. Fails if file exists.
- `removeFile(path)`: Delete a file. Creates backup before deletion.
- `moveFile(source_path, destination_path, overwrite=False)`: Move or rename a file. Creates destination directories if needed. Creates backup before moving. Fails if destination exists unless overwrite=True.
- `renameFile(source_path, destination_path, overwrite=False)`: Alias for moveFile. Use for renaming files.
- `undoFileChange(path)`: Restore a file from its most recent backup.
- `multiFileEdit(operations)`: Execute multiple file operations atomically. ALL changes succeed or NONE are applied.
  - Supports: edit, create, delete, rename operations
  - Edit uses targeted search-and-replace: `{"action": "edit", "path": "a.py", "old": "fragment", "new": "replacement"}`
    If `old` matches multiple locations, add `prologue` and/or `epilogue` — a single distinctive line copied verbatim from the file.
  - Multiple edits on the same file are allowed (applied sequentially).
  - Example: `multiFileEdit(operations=[{"action": "edit", "path": "a.py", "old": "old_name", "new": "new_name"},
                                        {"action": "rename", "from": "b.py", "to": "c.py"}])`
  - Use for refactoring that must succeed together (e.g., renaming a symbol across files)
- `findAndReplace(pattern, replacement, paths, dry_run=False, include_ignored=False)`: Find and replace across files.
  - `pattern`: Regex pattern to search for
  - `replacement`: Replacement string (supports backreferences \\1, \\2, etc.)
  - `paths`: Glob pattern (e.g., "src/**/*.py")
  - `dry_run`: If true, preview changes without applying
  - Respects .gitignore by default (use include_ignored=True to override)
- `restoreFile(path, backup_path=None)`: Restore a file from backup. Uses latest backup if backup_path not specified.
- `listBackups(path=None)`: List available backups. Filter by path or list all backups.

IMPORTANT: When using updateFile (full replacement mode) or writeNewFile, provide the raw file content directly.
Do NOT wrap the content in quotes, triple-quotes (''' or \"\"\"), or treat it as a string literal.

Backups are automatically created for file modifications."""

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that should be auto-approved.

        readFile, undoFileChange, restoreFile, and listBackups are read-only operations.
        """
        return ["readFile", "undoFileChange", "restoreFile", "listBackups"]

    def get_user_commands(self) -> List[UserCommand]:
        """File edit plugin provides model tools only."""
        return []

    def format_permission_request(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        channel_type: str
    ) -> Optional[PermissionDisplayInfo]:
        """Format permission request with diff display for file operations.

        This method is called by the permission system to get a custom
        display format for file editing tools.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            channel_type: Type of channel (console, webhook, file)

        Returns:
            PermissionDisplayInfo with formatted diff, or None for default display
        """
        if tool_name == "updateFile":
            return self._format_update_file(arguments)
        elif tool_name == "writeNewFile":
            return self._format_write_new_file(arguments)
        elif tool_name == "removeFile":
            return self._format_remove_file(arguments)
        elif tool_name in ("moveFile", "renameFile"):
            return self._format_move_file(arguments)
        elif tool_name == "multiFileEdit":
            return self._format_multi_file_edit(arguments)
        elif tool_name == "findAndReplace":
            return self._format_find_and_replace(arguments)

        return None

    def _format_update_file(self, arguments: Dict[str, Any]) -> Optional[PermissionDisplayInfo]:
        """Format updateFile for permission display.

        Supports two modes:
        - Targeted mode (old+new present): reads current file, applies the
          targeted edit, and diffs original vs result.
        - Full replacement mode (new_content present): diffs original vs
          new_content directly.
        """
        path = arguments.get("path", "")
        old_text = arguments.get("old")

        file_path = self._resolve_path(path)
        if not file_path.exists():
            # File doesn't exist - skip permission, let executor return the error
            return None

        try:
            old_content = read_text_verified(
                file_path, validate=self._sandbox_validator("read")
            )
        except OSError as e:
            logger.warning(f"Failed to read file for update permission display: {path}", exc_info=True)
            display_path = ellipsize_path(path, DEFAULT_MAX_PATH_WIDTH)
            return PermissionDisplayInfo(
                summary=f"Update file: {display_path} (read error)",
                details=f"Error reading file: {e}\nTraceback: {traceback.format_exc()}",
                format_hint="text"
            )

        if old_text is not None:
            # Targeted mode: compute the result via apply_edit
            new_text = arguments.get("new", "")
            prologue = arguments.get("prologue")
            epilogue = arguments.get("epilogue")
            # Enforce the per-profile span cap BEFORE attempting the anchor
            # match, so an oversized whole-file 'old'/'new' surfaces as a
            # guiding pre-validation error rather than a confusing
            # EditNotFoundError.  Same channel as the edit-error path below.
            span_error = self._check_edit_span(old_text, new_text)
            if span_error is not None:
                display_path = ellipsize_path(path, DEFAULT_MAX_PATH_WIDTH)
                return PermissionDisplayInfo(
                    summary=f"Update file: {display_path} (edit span exceeded)",
                    details=span_error,
                    format_hint="text",
                    pre_validation_error=span_error,
                )
            try:
                new_content = apply_edit(old_content, old_text, new_text, prologue, epilogue)
            except (EditNotFoundError, AmbiguousEditError) as e:
                display_path = ellipsize_path(path, DEFAULT_MAX_PATH_WIDTH)
                return PermissionDisplayInfo(
                    summary=f"Update file: {display_path} (edit error)",
                    details=str(e),
                    format_hint="text",
                    pre_validation_error=str(e),
                )
        else:
            # Full replacement mode
            # Gated profiles have no whole-file path — surface the guiding
            # error through the same pre_validation_error channel as edit errors.
            if not self._allow_full_replace:
                display_path = ellipsize_path(path, DEFAULT_MAX_PATH_WIDTH)
                return PermissionDisplayInfo(
                    summary=f"Update file: {display_path} (full replacement disabled)",
                    details=self._full_replace_gate_error(),
                    format_hint="text",
                    pre_validation_error=self._full_replace_gate_error(),
                )
            # Same absent-vs-empty distinction as the executor: a call that
            # names no content will be rejected there, so surface the guiding
            # error here as pre-validation rather than previewing a truncation
            # the executor will refuse.
            new_content = arguments.get("new_content")
            if new_content is None:
                new_content = arguments.get("content")
            if new_content is None:
                display_path = ellipsize_path(path, DEFAULT_MAX_PATH_WIDTH)
                missing_error = self._missing_content_error()
                return PermissionDisplayInfo(
                    summary=f"Update file: {display_path} (no content provided)",
                    details=missing_error,
                    format_hint="text",
                    pre_validation_error=missing_error,
                )

        diff_text, truncated, total_lines = generate_unified_diff(
            old_content, new_content, path, max_lines=DEFAULT_MAX_LINES
        )

        summary = summarize_diff(old_content, new_content, path)

        return PermissionDisplayInfo(
            summary=summary,
            details=diff_text,
            format_hint="diff",
            truncated=truncated,
            original_lines=total_lines if truncated else None
        )

    def _format_write_new_file(self, arguments: Dict[str, Any]) -> Optional[PermissionDisplayInfo]:
        """Format writeNewFile for permission display."""
        path = arguments.get("path", "")
        content = arguments.get("content", "")

        file_path = self._resolve_path(path)
        if file_path.exists():
            # File already exists - skip permission, let executor return the error
            return None

        diff_text, truncated, total_lines = generate_new_file_diff(
            content, path, max_lines=DEFAULT_MAX_LINES
        )

        lines = content.splitlines()
        display_path = ellipsize_path(path, DEFAULT_MAX_PATH_WIDTH)
        summary = f"Create new file: {display_path} ({len(lines)} lines)"

        return PermissionDisplayInfo(
            summary=summary,
            details=diff_text,
            format_hint="diff",
            truncated=truncated,
            original_lines=total_lines if truncated else None
        )

    def _format_remove_file(self, arguments: Dict[str, Any]) -> Optional[PermissionDisplayInfo]:
        """Format removeFile for permission display."""
        path = arguments.get("path", "")

        file_path = self._resolve_path(path)
        if not file_path.exists():
            # File doesn't exist - skip permission, let executor return the error
            return None

        try:
            content = read_text_verified(
                file_path, validate=self._sandbox_validator("read")
            )
        except OSError as e:
            logger.warning(f"Failed to read file for delete permission display: {path}", exc_info=True)
            display_path = ellipsize_path(path, DEFAULT_MAX_PATH_WIDTH)
            return PermissionDisplayInfo(
                summary=f"Delete file: {display_path} (read error)",
                details=f"Error reading file: {e}\nTraceback: {traceback.format_exc()}",
                format_hint="text"
            )

        diff_text, truncated, total_lines = generate_delete_file_diff(
            content, path, max_lines=DEFAULT_MAX_LINES
        )

        lines = content.splitlines()
        display_path = ellipsize_path(path, DEFAULT_MAX_PATH_WIDTH)
        summary = f"Delete file: {display_path} ({len(lines)} lines, backup will be created)"

        return PermissionDisplayInfo(
            summary=summary,
            details=diff_text,
            format_hint="diff",
            truncated=truncated,
            original_lines=total_lines if truncated else None
        )

    def _format_move_file(self, arguments: Dict[str, Any]) -> Optional[PermissionDisplayInfo]:
        """Format moveFile/renameFile for permission display."""
        source_path = arguments.get("source_path", "")
        destination_path = arguments.get("destination_path", "")
        overwrite = arguments.get("overwrite", False)

        source = self._resolve_path(source_path)
        destination = self._resolve_path(destination_path)

        if not source.exists():
            # Source doesn't exist - skip permission, let executor return the error
            return None

        if destination.exists() and not overwrite:
            # Destination exists and overwrite not set - skip permission, let executor return the error
            return None

        try:
            content = read_text_verified(
                source, validate=self._sandbox_validator("read")
            )
        except OSError as e:
            display_source = ellipsize_path(source_path, DEFAULT_MAX_PATH_WIDTH)
            return PermissionDisplayInfo(
                summary=f"Move file: {display_source} (read error)",
                details=f"Error reading source file: {e}",
                format_hint="text"
            )

        diff_text, truncated, total_lines = generate_move_file_diff(
            source_path, destination_path, content, max_lines=DEFAULT_MAX_LINES
        )

        lines = content.splitlines()
        # Use path pair ellipsization for source -> destination display
        path_pair = ellipsize_path_pair(source_path, destination_path, DEFAULT_MAX_PATH_WIDTH)
        if overwrite and destination.exists():
            summary = f"Move file: {path_pair} ({len(lines)} lines, overwrite enabled)"
        else:
            summary = f"Move file: {path_pair} ({len(lines)} lines)"

        return PermissionDisplayInfo(
            summary=summary,
            details=diff_text,
            format_hint="diff",
            truncated=truncated,
            original_lines=total_lines if truncated else None
        )

    # Tool executors

    def _execute_read_file(self, args: Dict[str, Any]) -> Any:
        """Execute readFile tool.

        Returns file content as a plain string (with a metadata header) so that
        provider converters never JSON-encode the body.  This prevents escaping
        artefacts (e.g. ``\\"`` for ``"``) that would make a subsequent
        ``updateFile`` fail to match the original text.

        Supports chunked reading via offset and limit parameters:
        - offset: Line number to start from (1-indexed, default: 1)
        - limit: Maximum lines to read (default: entire file)

        Returns metadata including has_more flag when limit is used.
        """
        path = args.get("path", "")
        offset = args.get("offset")  # 1-indexed line number
        limit = args.get("limit")    # Max lines to read
        self._trace(f"readFile: path={path}, offset={offset}, limit={limit}")

        if not path:
            return {"error": "path is required"}

        # Resolve path first, then check if allowed
        file_path = self._resolve_path(path)

        # Check if resolved path is allowed (within workspace or authorized for reading)
        if not self._is_path_allowed(str(file_path), mode="read"):
            return {"error": f"Path denied by sandbox: {path}"}

        if not file_path.exists():
            return {"error": f"File not found: {path}"}

        if not file_path.is_file():
            return {"error": f"Not a file: {path}"}

        # Check if file is an image - return as multimodal content
        IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico', '.svg'}
        IMAGE_MIME_TYPES = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.ico': 'image/x-icon',
            '.svg': 'image/svg+xml',
        }
        ext = file_path.suffix.lower()

        if ext in IMAGE_EXTENSIONS:
            self._trace(f"readFile: detected image file, returning as multimodal")
            try:
                data = read_bytes_verified(
                    file_path, validate=self._sandbox_validator("read")
                )
                mime_type = IMAGE_MIME_TYPES.get(ext, 'image/png')
                return {
                    "_multimodal": True,
                    "_multimodal_type": "image",
                    "image_data": data,
                    "mime_type": mime_type,
                    "display_name": file_path.name,
                    "path": normalize_result_path(path),
                    "size": len(data),
                    "type": "image",
                }
            except OSError as e:
                return {"error": f"Failed to read image: {e}"}

        if ext == '.pdf':
            self._trace("readFile: detected PDF, returning as multimodal file")
            try:
                data = read_bytes_verified(
                    file_path, validate=self._sandbox_validator("read")
                )
                return {
                    "_multimodal": True,
                    "_multimodal_type": "file",
                    "file_data": data,
                    "mime_type": "application/pdf",
                    "display_name": file_path.name,
                    "path": normalize_result_path(path),
                    "size": len(data),
                    "type": "file",
                }
            except OSError as e:
                return {"error": f"Failed to read PDF: {e}"}

        # Validate offset and limit if provided
        if offset is not None:
            if not isinstance(offset, int) or offset < 1:
                return {"error": "offset must be a positive integer (1-indexed)"}

        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                return {"error": "limit must be a positive integer"}

        try:
            content = read_text_verified(
                file_path, validate=self._sandbox_validator("read")
            )
            all_lines = content.splitlines(keepends=True)
            total_lines = len(all_lines)

            # Apply chunking if offset or limit specified
            if offset is not None or limit is not None:
                start_idx = (offset - 1) if offset else 0  # Convert to 0-indexed

                if limit is not None:
                    end_idx = start_idx + limit
                    selected_lines = all_lines[start_idx:end_idx]
                    has_more = end_idx < total_lines
                else:
                    selected_lines = all_lines[start_idx:]
                    has_more = False

                # Reconstruct content from selected lines
                chunk_content = "".join(selected_lines)

                # Calculate actual start/end line numbers (1-indexed)
                actual_start = start_idx + 1
                actual_end = min(start_idx + len(selected_lines), total_lines)

                header = (
                    f"path: {normalize_result_path(path)} | "
                    f"lines: {actual_start}-{actual_end} of {total_lines} | "
                    f"size: {len(chunk_content)} | "
                    f"has_more: {has_more}"
                )
                return f"{header}\n\n{chunk_content}"
            else:
                # No chunking - return entire file
                header = (
                    f"path: {normalize_result_path(path)} | "
                    f"lines: {total_lines} | "
                    f"size: {len(content)}"
                )
                return f"{header}\n\n{content}"
        except OSError as e:
            return {"error": f"Failed to read file: {e}"}

    def _write_line_ending(
        self,
        file_path: Path,
        existing_eol: Optional[str],
        content_loaded: bool,
    ) -> str:
        """Pick the line ending an ``updateFile`` write should produce.

        Args:
            file_path: Destination, used to find the enclosing repository.
            existing_eol: The ending the file was found to use.  Only
                meaningful when *content_loaded* is true.
            content_loaded: Whether the old content was read.  A targeted
                edit reads it and so already knows the ending; a whole-file
                replacement does not, and the file has to be sniffed
                instead.

        Returns:
            The ending to write.  See
            :class:`~.line_endings.LineEndingPolicy` for the precedence.
        """
        if content_loaded:
            return self._line_endings.ending_for(file_path, existing_eol)
        return self._line_endings.ending_for_file(
            file_path, validate=self._sandbox_validator("read")
        )

    def _execute_update_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute updateFile tool.

        Supports two modes determined by which parameters are present:

        - **Targeted mode** (``old`` + ``new``): reads the file, finds the
          ``old`` fragment (optionally anchored by ``prologue``/``epilogue``),
          and replaces it with ``new``.
        - **Full replacement mode** (``new_content`` or ``content``): replaces
          the entire file content.
        """
        path = args.get("path", "")
        old_text = args.get("old")

        if not path:
            return {"error": "path is required"}

        file_path = self._resolve_path(path)

        # Check if path is allowed for writing (enforces readonly restrictions)
        if not self._is_path_allowed(str(file_path), mode="write"):
            return {"error": f"Path denied by sandbox (write): {path}"}

        if not file_path.exists():
            return {"error": f"File not found: {path}. Use writeNewFile for new files."}

        if not file_path.is_file():
            return {"error": f"Not a file: {path}"}

        # Determine mode and compute final content.  ``existing_eol`` is
        # only meaningful once the old content has actually been read;
        # ``content_loaded`` records that, so the whole-file path does not
        # mistake "file has no line endings" for "not read yet".
        existing_eol: Optional[str] = None
        content_loaded = False
        if old_text is not None:
            # Targeted mode
            new_text = args.get("new")
            if new_text is None:
                return {"error": "'new' is required when 'old' is provided"}
            prologue = args.get("prologue")
            epilogue = args.get("epilogue")

            # Enforce the per-profile span cap before any disk read / match so
            # an oversized whole-file 'old'/'new' is rejected with a guiding
            # error the model can act on (localise, or switch to new_content).
            span_error = self._check_edit_span(old_text, new_text)
            if span_error is not None:
                self._trace(
                    f"updateFile(targeted) REJECTED span: path={path}, "
                    f"old_len={len(old_text)}, new_len={len(new_text)}, "
                    f"cap={self._max_edit_span_chars}"
                )
                return {"error": span_error}

            try:
                # LF-normalised for matching, with the file's own ending
                # handed back so the write can put it back (#805).
                current_content, existing_eol = self._line_endings.load(
                    file_path, validate=self._sandbox_validator("read")
                )
                content_loaded = True
            except OSError as e:
                return {"error": f"Failed to read file: {e}"}

            self._trace(f"updateFile(targeted): path={path}, old_len={len(old_text)}, new_len={len(new_text)}")

            try:
                new_content = apply_edit(current_content, old_text, new_text, prologue, epilogue)
            except EditNotFoundError as e:
                return {"error": f"Targeted edit failed: {e}"}
            except AmbiguousEditError as e:
                return {"error": f"Targeted edit failed: {e}"}
        else:
            # Full replacement mode
            # Gate: a profile with allow_full_replace=False has no whole-file
            # path.  Reject with a guiding error steering to targeted edits.
            # (Backstop for models that emit new_content despite it being
            # absent from the schema on this profile.)
            if not self._allow_full_replace:
                self._trace(f"updateFile(full) REJECTED — allow_full_replace=False: path={path}")
                return {"error": self._full_replace_gate_error()}
            # Accept both 'new_content' (canonical) and 'content' (alias).
            # Absent is NOT empty: a call that names no content is malformed
            # and must be rejected before any backup or write, or it silently
            # truncates the file and reports success.  An explicit empty
            # string remains the deliberate way to truncate.
            new_content = self._full_replacement_content(args)
            if new_content is None:
                self._trace(f"updateFile(full) REJECTED - no content: path={path}")
                return {"error": self._missing_content_error()}
            self._trace(f"updateFile(full): path={path}, content_len={len(new_content)}")

        new_content = restore(
            new_content,
            self._write_line_ending(file_path, existing_eol, content_loaded),
        )

        # Create backup before modification
        backup_path = None
        if self._backup_manager:
            backup_path = self._backup_manager.create_backup(
                file_path, validate=self._sandbox_validator("read")
            )

        try:
            write_text_verified(
                file_path,
                new_content,
                validate=self._sandbox_validator("write"),
                newline="",
            )
            result = {
                "success": True,
                "path": normalize_result_path(path),
                "size": len(new_content),
                "lines": len(new_content.splitlines())
            }
            if backup_path:
                result["backup"] = normalize_result_path(str(backup_path))
            # _telemetry: Convention-based telemetry
            result['_telemetry'] = {
                'jaato.file.operation': 'update',
                'jaato.file.path': normalize_result_path(path),
                'jaato.file.size_bytes': len(new_content),
                'jaato.file.lines': len(new_content.splitlines()),
                'jaato.file.had_backup': backup_path is not None,
            }
            return result
        except OSError as e:
            return {"error": f"Failed to write file: {e}"}

    def _execute_write_new_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute writeNewFile tool."""
        path = args.get("path", "")
        content = args.get("content", "")
        self._trace(f"writeNewFile: path={path}, content_len={len(content)}")

        if not path:
            return {"error": "path is required"}

        file_path = self._resolve_path(path)

        # Check if path is allowed for writing (enforces readonly restrictions)
        if not self._is_path_allowed(str(file_path), mode="write"):
            return {"error": f"Path denied by sandbox (write): {path}"}

        if file_path.exists():
            return {"error": f"File already exists: {path}. Use updateFile to modify existing files."}

        # A new file has no convention of its own to preserve, so only the
        # repository can have an opinion; absent one this is LF, as before.
        content = restore(content, self._line_endings.ending_for(file_path))

        try:
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # exclusive=True: O_EXCL|O_NOFOLLOW relative to the resolved
            # parent, so neither a symlink planted at the target nor a swap
            # of the parent directory can redirect the new file.
            write_text_verified(
                file_path,
                content,
                validate=self._sandbox_validator("write"),
                exclusive=True,
                newline="",
            )
            return {
                "success": True,
                "path": normalize_result_path(path),
                "size": len(content),
                "lines": len(content.splitlines()),
                # _telemetry: Convention-based telemetry
                "_telemetry": {
                    "jaato.file.operation": "write_new",
                    "jaato.file.path": normalize_result_path(path),
                    "jaato.file.size_bytes": len(content),
                    "jaato.file.lines": len(content.splitlines()),
                },
            }
        except OSError as e:
            return {"error": f"Failed to create file: {e}"}

    def _execute_remove_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute removeFile tool."""
        path = args.get("path", "")
        self._trace(f"removeFile: path={path}")

        if not path:
            return {"error": "path is required"}

        file_path = self._resolve_path(path)

        # Check if path is allowed for writing (enforces readonly restrictions)
        if not self._is_path_allowed(str(file_path), mode="write"):
            return {"error": f"Path denied by sandbox (write): {path}"}

        if not file_path.exists():
            return {"error": f"File not found: {path}"}

        if not file_path.is_file():
            return {"error": f"Not a file: {path}"}

        # Create backup before deletion
        backup_path = None
        if self._backup_manager:
            backup_path = self._backup_manager.create_backup(
                file_path, validate=self._sandbox_validator("read")
            )

        try:
            unlink_verified(file_path, validate=self._sandbox_validator("write"))
            result = {
                "success": True,
                "path": normalize_result_path(path),
                "deleted": True
            }
            if backup_path:
                result["backup"] = normalize_result_path(str(backup_path))
            # _telemetry: Convention-based telemetry
            result['_telemetry'] = {
                'jaato.file.operation': 'remove',
                'jaato.file.path': normalize_result_path(path),
            }
            return result
        except OSError as e:
            return {"error": f"Failed to delete file: {e}"}

    def _execute_move_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute moveFile/renameFile tool."""
        source_path = args.get("source_path", "")
        destination_path = args.get("destination_path", "")
        overwrite = args.get("overwrite", False)
        self._trace(f"moveFile: source={source_path}, dest={destination_path}, overwrite={overwrite}")

        if not source_path:
            return {"error": "source_path is required", "source": source_path}

        if not destination_path:
            return {"error": "destination_path is required", "source": source_path}

        source = self._resolve_path(source_path)
        destination = self._resolve_path(destination_path)

        # Check if both paths are allowed for writing (move = write on both ends)
        if not self._is_path_allowed(str(source), mode="write"):
            return {"error": f"Path denied by sandbox (write): {source_path}", "source": source_path}
        if not self._is_path_allowed(str(destination), mode="write"):
            return {"error": f"Path denied by sandbox (write): {destination_path}", "source": source_path}

        if not source.exists():
            return {"error": "Source file does not exist", "source": source_path}

        if not source.is_file():
            return {"error": f"Source is not a file: {source_path}", "source": source_path}

        if destination.exists() and not overwrite:
            return {
                "error": "Destination file already exists. Use overwrite=True to replace it.",
                "source": source_path,
                "destination": destination_path
            }

        # Create backup of source file before moving
        backup_path = None
        if self._backup_manager:
            backup_path = self._backup_manager.create_backup(
                source, validate=self._sandbox_validator("read")
            )

        # If overwriting, also backup the destination
        dest_backup_path = None
        if destination.exists() and overwrite and self._backup_manager:
            dest_backup_path = self._backup_manager.create_backup(
                destination, validate=self._sandbox_validator("read")
            )

        try:
            # Create destination parent directories if needed
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Move the file (this handles overwrite if destination exists).
            # Both ends are pinned to their resolved parents so neither can be
            # redirected between the sandbox check and the rename.
            move_verified(
                source,
                destination,
                validate=self._sandbox_validator("write"),
            )

            result = {
                "success": True,
                "source": normalize_result_path(source_path),
                "destination": normalize_result_path(destination_path)
            }
            if backup_path:
                result["source_backup"] = normalize_result_path(str(backup_path))
            if dest_backup_path:
                result["destination_backup"] = normalize_result_path(str(dest_backup_path))
            # _telemetry: Convention-based telemetry
            result['_telemetry'] = {
                'jaato.file.operation': 'move',
                'jaato.file.had_backup': backup_path is not None,
                'jaato.file.overwrite': overwrite,
            }
            return result
        except OSError as e:
            return {
                "error": f"Failed to move file: {e}",
                "source": normalize_result_path(source_path),
                "destination": normalize_result_path(destination_path)
            }

    def _execute_undo_file_change(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute undoFileChange tool."""
        path = args.get("path", "")
        self._trace(f"undoFileChange: path={path}")

        if not path:
            return {"error": "path is required"}

        if not self._backup_manager:
            return {"error": "Backup manager not initialized"}

        file_path = self._resolve_path(path)

        # Check if backup exists
        if not self._backup_manager.has_backup(file_path):
            return {"error": f"No backup found for: {path}"}

        # Get the backup path for reporting
        backup_path = self._backup_manager.get_latest_backup(file_path)

        # Restore from backup
        if self._backup_manager.restore_from_backup(file_path):
            return {
                "success": True,
                "path": normalize_result_path(path),
                "restored_from": normalize_result_path(str(backup_path)) if backup_path else "unknown",
                "message": f"File restored from backup",
                # _telemetry: Convention-based telemetry
                "_telemetry": {
                    "jaato.file.operation": "undo",
                    "jaato.file.path": normalize_result_path(path),
                },
            }
        else:
            return {"error": f"Failed to restore file from backup"}

    def _execute_multi_file_edit(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiFileEdit tool for atomic multi-file operations."""
        operations = args.get("operations", [])
        self._trace(f"multiFileEdit: {len(operations)} operations")

        if not operations:
            return {"error": "operations array is required and cannot be empty"}

        if not isinstance(operations, list):
            return {"error": "operations must be an array"}

        # Create executor with plugin's path resolution and sandbox checking
        executor = MultiFileExecutor(
            resolve_path_fn=self._resolve_path,
            is_path_allowed_fn=self._is_path_allowed,
            trace_fn=self._trace,
            line_endings=self._line_endings,
        )

        # Execute the batch
        result = executor.execute(operations)

        result_dict = result.to_dict()
        # _telemetry: Convention-based telemetry
        result_dict['_telemetry'] = {
            'jaato.file.operation': 'multi_edit',
            'jaato.file.files_count': len(operations),
            'jaato.file.all_succeeded': result_dict.get('success', False),
        }
        return result_dict

    def _execute_find_and_replace(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute findAndReplace tool for regex-based find/replace across files."""
        pattern = args.get("pattern", "")
        replacement = args.get("replacement", "")
        paths = args.get("paths", "")
        dry_run = args.get("dry_run", False)
        include_ignored = args.get("include_ignored", False)

        self._trace(f"findAndReplace: pattern='{pattern}', paths='{paths}', dry_run={dry_run}")

        if not pattern:
            return {"error": "pattern is required"}

        if replacement is None:
            return {"error": "replacement is required"}

        if not paths:
            return {"error": "paths glob pattern is required"}

        # Create backup function that uses our backup manager
        def backup_fn(file_path: Path) -> Optional[Path]:
            if self._backup_manager:
                return self._backup_manager.create_backup(
                    file_path, validate=self._sandbox_validator("read")
                )
            return None

        # Create executor
        workspace_root = Path(self._workspace_root) if self._workspace_root else None
        executor = FindReplaceExecutor(
            workspace_root=workspace_root,
            resolve_path_fn=self._resolve_path,
            is_path_allowed_fn=self._is_path_allowed,
            backup_fn=backup_fn,
            trace_fn=self._trace,
            line_endings=self._line_endings,
        )

        # Execute find/replace
        result = executor.execute(
            pattern=pattern,
            replacement=replacement,
            paths=paths,
            dry_run=dry_run,
            include_ignored=include_ignored
        )

        result_dict = result.to_dict()
        # _telemetry: Convention-based telemetry
        result_dict['_telemetry'] = {
            'jaato.file.operation': 'find_replace',
            'jaato.file.files_matched': result_dict.get('files_modified_count', 0),
            'jaato.file.replacements': result_dict.get('total_replacements', 0),
            'jaato.file.dry_run': dry_run,
        }
        return result_dict

    def _execute_restore_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute restoreFile tool to restore from a specific backup."""
        path = args.get("path", "")
        backup_path_str = args.get("backup_path")

        self._trace(f"restoreFile: path={path}, backup_path={backup_path_str}")

        if not path:
            return {"error": "path is required"}

        if not self._backup_manager:
            return {"error": "Backup manager not initialized"}

        file_path = self._resolve_path(path)

        # If backup_path specified, use it; otherwise use latest
        if backup_path_str:
            backup_path = Path(backup_path_str)
            if not backup_path.exists():
                return {"error": f"Backup file not found: {backup_path_str}"}
        else:
            backup_path = self._backup_manager.get_latest_backup(file_path)
            if not backup_path:
                # List available backups
                backups = self._backup_manager.list_backups(file_path)
                if not backups:
                    return {"error": f"No backups found for: {path}"}
                return {
                    "error": f"No backup specified. Available backups for {path}:",
                    "available_backups": [str(b) for b in backups]
                }

        # Restore from backup
        if self._backup_manager.restore_from_backup(file_path, backup_path):
            return {
                "success": True,
                "path": normalize_result_path(path),
                "restored_from": normalize_result_path(str(backup_path)),
                "message": "File restored from backup",
                # _telemetry: Convention-based telemetry
                "_telemetry": {
                    "jaato.file.operation": "restore",
                    "jaato.file.path": normalize_result_path(path),
                },
            }
        else:
            return {"error": "Failed to restore file from backup"}

    def _execute_list_backups(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute listBackups tool to list available backups."""
        path = args.get("path")

        self._trace(f"listBackups: path={path}")

        if not self._backup_manager:
            return {"error": "Backup manager not initialized"}

        if path:
            # List backups for specific file
            file_path = self._resolve_path(path)
            backups = self._backup_manager.list_backups(file_path)

            if not backups:
                return {
                    "path": normalize_result_path(path),
                    "backups": [],
                    "message": f"No backups found for {path}"
                }

            return {
                "path": normalize_result_path(path),
                "backups": [
                    {
                        "backup_path": normalize_result_path(str(b)),
                        "timestamp": b.stat().st_mtime,
                        "size": b.stat().st_size
                    }
                    for b in backups
                ],
                "count": len(backups),
                # _telemetry: Convention-based telemetry
                "_telemetry": {
                    "jaato.file.operation": "list_backups",
                    "jaato.file.count": len(backups),
                },
            }
        else:
            # List all backups
            all_backups = self._backup_manager.list_all_backups()

            if not all_backups:
                return {
                    "backups": [],
                    "message": "No backups found"
                }

            return {
                "backups": [b.to_dict() for b in all_backups],
                "count": len(all_backups),
                # _telemetry: Convention-based telemetry
                "_telemetry": {
                    "jaato.file.operation": "list_backups",
                    "jaato.file.count": len(all_backups),
                },
            }

    def _format_multi_file_edit(self, arguments: Dict[str, Any]) -> Optional[PermissionDisplayInfo]:
        """Format multiFileEdit for permission display."""
        operations = arguments.get("operations", [])

        if not operations:
            return None

        # Generate preview diff
        preview_text, truncated = generate_multi_file_diff_preview(
            operations,
            self._resolve_path,
            max_lines_per_file=30
        )

        # Count operations by type
        edit_count = sum(1 for op in operations if op.get("action") == "edit")
        create_count = sum(1 for op in operations if op.get("action") == "create")
        delete_count = sum(1 for op in operations if op.get("action") == "delete")
        rename_count = sum(1 for op in operations if op.get("action") == "rename")

        parts = []
        if edit_count:
            parts.append(f"{edit_count} edit{'s' if edit_count > 1 else ''}")
        if create_count:
            parts.append(f"{create_count} create{'s' if create_count > 1 else ''}")
        if delete_count:
            parts.append(f"{delete_count} delete{'s' if delete_count > 1 else ''}")
        if rename_count:
            parts.append(f"{rename_count} rename{'s' if rename_count > 1 else ''}")

        summary = f"Atomic multi-file operation: {', '.join(parts)}"

        return PermissionDisplayInfo(
            summary=summary,
            details=preview_text,
            format_hint="diff",
            truncated=truncated
        )

    def _format_find_and_replace(self, arguments: Dict[str, Any]) -> Optional[PermissionDisplayInfo]:
        """Format findAndReplace for permission display."""
        pattern = arguments.get("pattern", "")
        replacement = arguments.get("replacement", "")
        paths = arguments.get("paths", "")
        dry_run = arguments.get("dry_run", False)
        include_ignored = arguments.get("include_ignored", False)

        # If dry_run, no approval needed
        if dry_run:
            return None

        # Create executor to find matches for preview
        workspace_root = Path(self._workspace_root) if self._workspace_root else None
        executor = FindReplaceExecutor(
            workspace_root=workspace_root,
            resolve_path_fn=self._resolve_path,
            is_path_allowed_fn=self._is_path_allowed,
            trace_fn=self._trace
        )

        # Execute dry run to get preview
        result = executor.execute(
            pattern=pattern,
            replacement=replacement,
            paths=paths,
            dry_run=True,
            include_ignored=include_ignored
        )

        if not result.success:
            return PermissionDisplayInfo(
                summary=f"Find and replace: error",
                details=result.error or "Unknown error",
                format_hint="text"
            )

        if result.total_matches == 0:
            return PermissionDisplayInfo(
                summary=f"Find and replace: no matches found",
                details=f"Pattern '{pattern}' not found in files matching '{paths}'",
                format_hint="text"
            )

        # Generate preview
        preview_text, truncated = generate_find_replace_preview(
            result.file_matches,
            max_matches_per_file=5,
            max_files=10
        )

        summary = f"Find and replace: {result.total_matches} matches in {result.files_affected} files"

        return PermissionDisplayInfo(
            summary=summary,
            details=preview_text,
            format_hint="text",
            truncated=truncated
        )


def create_plugin() -> FileEditPlugin:
    """Factory function to create the file edit plugin instance."""
    return FileEditPlugin()
