"""Workspace Manager for Jaato Server.

Handles workspace discovery, creation, and configuration management
for web clients that need to select a workspace before starting a session.

Workspaces are directories under a configurable root that contain either:
- A .jaato/ directory
- A .env file

The manager persists workspace metadata to ~/.jaato/workspaces.json
"""

import json
import functools
import logging
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import dotenv_values

logger = logging.getLogger(__name__)

# Default registry path
DEFAULT_REGISTRY_PATH = Path.home() / ".jaato" / "workspaces.json"

# ---------------------------------------------------------------------------
# Provider catalog: derived from the providers' own contracts, never listed.
#
# This used to be a six-entry table (``anthropic``, ``google``, ``github``,
# ``antigravity``, ``ollama``, ``claude_cli``) written when those were the
# providers -- so the workspace picker offered six of the twenty-odd
# providers in the tree, and two of the six under names the runtime does
# not know (the modules are ``google_genai`` and ``github_models``; a
# ``.env`` written with ``JAATO_PROVIDER=google`` fails at ``load_provider``).
# Every provider already declares ``PROVIDER_AUTH_RESOLUTION`` -- the
# ordered credential chain ``explain provider`` renders -- and its ``env``
# steps ARE the env vars this module used to hardcode.  Reading them means
# a new provider appears here with no edit, the same way ``explain
# dependencies`` stopped hardcoding the distribution list (#966).
#
# The scan is AST-only (``shared.scaffold.introspect``), so it imports no
# provider SDK; the ``echo`` test double is excluded there too.
# ---------------------------------------------------------------------------

#: A provider with no ``env`` credential step still needs a way to be
#: DETECTED from a workspace ``.env`` that configures it without a secret.
#: Vertex AI is the one such case in the tree: it authenticates through
#: Application Default Credentials and is configured by ``PROJECT_ID``.
_DETECTION_HINTS: Dict[str, str] = {"PROJECT_ID": "google_genai"}


@functools.lru_cache(maxsize=1)
def provider_catalog() -> Dict[str, List[str]]:
    """Every model provider in the tree -> the env vars that carry its credential.

    Keys are the provider directory names, which are what a profile's
    ``provider:`` field and ``JAATO_PROVIDER`` name.  Values are the
    ``kind == "env"`` steps of ``PROVIDER_AUTH_RESOLUTION``, in resolution
    order; a provider whose credential is OAuth, ADC, an external CLI or
    nothing at all (``ollama``) has an empty list -- it can still be
    selected, and a key cannot be written for it (see ``update_config``).
    Cached for the daemon's lifetime: the contracts are source constants.
    """
    from jaato_server.shared.scaffold.introspect import providers

    out: Dict[str, List[str]] = {}
    for name, info in providers().items():
        out[name] = [a.name for a in info.auth if getattr(a, "kind", "") == "env" and a.name]
    return dict(sorted(out.items()))


def available_providers() -> List[str]:
    """The provider names the workspace picker may offer, sorted."""
    return list(provider_catalog().keys())


def credential_env_var(provider: str) -> Optional[str]:
    """The env var an API key for ``provider`` is written under, or ``None``.

    The FIRST ``env`` step of the provider's own chain -- the one its
    ``resolve_api_key`` consults first, so the key written here is the one
    it reads back.
    """
    vars_ = provider_catalog().get(provider) or []
    return vars_[0] if vars_ else None


def _detection_order() -> List[tuple]:
    """``(env_var, provider)`` pairs consulted by :meth:`_detect_provider`.

    ``JAATO_PROVIDER`` is explicit and wins.  Then every provider's declared
    credential vars, in sorted provider order so the answer does not
    depend on directory iteration order, then the detection hints.
    """
    order: List[tuple] = [("JAATO_PROVIDER", None)]
    for name, vars_ in provider_catalog().items():
        for v in vars_:
            order.append((v, name))
    order.extend(_DETECTION_HINTS.items())
    return order


class WorkspaceContainmentError(ValueError):
    """A workspace NAME resolved to a path outside the manager's root.

    Subclasses :class:`ValueError` deliberately: every WS handler that
    turns a workspace verb into an error frame already catches
    ``ValueError``, so containment refusals reach the client through the
    path that exists rather than through a new one.  It is a distinct type
    so a caller that wants to tell "escaped the root" from "does not
    exist" can, without parsing a message.
    """


class WorkspaceOwnershipError(ValueError):
    """A workspace belongs to a different authenticated user.

    A :class:`ValueError` for the reason :class:`WorkspaceContainmentError`
    is one: the WS handlers already turn a ``ValueError`` into an error
    frame.  Distinct so a caller can tell "not yours" from "does not exist"
    without parsing the message -- and so the two never share wording, since
    a refusal that reads like a missing workspace invites creating it.
    """


@dataclass
class WorkspaceInfo:
    """Information about a workspace."""
    name: str  # Relative path from root
    path: str  # Absolute path
    configured: bool  # Has valid provider config
    provider: Optional[str] = None
    model: Optional[str] = None
    last_accessed: Optional[str] = None
    #: The authenticated user who created it (``get_client_user`` at the
    #: time of ``workspace.create`` -- a #1074 ticket identity reads
    #: ``app:user``).  ``None`` for a workspace discovered on disk or created
    #: by a connection with no identity: UNOWNED, visible to everyone, the
    #: posture every pre-existing workspace keeps.  Persisted in the
    #: registry and preserved across re-discovery, which rebuilds every
    #: other field from the directory.
    owner: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkspaceManager:
    """Manages workspace discovery, creation, and configuration."""

    def __init__(
        self,
        workspace_root: str,
        registry_path: Optional[Path] = None,
    ):
        """Initialize the workspace manager.

        Args:
            workspace_root: Root directory containing workspaces.
            registry_path: Path to workspace registry file (default: ~/.jaato/workspaces.json).
        """
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.registry_path = registry_path or DEFAULT_REGISTRY_PATH

        # Ensure root exists
        if not self.workspace_root.exists():
            logger.warning(f"Workspace root does not exist: {self.workspace_root}")

        # In-memory cache
        self._workspaces: Dict[str, WorkspaceInfo] = {}

        # Currently selected workspace (legacy single-client mode)
        self._selected_workspace: Optional[str] = None

        # Per-client workspace selections for multi-client mode.
        # Maps client_id → workspace name.
        self._client_workspaces: Dict[str, str] = {}

        # Load registry
        self._load_registry()

    def _load_registry(self) -> None:
        """Load workspace registry from disk.

        The registry is keyed by workspace NAME and lives in one file per
        daemon user (``~/.jaato/workspaces.json``) whatever
        ``--workspace-root`` the daemon was started with, so a row written
        under one root outlives a move to another.  Such a row used to be
        loaded verbatim: it sat in ``_workspaces`` and so in every
        ``workspace.list``, while ``select`` and ``delete`` resolved the same
        NAME under the CURRENT root and answered "does not exist" -- a
        workspace nobody could open or remove.  The stale row is now the
        deployment's own root directory listed as a workspace of itself
        (root ``/srv/jaato`` -> ``/srv/jaato/workspaces``, then the root
        moved down one level), which is exactly the shape that produced it.

        So a row is loaded only when the path it records still resolves to
        ``<root>/<name>``; anything else is dropped with a WARNING naming
        the root it was written for, and is gone from the file at the next
        save.  A row whose directory no longer EXISTS is still loaded here
        -- it carries the owner and the last-opened time, which the
        directory cannot -- and :meth:`discover_workspaces` is what prunes
        it, since that is the point at which the daemon looks at the disk.
        """
        if not self.registry_path.exists():
            logger.debug(f"No workspace registry at {self.registry_path}")
            return

        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)

            recorded_root = data.get("root")
            dropped = []
            for ws_data in data.get("workspaces", []):
                name = ws_data.get("name")
                if not name:
                    continue
                if not self._registry_row_is_current(name, ws_data.get("path", "")):
                    dropped.append(name)
                    continue
                self._workspaces[name] = WorkspaceInfo(
                    name=name,
                    path=ws_data.get("path", ""),
                    configured=ws_data.get("configured", False),
                    provider=ws_data.get("provider"),
                    model=ws_data.get("model"),
                    last_accessed=ws_data.get("last_accessed"),
                    owner=ws_data.get("owner"),
                )

            if dropped:
                logger.warning(
                    "Ignoring %d workspace registry row(s) not under the workspace "
                    "root %s (registry written for root %s): %s",
                    len(dropped), self.workspace_root, recorded_root or "?",
                    ", ".join(sorted(dropped)),
                )
            logger.debug(f"Loaded {len(self._workspaces)} workspaces from registry")

        except Exception as e:
            logger.warning(f"Failed to load workspace registry: {e}")

    def _registry_row_is_current(self, name: str, path: str) -> bool:
        """Whether a registry row still describes ``<root>/<name>``.

        A row records the path it was analysed at.  It is current when that
        path is exactly the one the NAME resolves to under the current root
        -- the same resolution ``select`` and ``delete`` perform, so a row
        this accepts is one those verbs can act on.  A row with no path (a
        registry predating the field) is accepted on its name alone.
        """
        if not path:
            return True
        try:
            recorded = Path(path).expanduser().resolve()
            return recorded == (self.workspace_root / name).resolve()
        except (OSError, ValueError):
            return False

    def _save_registry(self) -> None:
        """Save workspace registry to disk."""
        try:
            # Ensure directory exists
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "root": str(self.workspace_root),
                "workspaces": [ws.to_dict() for ws in self._workspaces.values()],
            }

            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self._workspaces)} workspaces to registry")

        except Exception as e:
            logger.warning(f"Failed to save workspace registry: {e}")

    def _is_under_root(self, path: Path) -> bool:
        """Whether an ALREADY-RESOLVED path lives strictly beneath the root.

        Takes a resolved path rather than resolving one, so the single
        caller that has a path instead of a name (the registry branch of
        :meth:`get_workspace_path`) shares this comparison instead of
        carrying a second opinion about what "under the root" means.

        The root ITSELF is not under the root.  It used to be accepted, so
        an empty name (``root / ""`` is the root) selected the root as a
        workspace: ``_analyze_workspace`` named it after the root's own
        basename, the cache held it under the key ``""``, and every
        ``workspace.list`` showed a row -- ``workspaces`` on the live
        daemon -- that ``select`` and ``delete`` then refused by name.
        """
        return self.workspace_root in path.parents

    @staticmethod
    def _check_name(name: str) -> None:
        """The NAMING rule: a workspace name is one flat path component.

        Shared by the verbs that take a client-supplied name (``create``,
        ``select``, ``delete``) so the cache key and ``WorkspaceInfo.name``
        cannot disagree -- a nested name would be keyed as ``a/b`` and
        analysed as ``b``.  :meth:`_resolve_under_root` enforces the
        LOCATION rule; the two catch different things (``".."`` passes
        this one, ``"a/b"`` passes that one).
        """
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"Invalid workspace name: {name!r}")

    def _resolve_under_root(self, name: str) -> Path:
        """Turn a client-supplied workspace NAME into a path under the root.

        Workspace names arrive from clients (``workspace.select``,
        ``workspace.create``) and were joined onto ``workspace_root`` with
        nothing in between.  **The join contains nothing by itself**:
        pathlib discards the left operand when the right is absolute
        (``root / "/etc/passwd"`` is ``/etc/passwd``) and keeps ``..``
        verbatim (``root / ".."`` is the root's PARENT), so a name is a
        path traversal unless something resolves it and compares.

        Symlinks are resolved BEFORE the comparison, so a link planted
        inside the root is judged by its target.  That is not theoretical
        here: provisioned per-session workspaces live under this same root
        and the agent's own file tools write into them, so a link under
        the root is model-reachable.

        The cost is stated rather than hidden — a workspace an operator
        deliberately symlinked into the root is now REFUSED and must be
        moved or bind-mounted.  Such a workspace was already running
        without kernel confinement: the AppArmor/cgroup gate in
        ``websocket.py`` resolves both sides the same way and skips a
        session whose workspace does not resolve under the root.

        What this does NOT bound: every tenant's provisioned workspace is
        a sibling under one root, so containment stops a name from leaving
        the root and says nothing about which workspace *inside* it a
        client may select.

        Raises:
            WorkspaceContainmentError: the name resolves outside the root.
        """
        candidate = (self.workspace_root / name).expanduser()
        try:
            resolved = candidate.resolve()
        except OSError as e:          # symlink loop, ELOOP, name too long
            raise WorkspaceContainmentError(
                f"Cannot resolve workspace name {name!r}: {e}"
            ) from e

        if resolved == self.workspace_root:
            raise WorkspaceContainmentError(
                f"Workspace {name!r} resolves to the workspace root "
                f"{self.workspace_root} itself, which is not a workspace"
            )
        if not self._is_under_root(resolved):
            raise WorkspaceContainmentError(
                f"Workspace {name!r} resolves to {resolved}, which is outside "
                f"the workspace root {self.workspace_root}"
            )
        return resolved

    def discover_workspaces(self) -> List[WorkspaceInfo]:
        """Discover workspaces under the root directory.

        Scans for directories containing .jaato/ or .env files.
        Updates internal cache and persists to registry.

        Returns:
            List of discovered workspaces.
        """
        if not self.workspace_root.exists():
            logger.warning(f"Workspace root does not exist: {self.workspace_root}")
            return []

        # A cached row whose directory is gone -- removed out of band, or
        # carried over from a registry the daemon no longer has a directory
        # for -- would otherwise be listed forever while ``select`` and
        # ``delete`` refuse it by name.  This is the one point that looks at
        # the disk, so it is where the cache is reconciled with it.
        for name in list(self._workspaces):
            try:
                present = self._resolve_under_root(name).is_dir()
            except WorkspaceContainmentError:
                present = False
            if not present:
                logger.info("Forgetting workspace %r: no directory under %s", name, self.workspace_root)
                self._workspaces.pop(name, None)

        discovered = []

        for entry in self.workspace_root.iterdir():
            if not entry.is_dir():
                continue

            # Skip hidden directories (except we look inside for .jaato)
            if entry.name.startswith("."):
                continue

            # Check if this looks like a workspace
            has_jaato = (entry / ".jaato").is_dir()
            has_env = (entry / ".env").is_file()

            if has_jaato or has_env:
                ws_info = self._analyze_workspace(entry)
                discovered.append(ws_info)
                self._workspaces[ws_info.name] = ws_info

        # Save updated registry
        self._save_registry()

        logger.info(f"Discovered {len(discovered)} workspaces under {self.workspace_root}")
        return discovered

    def _analyze_workspace(self, path: Path, name: Optional[str] = None) -> WorkspaceInfo:
        """Analyze a workspace directory to determine its configuration status.

        Args:
            path: Absolute path to workspace directory.
            name: The workspace's NAME -- the cache key and what clients
                address it by.  Defaults to the directory's basename, which
                is right for discovery; a verb that resolved a name to a
                path passes the name it resolved, so a symlinked entry is
                still known by the name the client used.

        Returns:
            WorkspaceInfo with configuration details.
        """
        name = name or path.name
        env_file = path / ".env"

        provider = None
        model = None
        configured = False

        if env_file.exists():
            env_vars = dotenv_values(env_file)

            # Detect provider
            provider = self._detect_provider(env_vars)

            # Get model if set
            model = env_vars.get("MODEL_NAME") or env_vars.get("JAATO_MODEL")

            # Consider configured if we have a provider
            configured = provider is not None

        # Two fields the directory cannot tell us are carried over from the
        # registry entry: when it was last opened, and WHO owns it.  Dropping
        # the owner here would silently un-own every workspace on the first
        # ``workspace.list`` after a restart -- the failure that would make
        # the visibility rule cosmetic.
        existing = self._workspaces.get(name)
        last_accessed = existing.last_accessed if existing else None
        owner = existing.owner if existing else None

        return WorkspaceInfo(
            name=name,
            path=str(path),
            configured=configured,
            provider=provider,
            model=model,
            last_accessed=last_accessed,
            owner=owner,
        )

    def _detect_provider(self, env_vars: Dict[str, Optional[str]]) -> Optional[str]:
        """Detect the provider from environment variables.

        Args:
            env_vars: Dictionary of environment variables.

        Returns:
            Provider name or None if not detected.
        """
        for env_var, provider in _detection_order():
            if env_var in env_vars and env_vars[env_var]:
                if provider is None:
                    # JAATO_PROVIDER is explicit
                    return env_vars[env_var]
                return provider

        return None

    def owner_for_path(self, workspace_path: str) -> Optional[str]:
        """The qualified owner ``app:user`` of the workspace at ``workspace_path``.

        The lookup ``app://`` secret resolution uses (#1226): a session runs in
        an absolute workspace path, and the owner is what decides which
        application is asked to resolve its ``app://`` references.  Returns
        ``None`` for a path that is not a known workspace and for a known but
        UNOWNED one — both resolve nothing, correctly, since there is no owner
        to ask.  Compared on the resolved absolute path so a symlinked or
        non-normalised spelling of the same directory still matches.
        """
        if not workspace_path:
            return None
        try:
            target = os.path.realpath(workspace_path)
        except OSError:
            target = workspace_path
        for ws in self._workspaces.values():
            try:
                if os.path.realpath(ws.path) == target:
                    return ws.owner
            except OSError:
                continue
        return None

    @staticmethod
    def visible_to(ws_info: WorkspaceInfo, user: Optional[str]) -> bool:
        """Whether *user* may see and use *ws_info*.

        The one visibility rule, applied by :meth:`list_workspaces`,
        :meth:`select_workspace` and :meth:`delete_workspace` alike, so the
        list can never show a workspace the verbs then refuse or hide one
        they would accept:

        - a connection with NO identity (``user is None`` -- the shared
          bearer token, a daemon with no tickets configured) sees every
          workspace, which is what it always saw;
        - an authenticated user sees the workspaces they own and the
          UNOWNED ones (discovered on disk, or created before ownership was
          recorded), and never another user's.
        """
        return user is None or ws_info.owner is None or ws_info.owner == user

    def _check_owner(self, ws_info: WorkspaceInfo, user: Optional[str]) -> None:
        if not self.visible_to(ws_info, user):
            raise WorkspaceOwnershipError(
                f"Workspace {ws_info.name!r} belongs to another user"
            )

    def list_workspaces(self, for_user: Optional[str] = None) -> List[WorkspaceInfo]:
        """List the workspaces *for_user* may see.

        Combines cached workspaces with fresh discovery, then applies
        :meth:`visible_to`.  ``for_user=None`` is the unscoped listing.

        Returns:
            List of workspace info.
        """
        # Re-discover to get fresh state
        self.discover_workspaces()
        return [ws for ws in self._workspaces.values()
                if self.visible_to(ws, for_user)]

    def create_workspace(self, name: str, owner: Optional[str] = None) -> WorkspaceInfo:
        """Create a new workspace.

        Args:
            name: Name for the new workspace (becomes subdirectory name).
            owner: The creating connection's authenticated user, recorded so
                the workspace is theirs and hidden from other users; ``None``
                leaves it unowned.

        Returns:
            WorkspaceInfo for the created workspace.

        Raises:
            ValueError: If workspace already exists or name is invalid.
        """
        # Two checks that catch different things, so neither masks the
        # other: this one enforces the NAMING rule (a workspace name is one
        # flat component, which is what ``_analyze_workspace``'s
        # ``path.name`` keying assumes), and ``_resolve_under_root``
        # enforces the LOCATION rule.  ".." passes the first and is caught
        # by the second; "a/b" passes the second and is caught by the first.
        self._check_name(name)

        path = self._resolve_under_root(name)

        if path.exists():
            raise ValueError(f"Workspace already exists: {name}")

        # Create directory and .jaato subdirectory
        path.mkdir(parents=True)
        (path / ".jaato").mkdir()

        # Create empty .env file
        (path / ".env").touch()

        # ...and a GC strategy, because without one there is none at all.
        self._write_default_gc_config(path)

        ws_info = WorkspaceInfo(
            name=name,
            path=str(path),
            configured=False,
            last_accessed=datetime.now(timezone.utc).isoformat(),
            owner=owner,
        )

        self._workspaces[name] = ws_info
        self._save_registry()

        logger.info("Created workspace: %s at %s (owner=%s)", name, path, owner or "-")
        return ws_info

    #: What :meth:`create_workspace` writes to ``<workspace>/.jaato/gc.json``.
    #:
    #: ``type`` and nothing else, deliberately.  Every other key of that file
    #: has a framework default -- several of them behind an env var
    #: (``JAATO_GC_THRESHOLD`` / ``_TARGET`` / ``_PRESSURE`` /
    #: ``_MEDIA_BYTES``) -- and ``shared.plugins.gc._media_settings`` states
    #: the rule this follows: *omission has to mean "the dataclass decides",
    #: not "the default I happened to type"*.  A generated file that
    #: re-spelled today's 80.0 would outrank the env var for every workspace
    #: created before the number next moves, and would freeze each one on the
    #: value that was current the day it was made.
    DEFAULT_GC_CONFIG: Dict[str, Any] = {"type": "budget"}

    def _write_default_gc_config(self, path: Path) -> None:
        """Give a new workspace a GC strategy.

        A session gets its GC from its profile's ``gc:`` block, else from
        ``<workspace>/.jaato/gc.json``, else from ``~/.jaato/gc.json`` -- and
        if none of the three answers, ``JaatoServer.initialize`` leaves
        ``gc_result`` at ``None`` and the session runs with **no context
        garbage collection at all**.  A workspace created here is exactly
        that case: it is driven by a bare ``session.new`` against the
        ``JAATO_PROVIDER`` / ``MODEL_NAME`` pair in its ``.env``, with no
        profile, so nothing selected a strategy and the history grew until
        the pre-send guard refused it or the upstream did.

        **Not settable from ``.env``**, which is where one would first reach
        for it: there is no ``JAATO_GC_TYPE``.  The four ``JAATO_GC_*``
        variables are read by ``GCConfig``'s field defaults, and that object
        is only ever constructed once a strategy has been chosen -- so
        writing a threshold into ``.env`` and stopping there configures
        nothing, silently.  Choosing the strategy is the load-bearing act and
        ``gc.json`` is where it is expressed.

        ``budget`` rather than ``truncate`` because it dominates it: with an
        ``InstructionBudget`` it removes by GC policy (enrichment first,
        never LOCKED), and without one
        :meth:`BudgetGCPlugin.collect` falls back to the same turn-based
        truncation ``gc_truncate`` would have done.  There is no state in
        which it is the worse choice.

        Best-effort: a workspace that exists with no ``gc.json`` is the state
        every workspace was in before this, so a failure here is logged and
        the workspace is still created.  Existing workspaces are deliberately
        NOT migrated -- writing into a directory whose owner may have made
        their own choice is not this method's business, and the file is a
        starting point the user is meant to edit.

        Args:
            path: The workspace root; ``<path>/.jaato`` already exists.
        """
        target = path / ".jaato" / "gc.json"
        try:
            target.write_text(
                json.dumps(self.DEFAULT_GC_CONFIG, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Could not write default GC config to %s: %s -- the workspace "
                "is usable and its sessions will run with no GC until a "
                "gc.json or a profile gc: block supplies one", target, exc,
            )

    def delete_workspace(
        self,
        name: str,
        user: Optional[str] = None,
        in_use_by: Optional[List[str]] = None,
        client_id: Optional[str] = None,
    ) -> WorkspaceInfo:
        """Delete a workspace: its directory, and its registry entry.

        Destructive and unrecoverable -- the sessions persisted under
        ``<workspace>/.jaato`` go with it -- so the client is expected to
        have confirmed.  What the manager itself refuses, each as a
        ``ValueError`` the WS handler already turns into an error frame:

        - a name that leaves the root (:class:`WorkspaceContainmentError`,
          checked before existence, as for ``select``), or the root itself;
        - a workspace that does not exist;
        - another user's workspace (:class:`WorkspaceOwnershipError`);
        - a workspace something is still using -- the ``in_use_by`` ids the
          caller resolved (loaded sessions running in it), or another
          client's current selection.  Deleting a directory a runner is
          confined to is not a delete, it is a session failure with a
          delayed cause;
        - a workspace under RETENTION -- its profiles declared
          ``record_keeping.retention_days`` and audit files here have not
          reached it (EU AI Act Art. 19(1), #1119).  ``session.delete``
          removes a conversation and leaves the record, which is the verb
          for this case.

        The deleting client's own selection of it is cleared.

        Args:
            name: Workspace name (relative path from root).
            user: The caller's authenticated user, for the ownership check.
            in_use_by: Session ids the caller found loaded in this workspace.
            client_id: The deleting client, whose own selection of the
                workspace does not count as "in use" and is cleared.

        Returns:
            The deleted workspace's info, as it stood.
        """
        # Containment first, so a traversal is refused as one (and before
        # existence); then the naming rule, which containment cannot check.
        path = self._resolve_under_root(name)   # refuses the root itself
        self._check_name(name)
        if not path.exists():
            raise ValueError(f"Workspace does not exist: {name}")

        ws_info = self._analyze_workspace(path, name=name)
        self._check_owner(ws_info, user)

        if in_use_by:
            raise ValueError(
                f"Workspace {name!r} has {len(in_use_by)} loaded session(s): "
                f"{', '.join(sorted(in_use_by))} -- stop them first"
            )
        others = [cid for cid, sel in self._client_workspaces.items()
                  if sel == name and cid != client_id]
        if others:
            raise ValueError(
                f"Workspace {name!r} is selected by {len(others)} other client(s)"
            )

        # Art. 19(1) (#1119): a workspace whose profiles declared a
        # ``record_keeping.retention_days`` holds audit files a delete
        # would destroy before their minimum elapsed.  Refused rather than
        # preserved-as-orphans or overridden-with-a-warning -- see
        # ``record_retention.workspace_retention_hold`` for why.  A
        # workspace that declares nothing is unaffected, which is every
        # workspace that existed before this.
        from .record_retention import workspace_retention_hold
        hold = workspace_retention_hold(path)
        if hold:
            raise ValueError(f"Workspace {name!r} is under retention: {hold}")

        shutil.rmtree(path)
        self._workspaces.pop(name, None)
        if client_id is not None and self._client_workspaces.get(client_id) == name:
            self._client_workspaces.pop(client_id, None)
        if self._selected_workspace == name:
            self._selected_workspace = None
        self._save_registry()

        logger.info("Deleted workspace: %s at %s (by %s)", name, path, user or "-")
        return ws_info

    def select_workspace(
        self,
        name: str,
        client_id: Optional[str] = None,
        user: Optional[str] = None,
    ) -> WorkspaceInfo:
        """Select a workspace for a session or client.

        Args:
            name: Workspace name (relative path from root).
            client_id: Optional client identifier for per-client tracking.
                When provided, the selection is stored per-client.
                When None, uses the legacy single-workspace mode.
            user: The caller's authenticated user; another user's workspace
                is refused, so the list's visibility rule is also the
                verbs' (a filtered list over unguarded verbs is decoration).

        Returns:
            WorkspaceInfo with current configuration status.

        Raises:
            WorkspaceContainmentError: If the name resolves outside the
                workspace root.  Checked BEFORE existence, so the refusal
                does not double as an oracle for what exists out there.
            WorkspaceOwnershipError: If the workspace belongs to another user.
            ValueError: If workspace does not exist.
        """
        path = self._resolve_under_root(name)   # containment, before existence
        self._check_name(name)                  # one flat component

        if not path.exists():
            raise ValueError(f"Workspace does not exist: {name}")

        # Re-analyze to get fresh state
        ws_info = self._analyze_workspace(path, name=name)
        self._check_owner(ws_info, user)
        ws_info.last_accessed = datetime.now(timezone.utc).isoformat()

        self._workspaces[name] = ws_info

        if client_id:
            self._client_workspaces[client_id] = name
        else:
            self._selected_workspace = name

        self._save_registry()

        logger.info(f"Selected workspace: {name} (client={client_id or 'default'})")
        return ws_info

    def get_selected_workspace(
        self,
        client_id: Optional[str] = None,
    ) -> Optional[WorkspaceInfo]:
        """Get the currently selected workspace.

        Args:
            client_id: Optional client identifier.  When provided,
                returns the per-client selection.  Falls back to the
                legacy single-workspace selection.

        Returns:
            WorkspaceInfo or None if no workspace selected.
        """
        target = None
        if client_id:
            target = self._client_workspaces.get(client_id)
        if not target:
            target = self._selected_workspace
        if not target:
            return None
        return self._workspaces.get(target)

    def remove_client(self, client_id: str) -> None:
        """Remove per-client workspace tracking when a client disconnects.

        Args:
            client_id: Client identifier to remove.
        """
        self._client_workspaces.pop(client_id, None)

    def get_workspace_path(self, name: Optional[str] = None) -> Optional[Path]:
        """Get the absolute path to a workspace.

        Args:
            name: Workspace name, or None for selected workspace.

        Returns:
            Absolute path or None.
        """
        target = name or self._selected_workspace
        if not target:
            return None

        ws_info = self._workspaces.get(target)
        if ws_info and ws_info.path:
            # The registry is daemon-owned but its rows are whatever was
            # written into it, and one accepted out-of-root selection used
            # to persist across restarts -- so the stored path is checked
            # rather than trusted.
            stored = Path(ws_info.path)
            try:
                resolved = stored.resolve()
            except OSError:
                resolved = None
            if resolved is not None and self._is_under_root(resolved):
                return stored
            logger.warning(
                "Refusing registry path for workspace %r: %s is outside the "
                "workspace root %s", target, ws_info.path, self.workspace_root,
            )
            return None

        # Fallback to computed path.  An accessor stays total: a refusal is
        # None (its existing "no such workspace" answer), not an exception.
        try:
            return self._resolve_under_root(target)
        except WorkspaceContainmentError as e:
            logger.warning("Refusing workspace path for %r: %s", target, e)
            return None

    def get_env_file(self, name: Optional[str] = None) -> Optional[Path]:
        """Get the .env file path for a workspace.

        Args:
            name: Workspace name, or None for selected workspace.

        Returns:
            Path to .env file or None.
        """
        ws_path = self.get_workspace_path(name)
        if ws_path:
            return ws_path / ".env"
        return None

    def get_config_status(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration status for a workspace.

        Args:
            name: Workspace name, or None for selected workspace.

        Returns:
            Dictionary with configuration status.
        """
        target = name or self._selected_workspace
        if not target:
            return {
                "workspace": None,
                "configured": False,
                "available_providers": available_providers(),
                "missing_fields": ["workspace"],
            }

        ws_info = self._workspaces.get(target)
        if not ws_info:
            try:
                ws_path = self._resolve_under_root(target)
            except WorkspaceContainmentError as e:
                # Reporting a provider/model read out of an arbitrary .env
                # would make this an oracle for files outside the root.
                logger.warning("Refusing config status for %r: %s", target, e)
                return {
                    "workspace": target,
                    "configured": False,
                    "available_providers": available_providers(),
                    "missing_fields": ["workspace is outside the workspace root"],
                }
            if ws_path.exists():
                ws_info = self._analyze_workspace(ws_path, name=target)
            else:
                return {
                    "workspace": target,
                    "configured": False,
                    "available_providers": available_providers(),
                    "missing_fields": ["workspace does not exist"],
                }

        missing = []
        if not ws_info.configured:
            missing.append("provider")
        if not ws_info.model:
            missing.append("model")

        return {
            "workspace": target,
            "configured": ws_info.configured,
            "provider": ws_info.provider,
            "model": ws_info.model,
            "available_providers": available_providers(),
            "missing_fields": missing,
        }

    def update_config(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update workspace configuration.

        Writes provider and optional model/API key to the workspace's .env file.

        Args:
            provider: Provider name.
            model: Model name (optional).
            api_key: API key (optional, for non-OAuth providers).
            name: Workspace name, or None for selected workspace.

        Returns:
            Dictionary with update result.

        Raises:
            ValueError: If no workspace selected or invalid provider.
        """
        target = name or self._selected_workspace
        if not target:
            raise ValueError("No workspace selected")

        # The docstring always promised this refusal; nothing enforced it,
        # so a picker could write a JAATO_PROVIDER the runtime rejects at
        # load_provider() -- the failure then surfaced at session.new,
        # several steps from the form that caused it.
        if provider not in provider_catalog():
            raise ValueError(
                f"Unknown provider {provider!r}. Available: "
                f"{', '.join(available_providers())}"
            )

        env_file = self.get_env_file(target)
        if not env_file:
            raise ValueError(f"Cannot find .env for workspace: {target}")

        # Read existing env vars
        existing = {}
        if env_file.exists():
            existing = dict(dotenv_values(env_file))

        # Update with new values
        existing["JAATO_PROVIDER"] = provider

        if model:
            existing["MODEL_NAME"] = model

        if api_key:
            # Written under the FIRST env step of the provider's own
            # credential chain -- the var its resolve_api_key reads first.
            var = credential_env_var(provider)
            if not var:
                raise ValueError(
                    f"Provider {provider!r} takes no API key from the environment "
                    f"(its credential is OAuth, ADC, an external CLI, or none); "
                    f"sign in with its auth command instead."
                )
            existing[var] = api_key

        # Write back to .env file
        self._write_env_file(env_file, existing)

        # Re-analyze and update cache
        ws_path = self._resolve_under_root(target)
        ws_info = self._analyze_workspace(ws_path, name=target)
        ws_info.last_accessed = datetime.now(timezone.utc).isoformat()
        self._workspaces[target] = ws_info
        self._save_registry()

        logger.info(f"Updated config for workspace {target}: provider={provider}, model={model}")

        return {
            "workspace": target,
            "provider": provider,
            "model": model,
            "success": True,
        }

    def _write_env_file(self, path: Path, env_vars: Dict[str, Optional[str]]) -> None:
        """Write environment variables to a .env file.

        Args:
            path: Path to .env file.
            env_vars: Dictionary of environment variables.
        """
        lines = []
        for key, value in env_vars.items():
            if value is not None:
                # Quote values that contain spaces or special chars
                if " " in value or "=" in value or '"' in value:
                    value = f'"{value}"'
                lines.append(f"{key}={value}")

        path.write_text("\n".join(lines) + "\n")
