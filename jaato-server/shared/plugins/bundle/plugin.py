"""Top-level ``bundle`` user command — domain-agnostic bundle management.

This plugin owns the user-facing ``bundle`` command. Subcommands
operate across every domain plugin that has registered a
:class:`bundle_common.handler.BundleEntryHandler` (today: references;
soon: agents, tasks, profiles, services). The plugin is deliberately
thin — it parses arguments, dispatches into the registered handler
for the matching ``kind``, and formats the result. All on-disk state
lives in the domain plugins.

Surface (Phase 9 scope):

    bundle list                                          all bundles, all kinds
    bundle add <kind>:<id> --to <bundle-ref>             move an entry into a bundle
    bundle eject <kind>:<id>                             move an entry out, leave on disk
    bundle remove <kind>:<id>                            delete an entry from disk
    bundle help                                          detailed usage

Verbs deferred to a later phase (still reachable via the existing
``references bundle <verb>`` namespace until they're lifted):

    create / delete                          per-kind manifest writers
    reconcile                                per-kind sidecar refresh
    pack / unpack                            composite-aware archive ops
    merge                                    references-specific today
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from jaato_sdk.plugins.base import (
    CommandCompletion,
    CommandParameter,
    HelpLines,
    UserCommand,
)
from jaato_sdk.plugins.model_provider.types import ToolSchema

from ..bundle_common.bundle import (
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    VALID_BUNDLE_TIERS,
    AmbiguousBundleRefError,
    Bundle,
    BundleRef,
    find_bundle,
    parse_bundle_ref,
)
from ..bundle_common.handler import (
    BundleEntry,
    BundleEntryHandler,
    BundleEntryRegistry,
)
from ..bundle_common.handler import registry as default_registry
from ..bundle_common.pack import PackResult, pack_bundle_set
from ..bundle_common.unpack import (
    UnpackError,
    UnpackMode,
    UnpackResult,
    read_envelope,
    unpack_archive,
)

logger = logging.getLogger(__name__)


def _parse_kind_id(token: str) -> Tuple[str, str]:
    """Split ``<kind>:<id>`` user input.

    Bare ids are rejected — the design point is that names can clash
    across kinds (``references:weekly`` vs ``tasks:weekly``), so the
    user must qualify. The error message is friendly enough that a
    new user can recover without reading the help.
    """
    raw = (token or "").strip()
    if not raw:
        raise ValueError("entry reference is empty")
    if ":" not in raw:
        raise ValueError(
            f"entry reference {raw!r} must include a kind prefix — write "
            f"'<kind>:<id>' (e.g. 'references:api-spec' or 'agents:reviewer')"
        )
    kind, _, entry_id = raw.partition(":")
    kind = kind.strip()
    entry_id = entry_id.strip()
    if not kind or not entry_id:
        raise ValueError(
            f"entry reference {raw!r} must have a non-empty kind and id"
        )
    return kind, entry_id


class BundlePlugin:
    """Plugin providing the top-level ``bundle`` command.

    Holds no domain state — every subcommand reaches into the shared
    :class:`BundleEntryRegistry` for the relevant handler. The
    registry is injected at construction time so tests can provide
    isolated registries; production code uses the module-level
    singleton from :mod:`bundle_common.handler`.
    """

    def __init__(self, registry: Optional[BundleEntryRegistry] = None) -> None:
        self._registry = registry or default_registry
        self._workspace_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Plugin protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "bundle"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Plugin lifecycle — no-op; state lives in the registry/handlers."""
        return None

    def shutdown(self) -> None:
        """Plugin lifecycle — no-op; nothing to release."""
        return None

    def set_workspace_path(self, path: str) -> None:
        """Cache the workspace path for tier-relative output defaults."""
        self._workspace_path = path

    def get_tool_schemas(self) -> List[ToolSchema]:
        """No model-invokable tools — bundles are an operator concern."""
        return []

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        return {"bundle": self._execute_bundle_cmd}

    def get_auto_approved_tools(self) -> List[str]:
        return ["bundle"]

    def get_user_commands(self) -> List[UserCommand]:
        return [
            UserCommand(
                name="bundle",
                description=(
                    "Manage knowledge bundles across all domains "
                    "(list|add|eject|remove)"
                ),
                share_with_model=False,
                parameters=[
                    CommandParameter(
                        name="subcommand",
                        description=(
                            "Action: list, add, eject, remove, or help"
                        ),
                        required=False,
                    ),
                    CommandParameter(
                        name="target",
                        description="Subcommand-specific argument tail",
                        required=False,
                        capture_rest=True,
                    ),
                ],
            ),
        ]

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _execute_bundle_cmd(self, args: Dict[str, Any]) -> Any:
        """Dispatch into the right subcommand.

        ``subcommand`` defaults to ``"list"`` (matches the references
        command's symmetry: bare ``bundle`` shows what exists).
        Unknown subcommands return an error pointing at ``bundle help``.
        """
        subcommand = (args.get("subcommand") or "list").strip()
        target = args.get("target", "") or ""

        if subcommand == "list":
            return self._cmd_list()
        if subcommand == "add":
            return self._cmd_add(target)
        if subcommand == "eject":
            return self._cmd_eject(target)
        if subcommand == "remove":
            return self._cmd_remove(target)
        if subcommand == "create":
            return self._cmd_create(target)
        if subcommand == "delete":
            return self._cmd_delete(target)
        if subcommand == "reconcile":
            return self._cmd_reconcile(target)
        if subcommand == "pack":
            return self._cmd_pack(target)
        if subcommand == "unpack":
            return self._cmd_unpack(target)
        if subcommand == "help":
            return self._cmd_help()
        return {
            "error": (
                f"Unknown subcommand: {subcommand}. Use: list, add, eject, "
                f"remove, create, delete, reconcile, pack, unpack, help. "
                f"(Bundle merge remains under 'references bundle merge' — "
                f"it's references-specific by nature.)"
            )
        }

    # ------------------------------------------------------------------
    # Subcommands
    # ------------------------------------------------------------------

    def _cmd_list(self) -> HelpLines:
        """``bundle list`` — every bundle from every registered handler.

        Aggregates :meth:`BundleEntryHandler.list_bundles` across kinds.
        Output groups by kind (sorted) so the operator sees the
        references bundles first, then agents, then tasks, etc.
        """
        lines: List[Tuple[str, str]] = [("BUNDLES", "bold"), ("", "")]

        if not self._registry.kinds():
            lines.append(("    (no domain handlers registered)", "dim"))
            return HelpLines(lines=lines)

        any_rows = False
        for handler in self._registry.all_handlers():
            kind = handler.kind
            bundles = handler.list_bundles()
            entries = handler.list_entries()
            entry_count_by_bundle: Dict[Tuple[str, str], int] = {}
            for e in entries:
                key = (e.bundle_name, e.bundle_tier)
                entry_count_by_bundle[key] = entry_count_by_bundle.get(key, 0) + 1
            if not bundles:
                continue
            any_rows = True
            lines.append((f"  {kind}:", "bold"))
            for b in bundles:
                count = entry_count_by_bundle.get((b.name, b.tier), 0)
                lines.append((
                    f"    [{b.tier:<9}] {b.display_name:<18} {count:>3} entries",
                    "",
                ))
            lines.append(("", ""))

        if not any_rows:
            lines.append(("    (no bundles loaded across any kind)", "dim"))

        return HelpLines(lines=lines)

    def _cmd_add(self, raw_args: str) -> Dict[str, Any]:
        """``bundle add <kind>:<id> --to <bundle-ref>`` — move an entry into a bundle.

        Resolves the entry through the registry (kind-explicit so
        cross-kind name collisions can't masquerade as ambiguity).
        Resolves the target bundle by name across every kind — if the
        name is ambiguous (e.g. both references and agents have a
        ``teammate`` bundle), the user must qualify the target with
        the same kind as the entry. After moving the file, the
        handler is asked to ``reload_catalog()`` and ``reconcile_bundle``
        the source and target bundles so the catalog and sidecar
        stay in sync.
        """
        try:
            tokens = shlex.split(raw_args or "")
        except ValueError as e:
            return {"error": f"Failed to parse arguments: {e}"}

        entry_token: Optional[str] = None
        target_token: Optional[str] = None
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--to":
                if i + 1 >= len(tokens):
                    return {"error": "--to requires a bundle reference"}
                target_token = tokens[i + 1]
                i += 2
                continue
            if tok.startswith("--to="):
                target_token = tok.split("=", 1)[1]
                i += 1
                continue
            if entry_token is not None:
                return {"error": "Usage: bundle add <kind>:<id> --to <bundle-ref>"}
            entry_token = tok
            i += 1

        if entry_token is None or target_token is None:
            return {"error": "Usage: bundle add <kind>:<id> --to <bundle-ref>"}

        try:
            kind, entry_id = _parse_kind_id(entry_token)
        except ValueError as e:
            return {"error": str(e)}
        handler = self._registry.get(kind)
        if handler is None:
            return {
                "error": (
                    f"unknown kind {kind!r}; registered kinds: "
                    f"{', '.join(self._registry.kinds()) or '(none)'}"
                )
            }
        entry = handler.find_entry(entry_id)
        if entry is None:
            return {"error": f"Unknown {kind} entry {entry_id!r}"}

        try:
            target_ref = parse_bundle_ref(target_token)
        except ValueError as e:
            return {"error": f"--to: {e}"}
        try:
            target_bundle = find_bundle(
                handler.list_bundles(),
                target_ref,
                default_scope=BUNDLE_TIER_WORKSPACE,
            )
        except AmbiguousBundleRefError as e:
            return {"error": f"--to: {e}"}
        if target_bundle is None:
            return {
                "error": (
                    f"Unknown {kind} bundle {target_ref.display!r} — "
                    f"use 'references bundle create' to make it first"
                )
            }

        if entry.bundle_name == target_bundle.name and entry.bundle_tier == target_bundle.tier:
            return {
                "error": (
                    f"entry {entry_token!r} is already in "
                    f"{target_bundle.qualified_ref}"
                )
            }

        source_bundle = self._lookup_source_bundle(handler, entry)

        try:
            handler.move_entry_to_bundle(entry, target_bundle)
        except FileExistsError as e:
            return {"error": str(e)}
        except Exception as e:  # pragma: no cover - defensive
            return {"error": f"move failed: {e}"}

        return self._post_membership_change(
            handler=handler,
            verb="add",
            entry_token=entry_token,
            source_bundle=source_bundle,
            target_bundle=target_bundle,
        )

    def _cmd_eject(self, raw_args: str) -> Dict[str, Any]:
        """``bundle eject <kind>:<id>`` — move an entry out of its bundle.

        Lands the file at the kind's tier root in the same tier the
        entry currently lives in. The entry remains discoverable by
        the handler but is no longer counted in any bundle's manifest.
        Errors when the entry is already free or when its bundle is
        the tier-root bundle (no parent location to eject into).
        """
        try:
            tokens = shlex.split(raw_args or "")
        except ValueError as e:
            return {"error": f"Failed to parse arguments: {e}"}
        if len(tokens) != 1:
            return {"error": "Usage: bundle eject <kind>:<id>"}

        try:
            kind, entry_id = _parse_kind_id(tokens[0])
        except ValueError as e:
            return {"error": str(e)}
        handler = self._registry.get(kind)
        if handler is None:
            return {"error": f"unknown kind {kind!r}"}
        entry = handler.find_entry(entry_id)
        if entry is None:
            return {"error": f"Unknown {kind} entry {entry_id!r}"}
        if not entry.bundle_name:
            return {
                "error": (
                    f"entry {tokens[0]!r} is already free (not in any bundle)"
                )
            }

        source_bundle = self._lookup_source_bundle(handler, entry)
        try:
            handler.move_entry_to_free(entry, entry.bundle_tier)
        except FileExistsError as e:
            return {"error": str(e)}
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:  # pragma: no cover - defensive
            return {"error": f"eject failed: {e}"}

        return self._post_membership_change(
            handler=handler,
            verb="eject",
            entry_token=tokens[0],
            source_bundle=source_bundle,
            target_bundle=None,
        )

    def _cmd_remove(self, raw_args: str) -> Dict[str, Any]:
        """``bundle remove <kind>:<id>`` — delete an entry from disk.

        Permanently removes the entry's backing file. If the entry was
        in a bundle, that bundle is reconciled afterwards so its
        ``rows`` no longer cites the now-missing id.
        """
        try:
            tokens = shlex.split(raw_args or "")
        except ValueError as e:
            return {"error": f"Failed to parse arguments: {e}"}
        if len(tokens) != 1:
            return {"error": "Usage: bundle remove <kind>:<id>"}

        try:
            kind, entry_id = _parse_kind_id(tokens[0])
        except ValueError as e:
            return {"error": str(e)}
        handler = self._registry.get(kind)
        if handler is None:
            return {"error": f"unknown kind {kind!r}"}
        entry = handler.find_entry(entry_id)
        if entry is None:
            return {"error": f"Unknown {kind} entry {entry_id!r}"}

        source_bundle = self._lookup_source_bundle(handler, entry)
        try:
            handler.delete_entry(entry)
        except Exception as e:  # pragma: no cover - defensive
            return {"error": f"remove failed: {e}"}

        return self._post_membership_change(
            handler=handler,
            verb="remove",
            entry_token=tokens[0],
            source_bundle=source_bundle,
            target_bundle=None,
        )

    def _cmd_create(self, raw_args: str) -> Dict[str, Any]:
        """``bundle create <name> --kind <kind> [--scope workspace|user]``.

        Delegates to the chosen handler's
        :meth:`BundleEntryHandler.create_empty_bundle`. ``--kind`` is
        required because each kind writes its own manifest format
        (references manifests carry embedding metadata; agents/tasks
        manifests don't), so picking a default would force users
        through an error round-trip. Defaults: ``--scope workspace``.

        After the manifest is written the handler is asked to reload
        its catalog so subsequent commands see the new bundle.
        """
        try:
            tokens = shlex.split(raw_args or "")
        except ValueError as e:
            return {"error": f"Failed to parse arguments: {e}"}

        name_token: Optional[str] = None
        kind: Optional[str] = None
        scope: str = BUNDLE_TIER_WORKSPACE
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--kind":
                if i + 1 >= len(tokens):
                    return {"error": "--kind requires a value"}
                kind = tokens[i + 1]
                i += 2
                continue
            if tok.startswith("--kind="):
                kind = tok.split("=", 1)[1]
                i += 1
                continue
            if tok == "--scope":
                if i + 1 >= len(tokens):
                    return {"error": "--scope requires workspace or user"}
                value = tokens[i + 1]
                if value not in VALID_BUNDLE_TIERS:
                    return {"error": f"Unknown scope {value!r}; use workspace or user"}
                scope = value
                i += 2
                continue
            if tok.startswith("--scope="):
                value = tok.split("=", 1)[1]
                if value not in VALID_BUNDLE_TIERS:
                    return {"error": f"Unknown scope {value!r}; use workspace or user"}
                scope = value
                i += 1
                continue
            if name_token is not None:
                return {
                    "error": (
                        "Usage: bundle create <name> --kind <kind> "
                        "[--scope workspace|user]"
                    )
                }
            name_token = tok
            i += 1

        if name_token is None or kind is None:
            return {
                "error": (
                    "Usage: bundle create <name> --kind <kind> "
                    "[--scope workspace|user]"
                )
            }
        handler = self._registry.get(kind)
        if handler is None:
            return {
                "error": (
                    f"unknown kind {kind!r}; registered kinds: "
                    f"{', '.join(self._registry.kinds()) or '(none)'}"
                )
            }

        bundle_name = "" if name_token in ("root", "(root)") else name_token

        # Reject if a bundle with this name already exists in the
        # chosen tier under this kind.
        existing = next(
            (
                b for b in handler.list_bundles()
                if b.name == bundle_name and b.tier == scope
            ),
            None,
        )
        if existing is not None:
            return {
                "error": (
                    f"bundle '{existing.qualified_ref}' already exists "
                    f"under kind={kind!r} at {existing.directory}; use "
                    f"'bundle delete' first or pick a different name"
                )
            }

        ws_path = Path(self._workspace_path) if self._workspace_path else None
        if scope == BUNDLE_TIER_WORKSPACE and ws_path is None:
            return {
                "error": (
                    "create: cannot create a workspace bundle — no "
                    "workspace_path is bound. Pass --scope user, or load a "
                    "workspace first."
                )
            }
        try:
            new_bundle = handler.create_empty_bundle(
                bundle_name, scope, workspace_path=ws_path,
            )
        except NotImplementedError as e:
            return {"error": str(e)}
        except FileExistsError as e:
            return {"error": f"create: {e}"}
        except (RuntimeError, ValueError) as e:
            return {"error": f"create: {e}"}

        try:
            handler.reload_catalog()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("reload_catalog failed for %s: %s", kind, e)

        lines: List[Tuple[str, str]] = [
            ("CREATE", "bold"),
            ("", ""),
            (f"  kind: {kind}", ""),
            (f"  bundle: {new_bundle.qualified_ref}", ""),
            (f"  directory: {new_bundle.directory}", ""),
        ]

        return {
            "status": "ok",
            "kind": kind,
            "bundle": new_bundle.qualified_ref,
            "directory": str(new_bundle.directory),
            "help_lines": HelpLines(lines=lines),
        }

    def _cmd_delete(self, raw_args: str) -> Dict[str, Any]:
        """``bundle delete <bundle-ref> --kind <kind> [--force]``.

        Deletes a single kind's bundle. ``--kind`` is required even
        when only one kind is registered — the explicit qualifier
        protects against accidentally clobbering a composite bundle
        when more handlers come online. ``--force`` is required for
        non-empty bundles; without it the dispatcher errors out and
        leaves disk untouched.
        """
        try:
            tokens = shlex.split(raw_args or "")
        except ValueError as e:
            return {"error": f"Failed to parse arguments: {e}"}

        bundle_token: Optional[str] = None
        kind: Optional[str] = None
        force: bool = False
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--kind":
                if i + 1 >= len(tokens):
                    return {"error": "--kind requires a value"}
                kind = tokens[i + 1]
                i += 2
                continue
            if tok.startswith("--kind="):
                kind = tok.split("=", 1)[1]
                i += 1
                continue
            if tok == "--force":
                force = True
                i += 1
                continue
            if bundle_token is not None:
                return {
                    "error": (
                        "Usage: bundle delete <bundle-ref> --kind <kind> "
                        "[--force]"
                    )
                }
            bundle_token = tok
            i += 1

        if bundle_token is None or kind is None:
            return {
                "error": (
                    "Usage: bundle delete <bundle-ref> --kind <kind> "
                    "[--force]"
                )
            }
        handler = self._registry.get(kind)
        if handler is None:
            return {"error": f"unknown kind {kind!r}"}

        try:
            ref = parse_bundle_ref(bundle_token)
        except ValueError as e:
            return {"error": str(e)}
        try:
            bundle = find_bundle(
                handler.list_bundles(), ref,
                default_scope=BUNDLE_TIER_WORKSPACE,
            )
        except AmbiguousBundleRefError as e:
            return {"error": str(e)}
        if bundle is None:
            return {
                "error": (
                    f"Unknown {kind} bundle '{ref.display}'. Loaded: "
                    f"{[b.qualified_ref for b in handler.list_bundles()] or '(none)'}"
                )
            }

        try:
            handler.delete_bundle(bundle, force=force)
        except NotImplementedError as e:
            return {"error": str(e)}
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:  # pragma: no cover - defensive
            return {"error": f"delete failed: {e}"}

        try:
            handler.reload_catalog()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("reload_catalog failed for %s: %s", kind, e)

        lines: List[Tuple[str, str]] = [
            ("DELETE", "bold"),
            ("", ""),
            (f"  kind: {kind}", ""),
            (f"  removed: {bundle.qualified_ref}", ""),
            (f"  directory: {bundle.directory}", ""),
            (f"  forced: {force}", ""),
        ]

        return {
            "status": "ok",
            "kind": kind,
            "bundle": bundle.qualified_ref,
            "directory": str(bundle.directory),
            "forced": force,
            "help_lines": HelpLines(lines=lines),
        }

    def _cmd_reconcile(self, raw_args: str) -> Dict[str, Any]:
        """``bundle reconcile [<bundle-ref>] [--scope ws|user|all] [--kind <k>]``.

        With no bundle reference, reconciles every bundle that matches
        the scope filter (default: ``--scope workspace``). With a
        bundle reference, reconciles just that bundle — ``--kind`` is
        required in that case so the dispatcher knows which handler
        owns the manifest.

        Without ``--kind``, reconcile fans out across every registered
        handler that has a bundle matching the (name, tier) tuple, so
        a composite bundle can be brought up-to-date in one command.
        """
        try:
            tokens = shlex.split(raw_args or "")
        except ValueError as e:
            return {"error": f"Failed to parse arguments: {e}"}

        bundle_token: Optional[str] = None
        kind: Optional[str] = None
        scope_filter: Optional[str] = None  # None == default workspace
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--kind":
                if i + 1 >= len(tokens):
                    return {"error": "--kind requires a value"}
                kind = tokens[i + 1]
                i += 2
                continue
            if tok.startswith("--kind="):
                kind = tok.split("=", 1)[1]
                i += 1
                continue
            if tok == "--scope":
                if i + 1 >= len(tokens):
                    return {"error": "--scope requires workspace, user, or all"}
                value = tokens[i + 1]
                if value not in (*VALID_BUNDLE_TIERS, "all"):
                    return {
                        "error": (
                            f"Unknown scope {value!r}. Use workspace, user, or all."
                        )
                    }
                scope_filter = value
                i += 2
                continue
            if tok.startswith("--scope="):
                value = tok.split("=", 1)[1]
                if value not in (*VALID_BUNDLE_TIERS, "all"):
                    return {
                        "error": (
                            f"Unknown scope {value!r}. Use workspace, user, or all."
                        )
                    }
                scope_filter = value
                i += 1
                continue
            if bundle_token is not None:
                return {
                    "error": (
                        "Usage: bundle reconcile [<bundle-ref>] "
                        "[--scope workspace|user|all] [--kind <kind>]"
                    )
                }
            bundle_token = tok
            i += 1

        if bundle_token is not None and scope_filter is not None:
            return {
                "error": (
                    "Cannot combine a bundle reference with --scope; pick "
                    "one bundle (e.g. 'workspace:teammate') or pick a scope "
                    "(e.g. '--scope all')."
                )
            }

        # Determine which handlers to ask. With --kind, only that one;
        # otherwise every registered handler.
        if kind is not None:
            handler = self._registry.get(kind)
            if handler is None:
                return {"error": f"unknown kind {kind!r}"}
            handlers = [handler]
        else:
            handlers = self._registry.all_handlers()
        if not handlers:
            return {"error": "no kinds registered to reconcile"}

        # Collect target bundles per handler.
        targets: List[Tuple[BundleEntryHandler, Bundle]] = []
        if bundle_token is not None:
            try:
                ref = parse_bundle_ref(bundle_token)
            except ValueError as e:
                return {"error": str(e)}
            for h in handlers:
                try:
                    hit = find_bundle(
                        h.list_bundles(), ref,
                        default_scope=BUNDLE_TIER_WORKSPACE,
                    )
                except AmbiguousBundleRefError as e:
                    return {"error": str(e)}
                if hit is not None:
                    targets.append((h, hit))
            if not targets:
                return {
                    "error": (
                        f"no bundle '{ref.display}' found across "
                        f"{[h.kind for h in handlers]}"
                    )
                }
        else:
            effective_scope = scope_filter or BUNDLE_TIER_WORKSPACE
            for h in handlers:
                for b in h.list_bundles():
                    if effective_scope == "all" or b.tier == effective_scope:
                        targets.append((h, b))
            if not targets:
                return {
                    "status": "ok",
                    "results": [],
                    "help_lines": HelpLines(lines=[
                        ("RECONCILE", "bold"),
                        ("", ""),
                        ("  (no bundles to reconcile in the chosen scope)", "dim"),
                    ]),
                }

        results: List[Dict[str, Any]] = []
        lines: List[Tuple[str, str]] = [("RECONCILE", "bold"), ("", "")]
        for h, b in targets:
            try:
                rec = h.reconcile_bundle(b)
            except Exception as e:  # pragma: no cover - defensive
                lines.append((f"  [{h.kind}] {b.qualified_ref}: error — {e}", ""))
                results.append({
                    "kind": h.kind,
                    "bundle": b.qualified_ref,
                    "error": str(e),
                })
                continue
            summary = (
                rec.summary() if rec is not None and hasattr(rec, "summary")
                else "ok"
            )
            lines.append((f"  [{h.kind}] {b.qualified_ref}: {summary}", ""))
            results.append({
                "kind": h.kind,
                "bundle": b.qualified_ref,
                "summary": summary,
            })

        return {
            "status": "ok",
            "results": results,
            "help_lines": HelpLines(lines=lines),
        }

    def _cmd_pack(self, raw_args: str) -> Dict[str, Any]:
        """``bundle pack <name> [--scope workspace|user] [--to <archive>]``.

        Builds a composite v2 archive containing every registered
        kind's bundle named ``<name>`` in the chosen tier. Handlers
        with no matching bundle for that ``(name, tier)`` are silently
        skipped — packing ``teammate`` produces an archive with
        whatever combination of kinds happens to have a teammate
        bundle. The archive layout is documented in
        :mod:`bundle_common.pack`.

        Args:
            raw_args: ``<name> [--scope ws|user] [--to <archive>]``.
                Bare ``<name>`` defaults to ``--scope workspace`` and
                writes ``./<name>-<tier>.tar.gz`` under the workspace
                root (or current directory if no workspace is bound).
        """
        try:
            tokens = shlex.split(raw_args or "")
        except ValueError as e:
            return {"error": f"Failed to parse arguments: {e}"}

        name_token: Optional[str] = None
        scope: str = BUNDLE_TIER_WORKSPACE
        output_arg: Optional[str] = None
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--scope":
                if i + 1 >= len(tokens):
                    return {"error": "--scope requires workspace or user"}
                value = tokens[i + 1]
                if value not in VALID_BUNDLE_TIERS:
                    return {"error": f"Unknown scope {value!r}; use workspace or user"}
                scope = value
                i += 2
                continue
            if tok.startswith("--scope="):
                value = tok.split("=", 1)[1]
                if value not in VALID_BUNDLE_TIERS:
                    return {"error": f"Unknown scope {value!r}; use workspace or user"}
                scope = value
                i += 1
                continue
            if tok == "--to":
                if i + 1 >= len(tokens):
                    return {"error": "--to requires a path"}
                output_arg = tokens[i + 1]
                i += 2
                continue
            if tok.startswith("--to="):
                output_arg = tok.split("=", 1)[1]
                i += 1
                continue
            if name_token is not None:
                return {
                    "error": (
                        "Usage: bundle pack <name> [--scope workspace|user] "
                        "[--to <archive>]"
                    )
                }
            name_token = tok
            i += 1

        if name_token is None:
            return {
                "error": (
                    "Usage: bundle pack <name> [--scope workspace|user] "
                    "[--to <archive>]"
                )
            }

        # ``root`` aliases to the empty-string sentinel for symmetry
        # with parse_bundle_ref's accepted forms; users packing the
        # root bundle write 'bundle pack root'.
        bundle_name = "" if name_token in ("root", "(root)") else name_token

        # Determine the archive output path. Defaults match the legacy
        # references-side behaviour: ``<name>-<tier>.tar.gz`` under the
        # workspace root, or under cwd when no workspace is bound.
        if output_arg:
            archive_path = Path(output_arg).expanduser()
            if not archive_path.is_absolute():
                base = self._workspace_path
                if base:
                    archive_path = Path(base) / archive_path
        else:
            stem = bundle_name or "root"
            base = self._workspace_path or "."
            archive_path = Path(base) / f"{stem}-{scope}.tar.gz"

        # The packer needs concrete tier roots for resolve_bundle_roots.
        ws_path = Path(self._workspace_path) if self._workspace_path else None

        try:
            result = pack_bundle_set(
                bundle_name,
                scope,
                archive_path,
                registry=self._registry,
                workspace_path=ws_path,
                jaato_version=self._jaato_version_string(),
            )
        except FileNotFoundError as e:
            return {"error": f"pack: {e}"}
        except OSError as e:
            return {"error": f"pack: I/O error: {e}"}
        except ValueError as e:
            return {"error": f"pack: {e}"}

        size_kb = result.bytes_written / 1024
        kind_lines: List[str] = []
        total_entries = 0
        total_payloads = 0
        for kr in result.kinds:
            total_entries += kr.entry_count
            total_payloads += kr.payload_count
            kind_lines.append(
                f"    {kr.kind}: {kr.entry_count} entries, "
                f"{kr.payload_count} payloads"
            )

        if not result.kinds:
            return {
                "error": (
                    f"no bundle named {name_token!r} in scope {scope!r} "
                    f"across any registered kind"
                )
            }

        lines: List[Tuple[str, str]] = [
            ("PACK", "bold"),
            ("", ""),
            (f"  bundle: {scope}:{bundle_name or '(root)'}", ""),
            (f"  archive: {result.archive_path}", ""),
            (
                f"  total: {len(result.kinds)} kind(s), {total_entries} "
                f"entries, {total_payloads} payloads, {size_kb:.1f} KiB",
                "",
            ),
        ]
        for line in kind_lines:
            lines.append((line, ""))

        return {
            "status": "ok",
            "archive_path": str(result.archive_path),
            "source_name": result.source_name,
            "source_tier": result.source_tier,
            "kinds": [
                {
                    "kind": kr.kind,
                    "bundle_name": kr.bundle_name,
                    "entry_count": kr.entry_count,
                    "payload_count": kr.payload_count,
                }
                for kr in result.kinds
            ],
            "bytes_written": result.bytes_written,
            "help_lines": HelpLines(lines=lines),
        }

    def _cmd_unpack(self, raw_args: str) -> Dict[str, Any]:
        """``bundle unpack <archive> [--scope workspace|user]
        [--into <name>] [--overwrite|--merge] [--no-reconcile]``.

        Reads the archive's envelope, validates that every declared
        kind has a registered handler, then dispatches each kind's
        contents to its handler's domain root. Auto-reconciles each
        affected handler so freshly-installed bundles are immediately
        usable; pass ``--no-reconcile`` to skip when the recipient
        plans to run reconcile by hand later.

        ``--into <name>`` overrides the bundle name on the recipient
        side; without it the archive's recorded ``source_name`` is
        used. ``--scope`` overrides the recipient tier; default is
        ``workspace``.

        Tar-traversal attempts are rejected before any disk writes —
        the safety check lives in :mod:`bundle_common.unpack` and
        applies to every member, including symlink targets.
        """
        try:
            tokens = shlex.split(raw_args or "")
        except ValueError as e:
            return {"error": f"Failed to parse arguments: {e}"}

        archive_token: Optional[str] = None
        target_name: Optional[str] = None
        scope: str = BUNDLE_TIER_WORKSPACE
        mode: UnpackMode = UnpackMode.ERROR
        do_reconcile: bool = True
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--into":
                if i + 1 >= len(tokens):
                    return {"error": "--into requires a name"}
                target_name = tokens[i + 1]
                i += 2
                continue
            if tok.startswith("--into="):
                target_name = tok.split("=", 1)[1]
                i += 1
                continue
            if tok == "--scope":
                if i + 1 >= len(tokens):
                    return {"error": "--scope requires workspace or user"}
                value = tokens[i + 1]
                if value not in VALID_BUNDLE_TIERS:
                    return {"error": f"Unknown scope {value!r}; use workspace or user"}
                scope = value
                i += 2
                continue
            if tok.startswith("--scope="):
                value = tok.split("=", 1)[1]
                if value not in VALID_BUNDLE_TIERS:
                    return {"error": f"Unknown scope {value!r}; use workspace or user"}
                scope = value
                i += 1
                continue
            if tok == "--overwrite":
                mode = UnpackMode.OVERWRITE
                i += 1
                continue
            if tok == "--merge":
                mode = UnpackMode.MERGE
                i += 1
                continue
            if tok == "--no-reconcile":
                do_reconcile = False
                i += 1
                continue
            if archive_token is not None:
                return {
                    "error": (
                        "Usage: bundle unpack <archive> [--scope workspace|user] "
                        "[--into <name>] [--overwrite|--merge] [--no-reconcile]"
                    )
                }
            archive_token = tok
            i += 1

        if archive_token is None:
            return {
                "error": (
                    "Usage: bundle unpack <archive> [--scope workspace|user] "
                    "[--into <name>] [--overwrite|--merge] [--no-reconcile]"
                )
            }

        archive_path = Path(archive_token).expanduser()
        if not archive_path.is_absolute() and self._workspace_path:
            archive_path = Path(self._workspace_path) / archive_path

        ws_path = Path(self._workspace_path) if self._workspace_path else None
        if scope == BUNDLE_TIER_WORKSPACE and ws_path is None:
            return {
                "error": (
                    "unpack: cannot install into the workspace tier — no "
                    "workspace_path is bound. Pass --scope user, or load a "
                    "workspace first."
                )
            }

        try:
            result = unpack_archive(
                archive_path,
                registry=self._registry,
                target_tier=scope,
                target_name=target_name,
                mode=mode,
                workspace_path=ws_path,
            )
        except UnpackError as e:
            return {"error": f"unpack: {e}"}
        except FileNotFoundError as e:
            return {"error": f"unpack: {e}"}

        # Reload + reconcile each handler that received contents. The
        # reload lets the handler pick up the freshly-written manifests
        # and entry files; the reconcile self-heals any sidecar drift
        # (e.g. embedding model differs between packer and recipient).
        reconciled: List[str] = []
        for kr in result.kinds:
            handler = self._registry.get(kr.kind)
            if handler is None:  # pragma: no cover - validated above
                continue
            try:
                handler.reload_catalog()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(
                    "reload_catalog failed for %s: %s", kr.kind, e,
                )
            if not do_reconcile:
                continue
            # Find the freshly-installed bundle on the handler side.
            target_bundle = next(
                (
                    b for b in handler.list_bundles()
                    if b.name == kr.target_name and b.tier == kr.target_tier
                ),
                None,
            )
            if target_bundle is None:
                continue
            try:
                rec = handler.reconcile_bundle(target_bundle)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(
                    "reconcile_bundle failed for %s: %s", kr.kind, e,
                )
                continue
            if rec is not None and hasattr(rec, "summary"):
                reconciled.append(f"{kr.kind}: {rec.summary()}")
            else:
                reconciled.append(kr.kind)

        target_label = (
            f"{result.target_tier}:"
            f"{result.target_name or '(root)'}"
        )
        lines: List[Tuple[str, str]] = [
            ("UNPACK", "bold"),
            ("", ""),
            (f"  archive: {result.archive_path}", ""),
            (f"  installed: {target_label}", ""),
            (f"  format: v{result.format_version}", ""),
            (f"  mode: {result.mode.value}", ""),
        ]
        for kr in result.kinds:
            lines.append((
                f"    {kr.kind}: {kr.entry_count} entries -> {kr.target_dir}",
                "",
            ))
        if reconciled:
            lines.append(("", ""))
            for r in reconciled:
                lines.append((f"  reconciled: {r}", ""))
        elif do_reconcile:
            lines.append(("  reconciled: (no kinds were installed)", "dim"))
        else:
            lines.append(("  reconciled: skipped (--no-reconcile)", "dim"))

        return {
            "status": "ok",
            "archive_path": str(result.archive_path),
            "target": target_label,
            "target_tier": result.target_tier,
            "target_name": result.target_name,
            "format_version": result.format_version,
            "mode": result.mode.value,
            "kinds": [
                {
                    "kind": kr.kind,
                    "target_dir": str(kr.target_dir),
                    "entry_count": kr.entry_count,
                }
                for kr in result.kinds
            ],
            "reconciled": reconciled,
            "help_lines": HelpLines(lines=lines),
        }

    @staticmethod
    def _jaato_version_string() -> str:
        """Best-effort jaato package version for the archive envelope.

        Returns ``"unknown"`` when metadata isn't available (e.g.,
        editable install without installed metadata) — matches the
        references-side behaviour the legacy pack used.
        """
        try:
            from importlib.metadata import PackageNotFoundError, version
            return version("jaato-server")
        except PackageNotFoundError:
            return "unknown"
        except ImportError:
            return "unknown"

    def _cmd_help(self) -> HelpLines:
        return HelpLines(lines=[
            ("Bundle Command", "bold"),
            ("", ""),
            ("Manage knowledge bundles across every registered domain.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    bundle [subcommand] [args]", ""),
            ("", ""),
            ("SUBCOMMANDS", "bold"),
            ("    list", "dim"),
            ("        Show every bundle, grouped by kind. Tier column", "dim"),
            ("        distinguishes workspace from user bundles.", "dim"),
            ("", ""),
            ("    add <kind>:<id> --to <bundle-ref>", "dim"),
            ("        Move an entry into a bundle. <kind> is the domain", "dim"),
            ("        prefix (references, agents, tasks, profiles, services).", "dim"),
            ("        <bundle-ref> is '[<scope>:]<name>'.", "dim"),
            ("", ""),
            ("    eject <kind>:<id>", "dim"),
            ("        Move an entry out of its current bundle, into the", "dim"),
            ("        kind's tier root. The entry stays in the catalog as", "dim"),
            ("        a free (unbundled) item.", "dim"),
            ("", ""),
            ("    remove <kind>:<id>", "dim"),
            ("        Permanently delete an entry's backing file from disk.", "dim"),
            ("", ""),
            ("    create <name> --kind <kind> [--scope workspace|user]", "dim"),
            ("        Create an empty bundle of the given kind. --kind is", "dim"),
            ("        required because each kind has its own manifest format.", "dim"),
            ("", ""),
            ("    delete <bundle-ref> --kind <kind> [--force]", "dim"),
            ("        Remove a single kind's bundle from disk. --force is", "dim"),
            ("        required for non-empty bundles.", "dim"),
            ("", ""),
            ("    reconcile [<bundle-ref>] [--scope workspace|user|all] [--kind <k>]", "dim"),
            ("        Sync bundle manifests with the live catalog. With", "dim"),
            ("        no bundle ref, reconciles every bundle in scope (default", "dim"),
            ("        workspace). With a bundle ref, reconciles that bundle", "dim"),
            ("        across every kind that has it (or just --kind <k>).", "dim"),
            ("", ""),
            ("    pack <name> [--scope workspace|user] [--to <archive>]", "dim"),
            ("        Build a composite .tar.gz holding every kind's bundle", "dim"),
            ("        with the given <name> in the chosen tier. Kinds with", "dim"),
            ("        no matching bundle are silently skipped.", "dim"),
            ("        Default output: ./<name>-<tier>.tar.gz under the workspace.", "dim"),
            ("", ""),
            ("    unpack <archive> [--scope workspace|user] [--into <name>]", "dim"),
            ("                     [--overwrite|--merge] [--no-reconcile]", "dim"),
            ("        Install an archive's contents into the chosen tier.", "dim"),
            ("        Each kind in the archive is routed to its registered", "dim"),
            ("        handler. Auto-reconciles each affected handler so the", "dim"),
            ("        bundles are immediately usable.", "dim"),
            ("", ""),
            ("    help", "dim"),
            ("        Show this help message.", "dim"),
            ("", ""),
            ("ENTRY REFERENCES", "bold"),
            ("    Entries are addressed as <kind>:<id> because names may", "dim"),
            ("    clash across kinds (e.g. references:weekly and tasks:weekly).", "dim"),
            ("    The kind prefix is required for add/eject/remove.", "dim"),
            ("", ""),
            ("BUNDLE REFERENCES", "bold"),
            ("    Bundle containers are addressed as [<scope>:]<name>", "dim"),
            ("    where <scope> is 'workspace' or 'user'. Bare names default", "dim"),
            ("    to the workspace tier.", "dim"),
            ("", ""),
            ("RELATED", "bold"),
            ("    Bundle merge remains under 'references bundle merge' —", "dim"),
            ("    merging two bundles requires per-kind sidecar / index logic", "dim"),
            ("    (re-embedding, conflict resolution by metadata hash, etc.)", "dim"),
            ("    that doesn't generalize to non-references kinds today.", "dim"),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    bundle                                       (same as 'bundle list')", "dim"),
            ("    bundle list                                  Show all bundles", "dim"),
            ("    bundle add references:api-spec --to teammate", "dim"),
            ("    bundle eject references:api-spec", "dim"),
            ("    bundle remove references:api-spec", "dim"),
            ("    bundle pack teammate                         ./teammate-workspace.tar.gz", "dim"),
            ("    bundle pack teammate --scope user --to ~/share.tar.gz", "dim"),
            ("    bundle unpack ./teammate-workspace.tar.gz", "dim"),
            ("    bundle unpack share.tar.gz --into shared --overwrite", "dim"),
        ])

    # ------------------------------------------------------------------
    # Completions
    # ------------------------------------------------------------------

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        if command != "bundle":
            return []

        subcommands = [
            CommandCompletion("list", "Show every bundle, grouped by kind"),
            CommandCompletion("add", "Move an entry into a bundle"),
            CommandCompletion("eject", "Move an entry out of its bundle"),
            CommandCompletion("remove", "Permanently delete an entry"),
            CommandCompletion("create", "Create an empty bundle"),
            CommandCompletion("delete", "Remove a kind's bundle from disk"),
            CommandCompletion("reconcile", "Sync bundle manifests"),
            CommandCompletion("pack", "Build a composite distributable archive"),
            CommandCompletion("unpack", "Install an archive into a tier"),
            CommandCompletion("help", "Show detailed help"),
        ]

        if len(args) <= 1:
            partial = (args[0].lower() if args else "")
            return [s for s in subcommands if s.value.startswith(partial)]

        subcommand = args[0].lower()
        partial = args[-1].lower()

        if subcommand in ("add", "eject", "remove"):
            if len(args) == 2:
                # Offer <kind>:<id> for every loaded entry across kinds.
                options: List[CommandCompletion] = []
                for handler in self._registry.all_handlers():
                    for entry in handler.list_entries():
                        options.append(CommandCompletion(
                            f"{handler.kind}:{entry.id}",
                            f"{handler.kind} entry",
                        ))
                return [o for o in options if o.value.startswith(partial)]

            if subcommand == "add":
                if len(args) >= 3 and args[-2] == "--to":
                    # Bundle-name completions across all kinds.
                    options = []
                    for handler in self._registry.all_handlers():
                        for b in handler.list_bundles():
                            display = "root" if b.name == "" else b.name
                            options.append(CommandCompletion(
                                display,
                                f"Into {handler.kind} bundle {b.qualified_ref}",
                            ))
                            options.append(CommandCompletion(
                                f"{b.tier}:{display}",
                                f"Into {handler.kind} bundle {b.qualified_ref}",
                            ))
                    return [o for o in options if o.value.startswith(partial)]
                if partial.startswith("-") or not partial:
                    flags = [CommandCompletion("--to", "Target bundle (scope:name)")]
                    return [f for f in flags if f.value.startswith(partial or "-")]

        if subcommand == "pack":
            if len(args) == 2:
                # Offer bare bundle names — pack is composite, so the
                # name (not the kind-qualified form) is what matters.
                names: Set[str] = set()
                for handler in self._registry.all_handlers():
                    for b in handler.list_bundles():
                        names.add("root" if b.name == "" else b.name)
                options = [
                    CommandCompletion(n, f"Pack '{n}' across all kinds")
                    for n in sorted(names)
                ]
                return [o for o in options if o.value.startswith(partial)]
            if len(args) >= 3 and args[-2] == "--scope":
                scopes = [
                    CommandCompletion("workspace", "Workspace tier (default)"),
                    CommandCompletion("user", "User tier (~/.jaato/<kind>)"),
                ]
                return [s for s in scopes if s.value.startswith(partial)]
            if partial.startswith("-") or not partial:
                flags = [
                    CommandCompletion("--scope", "Source tier (workspace|user)"),
                    CommandCompletion("--to", "Output archive path"),
                ]
                return [f for f in flags if f.value.startswith(partial or "-")]

        if subcommand == "unpack":
            # First positional is a path — leave to the client's
            # filename completion.
            if len(args) >= 3 and args[-2] == "--scope":
                scopes = [
                    CommandCompletion("workspace", "Install into workspace tier"),
                    CommandCompletion("user", "Install into user tier"),
                ]
                return [s for s in scopes if s.value.startswith(partial)]
            if partial.startswith("-") or not partial:
                flags = [
                    CommandCompletion("--scope", "Destination tier (workspace|user)"),
                    CommandCompletion("--into", "Override the bundle name"),
                    CommandCompletion("--overwrite", "Replace existing bundles"),
                    CommandCompletion("--merge", "Merge into existing bundles"),
                    CommandCompletion("--no-reconcile", "Skip post-unpack reconcile"),
                ]
                return [f for f in flags if f.value.startswith(partial or "-")]

        if subcommand == "create":
            if len(args) >= 3 and args[-2] == "--kind":
                kinds = [
                    CommandCompletion(k, f"Create a {k} bundle")
                    for k in self._registry.kinds()
                ]
                return [c for c in kinds if c.value.startswith(partial)]
            if len(args) >= 3 and args[-2] == "--scope":
                scopes = [
                    CommandCompletion("workspace", "Workspace tier (default)"),
                    CommandCompletion("user", "User tier"),
                ]
                return [s for s in scopes if s.value.startswith(partial)]
            if partial.startswith("-") or not partial:
                flags = [
                    CommandCompletion("--kind", "Bundle kind (required)"),
                    CommandCompletion("--scope", "Tier (workspace|user)"),
                ]
                return [f for f in flags if f.value.startswith(partial or "-")]

        if subcommand == "delete":
            # First positional: bundle name across registered kinds.
            if len(args) == 2:
                names: Set[str] = set()
                for handler in self._registry.all_handlers():
                    for b in handler.list_bundles():
                        names.add("root" if b.name == "" else b.name)
                options = [
                    CommandCompletion(n, "Delete bundle (qualify with --kind)")
                    for n in sorted(names)
                ]
                return [o for o in options if o.value.startswith(partial)]
            if len(args) >= 3 and args[-2] == "--kind":
                kinds = [
                    CommandCompletion(k, f"Delete a {k} bundle")
                    for k in self._registry.kinds()
                ]
                return [c for c in kinds if c.value.startswith(partial)]
            if partial.startswith("-") or not partial:
                flags = [
                    CommandCompletion("--kind", "Bundle kind (required)"),
                    CommandCompletion("--force", "Delete non-empty bundles"),
                ]
                return [f for f in flags if f.value.startswith(partial or "-")]

        if subcommand == "reconcile":
            # First positional: bundle ref, but optional.
            if len(args) == 2 and not partial.startswith("-"):
                names = set()
                for handler in self._registry.all_handlers():
                    for b in handler.list_bundles():
                        names.add("root" if b.name == "" else b.name)
                options = [
                    CommandCompletion(n, "Reconcile this bundle")
                    for n in sorted(names)
                ]
                return [o for o in options if o.value.startswith(partial)]
            if len(args) >= 3 and args[-2] == "--scope":
                scopes = [
                    CommandCompletion("workspace", "Workspace tier (default)"),
                    CommandCompletion("user", "User tier"),
                    CommandCompletion("all", "Both tiers"),
                ]
                return [s for s in scopes if s.value.startswith(partial)]
            if len(args) >= 3 and args[-2] == "--kind":
                kinds = [
                    CommandCompletion(k, f"Limit to {k}")
                    for k in self._registry.kinds()
                ]
                return [c for c in kinds if c.value.startswith(partial)]
            if partial.startswith("-") or not partial:
                flags = [
                    CommandCompletion("--scope", "Tier filter (workspace|user|all)"),
                    CommandCompletion("--kind", "Limit to one kind"),
                ]
                return [f for f in flags if f.value.startswith(partial or "-")]

        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup_source_bundle(
        self, handler: BundleEntryHandler, entry: BundleEntry,
    ) -> Optional[Bundle]:
        """Return the :class:`Bundle` an entry currently belongs to.

        ``None`` when the entry is free. Used so the post-mutation
        reconcile knows which bundle's sidecar needs updating after
        the entry leaves.
        """
        if not entry.bundle_name:
            return None
        for b in handler.list_bundles():
            if b.name == entry.bundle_name and b.tier == entry.bundle_tier:
                return b
        return None

    def _post_membership_change(
        self,
        *,
        handler: BundleEntryHandler,
        verb: str,
        entry_token: str,
        source_bundle: Optional[Bundle],
        target_bundle: Optional[Bundle],
    ) -> Dict[str, Any]:
        """Reload the handler's catalog and reconcile both sides.

        The handler is responsible for re-attaching its own matchers
        inside :meth:`BundleEntryHandler.reconcile_bundle`; the bundle
        plugin only orchestrates the call order.
        """
        handler.reload_catalog()

        reconciled: List[str] = []
        for bundle in (source_bundle, target_bundle):
            if bundle is None:
                continue
            # Re-resolve against the post-reload bundle list — the
            # original Bundle dataclass we hold may be stale.
            live = next(
                (
                    b for b in handler.list_bundles()
                    if b.name == bundle.name and b.tier == bundle.tier
                ),
                None,
            )
            if live is None:
                continue
            try:
                rec = handler.reconcile_bundle(live)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("reconcile failed for %s: %s", live.qualified_ref, e)
                continue
            if rec is not None and hasattr(rec, "summary"):
                reconciled.append(f"{live.qualified_ref}: {rec.summary()}")
            else:
                reconciled.append(live.qualified_ref)

        lines: List[Tuple[str, str]] = [
            (verb.upper(), "bold"),
            ("", ""),
            (f"  entry: {entry_token}", ""),
        ]
        if source_bundle is not None:
            lines.append((f"  from: {source_bundle.qualified_ref}", ""))
        if target_bundle is not None:
            lines.append((f"  to: {target_bundle.qualified_ref}", ""))
        for summary in reconciled:
            lines.append((f"  reconciled: {summary}", ""))

        return {
            "status": "ok",
            "verb": verb,
            "entry": entry_token,
            "source_bundle": source_bundle.qualified_ref if source_bundle else None,
            "target_bundle": target_bundle.qualified_ref if target_bundle else None,
            "reconciled": reconciled,
            "help_lines": HelpLines(lines=lines),
        }


def create_plugin() -> BundlePlugin:
    """Factory called by the plugin registry."""
    return BundlePlugin()
