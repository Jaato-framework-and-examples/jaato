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
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        if subcommand == "help":
            return self._cmd_help()
        return {
            "error": (
                f"Unknown subcommand: {subcommand}. Use: list, add, eject, "
                f"remove, help. (Bundle creation, deletion, reconcile, "
                f"merge, pack and unpack remain under 'references bundle "
                f"<verb>' until they are generalized.)"
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
            ("    Bundle creation, deletion, reconcile, merge, pack, and", "dim"),
            ("    unpack remain under 'references bundle <verb>' until the", "dim"),
            ("    cross-kind generalization lands.", "dim"),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    bundle                                       (same as 'bundle list')", "dim"),
            ("    bundle list                                  Show all bundles", "dim"),
            ("    bundle add references:api-spec --to teammate", "dim"),
            ("    bundle eject references:api-spec", "dim"),
            ("    bundle remove references:api-spec", "dim"),
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
