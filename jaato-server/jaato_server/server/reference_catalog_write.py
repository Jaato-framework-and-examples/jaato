"""Write one file of a workspace's reference catalog, then reconcile its index (#1422).

The one writer of ``<workspace>/.jaato/references/**`` on a person's
behalf.  The daemon decides WHAT to write -- the owner gate, the claim's
re-validation, the origin stamp (``reference_curation``), a reference's new
links (``reference_catalog``) -- and hands the bytes to the runner serving a
session in that workspace (``session.write_reference``), which calls
:func:`write_catalog_file` in its BASE profile.  Template v44 lets base
write the catalog; model-called tool bodies run in ``tool_hat``, which
still denies it.

The runner is where the embedding model is loaded, so the bundle's vector
index is reconciled there with the references plugin's own provider:
``reconcile_bundle`` runs unchanged, with no vectors crossing a process
boundary.

With no session in the workspace to ask, the daemon calls the same function
itself with no provider: the entry is placed and an indexed bundle is
reported ``unavailable``, as before #1422.

Stdlib plus the references plugin's helpers and :mod:`.contained_write`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from jaato_server.shared.plugins.references.bundle import (
    BUNDLE_TIER_WORKSPACE,
    ROOT_BUNDLE_NAME,
    ReferenceBundle,
    load_reference_bundle,
)
from jaato_server.shared.plugins.references.config_loader import discover_references

from .contained_write import PathLeavesRoot, contained_dir, write_contained

logger = logging.getLogger(__name__)

#: Where the catalog lives, relative to the workspace.
CATALOG_REL = ".jaato/references"

#: ``reconcile_bundle``'s status values, as the promotion answer spells them.
_RECONCILE_STATUS = {"updated": "updated", "clean": "clean",
                     "skipped_busy": "busy", "unavailable": "unavailable",
                     "error": "error"}


def destination_rel(bundle: str) -> str:
    """The workspace-relative directory of ``bundle`` (``""`` = the catalog root)."""
    return f"{CATALOG_REL}/{bundle}" if bundle else CATALOG_REL


def destination_bundle(root: str, bundle: str) -> Optional[ReferenceBundle]:
    """The workspace-tier bundle ``bundle`` names, or ``None`` when there is none."""
    try:
        directory = contained_dir(root, destination_rel(bundle), create=False)
    except PathLeavesRoot:
        return None
    if not directory:
        return None
    return load_reference_bundle(Path(directory), name=bundle or ROOT_BUNDLE_NAME,
                                 tier=BUNDLE_TIER_WORKSPACE)


def _usable(provider: Any) -> bool:
    """Whether ``provider`` can embed now, loading its model if it must."""
    if provider is None:
        return False
    if not getattr(provider, "available", False):
        try:
            provider.load_model()
        except Exception:  # noqa: BLE001 -- reported as unavailable
            logger.warning("reference reconcile: the embedding model did not load",
                           exc_info=True)
            return False
    return bool(getattr(provider, "available", False))


def reconcile_destination(
    root: str, bundle: str, provider: Any, ref_id: str,
) -> Tuple[str, str]:
    """Bring ``bundle``'s vector index up to date: ``(outcome, detail)``.

    ``none`` when the bundle declares no index.  ``unavailable`` with no
    provider, or one whose model is not the index's (vectors from two
    models are not comparable).  Otherwise ``reconcile_bundle`` runs with
    ``provider``; ``updated`` means ``ref_id`` got its row, and a reconcile
    that skipped it is ``error`` with the reason.
    """
    from jaato_server.shared.plugins.references.reconcile import reconcile_bundle

    dest = destination_bundle(root, bundle)
    if dest is None or not dest.has_index:
        return "none", ""
    if not _usable(provider):
        return "unavailable", "no embedding provider is available in this session"
    if provider.model_name != dest.embedding_model:
        return "unavailable", (f"the session embeds with {provider.model_name!r} and the "
                               f"index was built with {dest.embedding_model!r}")
    sources = discover_references(str(dest.directory), base_path=str(dest.directory.parent),
                                  project_root=root)
    for source in sources:
        source.bundle_name = dest.name
    result = reconcile_bundle(dest, sources, provider)
    skipped = dict(result.skipped)
    if ref_id in skipped:
        return "error", f"'{ref_id}' was not embedded: {skipped[ref_id]}"
    return _RECONCILE_STATUS.get(result.status.value, "error"), result.error or ""


def write_catalog_file(
    root: str, rel_file: str, data: str, *, replace: bool = False,
    bundle: str = "", ref_id: str = "", reconcile: bool = False,
    provider: Any = None,
) -> Dict[str, Any]:
    """Write ``rel_file`` under ``root`` and, when asked, reconcile ``bundle``.

    Args:
        root: The workspace, resolved.
        rel_file: Workspace-relative path; must lie under ``.jaato/references``.
        data: The file's text.
        replace: Overwrite an existing file (a links edit); otherwise an
            existing file is a ``collision``.
        bundle: The bundle the file belongs to (``""`` = the root).
        ref_id: The reference the file defines, for the reconcile report.
        reconcile: Reconcile the bundle's vector index afterwards.
        provider: The embedding provider, ``None`` when this process has
            none (the index is then ``unavailable``).

    Returns:
        ``{"ok", "category", "error", "reconcile", "reconcile_detail"}``.
        Categories: ``invalid_request``, ``collision``, ``unsafe_path``,
        ``io_error``.  Never raises for a refusal.
    """
    answer: Dict[str, Any] = {"ok": False, "category": "", "error": "",
                              "reconcile": "", "reconcile_detail": ""}
    norm = os.path.normpath(rel_file or "")
    if not isinstance(data, str) or not norm.startswith(CATALOG_REL + os.sep):
        answer.update(category="invalid_request",
                      error=f"{rel_file!r} is not a file under {CATALOG_REL}")
        return answer
    if not replace and os.path.lexists(os.path.join(root, norm)):
        answer.update(category="collision", error=f"{norm} already exists")
        return answer
    try:
        write_contained(root, norm, data.encode("utf-8"))
    except PathLeavesRoot as exc:
        answer.update(category="unsafe_path", error=str(exc))
        return answer
    except OSError as exc:
        answer.update(category="io_error", error=f"could not write {norm}: {exc}")
        return answer
    answer["ok"] = True
    if reconcile:
        answer["reconcile"], answer["reconcile_detail"] = reconcile_destination(
            root, bundle, provider, ref_id)
    return answer
