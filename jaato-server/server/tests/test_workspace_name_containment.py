"""A workspace NAME may not leave the workspace root.

``WorkspaceManager`` turns a client-supplied name into a path by joining
it onto ``workspace_root``.  The join contains nothing by itself, and
``select_workspace`` validated nothing at all, so on a WS daemon any
authenticated client could name a path outside the root that the server
would then open, analyse, and persist into its registry:

    root / "../../etc"    ->  <root>/../../etc     (".." kept verbatim)
    root / "/etc/passwd"  ->  /etc/passwd          (left operand discarded)
    root / ".."           ->  the root's PARENT    (no separator involved)

Every deny case here points at a directory that EXISTS, because the
refusal must come from the containment check rather than from the
``path.exists()`` test one line below it -- against the unfixed code
those selections SUCCEEDED, and a test using a non-existent target would
pass either way and prove nothing.

Provisioning is the default on WS (``session.new`` auto-provisions under
``{root}/sessions/{id}``), so this is the escape hatch a tenant opts into
via ``workspace.select``, not the normal path.  Note what it does NOT
bound: every tenant's provisioned workspace is a sibling under one root,
so containment stops a name leaving the root and says nothing about which
workspace *inside* it a client may select.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from server.workspace_manager import WorkspaceContainmentError, WorkspaceManager


@pytest.fixture
def tree(tmp_path: Path):
    """A workspace root, plus a populated directory OUTSIDE it.

    ``outside`` carries a ``.env`` naming a provider, so a leak through
    ``get_config_status`` would be visible as a value rather than only as
    a path.
    """
    root = tmp_path / "ws-root"
    (root / "alpha" / ".jaato").mkdir(parents=True)
    (root / "alpha" / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-inside\n")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ".env").write_text("GITHUB_TOKEN=ghp-leaked\nMODEL_NAME=secret-model\n")

    return root, outside, tmp_path / "registry.json"


def _manager(tree) -> WorkspaceManager:
    root, _outside, registry = tree
    # An explicit registry path: the default is the REAL ~/.jaato/workspaces.json
    # and a test must not write the developer's own registry.
    return WorkspaceManager(str(root), registry_path=registry)


# ----------------------------------------------------------------------
# The control.  Without it every refusal below proves only that something
# said no.
# ----------------------------------------------------------------------


def test_a_workspace_under_the_root_still_selects(tree) -> None:
    manager = _manager(tree)
    info = manager.select_workspace("alpha")
    assert Path(info.path).resolve() == (tree[0] / "alpha").resolve()


# ----------------------------------------------------------------------
# select_workspace
# ----------------------------------------------------------------------


def test_a_relative_escape_is_refused(tree) -> None:
    root, outside, _ = tree
    name = os.path.relpath(outside, root)          # "../outside"
    assert (root / name).exists(), "the target must exist or this proves nothing"

    with pytest.raises(WorkspaceContainmentError):
        _manager(tree).select_workspace(name)


def test_an_absolute_name_is_refused(tree) -> None:
    """pathlib discards the root for an absolute right operand, so this
    never touched the workspace root at all."""
    _root, outside, _ = tree
    assert outside.exists()

    with pytest.raises(WorkspaceContainmentError):
        _manager(tree).select_workspace(str(outside))


def test_a_bare_dotdot_is_refused_although_it_carries_no_separator(tree) -> None:
    """Why the rule is containment rather than a separator check: ``..``
    resolves to the root's parent and contains neither "/" nor "\\"."""
    with pytest.raises(WorkspaceContainmentError):
        _manager(tree).select_workspace("..")


def test_a_symlink_is_judged_by_its_target(tree) -> None:
    """A link planted inside the root is model-reachable -- the agent's own
    file tools write into provisioned workspaces under this same root."""
    root, outside, _ = tree
    (root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkspaceContainmentError):
        _manager(tree).select_workspace("linked")


def test_a_symlink_that_stays_inside_the_root_is_accepted(tree) -> None:
    """The control for the case above: resolving is not the same as
    refusing every link."""
    root, _outside, _ = tree
    (root / "inside-link").symlink_to(root / "alpha", target_is_directory=True)

    info = _manager(tree).select_workspace("inside-link")
    assert Path(info.path).resolve() == (root / "alpha").resolve()


def test_the_refusal_is_a_valueerror_so_the_ws_handler_still_reports_it(
    tree,
) -> None:
    """``_handle_workspace_select`` converts ``ValueError`` into an error
    frame; a refusal outside that hierarchy would crash the handler
    instead of answering the client."""
    assert issubclass(WorkspaceContainmentError, ValueError)

    try:
        _manager(tree).select_workspace("..")
    except ValueError as e:
        assert "outside the workspace root" in str(e)
    else:                                          # pragma: no cover
        pytest.fail("expected a refusal")


def test_containment_is_checked_before_existence(tree) -> None:
    """So a refusal does not double as an oracle for what exists out
    there: a present and an absent out-of-root target answer alike."""
    root, outside, _ = tree
    manager = _manager(tree)

    present = os.path.relpath(outside, root)
    absent = os.path.relpath(outside.parent / "no-such-dir", root)

    errors = []
    for name in (present, absent):
        with pytest.raises(WorkspaceContainmentError) as exc:
            manager.select_workspace(name)
        errors.append(type(exc.value))
    assert errors[0] is errors[1]


# ----------------------------------------------------------------------
# create_workspace -- two checks, each catching what the other does not
# ----------------------------------------------------------------------


def test_create_refuses_dotdot_which_its_name_check_allows(tree) -> None:
    """``".."`` has no separator, so the pre-existing character check
    passes it; containment is what refuses it."""
    with pytest.raises(WorkspaceContainmentError):
        _manager(tree).create_workspace("..")


def test_create_still_refuses_a_nested_name_which_containment_allows(
    tree,
) -> None:
    """The other half: ``"a/b"`` resolves under the root perfectly well,
    and the naming check is what refuses it -- so neither check is
    masking the other."""
    manager = _manager(tree)
    with pytest.raises(ValueError) as exc:
        manager.create_workspace("a/b")
    assert not isinstance(exc.value, WorkspaceContainmentError)


def test_create_still_works(tree) -> None:
    info = _manager(tree).create_workspace("beta")
    assert Path(info.path).resolve() == (tree[0] / "beta").resolve()
    assert (tree[0] / "beta" / ".jaato").is_dir()


# ----------------------------------------------------------------------
# The two accessors, which refuse by ANSWERING rather than by raising
# ----------------------------------------------------------------------


def test_config_status_does_not_read_an_env_outside_the_root(tree) -> None:
    root, outside, _ = tree
    status = _manager(tree).get_config_status(os.path.relpath(outside, root))

    assert status["configured"] is False
    assert status.get("provider") is None
    assert "secret-model" not in repr(status)
    assert status["missing_fields"] == ["workspace is outside the workspace root"]


def test_config_status_still_reports_a_workspace_inside_the_root(tree) -> None:
    status = _manager(tree).get_config_status("alpha")
    assert status["configured"] is True
    assert status["provider"] == "anthropic"


def test_workspace_path_refuses_an_escaping_name_with_none(tree) -> None:
    """An accessor stays total -- a refusal is its existing "no such
    workspace" answer, not an exception its callers never expected."""
    root, outside, _ = tree
    assert _manager(tree).get_workspace_path(os.path.relpath(outside, root)) is None


def test_workspace_path_refuses_a_poisoned_registry_row(tree) -> None:
    """One accepted out-of-root selection used to be PERSISTED, so the
    stored path is checked rather than trusted.  Since the stale-row
    reconciliation (``test_workspace_registry_stale_rows.py``) the row is
    refused one step earlier -- ``_load_registry`` keeps only rows whose
    path is ``<root>/<name>`` -- and the accessor's own check stays as the
    second door, so neither answer may ever be the out-of-root path."""
    root, outside, registry = tree
    registry.write_text(json.dumps({
        "root": str(root),
        "workspaces": [{"name": "alpha", "path": str(outside), "configured": True}],
    }))

    manager = _manager(tree)                       # the row is not loaded
    assert "alpha" not in manager._workspaces
    answer = manager.get_workspace_path("alpha")
    assert answer != Path(outside)
    assert answer is None or answer == (root / "alpha").resolve()


def test_workspace_path_still_answers_for_a_contained_row(tree) -> None:
    manager = _manager(tree)
    manager.discover_workspaces()
    path = manager.get_workspace_path("alpha")
    assert path is not None and path.resolve() == (tree[0] / "alpha").resolve()
