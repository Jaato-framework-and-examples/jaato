"""A rejected ``category_id`` must hand back the ids that would have worked.

Category ids are hashed on purpose -- ``filesystem`` is ``c_cbf61858`` --
so that tool discovery cannot be short-circuited by a model reasoning from
names.  The cost of that design is that a wrong ``category_id`` is not a
typo: it is the certain first move of any model that has not already called
``list_tools()`` with no arguments.  Observed on a live session's FIRST
turn, in parallel with two other calls, before any listing had returned::

    {"name": "t_6fdebc90", "arguments": "{\\"category_id\\": \\"filesystem\\"}"}
    -> {"error": "Unknown category_id 'filesystem'."}

The parameter description already says, emphatically, that the id must come
from a prior ``list_tools()`` call.  Prose does not survive a model that
guesses; the error is the second line of defence, and it used to be a bare
string -- naming neither the valid ids nor how to obtain them, while
holding every one of them in ``cat_id_to_name`` at that exact line.

These pin the recoverable shape.  Note what they do NOT assert: that an
error is returned at all.  That assertion passed against the dead-end
version and would keep passing through a regression back to it.  What has
to hold is that the reply carries a usable id.

Same family as #750's ``unreadable_arguments_error``, and the same bar the
scaffold holds itself to in
``test_scaffold_archetype_docs.py::test_unknown_archetype_names_the_accepted_ones``.
"""

from shared.tool_id_map import name_to_id

from .test_plugin import _make_test_env


def _list_tools(plugin, **args):
    return plugin.get_executors()["list_tools"](args)


class _SessionWithPlugins:
    """Minimal stand-in for a session that restricts its plugin set."""

    def __init__(self, plugins):
        self._tool_plugins = set(plugins)


def test_unknown_category_id_returns_valid_ids() -> None:
    """The rejection must carry at least one id the caller can retry with."""
    plugin, _ = _make_test_env()

    result = _list_tools(plugin, category_id="filesystem_nope")

    ids = [c["id"] for c in result.get("categories", [])]
    assert ids, (
        "list_tools rejected an unknown category_id without returning a "
        "single valid one, so the model's only move is another guess "
        "against hashed ids it cannot derive.  Every valid id is in "
        "cat_id_to_name at the line that raises this error -- return them."
    )
    assert all(i.startswith("c_") for i in ids), (
        f"Recovery payload lists something other than category ids: {ids}"
    )


def test_returned_id_actually_works() -> None:
    """An id from the rejection must succeed on the very next call.

    Naming ids that don't resolve would be a worse dead end than naming
    none, so the round-trip is closed here rather than assumed.
    """
    plugin, _ = _make_test_env()

    rejection = _list_tools(plugin, category_id="not-an-id")
    first = rejection["categories"][0]

    retry = _list_tools(plugin, category_id=first["id"])

    assert retry.get("category") == first["name"], (
        f"Retrying with the id list_tools itself offered "
        f"({first['id']} for {first['name']!r}) did not resolve: {retry}"
    )
    assert "error" not in retry


def test_error_string_is_kept_alongside_the_ids() -> None:
    """The call still has to read as a failure, not a silent summary."""
    plugin, _ = _make_test_env()

    result = _list_tools(plugin, category_id="bogus")

    assert "bogus" in result.get("error", ""), (
        "The rejection no longer echoes the offending value, so a model "
        "issuing several calls in parallel cannot tell which one failed."
    )


def test_a_category_name_is_answered_with_its_id() -> None:
    """Passing the NAME instead of the id is the observed failure; say so.

    ``filesystem`` was guessed as a category_id on turn one.  When the
    guess IS a real category's name, the correction is one specific id --
    surface it rather than making the model rescan the list.
    """
    plugin, _ = _make_test_env()

    result = _list_tools(plugin, category_id="filesystem")

    assert result.get("did_you_mean", {}).get("category_id") == name_to_id(
        "filesystem", prefix="c"
    ), (
        "A guess that exactly matches a category name was rejected without "
        "naming that category's id, which is the single most likely "
        f"correction: {result.get('did_you_mean')}"
    )


def test_recovery_does_not_leak_categories_the_session_cannot_use() -> None:
    """The recovery payload inherits the summary's session filtering.

    The summary hides categories whose tools belong to plugins outside the
    session's profile -- listing them primes the model to invoke tools it
    cannot reach.  Recovery replays that same payload precisely so it
    cannot become a back door around the filter.
    """
    plugin, _ = _make_test_env()
    plugin.set_session(_SessionWithPlugins({"file_edit"}))

    result = _list_tools(plugin, category_id="whatever")

    names = {c["name"] for c in result.get("categories", [])}
    assert "filesystem" in names
    assert "search" not in names and "coordination" not in names, (
        f"Recovery disclosed categories the session has no access to: {names}"
    )
    assert "did_you_mean" not in _list_tools(plugin, category_id="search"), (
        "The did_you_mean correction resolved a hidden category's name to "
        "its id, leaking exactly what the summary filter withholds."
    )
