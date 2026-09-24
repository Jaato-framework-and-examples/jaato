"""``server.session_groups``: which sessions may message each other.

The predicate is derived from two facts the daemon stamps -- the cascade
(``cascade_driver_id``) and the authenticated creator (``created_by``) --
and the rules pinned here are the ones a wrong predicate would break
silently: two anonymous sessions must NOT form a group, an empty string is
an absent fact, and a persisted record missing the owner field joins no
user group rather than raising.
"""
from types import SimpleNamespace as NS

from jaato_server.server.session_groups import (
    CASCADE_KEY_PREFIX,
    USER_KEY_PREFIX,
    common_groups,
    group_keys,
    group_keys_for,
    same_group,
)


def test_a_cascade_and_an_owner_each_yield_a_key():
    keys = group_keys(NS(cascade_driver_id="c1", created_by="app:alice"))
    assert keys == {CASCADE_KEY_PREFIX + "c1", USER_KEY_PREFIX + "app:alice"}


def test_none_never_matches_none():
    """Two anonymous IPC sessions must not form a daemon-wide group."""
    a = NS(cascade_driver_id=None, created_by=None)
    b = NS(cascade_driver_id=None, created_by=None)
    assert group_keys(a) == frozenset()
    assert same_group(a, b) is False


def test_an_empty_string_is_an_absent_fact():
    """``created_by=""`` is the fail-open shape #1074 refuses at the door;
    the predicate reads it the same way."""
    assert group_keys_for("", "") == frozenset()
    assert same_group(NS(cascade_driver_id="", created_by=""),
                      NS(cascade_driver_id="", created_by="")) is False


def test_same_cascade_is_a_group_whatever_the_owners():
    a = NS(cascade_driver_id="c1", created_by="app:alice")
    b = NS(cascade_driver_id="c1", created_by=None)
    assert same_group(a, b)
    assert common_groups(a, b) == {CASCADE_KEY_PREFIX + "c1"}


def test_same_owner_is_a_group_across_cascades_and_workspaces():
    a = NS(cascade_driver_id="c1", created_by="app:alice")
    b = NS(cascade_driver_id="c2", created_by="app:alice")
    assert common_groups(a, b) == {USER_KEY_PREFIX + "app:alice"}


def test_owners_are_application_qualified_so_apps_do_not_collide():
    """``alice`` of one application is not ``alice`` of another."""
    a = NS(cascade_driver_id=None, created_by="app1:alice")
    b = NS(cascade_driver_id=None, created_by="app2:alice")
    assert same_group(a, b) is False


def test_a_record_without_the_owner_field_joins_no_user_group():
    """A ``SessionInfo`` predating ``created_by`` has no attribute at all;
    it must read as absent, never raise."""
    old = NS(cascade_driver_id="c1")
    assert group_keys(old) == {CASCADE_KEY_PREFIX + "c1"}
    assert same_group(old, NS(cascade_driver_id=None, created_by="app:x")) is False


def test_the_predicate_is_symmetric():
    a = NS(cascade_driver_id="c1", created_by="app:a")
    b = NS(cascade_driver_id="c2", created_by="app:a")
    assert same_group(a, b) == same_group(b, a)
