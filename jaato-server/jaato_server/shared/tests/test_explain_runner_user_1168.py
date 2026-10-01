"""``jaato-scaffold explain runner-user`` describes the policy the code runs.

#1168 step 3 added ``--runner-uid-policy {daemon,peer,workspace-owner}``,
and the only ``explain`` surface that named it was one line in ``explain
env``: nothing said when a policy keeps the daemon's uid, when a session
is refused, what is handed to the target or where credentials go.

The topic READS ``server/runner_user.py`` rather than restating it, and
these tests hold that: every policy the resolver accepts is described,
and every reason the resolver gives for keeping the daemon's uid is one
the topic shows.
"""

from __future__ import annotations

from jaato_server.server import runner_user as ru
from jaato_server.shared.scaffold import explain
from jaato_server.shared.scaffold.introspection_verbs import _SCOPES
from jaato_server.shared.tests.reversion import Reversion

_RU = "jaato-server/jaato_server/server/runner_user.py"

REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/introspection_verbs.py",
        find='    "runner-user": ExplainScope(_explain.runner_user,\n',
        replace='    "runner-user-gone": ExplainScope(_explain.runner_user,\n',
        test="test_the_topic_is_registered",
        because="the topic leaves the explain dispatch, so the three "
                "policies are back to one line in `explain env`",
    ),
    Reversion(
        target=_RU,
        find="            return None, None, REASON_NO_PEER",
        replace='            return None, None, "peer absent"',
        test="test_every_fallback_reason_is_one_the_topic_shows",
        because="the resolver gives a reason for keeping root that the "
                "topic does not name, so the two describe different rules",
    ),
    Reversion(
        target=_RU,
        find="        return None, None, REASON_NO_WORKSPACE",
        replace='        return None, None, "workspace missing"',
        test="test_every_fallback_reason_is_one_the_topic_shows",
        because="as above, for workspace-owner without a workspace",
    ),
]


def test_the_topic_is_registered():
    assert "runner-user" in _SCOPES


def test_every_policy_is_described():
    data, text = explain.runner_user()
    assert set(data["policies"]) == set(ru.POLICIES)
    assert data["default"] == ru.POLICY_DAEMON
    for policy in ru.POLICIES:
        assert policy in text


def test_every_fallback_reason_is_one_the_topic_shows():
    data, _text = explain.runner_user()
    shown = " ".join(data["keeps_daemon_uid_when"])
    reasons = [
        ru._target_uid_gid(ru.POLICY_PEER, None, None)[2],
        ru._target_uid_gid(ru.POLICY_WORKSPACE_OWNER, None, None)[2],
        ru._target_uid_gid(ru.POLICY_WORKSPACE_OWNER, None,
                           "/nonexistent/jaato-1168")[2],
    ]
    for reason in reasons:
        # The stat failure appends the OSError; the stable part is the prefix.
        stable = reason.split(" (")[0] if reason.startswith(
            ru.REASON_WORKSPACE_UNSTATABLE) else reason
        assert stable in shown, reason


def test_the_handed_over_paths_and_refusal_inputs_come_from_the_resolver():
    data, text = explain.runner_user()
    dirs, files = ru.runner_owned_paths(
        session_id="<session_id>", workspace_path="<ws>",
        session_tmp="<session tmpdir>", private_tmp="<ws>/.tmp",
        workspace_home="<ws>/.home", log_path="<runner log>",
    )
    assert data["handed_to_target"] == {"directories": dirs, "files": files}
    assert data["must_be_readable"] == ru.runner_import_paths()
    assert "<ws>/.jaato/logs/" in text
