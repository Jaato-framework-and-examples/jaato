"""Path containment for interactive_shell (jaato issues #722, #503).

`cli` has always refused a command naming a path outside the session
workspace; `interactive_shell`, which spawns a real PTY and is strictly
more capable, refused nothing.  These tests pin the three things #722
changed:

1. a spawn command is checked exactly as a `cli` command is, and **fails
   closed** when the analyzer cannot model it;
2. text typed into a live session is checked too, but **fails open** on a
   parse failure — that text is whatever the running program reads;
3. the absence of kernel confinement is announced, and can be made to
   fail closed with ``require_confinement``.

Plus the `cwd` half of #503: ``ShellSession`` verifies its working
directory against the workspace before spawning.

Most cases need no PTY backend, because a refusal happens *before* any
process is spawned — the ones that do drive a live session are marked and
skipped when ``pexpect`` (the ``jaato-server[interactive]`` extra) is
absent.
"""

import logging

import pytest

from jaato_sdk.plugins.model_provider.types import WithMetadata

from shared.plugins.interactive_shell.plugin import create_plugin
from shared.plugins.interactive_shell.session import (
    ShellSession,
    _BACKEND,
    _verify_cwd_within,
)


needs_backend = pytest.mark.skipif(
    _BACKEND is None,
    reason="no PTY backend installed (jaato-server[interactive])",
)


def _unpack(result):
    """Unpack an executor result to its model-facing dict."""
    if isinstance(result, WithMetadata):
        return result.result
    if isinstance(result, tuple):
        return result[1] if isinstance(result[0], bool) else result[0]
    return result


@pytest.fixture
def plugin(tmp_path):
    """An initialized plugin sandboxed to a temp workspace."""
    p = create_plugin()
    p.initialize({
        'workspace_root': str(tmp_path),
        'max_sessions': 4,
        'max_lifetime': 30,
        'max_idle': 15,
        'idle_timeout': 0.3,
    })
    yield p
    p.shutdown()


class TestSpawnContainment:
    """shell_spawn refuses what cli_based_tool refuses."""

    def test_absolute_path_outside_workspace_is_refused(self, plugin):
        result = plugin._exec_spawn({'command': 'cat /etc/hostname'})
        assert 'error' in result
        assert '/etc/hostname' in result['error']
        assert 'session_id' not in result

    def test_traversal_out_of_workspace_is_refused(self, plugin):
        # Enough ``..`` to clamp at ``/`` whatever the workspace depth.
        # A shallower traversal from a tmp_path workspace lands back under
        # /tmp, which is inside the default temp allowance — the trap
        # ``shared/plugins/CLAUDE.md`` warns about in exactly these words.
        result = plugin._exec_spawn({
            'command': 'cat ../../../../../../../../etc/hostname',
        })
        assert 'error' in result

    def test_write_target_outside_workspace_is_refused(self, plugin):
        result = plugin._exec_spawn({'command': 'echo x > /etc/nope'})
        assert 'error' in result
        assert '/etc/nope' in result['error']

    def test_path_inside_a_quoted_subcommand_is_NOT_caught(self, plugin):
        """A known, shared limit of the string layer — recorded, not implied.

        ``analyze_command`` does not descend into the quoted argument of
        ``sh -c``, so the path inside it is never seen.  ``cli`` has the
        same blind spot (this is its analyzer, unchanged), and it is the
        concrete reason the plugin announces the absence of kernel
        confinement instead of presenting the string check as a boundary.
        Asserted so that closing the gap is a deliberate change to this
        test rather than a silent one.
        """
        assert plugin._containment_refusal(
            "sh -c 'cat /etc/hostname'", 'shell_spawn', on_parse_error="deny"
        ) is None

    def test_second_segment_is_judged_too(self, plugin):
        """A compound command is judged segment by segment, as in cli."""
        result = plugin._exec_spawn({'command': 'echo ok && cat /etc/hostname'})
        assert 'error' in result
        assert '/etc/hostname' in result['error']

    def test_unparseable_command_fails_closed(self, plugin):
        result = plugin._exec_spawn({'command': 'cat "/etc/passwd'})
        assert 'error' in result
        assert 'cannot be parsed' in result['error']

    def test_no_workspace_root_means_no_sandbox(self):
        """Matching cli: with no workspace configured, nothing is refused."""
        p = create_plugin()
        p.initialize({'idle_timeout': 0.3})
        try:
            assert p._containment_refusal(
                'cat /etc/hostname', 'shell_spawn', on_parse_error="deny"
            ) is None
        finally:
            p.shutdown()

    @needs_backend
    def test_inside_workspace_still_spawns(self, plugin, tmp_path):
        (tmp_path / "hello.txt").write_text("inside\n")
        result = _unpack(plugin._exec_spawn({
            'command': f'cat {tmp_path}/hello.txt',
        }))
        assert 'error' not in result
        assert 'inside' in result['output']
        plugin._exec_close({'session_id': result['session_id']})


class TestInputContainment:
    """shell_input is checked, and is deliberately more forgiving."""

    def test_outside_path_typed_into_a_live_session_is_refused(self, plugin):
        refusal = plugin._containment_refusal(
            'cat /etc/hostname\n', 'shell_input', on_parse_error="allow"
        )
        assert refusal is not None
        assert '/etc/hostname' in refusal['error']

    def test_unparseable_input_fails_open(self, plugin):
        """A REPL line is not shell syntax; refusing it would break REPLs."""
        assert plugin._containment_refusal(
            "print('unbalanced\n", 'shell_input', on_parse_error="allow"
        ) is None

    def test_ordinary_input_passes(self, plugin):
        assert plugin._containment_refusal(
            'echo hello\n', 'shell_input', on_parse_error="allow"
        ) is None
        assert plugin._containment_refusal(
            'hunter2\n', 'shell_input', on_parse_error="allow"
        ) is None

    def test_missing_session_outranks_containment(self, plugin):
        """A bad session_id reports the session, not a path verdict."""
        result = plugin._exec_input({
            'session_id': 'nonexistent',
            'input': 'cat /etc/hostname\n',
        })
        assert 'No session' in result['error']

    @needs_backend
    def test_live_session_refuses_and_sends_nothing(self, plugin, tmp_path):
        spawn = _unpack(plugin._exec_spawn({
            'command': 'bash --norc --noprofile',
        }))
        sid = spawn['session_id']
        try:
            refused = _unpack(plugin._exec_input({
                'session_id': sid,
                'input': 'cat /etc/hostname\n',
            }))
            assert 'error' in refused
            assert refused['is_alive'] is True

            # The session is untouched and still usable.
            ok = _unpack(plugin._exec_input({
                'session_id': sid,
                'input': 'echo still_here\n',
            }))
            assert 'still_here' in ok['output']
        finally:
            plugin._exec_close({'session_id': sid})


class TestConfinementPosture:
    """Running without a kernel boundary is announced, or refused."""

    def test_unconfined_spawn_warns_once(self, plugin, caplog):
        with caplog.at_level(logging.WARNING):
            plugin._confinement_refusal()
            plugin._confinement_refusal()
        warnings = [
            r for r in caplog.records
            if 'WITHOUT kernel confinement' in r.getMessage()
        ]
        assert len(warnings) == 1

    def test_confined_spawn_does_not_warn(self, plugin, caplog):
        plugin.set_apparmor_child_transition_callback(lambda: None)
        with caplog.at_level(logging.WARNING):
            assert plugin._confinement_refusal() is None
        assert not [
            r for r in caplog.records
            if 'WITHOUT kernel confinement' in r.getMessage()
        ]

    def test_require_confinement_refuses_when_unconfined(self, tmp_path):
        p = create_plugin()
        p.initialize({
            'workspace_root': str(tmp_path),
            'require_confinement': True,
        })
        try:
            result = p._exec_spawn({'command': 'echo hi'})
            assert 'error' in result
            assert 'require_confinement' in result['error']
        finally:
            p.shutdown()

    def test_require_confinement_allows_when_confined(self, tmp_path):
        p = create_plugin()
        p.initialize({
            'workspace_root': str(tmp_path),
            'require_confinement': True,
        })
        p.set_apparmor_child_transition_callback(lambda: None)
        try:
            assert p._confinement_refusal() is None
        finally:
            p.shutdown()

    def test_knob_is_declared_in_the_config_schema(self, plugin):
        schema = plugin.get_config_schema()
        assert 'require_confinement' in schema['properties']


class TestSpawnCwdContainment:
    """The cwd half of #503, enforced where the process is spawned."""

    def test_cwd_outside_workspace_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            _verify_cwd_within('/etc', str(tmp_path))

    def test_cwd_inside_workspace_is_accepted(self, tmp_path):
        (tmp_path / "sub").mkdir()
        _verify_cwd_within(str(tmp_path / "sub"), str(tmp_path))
        _verify_cwd_within(str(tmp_path), str(tmp_path))

    def test_symlinked_cwd_is_judged_by_its_target(self, tmp_path):
        link = tmp_path / "escape"
        link.symlink_to("/etc")
        with pytest.raises(ValueError):
            _verify_cwd_within(str(link), str(tmp_path))

    def test_no_boundary_asserted_means_no_check(self, tmp_path):
        _verify_cwd_within('/etc', None)
        _verify_cwd_within(None, str(tmp_path))

    @needs_backend
    def test_shell_session_refuses_an_escaping_cwd(self, tmp_path):
        with pytest.raises(ValueError):
            ShellSession(
                command='echo x',
                session_id='escaping',
                cwd='/etc',
                workspace_root=str(tmp_path),
            )
