"""Tests for the InteractiveShellPlugin — tool schemas and executors."""

import time

import pytest

from shared.plugins.interactive_shell.plugin import (
    InteractiveShellPlugin,
    create_plugin,
)


def _unpack(result):
    """Unpack executor result — may be a plain dict or (dict, metadata) tuple."""
    if isinstance(result, tuple):
        return result[0]
    return result


@pytest.fixture
def plugin(tmp_path):
    """Create an initialized plugin with a temp workspace."""
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


class TestPluginMetadata:
    """Test plugin identity and discovery."""

    def test_name(self, plugin):
        assert plugin.name == "interactive_shell"

    def test_create_plugin_factory(self):
        p = create_plugin()
        assert isinstance(p, InteractiveShellPlugin)
        assert p.name == "interactive_shell"

    def test_tool_schemas_count(self, plugin):
        schemas = plugin.get_tool_schemas()
        assert len(schemas) == 6

    def test_tool_schema_names(self, plugin):
        schemas = plugin.get_tool_schemas()
        names = {s.name for s in schemas}
        assert names == {
            'shell_spawn', 'shell_input', 'shell_read',
            'shell_control', 'shell_close', 'shell_list',
        }

    def test_executor_names_match_schemas(self, plugin):
        schemas = plugin.get_tool_schemas()
        executors = plugin.get_executors()
        schema_names = {s.name for s in schemas}
        executor_names = set(executors.keys())
        assert schema_names == executor_names

    def test_system_instructions_not_empty(self, plugin):
        instructions = plugin.get_system_instructions()
        assert instructions is not None
        assert "shell_spawn" in instructions
        assert "shell_input" in instructions

    def test_auto_approved_tools(self, plugin):
        approved = plugin.get_auto_approved_tools()
        assert 'shell_list' in approved
        # Interactive tools should NOT be auto-approved
        assert 'shell_spawn' not in approved
        assert 'shell_input' not in approved

    def test_discoverability(self, plugin):
        """All tools should be discoverable, not core."""
        for schema in plugin.get_tool_schemas():
            assert schema.discoverability == "discoverable"


class TestSpawnTool:
    """Test the shell_spawn executor."""

    def test_spawn_returns_session_id(self, plugin):
        result = _unpack(plugin._exec_spawn({'command': 'echo hi'}))
        assert 'session_id' in result
        assert 'output' in result
        assert 'hi' in result['output']

    def test_spawn_custom_session_name(self, plugin):
        result = _unpack(plugin._exec_spawn({
            'command': 'echo named',
            'session_name': 'myshell',
        }))
        assert result['session_id'] == 'myshell'

    def test_spawn_duplicate_name_deduplicates(self, plugin):
        r1 = _unpack(plugin._exec_spawn({
            'command': 'bash --norc --noprofile',
            'session_name': 'dup',
        }))
        r2 = _unpack(plugin._exec_spawn({
            'command': 'bash --norc --noprofile',
            'session_name': 'dup',
        }))
        assert r1['session_id'] != r2['session_id']
        assert r2['session_id'].startswith('dup_')

        # Cleanup
        plugin._exec_close({'session_id': r1['session_id']})
        plugin._exec_close({'session_id': r2['session_id']})

    def test_spawn_missing_command(self, plugin):
        result = plugin._exec_spawn({})
        assert 'error' in result

    def test_spawn_session_limit(self, plugin):
        """Exceeding max_sessions returns error."""
        sessions = []
        for i in range(4):
            r = _unpack(plugin._exec_spawn({
                'command': 'bash --norc --noprofile',
                'session_name': f's{i}',
            }))
            assert 'error' not in r
            sessions.append(r['session_id'])

        # 5th should fail
        r = _unpack(plugin._exec_spawn({
            'command': 'bash --norc --noprofile',
            'session_name': 's4',
        }))
        assert 'error' in r
        assert 'Maximum concurrent sessions' in r['error']

        for sid in sessions:
            plugin._exec_close({'session_id': sid})


class TestInputTool:
    """Test the shell_input executor."""

    def test_input_and_response(self, plugin):
        spawn = _unpack(plugin._exec_spawn({'command': 'bash --norc --noprofile'}))
        sid = spawn['session_id']

        result = _unpack(plugin._exec_input({
            'session_id': sid,
            'input': 'echo tool_test_123\n',
        }))
        assert 'tool_test_123' in result['output']
        assert result['is_alive']

        plugin._exec_close({'session_id': sid})

    def test_input_missing_session_id(self, plugin):
        result = plugin._exec_input({'input': 'hello\n'})
        assert 'error' in result

    def test_input_nonexistent_session(self, plugin):
        result = plugin._exec_input({
            'session_id': 'nonexistent',
            'input': 'hello\n',
        })
        assert 'error' in result
        assert 'No session' in result['error']

    def test_input_to_exited_session(self, plugin):
        spawn = _unpack(plugin._exec_spawn({'command': 'echo done'}))
        sid = spawn['session_id']
        time.sleep(0.5)  # let it exit

        result = _unpack(plugin._exec_input({
            'session_id': sid,
            'input': 'hello\n',
        }))
        assert 'error' in result or not result.get('is_alive', True)

        plugin._exec_close({'session_id': sid})


class TestReadTool:
    """Test the shell_read executor."""

    def test_read_pending_output(self, plugin):
        spawn = _unpack(plugin._exec_spawn({'command': 'bash --norc --noprofile'}))
        sid = spawn['session_id']

        result = _unpack(plugin._exec_read({'session_id': sid, 'timeout': 0.3}))
        assert 'output' in result
        assert isinstance(result['output'], str)

        plugin._exec_close({'session_id': sid})

    def test_read_missing_session(self, plugin):
        result = plugin._exec_read({'session_id': 'nope'})
        assert 'error' in result


class TestControlTool:
    """Test the shell_control executor."""

    def test_ctrl_c(self, plugin):
        spawn = _unpack(plugin._exec_spawn({'command': 'bash --norc --noprofile'}))
        sid = spawn['session_id']

        # Start a sleep, interrupt it
        _unpack(plugin._exec_input({'session_id': sid, 'input': 'sleep 60\n'}))
        time.sleep(0.2)

        result = _unpack(plugin._exec_control({'session_id': sid, 'key': 'c-c'}))
        assert result.get('is_alive', False)

        plugin._exec_close({'session_id': sid})

    def test_control_missing_key(self, plugin):
        spawn = _unpack(plugin._exec_spawn({'command': 'bash --norc --noprofile'}))
        sid = spawn['session_id']

        result = _unpack(plugin._exec_control({'session_id': sid}))
        assert 'error' in result

        plugin._exec_close({'session_id': sid})


class TestCloseTool:
    """Test the shell_close executor."""

    def test_close_returns_exit_status(self, plugin):
        spawn = _unpack(plugin._exec_spawn({
            'command': "bash --norc --noprofile -c 'exit 7'",
        }))
        sid = spawn['session_id']
        time.sleep(0.5)  # let it exit

        result = _unpack(plugin._exec_close({'session_id': sid}))
        assert result['exit_status'] == 7

    def test_close_removes_session(self, plugin):
        spawn = _unpack(plugin._exec_spawn({'command': 'bash --norc --noprofile'}))
        sid = spawn['session_id']

        plugin._exec_close({'session_id': sid})

        # Session should be gone
        listing = plugin._exec_list({})
        ids = [s['session_id'] for s in listing['sessions']]
        assert sid not in ids

    def test_close_missing_session(self, plugin):
        result = plugin._exec_close({'session_id': 'nope'})
        assert 'error' in result


class TestListTool:
    """Test the shell_list executor."""

    def test_list_empty(self, plugin):
        result = plugin._exec_list({})
        assert result['count'] == 0
        assert result['sessions'] == []

    def test_list_shows_active_sessions(self, plugin):
        r1 = _unpack(plugin._exec_spawn({
            'command': 'bash --norc --noprofile',
            'session_name': 'alpha',
        }))
        r2 = _unpack(plugin._exec_spawn({
            'command': 'bash --norc --noprofile',
            'session_name': 'beta',
        }))

        result = plugin._exec_list({})
        assert result['count'] == 2
        ids = {s['session_id'] for s in result['sessions']}
        assert 'alpha' in ids
        assert 'beta' in ids

        for s in result['sessions']:
            assert 'command' in s
            assert 'is_alive' in s
            assert 'age_seconds' in s

        plugin._exec_close({'session_id': 'alpha'})
        plugin._exec_close({'session_id': 'beta'})


class TestWorkspacePath:
    """Test workspace path wiring."""

    def test_set_workspace_path(self, plugin, tmp_path):
        new_path = str(tmp_path / "subdir")
        import os
        os.makedirs(new_path)

        plugin.set_workspace_path(new_path)

        spawn = _unpack(plugin._exec_spawn({'command': 'bash --norc --noprofile'}))
        sid = spawn['session_id']

        result = _unpack(plugin._exec_input({
            'session_id': sid,
            'input': 'pwd\n',
        }))
        assert new_path in result['output']

        plugin._exec_close({'session_id': sid})
