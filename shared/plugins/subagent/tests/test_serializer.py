"""Tests for subagent serialization utilities."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from ..serializer import (
    serialize_subagent_state,
    deserialize_subagent_state,
    serialize_subagent_registry,
    deserialize_subagent_registry,
)
from ..config import SubagentProfile, GCProfileConfig
from ...model_provider.types import Message, Part, Role


class TestSerializeSubagentState:
    """Tests for serialize_subagent_state."""

    def test_serialize_minimal_state(self):
        """Test serialization with minimal data."""
        session_info = {
            'agent_id': 'subagent_1',
            'profile': None,
            'session': None,
            'created_at': datetime(2024, 1, 28, 10, 30, 0),
            'last_activity': datetime(2024, 1, 28, 10, 35, 0),
            'turn_count': 0,
            'max_turns': 10,
        }

        result = serialize_subagent_state(session_info)

        assert result['version'] == '1.0'
        assert result['agent_id'] == 'subagent_1'
        assert result['profile'] == {}
        assert result['history'] == []
        assert result['turn_accounting'] == []
        assert result['created_at'] == '2024-01-28T10:30:00'
        assert result['last_activity'] == '2024-01-28T10:35:00'
        assert result['turn_count'] == 0
        assert result['max_turns'] == 10

    def test_serialize_with_profile(self):
        """Test serialization with a full profile."""
        profile = SubagentProfile(
            name='investigator',
            description='A research assistant',
            plugins=['cli', 'web_search'],
            plugin_configs={'cli': {'timeout': 30}},
            system_instructions='You are a researcher.',
            model='gemini-2.5-flash',
            provider='google_genai',
            max_turns=5,
            auto_approved=True,
            icon=['[R]', '[E]', '[S]'],
            icon_name='research',
        )

        session_info = {
            'agent_id': 'subagent_2',
            'profile': profile,
            'session': None,
            'created_at': datetime(2024, 1, 28, 10, 30, 0),
            'last_activity': datetime(2024, 1, 28, 10, 35, 0),
            'turn_count': 3,
            'max_turns': 5,
        }

        result = serialize_subagent_state(session_info)

        assert result['profile']['name'] == 'investigator'
        assert result['profile']['description'] == 'A research assistant'
        assert result['profile']['plugins'] == ['cli', 'web_search']
        assert result['profile']['plugin_configs'] == {'cli': {'timeout': 30}}
        assert result['profile']['system_instructions'] == 'You are a researcher.'
        assert result['profile']['model'] == 'gemini-2.5-flash'
        assert result['profile']['provider'] == 'google_genai'
        assert result['profile']['max_turns'] == 5
        assert result['profile']['auto_approved'] is True
        assert result['profile']['icon'] == ['[R]', '[E]', '[S]']
        assert result['profile']['icon_name'] == 'research'

    def test_serialize_with_gc_config(self):
        """Test serialization with GC configuration."""
        gc_config = GCProfileConfig(
            type='hybrid',
            threshold_percent=70.0,
            preserve_recent_turns=3,
            notify_on_gc=False,
            summarize_middle_turns=5,
            max_turns=20,
            plugin_config={'model': 'gemini-flash'},
        )
        profile = SubagentProfile(
            name='test',
            description='Test profile',
            gc=gc_config,
        )

        session_info = {
            'agent_id': 'subagent_3',
            'profile': profile,
            'session': None,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'turn_count': 0,
            'max_turns': 10,
        }

        result = serialize_subagent_state(session_info)

        assert 'gc' in result['profile']
        gc = result['profile']['gc']
        assert gc['type'] == 'hybrid'
        assert gc['threshold_percent'] == 70.0
        assert gc['preserve_recent_turns'] == 3
        assert gc['notify_on_gc'] is False
        assert gc['summarize_middle_turns'] == 5
        assert gc['max_turns'] == 20
        assert gc['plugin_config'] == {'model': 'gemini-flash'}

    def test_serialize_with_session_history(self):
        """Test serialization with session history and turn accounting."""
        # Create mock session
        mock_session = MagicMock()
        mock_session.get_history.return_value = [
            Message(role=Role.USER, parts=[Part(text='Hello')]),
            Message(role=Role.MODEL, parts=[Part(text='Hi there!')]),
        ]
        mock_session.get_turn_accounting.return_value = [
            {'prompt': 100, 'output': 50, 'total': 150, 'turn': 0},
        ]

        session_info = {
            'agent_id': 'subagent_4',
            'profile': SubagentProfile(name='test', description='Test'),
            'session': mock_session,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'turn_count': 1,
            'max_turns': 10,
        }

        result = serialize_subagent_state(session_info)

        assert len(result['history']) == 2
        assert result['history'][0]['role'] == 'user'
        assert result['history'][0]['parts'][0]['text'] == 'Hello'
        assert result['history'][1]['role'] == 'model'
        assert result['turn_accounting'] == [
            {'prompt': 100, 'output': 50, 'total': 150, 'turn': 0},
        ]


class TestDeserializeSubagentState:
    """Tests for deserialize_subagent_state."""

    def test_deserialize_minimal_state(self):
        """Test deserialization with minimal data."""
        data = {
            'version': '1.0',
            'agent_id': 'subagent_1',
            'profile': {},
            'history': [],
            'turn_accounting': [],
            'created_at': '2024-01-28T10:30:00',
            'last_activity': '2024-01-28T10:35:00',
            'turn_count': 0,
            'max_turns': 10,
        }

        result = deserialize_subagent_state(data)

        assert result['agent_id'] == 'subagent_1'
        assert result['profile'] is None  # Empty profile becomes None
        assert result['history'] == []
        assert result['turn_accounting'] == []
        assert result['created_at'] == datetime(2024, 1, 28, 10, 30, 0)
        assert result['last_activity'] == datetime(2024, 1, 28, 10, 35, 0)
        assert result['turn_count'] == 0
        assert result['max_turns'] == 10
        assert result['session'] is None

    def test_deserialize_with_profile(self):
        """Test deserialization with profile data."""
        data = {
            'version': '1.0',
            'agent_id': 'subagent_2',
            'profile': {
                'name': 'investigator',
                'description': 'Research assistant',
                'plugins': ['cli', 'web_search'],
                'plugin_configs': {'cli': {'timeout': 30}},
                'system_instructions': 'You are a researcher.',
                'model': 'gemini-2.5-flash',
                'provider': 'google_genai',
                'max_turns': 5,
                'auto_approved': True,
            },
            'history': [],
            'turn_accounting': [],
            'created_at': '2024-01-28T10:30:00',
            'last_activity': '2024-01-28T10:35:00',
            'turn_count': 3,
            'max_turns': 5,
        }

        result = deserialize_subagent_state(data)
        profile = result['profile']

        assert profile is not None
        assert profile.name == 'investigator'
        assert profile.description == 'Research assistant'
        assert profile.plugins == ['cli', 'web_search']
        assert profile.plugin_configs == {'cli': {'timeout': 30}}
        assert profile.system_instructions == 'You are a researcher.'
        assert profile.model == 'gemini-2.5-flash'
        assert profile.provider == 'google_genai'
        assert profile.max_turns == 5
        assert profile.auto_approved is True

    def test_deserialize_with_gc_config(self):
        """Test deserialization with GC configuration."""
        data = {
            'version': '1.0',
            'agent_id': 'subagent_3',
            'profile': {
                'name': 'test',
                'description': 'Test',
                'gc': {
                    'type': 'hybrid',
                    'threshold_percent': 70.0,
                    'preserve_recent_turns': 3,
                    'notify_on_gc': False,
                    'summarize_middle_turns': 5,
                    'max_turns': 20,
                    'plugin_config': {'model': 'gemini-flash'},
                },
            },
            'history': [],
            'turn_accounting': [],
            'created_at': '2024-01-28T10:30:00',
            'last_activity': '2024-01-28T10:35:00',
            'turn_count': 0,
            'max_turns': 10,
        }

        result = deserialize_subagent_state(data)
        gc = result['profile'].gc

        assert gc is not None
        assert gc.type == 'hybrid'
        assert gc.threshold_percent == 70.0
        assert gc.preserve_recent_turns == 3
        assert gc.notify_on_gc is False
        assert gc.summarize_middle_turns == 5
        assert gc.max_turns == 20
        assert gc.plugin_config == {'model': 'gemini-flash'}

    def test_deserialize_with_history(self):
        """Test deserialization with conversation history."""
        data = {
            'version': '1.0',
            'agent_id': 'subagent_4',
            'profile': {'name': 'test', 'description': 'Test'},
            'history': [
                {'role': 'user', 'parts': [{'type': 'text', 'text': 'Hello'}]},
                {'role': 'model', 'parts': [{'type': 'text', 'text': 'Hi!'}]},
            ],
            'turn_accounting': [
                {'prompt': 100, 'output': 50, 'total': 150},
            ],
            'created_at': '2024-01-28T10:30:00',
            'last_activity': '2024-01-28T10:35:00',
            'turn_count': 1,
            'max_turns': 10,
        }

        result = deserialize_subagent_state(data)

        assert len(result['history']) == 2
        assert result['history'][0].role == Role.USER
        assert result['history'][0].parts[0].text == 'Hello'
        assert result['history'][1].role == Role.MODEL
        assert result['history'][1].parts[0].text == 'Hi!'
        assert result['turn_accounting'] == [
            {'prompt': 100, 'output': 50, 'total': 150},
        ]

    def test_deserialize_unsupported_version(self):
        """Test that unsupported version raises error."""
        data = {
            'version': '2.0',
            'agent_id': 'subagent_1',
        }

        with pytest.raises(ValueError, match="Unsupported subagent state version"):
            deserialize_subagent_state(data)

    def test_roundtrip(self):
        """Test serialization followed by deserialization."""
        profile = SubagentProfile(
            name='roundtrip_test',
            description='Testing roundtrip',
            plugins=['cli'],
            max_turns=7,
            gc=GCProfileConfig(type='truncate', threshold_percent=85.0),
        )

        original = {
            'agent_id': 'subagent_roundtrip',
            'profile': profile,
            'session': None,
            'created_at': datetime(2024, 1, 28, 10, 30, 0),
            'last_activity': datetime(2024, 1, 28, 10, 35, 0),
            'turn_count': 2,
            'max_turns': 7,
        }

        serialized = serialize_subagent_state(original)
        deserialized = deserialize_subagent_state(serialized)

        assert deserialized['agent_id'] == original['agent_id']
        assert deserialized['turn_count'] == original['turn_count']
        assert deserialized['max_turns'] == original['max_turns']
        assert deserialized['created_at'] == original['created_at']
        assert deserialized['last_activity'] == original['last_activity']
        assert deserialized['profile'].name == profile.name
        assert deserialized['profile'].description == profile.description
        assert deserialized['profile'].plugins == profile.plugins
        assert deserialized['profile'].gc.type == profile.gc.type
        assert deserialized['profile'].gc.threshold_percent == profile.gc.threshold_percent


class TestSerializeSubagentRegistry:
    """Tests for serialize_subagent_registry."""

    def test_serialize_empty_registry(self):
        """Test serialization of empty registry."""
        result = serialize_subagent_registry({})

        assert result['version'] == '1.0'
        assert result['agents'] == []

    def test_serialize_registry_with_agents(self):
        """Test serialization of registry with multiple agents."""
        mock_session_1 = MagicMock()
        mock_session_1.is_running = False

        mock_session_2 = MagicMock()
        mock_session_2.is_running = True

        active_sessions = {
            'subagent_1': {
                'session': mock_session_1,
                'profile': SubagentProfile(name='researcher', description='Research'),
                'agent_id': 'subagent_1',
                'created_at': datetime(2024, 1, 28, 10, 30, 0),
                'last_activity': datetime(2024, 1, 28, 10, 35, 0),
                'turn_count': 3,
                'max_turns': 10,
            },
            'subagent_2': {
                'session': mock_session_2,
                'profile': SubagentProfile(name='coder', description='Coding'),
                'agent_id': 'subagent_2',
                'created_at': datetime(2024, 1, 28, 11, 0, 0),
                'last_activity': datetime(2024, 1, 28, 11, 5, 0),
                'turn_count': 1,
                'max_turns': 5,
            },
        }

        result = serialize_subagent_registry(active_sessions)

        assert result['version'] == '1.0'
        assert len(result['agents']) == 2

        # Find agents by id
        agent_1 = next(a for a in result['agents'] if a['agent_id'] == 'subagent_1')
        agent_2 = next(a for a in result['agents'] if a['agent_id'] == 'subagent_2')

        assert agent_1['profile_name'] == 'researcher'
        assert agent_1['status'] == 'idle'
        assert agent_1['turn_count'] == 3
        assert agent_1['max_turns'] == 10
        assert agent_1['created_at'] == '2024-01-28T10:30:00'
        assert agent_1['last_activity'] == '2024-01-28T10:35:00'

        assert agent_2['profile_name'] == 'coder'
        assert agent_2['status'] == 'running'
        assert agent_2['turn_count'] == 1
        assert agent_2['max_turns'] == 5


class TestDeserializeSubagentRegistry:
    """Tests for deserialize_subagent_registry."""

    def test_deserialize_empty_registry(self):
        """Test deserialization of empty registry."""
        data = {'version': '1.0', 'agents': []}

        result = deserialize_subagent_registry(data)

        assert result == []

    def test_deserialize_registry_with_agents(self):
        """Test deserialization of registry with agents."""
        data = {
            'version': '1.0',
            'agents': [
                {
                    'agent_id': 'subagent_1',
                    'profile_name': 'researcher',
                    'status': 'idle',
                    'created_at': '2024-01-28T10:30:00',
                    'last_activity': '2024-01-28T10:35:00',
                    'turn_count': 3,
                    'max_turns': 10,
                },
                {
                    'agent_id': 'subagent_2',
                    'profile_name': 'coder',
                    'status': 'running',
                    'created_at': '2024-01-28T11:00:00',
                    'last_activity': '2024-01-28T11:05:00',
                    'turn_count': 1,
                    'max_turns': 5,
                },
            ],
        }

        result = deserialize_subagent_registry(data)

        assert len(result) == 2

        assert result[0]['agent_id'] == 'subagent_1'
        assert result[0]['profile_name'] == 'researcher'
        assert result[0]['status'] == 'idle'
        assert result[0]['created_at'] == datetime(2024, 1, 28, 10, 30, 0)
        assert result[0]['last_activity'] == datetime(2024, 1, 28, 10, 35, 0)
        assert result[0]['turn_count'] == 3
        assert result[0]['max_turns'] == 10

        assert result[1]['agent_id'] == 'subagent_2'
        assert result[1]['profile_name'] == 'coder'
        assert result[1]['status'] == 'running'

    def test_deserialize_unsupported_version(self):
        """Test that unsupported version raises error."""
        data = {
            'version': '2.0',
            'agents': [],
        }

        with pytest.raises(ValueError, match="Unsupported subagent registry version"):
            deserialize_subagent_registry(data)

    def test_registry_roundtrip(self):
        """Test serialization followed by deserialization for registry."""
        mock_session = MagicMock()
        mock_session.is_running = False

        original = {
            'subagent_test': {
                'session': mock_session,
                'profile': SubagentProfile(name='test_profile', description='Test'),
                'agent_id': 'subagent_test',
                'created_at': datetime(2024, 1, 28, 10, 30, 0),
                'last_activity': datetime(2024, 1, 28, 10, 35, 0),
                'turn_count': 5,
                'max_turns': 10,
            },
        }

        serialized = serialize_subagent_registry(original)
        deserialized = deserialize_subagent_registry(serialized)

        assert len(deserialized) == 1
        assert deserialized[0]['agent_id'] == 'subagent_test'
        assert deserialized[0]['profile_name'] == 'test_profile'
        assert deserialized[0]['status'] == 'idle'
        assert deserialized[0]['turn_count'] == 5
        assert deserialized[0]['max_turns'] == 10
        assert deserialized[0]['created_at'] == datetime(2024, 1, 28, 10, 30, 0)
        assert deserialized[0]['last_activity'] == datetime(2024, 1, 28, 10, 35, 0)
