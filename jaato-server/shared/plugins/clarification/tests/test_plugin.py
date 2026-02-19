"""Tests for the clarification plugin."""

import pytest

from ..plugin import ClarificationPlugin, create_plugin
from ..channels import AutoChannel


class TestClarificationPluginInitialization:
    """Tests for plugin initialization."""

    def test_create_plugin_factory(self):
        plugin = create_plugin()
        assert isinstance(plugin, ClarificationPlugin)

    def test_plugin_name(self):
        plugin = ClarificationPlugin()
        assert plugin.name == "clarification"

    def test_initialize_without_config(self):
        plugin = ClarificationPlugin()
        plugin.initialize()
        assert plugin._initialized is True
        assert plugin._channel is not None

    def test_initialize_with_console_channel(self):
        plugin = ClarificationPlugin()
        plugin.initialize({"channel_type": "console"})
        assert plugin._initialized is True

    def test_initialize_with_auto_channel(self):
        plugin = ClarificationPlugin()
        plugin.initialize({"channel_type": "auto"})
        assert plugin._initialized is True
        assert isinstance(plugin._channel, AutoChannel)

    def test_initialize_with_auto_channel_config(self):
        plugin = ClarificationPlugin()
        plugin.initialize({
            "channel_type": "auto",
            "channel_config": {"default_free_text": "custom"},
        })
        assert plugin._initialized is True

    def test_shutdown(self):
        plugin = ClarificationPlugin()
        plugin.initialize()
        plugin.shutdown()

        assert plugin._initialized is False
        assert plugin._channel is None


class TestClarificationPluginToolSchemas:
    """Tests for tool schemas."""

    def test_get_tool_schemas(self):
        plugin = ClarificationPlugin()
        schemas = plugin.get_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0].name == "request_clarification"

    def test_request_clarification_schema(self):
        plugin = ClarificationPlugin()
        schemas = plugin.get_tool_schemas()
        schema_obj = schemas[0]
        schema = schema_obj.parameters

        assert schema["type"] == "object"
        assert "context" in schema["properties"]
        assert "questions" in schema["properties"]
        assert "context" in schema["required"]
        assert "questions" in schema["required"]

    def test_question_schema_structure(self):
        plugin = ClarificationPlugin()
        schemas = plugin.get_tool_schemas()
        schema = schemas[0].parameters

        question_schema = schema["properties"]["questions"]["items"]
        assert "text" in question_schema["properties"]
        assert "question_type" in question_schema["properties"]
        assert "choices" in question_schema["properties"]
        assert "required" in question_schema["properties"]
        assert "default_choice" in question_schema["properties"]

    def test_question_type_enum(self):
        plugin = ClarificationPlugin()
        schemas = plugin.get_tool_schemas()
        schema = schemas[0].parameters

        question_type_schema = schema["properties"]["questions"]["items"]["properties"]["question_type"]
        assert "enum" in question_type_schema
        assert "single_choice" in question_type_schema["enum"]
        assert "multiple_choice" in question_type_schema["enum"]
        assert "free_text" in question_type_schema["enum"]


class TestClarificationPluginExecutors:
    """Tests for executor methods."""

    def test_get_executors(self):
        plugin = ClarificationPlugin()
        executors = plugin.get_executors()

        assert "request_clarification" in executors
        assert callable(executors["request_clarification"])


class TestRequestClarificationExecutor:
    """Tests for request_clarification executor."""

    def test_execute_not_initialized(self):
        plugin = ClarificationPlugin()
        executors = plugin.get_executors()

        result = executors["request_clarification"]({
            "context": "Test",
            "questions": [{"text": "Q1"}],
        })

        assert "error" in result

    def test_execute_single_choice_question(self):
        plugin = ClarificationPlugin()
        plugin.initialize({"channel_type": "auto"})
        executors = plugin.get_executors()

        result = executors["request_clarification"]({
            "context": "Need to know your preference",
            "questions": [
                {
                    "text": "Which environment?",
                    "question_type": "single_choice",
                    "choices": ["Development", "Production"],
                    "default_choice": 1,
                },
            ],
        })

        assert "error" not in result
        assert "responses" in result
        assert "1" in result["responses"]
        assert result["responses"]["1"]["selected"] == 1
        assert result["responses"]["1"]["type"] == "single_choice"

    def test_execute_multiple_choice_question(self):
        plugin = ClarificationPlugin()
        plugin.initialize({"channel_type": "auto"})
        executors = plugin.get_executors()

        result = executors["request_clarification"]({
            "context": "Select features",
            "questions": [
                {
                    "text": "Which features?",
                    "question_type": "multiple_choice",
                    "choices": ["Logging", "Metrics", "Tracing"],
                    "default_choice": 1,
                },
            ],
        })

        assert "error" not in result
        assert "responses" in result
        assert "1" in result["responses"]
        assert result["responses"]["1"]["type"] == "multiple_choice"
        assert 1 in result["responses"]["1"]["selected"]

    def test_execute_free_text_question(self):
        plugin = ClarificationPlugin()
        plugin.initialize({
            "channel_type": "auto",
            "channel_config": {"default_free_text": "my custom answer"},
        })
        executors = plugin.get_executors()

        result = executors["request_clarification"]({
            "context": "Need description",
            "questions": [
                {
                    "text": "Describe your requirements",
                    "question_type": "free_text",
                },
            ],
        })

        assert "error" not in result
        assert "responses" in result
        assert result["responses"]["1"]["value"] == "my custom answer"
        assert result["responses"]["1"]["type"] == "free_text"

    def test_execute_multiple_questions(self):
        plugin = ClarificationPlugin()
        plugin.initialize({"channel_type": "auto"})
        executors = plugin.get_executors()

        result = executors["request_clarification"]({
            "context": "Configuration needed",
            "questions": [
                {
                    "text": "Environment",
                    "question_type": "single_choice",
                    "choices": ["Dev", "Prod"],
                },
                {
                    "text": "Description",
                    "question_type": "free_text",
                },
                {
                    "text": "Features",
                    "question_type": "multiple_choice",
                    "choices": ["A", "B"],
                },
            ],
        })

        assert "error" not in result
        assert "responses" in result
        assert "1" in result["responses"]
        assert "2" in result["responses"]
        assert "3" in result["responses"]

    def test_execute_with_defaults(self):
        plugin = ClarificationPlugin()
        plugin.initialize({"channel_type": "auto"})
        executors = plugin.get_executors()

        result = executors["request_clarification"]({
            "context": "Quick config",
            "questions": [
                {
                    "text": "Select",
                    "choices": ["A", "B"],
                    "default_choice": 2,
                },
            ],
        })

        assert result["responses"]["1"]["selected"] == 2

    def test_execute_question_type_defaults_to_single_choice(self):
        plugin = ClarificationPlugin()
        plugin.initialize({"channel_type": "auto"})
        executors = plugin.get_executors()

        result = executors["request_clarification"]({
            "context": "Test",
            "questions": [
                {
                    "text": "Pick",
                    "choices": ["X"],
                    # No question_type specified
                },
            ],
        })

        assert "error" not in result
        assert result["responses"]["1"]["type"] == "single_choice"


class TestClarificationPluginSystemInstructions:
    """Tests for system instructions."""

    def test_get_system_instructions(self):
        plugin = ClarificationPlugin()
        instructions = plugin.get_system_instructions()

        assert instructions is not None
        assert "request_clarification" in instructions
        assert "single_choice" in instructions
        assert "multiple_choice" in instructions
        assert "free_text" in instructions

    def test_system_instructions_has_example(self):
        plugin = ClarificationPlugin()
        instructions = plugin.get_system_instructions()

        assert "Example" in instructions or "example" in instructions


class TestClarificationPluginAutoApproved:
    """Tests for auto-approved tools."""

    def test_get_auto_approved_tools(self):
        plugin = ClarificationPlugin()
        auto_approved = plugin.get_auto_approved_tools()

        assert "request_clarification" in auto_approved


class TestClarificationPluginWorkflow:
    """Tests for complete clarification workflows."""

    def test_full_workflow_deployment_config(self):
        plugin = ClarificationPlugin()
        plugin.initialize({"channel_type": "auto"})
        executors = plugin.get_executors()

        # Simulate model asking for deployment configuration
        result = executors["request_clarification"]({
            "context": "I need to configure the deployment. Please provide the following information.",
            "questions": [
                {
                    "text": "Which environment should I deploy to?",
                    "question_type": "single_choice",
                    "choices": ["Development", "Staging", "Production"],
                    "default_choice": 1,
                },
                {
                    "text": "Which optional features should be enabled?",
                    "question_type": "multiple_choice",
                    "choices": ["Enhanced logging", "Metrics collection", "Debug mode"],
                    "required": False,
                    "default_choice": 1,
                },
                {
                    "text": "Any additional deployment notes?",
                    "question_type": "free_text",
                    "required": False,
                },
            ],
        })

        assert "error" not in result
        assert "responses" in result

        # Check environment response
        env = result["responses"]["1"]
        assert env["selected"] == 1
        assert env["text"] == "Development"

        # Check features response
        features = result["responses"]["2"]
        assert 1 in features["selected"]

    def test_workflow_ambiguous_request(self):
        plugin = ClarificationPlugin()
        plugin.initialize({"channel_type": "auto"})
        executors = plugin.get_executors()

        # Simulate model clarifying an ambiguous user request
        result = executors["request_clarification"]({
            "context": "Your request to 'add authentication' could be implemented in several ways. Please clarify your preferences.",
            "questions": [
                {
                    "text": "What type of authentication?",
                    "question_type": "single_choice",
                    "choices": ["Session-based (cookies)", "JWT tokens", "OAuth 2.0"],
                },
                {
                    "text": "Which OAuth providers should be supported?",
                    "question_type": "multiple_choice",
                    "choices": ["Google", "GitHub", "Microsoft"],
                },
            ],
        })

        assert "error" not in result
        assert "1" in result["responses"]
        assert "2" in result["responses"]
