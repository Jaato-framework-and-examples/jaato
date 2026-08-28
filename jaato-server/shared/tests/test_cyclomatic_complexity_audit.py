"""Audit test — cyclomatic-complexity ratchet over the Python packages.

WHY A RATCHET AND NOT A THRESHOLD.  A flat "no function may exceed N"
gate cannot be adopted here: at the time this guard was written the tree
carried 416 blocks above the ceiling (and 873 above 10), topping out at
``output_buffer._render_impl`` on 117.  A gate that is red on its first
run gets switched off within a week.  So this test freezes the current
numbers as a BASELINE and guards the *derivative* instead:

  - **NEW** functions above CEILING fail.  New code is held to the bar
    even while the old code is not.
  - **EXISTING** baselined functions may not get *worse*.  Recording the
    number, not just the name, is what makes this a ratchet: a function
    sitting at 117 may go to 110, but not to 118.
  - **IMPROVED / DELETED** baselined functions fail as stale, asking the
    dev to lower or drop the recorded number.  This is what walks the
    debt down instead of letting the baseline calcify — the same
    stale-entry discipline ``test_session_env_audit.py`` applies to its
    ALLOWLIST.

CHOICE OF METRIC.  radon, ceiling 15.  radon counts boolean operators
and comprehensions as decision points, which mccabe (ruff C901) does
not, so radon scores run above mccabe ones — median +3 in the 11-15
band, p90 +7.  That inflation is why the ceiling is 15 rather than 10:
measured over that band, 26% of the blocks a ceiling of 10 would have
caught score <=7 under mccabe — not branchy at all, just written with
defensive ``x.get(k) or ""`` defaults.  ``command_router._handle_session_send``
(radon 12, mccabe 3) is the canonical example.  Rejecting that class of
function teaches people to drop the defaults or split coherent code in
half to game the counter, so the ceiling is set where the metric still
means what it says.  A tighter bar wants the metric changed first (drop
``ast.BoolOp`` from the count), not just the number lowered.

SCOPE.  The three installed Python packages — ``jaato-server``,
``jaato-sdk``, ``jaato-tui``.  ``scripts/``, ``examples/`` and
``out-of-tree-plugins/`` are out of scope: they are not shipped and not
otherwise covered by CI.  Test files ARE in scope; they were only 16 of
the 416 initial entries, so carving them out would buy nothing and would
leave complex test helpers unguarded.

REGENERATING THE BASELINE.  Run this file as a script to print a fresh
BASELINE body, then paste it over the literal below::

    python jaato-server/shared/tests/test_cyclomatic_complexity_audit.py

Do that only for a deliberate re-freeze.  Routine work should be
touching individual lines, and each touched line is a reviewable claim
about one function.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import pytest

try:
    from radon.complexity import cc_visit
    from radon.visitors import Function
except ImportError as exc:  # pragma: no cover - dev env is misconfigured
    raise ImportError(
        "The complexity audit needs radon, which ships in the server's `dev` "
        "extra alongside pytest:\n"
        '    pip install -e "jaato-server/.[dev]"\n'
        "This is a hard error rather than a skip on purpose: a guard that "
        "silently no-ops when its dependency is missing is exactly the "
        "failure mode it exists to prevent."
    ) from exc


# Functions authored from here on may not exceed this radon score.  See the
# module docstring for why 15 and not 10.
CEILING = 15

# Package roots scanned, relative to the repo root.
PACKAGES: Tuple[str, ...] = ("jaato-server", "jaato-sdk", "jaato-tui")

# ---------------------------------------------------------------------------
# Baseline — frozen 2026-08-28 against radon 6.0.1, ceiling 15.
#
# Every entry is a function that was ALREADY over the ceiling when the guard
# went in.  The number is the score at freeze time and is a ceiling of its
# own: see test_baselined_functions_do_not_get_worse.
#
# Adding an entry is allowed but is a claim, not a formality — pair it with a
# comment explaining why the complexity is irreducible.  Removing one (or
# lowering its number) is the normal, encouraged direction of travel.
# ---------------------------------------------------------------------------

BASELINE: Dict[str, int] = {
    "jaato-sdk/jaato_sdk/client/ipc.py::IPCClient.connect": 24,
    "jaato-sdk/jaato_sdk/doctor.py::check_session": 20,
    "jaato-server/server/__main__.py::JaatoDaemon.start": 25,
    "jaato-server/server/__main__.py::main": 33,
    "jaato-server/server/apparmor.py::AppArmorManager._render_profile": 18,
    "jaato-server/server/command_router.py::CommandRouter._dispatch": 35,
    "jaato-server/server/command_router.py::CommandRouter._execute_daemon_command": 16,
    "jaato-server/server/command_router.py::CommandRouter._handle_session_bind_wake": 18,
    "jaato-server/server/command_router.py::CommandRouter._handle_session_new": 18,
    "jaato-server/server/command_router.py::CommandRouter.get_command_list": 52,
    "jaato-server/server/core.py::JaatoServer._build_profile_session_kwargs": 17,
    "jaato-server/server/core.py::JaatoServer._build_send_message_notification_handler._handle": 93,
    "jaato-server/server/core.py::JaatoServer._check_auth_completion": 17,
    "jaato-server/server/core.py::JaatoServer._emit_conversation_replay": 19,
    "jaato-server/server/core.py::JaatoServer._setup_permission_hooks.on_permission_requested": 35,
    "jaato-server/server/core.py::JaatoServer._start_model_thread.model_thread": 36,
    "jaato-server/server/core.py::JaatoServer.execute_command": 28,
    "jaato-server/server/core.py::JaatoServer.initialize": 44,
    "jaato-server/server/core.py::JaatoServer.initialize._run_load_plugins": 17,
    "jaato-server/server/core.py::JaatoServer.shutdown": 35,
    "jaato-server/server/egress_proxy/config.py::validate_allowlist": 22,
    "jaato-server/server/ipc.py::JaatoIPCServer._handle_message": 30,
    "jaato-server/server/runner/rpc.py::RunnerRPC._dispatch_method": 53,
    "jaato-server/server/runner/rpc.py::RunnerRPC._handle_session_register_client_tools": 17,
    "jaato-server/server/runner/rpc.py::RunnerRPC._handle_session_resolve_fork_point": 17,
    "jaato-server/server/runner/rpc.py::RunnerRPC._handle_session_send_message": 20,
    "jaato-server/server/runner/rpc.py::RunnerRPC._handle_subagent_forward_event": 19,
    "jaato-server/server/runner/rpc.py::RunnerRPC._install_session_notification_callbacks": 20,
    "jaato-server/server/runner/rpc.py::RunnerRPC._restore_session_notification_callbacks": 29,
    "jaato-server/server/runner/rpc.py::RunnerRPC.serve": 18,
    "jaato-server/server/runner/session.py::_build_session": 17,
    "jaato-server/server/runner/session.py::_configure_runtime_plugins": 19,
    "jaato-server/server/runner/tests/test_session_dispatch_lifecycle_e2e.py::test_full_session_lifecycle_through_dispatch_surface": 21,
    "jaato-server/server/runner/tests/test_session_send_message_rpc.py::test_send_message_fires_post_turn_notifications_in_order": 26,
    "jaato-server/server/runner_rpc_client.py::RunnerRPCClient._read_loop": 23,
    "jaato-server/server/runner_rpc_handlers/profile_payload_schema.py::validate_profile_payload": 16,
    "jaato-server/server/runner_rpc_handlers/spawn_isolated_runner.py::SpawnIsolatedRunnerHandler.handle": 26,
    "jaato-server/server/runner_spawn.py::build_session_envelope": 42,
    "jaato-server/server/runner_spawn.py::spawn_session_runner": 19,
    "jaato-server/server/session_manager.py::SessionManager._apply_client_config": 16,
    "jaato-server/server/session_manager.py::SessionManager._build_isolated_envelope": 23,
    "jaato-server/server/session_manager.py::SessionManager._build_session_info_event": 22,
    "jaato-server/server/session_manager.py::SessionManager._cascade_teardown_isolated_subagents": 20,
    "jaato-server/server/session_manager.py::SessionManager._create_session_impl": 61,
    "jaato-server/server/session_manager.py::SessionManager._expand_prompt_references": 18,
    "jaato-server/server/session_manager.py::SessionManager._handle_turn_tracking_event": 21,
    "jaato-server/server/session_manager.py::SessionManager._intercept_prompt_help_refs": 17,
    "jaato-server/server/session_manager.py::SessionManager._load_session_impl": 58,
    "jaato-server/server/session_manager.py::SessionManager._provision_ipc_apparmor_and_spawn_runner": 25,
    "jaato-server/server/session_manager.py::SessionManager._run_ephemeral_session_impl": 21,
    "jaato-server/server/session_manager.py::SessionManager._save_session": 42,
    "jaato-server/server/session_manager.py::SessionManager._spawn_isolated_runner": 17,
    "jaato-server/server/session_manager.py::SessionManager.attach_session": 16,
    "jaato-server/server/session_manager.py::SessionManager.build_sibling_roster": 18,
    "jaato-server/server/session_manager.py::SessionManager.deliver_sibling_message": 17,
    "jaato-server/server/session_manager.py::SessionManager.handle_request": 103,
    "jaato-server/server/session_manager.py::SessionManager.wake_session": 17,
    "jaato-server/server/test_client.py::format_event": 29,
    "jaato-server/server/tests/test_runner_cgroup_attach_7d.py::test_websocket_pre_init_hook_provisions_cgroup_before_spawn": 23,
    "jaato-server/server/tests/test_the_heal_path_says_what_it_did.py::test_both_notification_hooks_log_the_miss": 16,
    "jaato-server/server/tests/test_the_heal_path_says_what_it_did.py::test_every_cache_writer_either_logs_or_is_named_here": 17,
    "jaato-server/server/tests/test_transport_timeout_does_not_kill_the_session.py::test_the_terminal_assignment_is_unreachable_for_transport_timeouts": 16,
    "jaato-server/server/wake_ingress.py::process_wake": 19,
    "jaato-server/server/websocket.py::JaatoWSServer._handle_message": 30,
    "jaato-server/server/websocket.py::JaatoWSServer._handle_message_daemon": 35,
    "jaato-server/server/websocket.py::JaatoWSServer._register_client_tools": 25,
    "jaato-server/server/websocket.py::JaatoWSServer.set_command_router._apparmor_pre_init_hook": 17,
    "jaato-server/server/websocket.py::JaatoWSServer.set_command_router._apparmor_session_hook": 19,
    "jaato-server/server/websocket.py::JaatoWSServer.start": 21,
    "jaato-server/server/workspace_monitor.py::WorkspaceMonitor._on_fs_event": 18,
    "jaato-server/server/workspace_monitor.py::WorkspaceMonitor.reconcile": 30,
    "jaato-server/shared/ai_tool_runner.py::ToolExecutor._execute_impl": 73,
    "jaato-server/shared/ai_tool_runner.py::ToolExecutor._execute_with_auto_background": 17,
    "jaato-server/shared/budget_control.py::DegradeRung.from_dict": 16,
    "jaato-server/shared/change_tools.py::changed_lines_tool": 24,
    "jaato-server/shared/client_commands.py::parse_user_input": 24,
    "jaato-server/shared/completion_processors.py::invoke_processors": 30,
    "jaato-server/shared/event_bus_tools.py::_format_event_notification": 18,
    "jaato-server/shared/instruction_budget_builder.py::collect_instruction_texts": 28,
    "jaato-server/shared/jaato_runtime.py::JaatoRuntime._cache_tool_configuration": 26,
    "jaato-server/shared/jaato_runtime.py::JaatoRuntime.create_provider": 19,
    "jaato-server/shared/jaato_runtime.py::JaatoRuntime.get_system_instructions": 32,
    "jaato-server/shared/jaato_runtime.py::JaatoRuntime.get_tool_schemas": 16,
    "jaato-server/shared/jaato_session.py::JaatoSession._execute_function_calls_parallel": 20,
    "jaato-server/shared/jaato_session.py::JaatoSession._execute_single_tool": 33,
    "jaato-server/shared/jaato_session.py::JaatoSession._execute_single_tool_for_parallel": 30,
    "jaato-server/shared/jaato_session.py::JaatoSession._execute_tools_and_continue": 22,
    "jaato-server/shared/jaato_session.py::JaatoSession._get_framework_enrichments": 16,
    "jaato-server/shared/jaato_session.py::JaatoSession._handle_cancellation": 18,
    "jaato-server/shared/jaato_session.py::JaatoSession._run_chat_loop": 89,
    "jaato-server/shared/jaato_session.py::JaatoSession._run_chat_loop_with_parts": 35,
    "jaato-server/shared/jaato_session.py::JaatoSession._send_tool_results_and_continue": 27,
    "jaato-server/shared/jaato_session.py::JaatoSession._track_activated_tools_in_budget": 20,
    "jaato-server/shared/jaato_session.py::JaatoSession._update_conversation_budget": 37,
    "jaato-server/shared/jaato_session.py::JaatoSession.activate_discovered_tools": 21,
    "jaato-server/shared/jaato_session.py::JaatoSession.configure": 87,
    "jaato-server/shared/jaato_session.py::JaatoSession.resolve_fork_point": 17,
    "jaato-server/shared/jaato_session.py::JaatoSession.send_message": 29,
    "jaato-server/shared/lifecycle_tools.py::LifecycleTools._describe_pending_field": 28,
    "jaato-server/shared/lifecycle_tools.py::LifecycleTools._execute_prepare_completion": 19,
    "jaato-server/shared/lifecycle_tools.py::LifecycleTools._execute_signal_completion": 27,
    "jaato-server/shared/lifecycle_tools.py::LifecycleTools._walk_required": 17,
    "jaato-server/shared/plugins/artifact_tracker/plugin.py::ArtifactTrackerPlugin.enrich_tool_result": 17,
    "jaato-server/shared/plugins/ast_search/plugin.py::ASTSearchPlugin._execute_ast_search": 43,
    "jaato-server/shared/plugins/ast_search/plugin.py::ASTSearchPlugin.execute_streaming": 46,
    "jaato-server/shared/plugins/background/mixin.py::BackgroundCapableMixin.get_task": 19,
    "jaato-server/shared/plugins/bundle/plugin.py::BundlePlugin._cmd_add": 21,
    "jaato-server/shared/plugins/bundle/plugin.py::BundlePlugin._cmd_create": 29,
    "jaato-server/shared/plugins/bundle/plugin.py::BundlePlugin._cmd_delete": 21,
    "jaato-server/shared/plugins/bundle/plugin.py::BundlePlugin._cmd_pack": 29,
    "jaato-server/shared/plugins/bundle/plugin.py::BundlePlugin._cmd_reconcile": 35,
    "jaato-server/shared/plugins/bundle/plugin.py::BundlePlugin._cmd_unpack": 41,
    "jaato-server/shared/plugins/bundle/plugin.py::BundlePlugin.get_command_completions": 108,
    "jaato-server/shared/plugins/bundle_common/bundle.py::_load_bundle_from_manifest": 18,
    "jaato-server/shared/plugins/clarification/channels.py::ClarificationChannel._parse_answer": 20,
    "jaato-server/shared/plugins/clarification/channels.py::ConsoleChannel._ask_multiple_choice": 20,
    "jaato-server/shared/plugins/clarification/channels.py::QueueChannel.request_clarification": 19,
    "jaato-server/shared/plugins/clarification/plugin.py::ClarificationPlugin._execute_clarification": 32,
    "jaato-server/shared/plugins/cli/plugin.py::CLIToolPlugin._classify_path_modes": 18,
    "jaato-server/shared/plugins/cli/plugin.py::CLIToolPlugin._execute": 23,
    "jaato-server/shared/plugins/cli/plugin.py::CLIToolPlugin._execute_streaming": 24,
    "jaato-server/shared/plugins/cli/plugin.py::CLIToolPlugin.initialize": 18,
    "jaato-server/shared/plugins/code_block_formatter/plugin.py::CodeBlockFormatterPlugin._render_code_block": 23,
    "jaato-server/shared/plugins/diff_formatter/renderers/side_by_side.py::SideBySideRenderer._render_pair": 17,
    "jaato-server/shared/plugins/enrichment_formatter.py::_word_wrap": 16,
    "jaato-server/shared/plugins/environment/plugin.py::EnvironmentPlugin._get_network_info": 18,
    "jaato-server/shared/plugins/file_edit/multi_file.py::FileOperation.from_dict": 17,
    "jaato-server/shared/plugins/file_edit/multi_file.py::MultiFileExecutor._rollback": 17,
    "jaato-server/shared/plugins/file_edit/multi_file.py::MultiFileExecutor.validate_operations": 36,
    "jaato-server/shared/plugins/file_edit/multi_file.py::generate_multi_file_diff_preview": 16,
    "jaato-server/shared/plugins/file_edit/plugin.py::FileEditPlugin._execute_move_file": 16,
    "jaato-server/shared/plugins/file_edit/plugin.py::FileEditPlugin._execute_read_file": 20,
    "jaato-server/shared/plugins/file_edit/plugin.py::FileEditPlugin._execute_update_file": 16,
    "jaato-server/shared/plugins/file_edit/plugin.py::FileEditPlugin._format_multi_file_edit": 18,
    "jaato-server/shared/plugins/filesystem_query/config_loader.py::load_config": 18,
    "jaato-server/shared/plugins/filesystem_query/config_loader.py::validate_config": 22,
    "jaato-server/shared/plugins/filesystem_query/plugin.py::FilesystemQueryPlugin._execute_glob_files": 21,
    "jaato-server/shared/plugins/filesystem_query/plugin.py::FilesystemQueryPlugin._execute_grep_content": 32,
    "jaato-server/shared/plugins/filesystem_query/plugin.py::FilesystemQueryPlugin._stream_glob_files": 20,
    "jaato-server/shared/plugins/filesystem_query/plugin.py::FilesystemQueryPlugin.execute_streaming": 30,
    "jaato-server/shared/plugins/gc/__init__.py::load_gc_from_file": 20,
    "jaato-server/shared/plugins/gc/utils.py::dedup_identical_tool_results": 16,
    "jaato-server/shared/plugins/gc/utils.py::ensure_tool_call_integrity": 41,
    "jaato-server/shared/plugins/gc_budget/plugin.py::BudgetGCPlugin._build_tool_call_pair_map": 17,
    "jaato-server/shared/plugins/gc_budget/plugin.py::BudgetGCPlugin.collect": 28,
    "jaato-server/shared/plugins/gc_budget/tests/test_budget_gc.py::TestEndToEndPairAwareGC.test_gc_removes_tool_result_keeps_pairing": 21,
    "jaato-server/shared/plugins/interactive_shell/session.py::ShellSession._read_until_idle_windows": 18,
    "jaato-server/shared/plugins/introspection/plugin.py::IntrospectionPlugin._execute_list_tools": 50,
    "jaato-server/shared/plugins/lsp/plugin.py::LSPToolPlugin._build_empty_result_error": 19,
    "jaato-server/shared/plugins/lsp/plugin.py::LSPToolPlugin._build_no_server_error": 16,
    "jaato-server/shared/plugins/lsp/plugin.py::LSPToolPlugin._call_lsp_method": 35,
    "jaato-server/shared/plugins/lsp/plugin.py::LSPToolPlugin._compose_lsp_server_data_dir_rules": 18,
    "jaato-server/shared/plugins/lsp/plugin.py::LSPToolPlugin._exec_apply_code_action": 25,
    "jaato-server/shared/plugins/lsp/plugin.py::LSPToolPlugin._extract_first_data_dir_from_args": 17,
    "jaato-server/shared/plugins/lsp/plugin.py::LSPToolPlugin._thread_main.run_lsp": 29,
    "jaato-server/shared/plugins/lsp/plugin.py::LSPToolPlugin.get_file_dependents": 25,
    "jaato-server/shared/plugins/lsp/tests/test_plugin.py::TestLSPClientUtilities.test_guess_language_id": 19,
    "jaato-server/shared/plugins/mcp/plugin.py::MCPToolPlugin._cmd_logs": 18,
    "jaato-server/shared/plugins/mcp/plugin.py::MCPToolPlugin._cmd_reload": 26,
    "jaato-server/shared/plugins/mcp/plugin.py::MCPToolPlugin._execute": 23,
    "jaato-server/shared/plugins/mcp/plugin.py::MCPToolPlugin._thread_main.run_mcp_server": 41,
    "jaato-server/shared/plugins/mcp/plugin.py::MCPToolPlugin.execute_streaming": 26,
    "jaato-server/shared/plugins/mcp/plugin.py::MCPToolPlugin.get_command_completions": 22,
    "jaato-server/shared/plugins/memory/plugin.py::MemoryPlugin._enrich_text": 23,
    "jaato-server/shared/plugins/memory/plugin.py::MemoryPlugin._execute_retrieve": 31,
    "jaato-server/shared/plugins/memory/plugin.py::MemoryPlugin._execute_update": 20,
    "jaato-server/shared/plugins/memory/plugin.py::MemoryPlugin._memory_edit": 23,
    "jaato-server/shared/plugins/memory/plugin.py::MemoryPlugin._validate_memory_schema": 24,
    "jaato-server/shared/plugins/memory/test_standalone.py::test_basic_functionality": 17,
    "jaato-server/shared/plugins/mermaid_formatter/backends/sixel.py::SixelBackend.render": 17,
    "jaato-server/shared/plugins/model_provider/_openai_compat/base.py::OpenAICompatProvider._handle_api_error": 16,
    "jaato-server/shared/plugins/model_provider/_openai_compat/base.py::OpenAICompatProvider._stream_response": 44,
    "jaato-server/shared/plugins/model_provider/_openai_compat/base.py::OpenAICompatProvider.complete": 19,
    "jaato-server/shared/plugins/model_provider/_openai_compat/converters.py::message_to_openai": 26,
    "jaato-server/shared/plugins/model_provider/anthropic/converters.py::validate_tool_use_pairing": 19,
    "jaato-server/shared/plugins/model_provider/anthropic/provider.py::AnthropicProvider._handle_api_error": 22,
    "jaato-server/shared/plugins/model_provider/anthropic/provider.py::AnthropicProvider._stream_response": 56,
    "jaato-server/shared/plugins/model_provider/anthropic/provider.py::AnthropicProvider.complete": 37,
    "jaato-server/shared/plugins/model_provider/anthropic/provider.py::AnthropicProvider.initialize": 25,
    "jaato-server/shared/plugins/model_provider/anthropic/provider.py::AnthropicProvider.verify_auth": 16,
    "jaato-server/shared/plugins/model_provider/antigravity/provider.py::AntigravityProvider._make_request": 17,
    "jaato-server/shared/plugins/model_provider/antigravity/provider.py::AntigravityProvider._process_stream": 23,
    "jaato-server/shared/plugins/model_provider/antigravity/provider.py::AntigravityProvider.initialize": 21,
    "jaato-server/shared/plugins/model_provider/chrome_ai/provider.py::ChromeAIProvider.complete": 28,
    "jaato-server/shared/plugins/model_provider/claude_cli/provider.py::ClaudeCLIProvider._build_cli_args": 19,
    "jaato-server/shared/plugins/model_provider/claude_cli/provider.py::ClaudeCLIProvider._execute_query": 18,
    "jaato-server/shared/plugins/model_provider/claude_cli/provider.py::ClaudeCLIProvider._execute_query_streaming": 58,
    "jaato-server/shared/plugins/model_provider/claude_cli/provider.py::ClaudeCLIProvider._stream_cli_messages": 27,
    "jaato-server/shared/plugins/model_provider/doubleword/auth.py::validate_api_key": 16,
    "jaato-server/shared/plugins/model_provider/doubleword/provider.py::DoublewordProvider.verify_auth": 17,
    "jaato-server/shared/plugins/model_provider/github_models/converters.py::message_from_sdk": 17,
    "jaato-server/shared/plugins/model_provider/github_models/copilot_client.py::CopilotClient._make_request": 24,
    "jaato-server/shared/plugins/model_provider/github_models/copilot_client.py::CopilotClient.complete_responses_stream": 22,
    "jaato-server/shared/plugins/model_provider/github_models/copilot_client.py::CopilotClient.list_models_with_info": 17,
    "jaato-server/shared/plugins/model_provider/github_models/provider.py::GitHubModelsProvider._build_copilot_messages_from": 19,
    "jaato-server/shared/plugins/model_provider/github_models/provider.py::GitHubModelsProvider._complete_azure_streaming": 42,
    "jaato-server/shared/plugins/model_provider/github_models/provider.py::GitHubModelsProvider._copilot_responses_streaming": 22,
    "jaato-server/shared/plugins/model_provider/github_models/provider.py::GitHubModelsProvider._copilot_streaming_response": 21,
    "jaato-server/shared/plugins/model_provider/github_models/provider.py::GitHubModelsProvider._fetch_models_from_api.parse_model": 16,
    "jaato-server/shared/plugins/model_provider/github_models/provider.py::GitHubModelsProvider._handle_api_error": 27,
    "jaato-server/shared/plugins/model_provider/github_models/provider.py::GitHubModelsProvider._responses_api_response_to_provider": 21,
    "jaato-server/shared/plugins/model_provider/github_models/provider.py::GitHubModelsProvider.initialize": 18,
    "jaato-server/shared/plugins/model_provider/google_genai/converters.py::part_from_sdk": 23,
    "jaato-server/shared/plugins/model_provider/google_genai/provider.py::GoogleGenAIProvider._complete_streaming": 44,
    "jaato-server/shared/plugins/model_provider/google_genai/provider.py::GoogleGenAIProvider.initialize": 16,
    "jaato-server/shared/plugins/model_provider/nebius/auth.py::validate_api_key": 16,
    "jaato-server/shared/plugins/model_provider/nebius/converters.py::message_to_openai": 26,
    "jaato-server/shared/plugins/model_provider/nebius/provider.py::NebiusProvider.verify_auth": 17,
    "jaato-server/shared/plugins/model_provider/nim/auth.py::validate_api_key": 16,
    "jaato-server/shared/plugins/model_provider/nim/provider.py::NIMProvider.verify_auth": 17,
    "jaato-server/shared/plugins/model_provider/openrouter/converters.py::message_to_openai": 27,
    "jaato-server/shared/plugins/model_provider/openrouter/provider.py::OpenRouterProvider._handle_api_error": 16,
    "jaato-server/shared/plugins/model_provider/openrouter/provider.py::OpenRouterProvider._stream_response": 51,
    "jaato-server/shared/plugins/model_provider/openrouter/provider.py::OpenRouterProvider.complete": 27,
    "jaato-server/shared/plugins/model_provider/openrouter/provider.py::OpenRouterProvider.initialize": 45,
    "jaato-server/shared/plugins/model_provider/ovhcloud/auth.py::validate_api_key": 16,
    "jaato-server/shared/plugins/model_provider/ovhcloud/provider.py::OVHcloudProvider.verify_auth": 23,
    "jaato-server/shared/plugins/model_provider/vllm/provider.py::VLLMProvider._coerce_args_to_schema": 20,
    "jaato-server/shared/plugins/model_provider/vllm/provider.py::VLLMProvider._stream_response": 43,
    "jaato-server/shared/plugins/model_provider/vllm/provider.py::VLLMProvider.complete": 30,
    "jaato-server/shared/plugins/model_provider/zhipuai/auth.py::validate_api_key": 16,
    "jaato-server/shared/plugins/model_provider/zhipuai/provider.py::ZhipuAIProvider.initialize": 20,
    "jaato-server/shared/plugins/notebook/backends/kaggle.py::KaggleBackend._get_kernel_output": 18,
    "jaato-server/shared/plugins/notebook/backends/kaggle.py::KaggleBackend._parse_kernel_output": 18,
    "jaato-server/shared/plugins/notebook/backends/kaggle.py::KaggleBackend.execute": 36,
    "jaato-server/shared/plugins/notebook/plugin.py::NotebookPlugin._execute_code": 45,
    "jaato-server/shared/plugins/notebook/plugin.py::NotebookPlugin._execute_streaming_impl": 28,
    "jaato-server/shared/plugins/notebook/plugin.py::NotebookPlugin.execute_streaming": 18,
    "jaato-server/shared/plugins/notebook/plugin.py::NotebookPlugin.format_permission_request": 17,
    "jaato-server/shared/plugins/permission/channels.py::ConsoleChannel.request_permission": 16,
    "jaato-server/shared/plugins/permission/plugin.py::PermissionPlugin._get_tool_completions": 16,
    "jaato-server/shared/plugins/permission/plugin.py::PermissionPlugin._handle_channel_response": 16,
    "jaato-server/shared/plugins/permission/plugin.py::PermissionPlugin.check_permission": 88,
    "jaato-server/shared/plugins/permission/plugin.py::PermissionPlugin.execute_permissions": 17,
    "jaato-server/shared/plugins/permission/policy.py::PermissionPolicy.check": 18,
    "jaato-server/shared/plugins/permission/runner_rpc_channel.py::RunnerRPCChannel.request_permission": 17,
    "jaato-server/shared/plugins/permission/sanitization.py::check_path_scope": 19,
    "jaato-server/shared/plugins/prompt_library/plugin.py::PromptLibraryPlugin._execute_prompt_command": 33,
    "jaato-server/shared/plugins/prompt_library/plugin.py::PromptLibraryPlugin._fetch_from_git": 25,
    "jaato-server/shared/plugins/prompt_library/plugin.py::PromptLibraryPlugin._fetch_from_npx": 17,
    "jaato-server/shared/plugins/prompt_library/plugin.py::PromptLibraryPlugin.get_command_completions": 29,
    "jaato-server/shared/plugins/references/config_loader.py::discover_references": 26,
    "jaato-server/shared/plugins/references/config_loader.py::load_config": 21,
    "jaato-server/shared/plugins/references/config_loader.py::validate_config": 19,
    "jaato-server/shared/plugins/references/config_loader.py::validate_reference_file": 33,
    "jaato-server/shared/plugins/references/config_loader.py::validate_source": 22,
    "jaato-server/shared/plugins/references/merge.py::merge_bundle": 51,
    "jaato-server/shared/plugins/references/models.py::ReferenceSource.to_instruction": 18,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._apply_transitive_selection": 21,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._build_contents_annotation": 18,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._bundle_completions": 87,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._cmd_bundle_add": 22,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._cmd_bundle_create": 28,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._cmd_bundle_delete": 24,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._cmd_bundle_merge": 55,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._cmd_bundle_pack": 28,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._cmd_bundle_reconcile": 35,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._cmd_bundle_unpack": 44,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._cmd_references_reload": 34,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._discover_and_load_bundles": 20,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._enrich_content": 74,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._execute_list": 23,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._execute_select": 54,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._get_reference_content": 17,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._post_membership_change": 19,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._references_completions": 19,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin._resolve_transitive_references": 17,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin.get_system_instructions": 29,
    "jaato-server/shared/plugins/references/plugin.py::ReferencesPlugin.initialize": 73,
    "jaato-server/shared/plugins/references/reconcile.py::_reconcile_locked": 31,
    "jaato-server/shared/plugins/registry.py::PluginRegistry._discover_via_directory": 21,
    "jaato-server/shared/plugins/registry.py::PluginRegistry._discover_via_entry_points": 16,
    "jaato-server/shared/plugins/registry.py::PluginRegistry._generate_fallback_message": 42,
    "jaato-server/shared/plugins/registry.py::PluginRegistry.expose_all": 17,
    "jaato-server/shared/plugins/registry.py::PluginRegistry.expose_tool": 33,
    "jaato-server/shared/plugins/reliability/plugin.py::ReliabilityPlugin._apply_profile_overrides": 17,
    "jaato-server/shared/plugins/reliability/plugin.py::ReliabilityPlugin._check_model_switch_suggestion": 17,
    "jaato-server/shared/plugins/reliability/plugin.py::ReliabilityPlugin._execute_status": 24,
    "jaato-server/shared/plugins/reliability/plugin.py::ReliabilityPlugin._handle_failure": 22,
    "jaato-server/shared/plugins/reliability/plugin.py::ReliabilityPlugin.check_escalation": 17,
    "jaato-server/shared/plugins/reliability/plugin.py::ReliabilityPlugin.get_command_completions": 64,
    "jaato-server/shared/plugins/reliability/policy_config.py::_parse_pattern_detection_kwargs": 27,
    "jaato-server/shared/plugins/reliability/policy_config.py::_parse_single_policy": 27,
    "jaato-server/shared/plugins/reliability/types.py::FailureKey._extract_key_params": 20,
    "jaato-server/shared/plugins/reliability/types.py::classify_failure": 22,
    "jaato-server/shared/plugins/sandbox_manager/plugin.py::SandboxManagerPlugin._cmd_add": 16,
    "jaato-server/shared/plugins/sandbox_manager/plugin.py::SandboxManagerPlugin._cmd_remove": 17,
    "jaato-server/shared/plugins/sandbox_manager/plugin.py::SandboxManagerPlugin._execute_sandbox_command": 16,
    "jaato-server/shared/plugins/sandbox_manager/plugin.py::SandboxManagerPlugin._replay_pending_paths": 19,
    "jaato-server/shared/plugins/sandbox_manager/plugin.py::SandboxManagerPlugin.add_path_programmatic": 17,
    "jaato-server/shared/plugins/sandbox_utils.py::check_path_with_jaato_containment": 31,
    "jaato-server/shared/plugins/service_connector/auth.py::AuthManager.get_auth_headers": 19,
    "jaato-server/shared/plugins/service_connector/bruno_import.py::parse_bru_file": 18,
    "jaato-server/shared/plugins/service_connector/bruno_import.py::parse_bruno_collection": 28,
    "jaato-server/shared/plugins/service_connector/http_client.py::ServiceHttpClient.build_request": 26,
    "jaato-server/shared/plugins/service_connector/http_client.py::ServiceHttpClient.execute": 40,
    "jaato-server/shared/plugins/service_connector/openapi_parser.py::_extract_json_schema": 19,
    "jaato-server/shared/plugins/service_connector/plugin.py::ServiceConnectorPlugin._build_auth_context": 30,
    "jaato-server/shared/plugins/service_connector/plugin.py::ServiceConnectorPlugin._cmd_auth": 16,
    "jaato-server/shared/plugins/service_connector/plugin.py::ServiceConnectorPlugin._cmd_endpoints": 18,
    "jaato-server/shared/plugins/service_connector/plugin.py::ServiceConnectorPlugin._cmd_show": 18,
    "jaato-server/shared/plugins/service_connector/plugin.py::ServiceConnectorPlugin._execute_call_service": 69,
    "jaato-server/shared/plugins/service_connector/plugin.py::ServiceConnectorPlugin._execute_discover_service": 16,
    "jaato-server/shared/plugins/service_connector/plugin.py::ServiceConnectorPlugin._execute_list_endpoints": 19,
    "jaato-server/shared/plugins/service_connector/plugin.py::ServiceConnectorPlugin._execute_preview_request": 17,
    "jaato-server/shared/plugins/service_connector/validation.py::_validate_schema": 44,
    "jaato-server/shared/plugins/subagent/config.py::_discover_premium_profiles": 29,
    "jaato-server/shared/plugins/subagent/config.py::_merge_profiles": 44,
    "jaato-server/shared/plugins/subagent/config.py::_scan_profiles_dir": 35,
    "jaato-server/shared/plugins/subagent/config.py::build_inline_profile": 17,
    "jaato-server/shared/plugins/subagent/config.py::resolve_agent": 25,
    "jaato-server/shared/plugins/subagent/config.py::validate_profile": 53,
    "jaato-server/shared/plugins/subagent/plugin.py::SubagentPlugin._dispatch_isolated_spawn": 22,
    "jaato-server/shared/plugins/subagent/plugin.py::SubagentPlugin._execute_spawn_subagent": 64,
    "jaato-server/shared/plugins/subagent/plugin.py::SubagentPlugin._run_subagent_async": 75,
    "jaato-server/shared/plugins/subagent/plugin.py::SubagentPlugin.initialize": 17,
    "jaato-server/shared/plugins/subagent/plugin.py::SubagentPlugin.restore_persistence_state": 29,
    "jaato-server/shared/plugins/subagent/tests/test_serializer.py::TestSerializeSubagentRegistry.test_serialize_registry_with_agents": 17,
    "jaato-server/shared/plugins/table_formatter/plugin.py::TableFormatterPlugin.process_chunk": 16,
    "jaato-server/shared/plugins/template/plugin.py::TemplatePlugin._check_item_against_item_keys": 30,
    "jaato-server/shared/plugins/template/plugin.py::TemplatePlugin._discover_standalone_templates": 16,
    "jaato-server/shared/plugins/template/plugin.py::TemplatePlugin._enrich_text_with_template_hints": 19,
    "jaato-server/shared/plugins/template/plugin.py::TemplatePlugin._execute_list_template_variables": 20,
    "jaato-server/shared/plugins/template/plugin.py::TemplatePlugin._execute_render_template_to_file": 30,
    "jaato-server/shared/plugins/template/plugin.py::TemplatePlugin._parse_mustache_structure": 45,
    "jaato-server/shared/plugins/template/plugin.py::TemplatePlugin._preprocess_mustache_dotted_paths": 17,
    "jaato-server/shared/plugins/template/plugin.py::TemplatePlugin._resolve_template_path": 16,
    "jaato-server/shared/plugins/template/plugin.py::TemplatePlugin._validate_render_inputs_against_structure": 31,
    "jaato-server/shared/plugins/template/plugin.py::TemplatePlugin._validate_template_index": 26,
    "jaato-server/shared/plugins/template/plugin.py::TemplatePlugin.enrich_tool_result": 29,
    "jaato-server/shared/plugins/todo/config_loader.py::validate_config": 26,
    "jaato-server/shared/plugins/todo/plugin.py::TodoPlugin._execute_add_dependent_step": 19,
    "jaato-server/shared/plugins/todo/plugin.py::TodoPlugin._execute_create_plan": 20,
    "jaato-server/shared/plugins/todo/plugin.py::TodoPlugin._execute_set_step_status": 29,
    "jaato-server/shared/plugins/waypoint/plugin.py::WaypointPlugin.get_command_completions": 22,
    "jaato-server/shared/plugins/waypoint/tests/test_plugin.py::TestCommandExecution.test_list_shows_current_waypoint": 18,
    "jaato-server/shared/plugins/web_fetch/plugin.py::WebFetchPlugin._detect_content_type": 21,
    "jaato-server/shared/plugins/web_fetch/plugin.py::WebFetchPlugin._execute": 65,
    "jaato-server/shared/plugins/web_fetch/plugin.py::WebFetchPlugin._extract_forms": 17,
    "jaato-server/shared/plugins/web_fetch/plugin.py::WebFetchPlugin._fetch_url": 30,
    "jaato-server/shared/plugins/web_fetch/plugin.py::WebFetchPlugin._html_to_markdown": 36,
    "jaato-server/shared/plugins/web_fetch/plugin.py::WebFetchPlugin.initialize": 18,
    "jaato-server/shared/plugins/webhook/config.py::validate_config": 40,
    "jaato-server/shared/retry_utils.py::classify_error": 22,
    "jaato-server/shared/retry_utils.py::is_context_limit_error": 22,
    "jaato-server/shared/retry_utils.py::with_retry": 23,
    "jaato-server/shared/rewind.py::detect_truncated_tool_call": 16,
    "jaato-server/shared/runtime_limits.py::RuntimeLimits.__post_init__": 18,
    "jaato-server/shared/scaffold/__main__.py::_cmd_explain": 24,
    "jaato-server/shared/scaffold/__main__.py::_cmd_validate": 16,
    "jaato-server/shared/scaffold/build.py::_compose_env": 17,
    "jaato-server/shared/scaffold/build.py::_new_client_archetype": 30,
    "jaato-server/shared/scaffold/build.py::_new_profile_set": 29,
    "jaato-server/shared/scaffold/explain.py::event": 21,
    "jaato-server/shared/scaffold/explain.py::events": 17,
    "jaato-server/shared/scaffold/explain.py::plugin": 17,
    "jaato-server/shared/scaffold/explain.py::profile_cost": 23,
    "jaato-server/shared/scaffold/explain.py::provider": 16,
    "jaato-server/shared/scaffold/explain.py::sets": 17,
    "jaato-server/shared/scaffold/introspect.py::_env_reads": 21,
    "jaato-server/shared/scaffold/introspect.py::events": 32,
    "jaato-server/shared/scaffold/introspect.py::plugins": 24,
    "jaato-server/shared/scaffold/validate.py::_check_prefetch_directives": 18,
    "jaato-server/shared/scaffold/validate.py::validate_profile": 77,
    "jaato-server/shared/session_envelope.py::SessionInitEnvelope.from_dict": 19,
    "jaato-server/shared/subprocess_runner.py::run_command": 28,
    "jaato-server/shared/tests/test_session_envelope.py::test_bootstrap_envelope_minimal_construction": 20,
    "jaato-server/shared/tests/test_unreachable_says_which_kind.py::test_every_unreachable_producer_logs_a_cause": 21,
    "jaato-server/shared/token_accounting.py::TokenLedger.summarize": 18,
    "jaato-server/shared/ui_utils.py::ellipsize_path": 23,
    "jaato-tui/agent_registry.py::AgentRegistry.create_agent": 16,
    "jaato-tui/agent_tab_bar.py::AgentTabBar.get_selected_agent_tab_offset": 19,
    "jaato-tui/agent_tab_bar.py::AgentTabBar.render": 20,
    "jaato-tui/agent_tab_bar.py::AgentTabBar.render_pane_aligned": 18,
    "jaato-tui/client_commands.py::parse_user_input": 24,
    "jaato-tui/color_picker.py::HexColorInput.handle_key": 19,
    "jaato-tui/command_mode.py::run_command_mode": 34,
    "jaato-tui/editor_utils.py::format_for_editing": 19,
    "jaato-tui/editor_utils.py::parse_edited_content": 17,
    "jaato-tui/file_completer.py::AtFileCompleter.get_completions": 16,
    "jaato-tui/file_completer.py::CommandCompleter.get_completions": 43,
    "jaato-tui/headless_mode.py::run_headless_mode.handle_events": 63,
    "jaato-tui/j_markup_renderer.py::_render_table_to_ansi": 19,
    "jaato-tui/keybindings.py::KeybindingConfig.from_file": 16,
    "jaato-tui/keybindings.py::detect_terminal": 20,
    "jaato-tui/output_buffer.py::OutputBuffer._calculate_tool_tree_height": 33,
    "jaato-tui/output_buffer.py::OutputBuffer._finalize_completed_tools": 16,
    "jaato-tui/output_buffer.py::OutputBuffer._measure_display_lines": 16,
    "jaato-tui/output_buffer.py::OutputBuffer._render_active_tools_inline": 79,
    "jaato-tui/output_buffer.py::OutputBuffer._render_impl": 117,
    "jaato-tui/output_buffer.py::OutputBuffer._render_single_notebook_row": 17,
    "jaato-tui/output_buffer.py::OutputBuffer._render_tool_block": 53,
    "jaato-tui/output_buffer.py::OutputBuffer._scroll_to_selected_tool": 21,
    "jaato-tui/output_buffer.py::OutputBuffer.add_active_tool": 31,
    "jaato-tui/output_buffer.py::OutputBuffer.append": 48,
    "jaato-tui/output_buffer.py::OutputBuffer.mark_tool_completed": 24,
    "jaato-tui/output_buffer.py::format_turn_for_clipboard": 20,
    "jaato-tui/plan_panel.py::PlanPanel.render_popup": 27,
    "jaato-tui/pt_display.py::PTDisplay._get_status_bar_content": 25,
    "jaato-tui/pt_display.py::PTDisplay._maybe_retrigger_completion": 18,
    "jaato-tui/pt_display.py::ScrollableBufferControl.mouse_handler": 33,
    "jaato-tui/pt_display.py::StyledOutputProcessor.apply_transformation": 22,
    "jaato-tui/renderers/headless.py::HeadlessFileRenderer.on_plan_updated": 18,
    "jaato-tui/rich_client.py::handle_screenshot_command_ipc": 39,
    "jaato-tui/rich_client.py::main": 16,
    "jaato-tui/rich_client.py::run_ipc_mode.command_completion_provider": 24,
    "jaato-tui/rich_client.py::run_ipc_mode.handle_events": 217,
    "jaato-tui/rich_client.py::run_ipc_mode.handle_input": 71,
    "jaato-tui/tests/test_output_buffer.py::TestStreamingToolTreePositioning.test_multiple_streaming_rounds_with_tools": 20,
    "jaato-tui/tests/test_output_buffer.py::TestToolTreePositioning.test_multiple_tool_batches_maintain_order": 20,
    "jaato-tui/theme.py::validate_theme": 19,
    "jaato-tui/theme_editor.py::ThemeEditor.handle_key": 22,
    "jaato-tui/theme_editor.py::ThemeEditor.render_simple": 23,
    "jaato-tui/tool_output_popup.py::ToolOutputPopup.render": 16,
    "jaato-tui/ui_utils.py::ellipsize_path": 23,
    "jaato-tui/workspace_panel.py::WorkspacePanel.render_popup": 30,
}


def _repo_root() -> Path:
    """Return the repo root.

    This file lives at ``<root>/jaato-server/shared/tests/`` — three
    parents up from the test file is the checkout root that holds all
    three package directories.
    """
    return Path(__file__).resolve().parents[3]


def _iter_functions(blocks: Iterable[object], prefix: str = "") -> Iterator[Tuple[str, int]]:
    """Flatten radon blocks into ``(qualified_name, complexity)`` pairs.

    ``cc_visit`` returns a mix of ``Function`` and ``Class`` blocks, and
    hangs nested functions off their parent's ``closures`` rather than
    surfacing them at the top level.  Two consequences are handled here:

    - **Class blocks are skipped.**  A ``Class`` block's complexity is
      the aggregate over its methods, so keeping it would double-count
      and would make a class fail for reasons no single function owns.
      The methods themselves come back as ``Function`` blocks with
      ``fullname`` already spelled ``Class.method``.
    - **Closures are recursed into**, qualified as ``outer.inner``, so a
      complex nested helper cannot hide inside a simple-looking parent.
    """
    for block in blocks:
        if not isinstance(block, Function):
            continue
        name = prefix + block.fullname
        yield name, block.complexity
        yield from _iter_functions(getattr(block, "closures", []), name + ".")


@lru_cache(maxsize=1)
def _scan() -> Dict[str, int]:
    """Scan every package .py file, returning ``{"path::Qual.name": cc}``.

    Cached because all three tests need the same scan and it walks ~1150
    files.  Keys are repo-relative POSIX paths so the baseline reads the
    same on Windows checkouts.

    A handful of qualified names (10 at the time of writing, all in
    ``notebook/plugin.py``) appear twice in one file — the same name
    bound under different branches.  Those collapse to one key holding
    the **higher** score, which keeps the ratchet conservative: the
    guard can only be satisfied by the worst of the pair.
    """
    root = _repo_root()
    scores: Dict[str, int] = {}
    for package in PACKAGES:
        for path in sorted((root / package).rglob("*.py")):
            try:
                blocks = cc_visit(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                # A file this interpreter cannot parse is not this guard's
                # business — the test suites that import it will say so
                # far more usefully than a complexity number would.
                continue
            rel = path.relative_to(root).as_posix()
            for name, complexity in _iter_functions(blocks):
                key = f"{rel}::{name}"
                scores[key] = max(scores.get(key, 0), complexity)
    return scores


@pytest.fixture(scope="module")
def scores() -> Dict[str, int]:
    """Current complexity of every function in the scanned packages."""
    root = _repo_root()
    missing = [p for p in PACKAGES if not (root / p).is_dir()]
    if missing:
        pytest.skip(
            f"package directories not present at {root}: {', '.join(missing)} "
            "— the audit only runs against a full checkout"
        )
    return _scan()


def test_no_new_functions_over_ceiling(scores: Dict[str, int]) -> None:
    """Fail when a function above CEILING is not in the frozen BASELINE.

    This is the half of the guard that applies to new code: anything
    written from now on either stays at or below CEILING, or earns an
    explicit, reviewable baseline line.
    """
    offenders = sorted(
        (key, cc)
        for key, cc in scores.items()
        if cc > CEILING and key not in BASELINE
    )
    if offenders:
        details = "\n  ".join(f"{key}: {cc}" for key, cc in offenders)
        raise AssertionError(
            f"Functions above the cyclomatic-complexity ceiling ({CEILING}) "
            "that are not in BASELINE:\n  "
            f"{details}\n\n"
            "Preferred fix: split the function.  The usual shapes are a long\n"
            "if/elif dispatch (extract a table or per-branch helpers) and a\n"
            "function doing validation + work (split the two).\n\n"
            "If the complexity is genuinely irreducible, add the entry to\n"
            "BASELINE in this file WITH A COMMENT saying why — that line is a\n"
            "claim a reviewer can push back on.\n\n"
            "Note that radon counts `and`/`or` and comprehensions, so a run of\n"
            "defensive `x.get(k) or \"\"` defaults can push an otherwise flat\n"
            "function over the line; see this module's docstring."
        )


def test_baselined_functions_do_not_get_worse(scores: Dict[str, int]) -> None:
    """Fail when a baselined function's complexity has increased.

    The recorded number — not merely the name — is what makes BASELINE a
    ratchet.  Existing debt is tolerated at the level it was frozen at;
    growing it is not.
    """
    regressions = sorted(
        (key, BASELINE[key], scores[key])
        for key in BASELINE
        if key in scores and scores[key] > BASELINE[key]
    )
    if regressions:
        details = "\n  ".join(
            f"{key}: {was} -> {now} (+{now - was})" for key, was, now in regressions
        )
        raise AssertionError(
            "Baselined functions got MORE complex:\n  "
            f"{details}\n\n"
            "These are already over the ceiling; the baseline tolerates them\n"
            "at their frozen size only.  Add the new logic in a helper rather\n"
            "than growing the function, or take the opportunity to split it.\n\n"
            "Raising the recorded number is a last resort and should be called\n"
            "out in the PR description — it moves the ratchet the wrong way."
        )


def test_baseline_has_no_stale_entries(scores: Dict[str, int]) -> None:
    """Fail when a BASELINE entry has improved or disappeared.

    Rationale (mirrors ``test_session_env_audit``'s stale-entry test):
    stale entries hide progress and let the baseline calcify into
    permanent permission.  When a function is simplified or deleted, the
    line comes down with it — that is the mechanism by which the 416
    initial entries are meant to shrink.
    """
    improved: List[Tuple[str, int, int]] = []
    removed: List[str] = []
    for key, recorded in sorted(BASELINE.items()):
        current = scores.get(key)
        if current is None:
            removed.append(key)
        elif current < recorded:
            improved.append((key, recorded, current))

    if not improved and not removed:
        return

    parts: List[str] = []
    if improved:
        lines = "\n  ".join(
            f"{key}: {was} -> {now}"
            + ("  (now at or below the ceiling — DELETE the line)" if now <= CEILING
               else f"  (lower the recorded number to {now})")
            for key, was, now in improved
        )
        parts.append(f"Baselined functions that got SIMPLER:\n  {lines}")
    if removed:
        lines = "\n  ".join(removed)
        parts.append(
            "BASELINE entries with no matching function — deleted, renamed, or "
            f"moved:\n  {lines}\n\nDelete these lines from BASELINE."
        )

    raise AssertionError(
        "\n\n".join(parts)
        + "\n\nThis failure means the tree improved and the baseline did not "
        "keep up.\nUpdating it is the point of the guard, not a chore around it."
    )


if __name__ == "__main__":
    # Regeneration helper — see the module docstring.  Prints a fresh
    # BASELINE body for pasting over the literal above.
    current = _scan()
    print("BASELINE: Dict[str, int] = {")
    for _key, _cc in sorted(current.items()):
        if _cc > CEILING:
            print(f'    "{_key}": {_cc},')
    print("}")
