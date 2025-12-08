
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING
import subprocess
import os
import pathlib
from google import genai
from google.genai import types
from shared.token_accounting import TokenLedger
from shared.plugins.base import OutputCallback
from shared.plugins.model_provider.google_genai.converters import tool_schema_to_sdk

if TYPE_CHECKING:
    from shared.plugins.registry import PluginRegistry
    from shared.plugins.permission import PermissionPlugin
    from shared.plugins.background.protocol import BackgroundCapable


class ToolExecutor:
    """Registry mapping tool names to callables.

    Executors should accept a single dict-like argument and return a JSON-serializable result.

    Supports optional permission checking via a PermissionPlugin. When a permission
    plugin is set, all tool executions are checked against the permission policy
    before execution.

    Supports auto-backgrounding for BackgroundCapable plugins. When a tool execution
    exceeds the plugin's configured threshold, it is automatically converted to a
    background task and a handle is returned.
    """
    def __init__(
        self,
        ledger: Optional[TokenLedger] = None,
        auto_background_enabled: bool = True,
        auto_background_pool_size: int = 4
    ):
        self._map: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._permission_plugin: Optional['PermissionPlugin'] = None
        self._permission_context: Dict[str, Any] = {}
        self._ledger: Optional[TokenLedger] = ledger

        # Registry reference for plugin lookups (set via set_registry)
        self._registry: Optional['PluginRegistry'] = None

        # Output callback for real-time output from plugins
        self._output_callback: Optional[OutputCallback] = None

        # Auto-background support
        self._auto_background_enabled = auto_background_enabled
        self._auto_background_pool: Optional[ThreadPoolExecutor] = None
        self._auto_background_pool_size = auto_background_pool_size

    def register(self, name: str, fn: Callable[[Dict[str, Any]], Any]) -> None:
        self._map[name] = fn

    def set_ledger(self, ledger: Optional[TokenLedger]) -> None:
        """Set the ledger for recording events."""
        self._ledger = ledger

    def set_permission_plugin(
        self,
        plugin: Optional['PermissionPlugin'],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set the permission plugin for access control.

        Args:
            plugin: PermissionPlugin instance, or None to disable permission checking.
            context: Optional context dict passed to permission checks (e.g., session_id).
        """
        self._permission_plugin = plugin
        self._permission_context = context or {}

    def set_registry(self, registry: Optional['PluginRegistry']) -> None:
        """Set the plugin registry for plugin lookups.

        Required for auto-background support to find BackgroundCapable plugins.

        Args:
            registry: PluginRegistry instance, or None to disable.
        """
        self._registry = registry

    def set_output_callback(self, callback: Optional[OutputCallback]) -> None:
        """Set the output callback for real-time plugin output.

        When set, plugins that support output callbacks will receive this
        callback to emit real-time output during tool execution.

        The callback is passed to plugins via their set_output_callback()
        method if they implement it.

        Args:
            callback: OutputCallback function, or None to clear.
        """
        self._output_callback = callback

        # Forward callback to exposed plugins that support it
        if self._registry:
            for plugin_name in self._registry.list_exposed():
                plugin = self._registry.get_plugin(plugin_name)
                if plugin and hasattr(plugin, 'set_output_callback'):
                    plugin.set_output_callback(callback)

        # Also set on permission plugin if configured
        if self._permission_plugin and hasattr(self._permission_plugin, 'set_output_callback'):
            self._permission_plugin.set_output_callback(callback)

    def get_output_callback(self) -> Optional[OutputCallback]:
        """Get the current output callback.

        Returns:
            The current OutputCallback, or None if not set.
        """
        return self._output_callback

    def _get_auto_background_pool(self) -> ThreadPoolExecutor:
        """Get or create the thread pool for auto-background execution."""
        if self._auto_background_pool is None:
            self._auto_background_pool = ThreadPoolExecutor(
                max_workers=self._auto_background_pool_size
            )
        return self._auto_background_pool

    def _get_plugin_for_tool(self, tool_name: str) -> Optional['BackgroundCapable']:
        """Get the BackgroundCapable plugin that provides a tool.

        Args:
            tool_name: Name of the tool to look up.

        Returns:
            The BackgroundCapable plugin, or None if not found or not capable.
        """
        if not self._registry:
            return None

        # Import here to avoid circular imports
        from shared.plugins.background.protocol import BackgroundCapable

        plugin = self._registry.get_plugin_for_tool(tool_name)
        if plugin and isinstance(plugin, BackgroundCapable):
            return plugin
        return None

    def _execute_sync(self, name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        """Execute a tool synchronously (internal helper).

        This is the core execution logic, extracted to support auto-backgrounding.

        Args:
            name: Tool name.
            args: Arguments dict.

        Returns:
            Tuple of (success, result).
        """
        fn = self._map.get(name)
        if not fn:
            # Check if generic execution is allowed
            if os.environ.get('AI_EXECUTE_TOOLS', '').lower() in ('1', 'true', 'yes'):
                try:
                    return _generic_executor(name, args, debug=False)
                except Exception as exc:
                    return False, {'error': str(exc)}
            return False, {'error': f'No executor registered for {name}'}

        try:
            if fn.__name__ == 'mcp_based_tool':
                result = fn(name, args)
            else:
                result = fn(args)
            return True, result
        except Exception as exc:
            return False, {'error': str(exc)}

    def _execute_with_auto_background(
        self,
        name: str,
        args: Dict[str, Any],
        plugin: 'BackgroundCapable',
        threshold: float,
        permission_meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Any]:
        """Execute a tool with auto-background on timeout.

        Args:
            name: Tool name.
            args: Arguments dict.
            plugin: The BackgroundCapable plugin.
            threshold: Timeout threshold in seconds.
            permission_meta: Optional permission metadata to inject.

        Returns:
            Tuple of (success, result). If auto-backgrounded, result contains
            task handle info with auto_backgrounded=True.
        """
        pool = self._get_auto_background_pool()

        # Submit to thread pool
        future = pool.submit(self._execute_sync, name, args)

        try:
            # Wait up to threshold seconds
            ok, result = future.result(timeout=threshold)

            # Inject permission metadata if available
            if permission_meta and isinstance(result, dict):
                result['_permission'] = permission_meta

            return ok, result

        except FuturesTimeoutError:
            # Task exceeded threshold - convert to background
            try:
                handle = plugin.register_running_task(future, name, args)

                result = {
                    "auto_backgrounded": True,
                    "task_id": handle.task_id,
                    "plugin_name": handle.plugin_name,
                    "tool_name": handle.tool_name,
                    "threshold_seconds": threshold,
                    "message": f"Task exceeded {threshold}s threshold, continuing in background. "
                               f"Use task_id '{handle.task_id}' to check status."
                }

                # Inject permission metadata
                if permission_meta:
                    result['_permission'] = permission_meta

                # Record auto-background event
                if self._ledger:
                    self._ledger._record('auto-background', {
                        'tool': name,
                        'task_id': handle.task_id,
                        'threshold': threshold,
                    })

                return True, result

            except Exception as e:
                # Failed to register - wait for result and return normally
                try:
                    ok, result = future.result(timeout=300)  # Give it more time
                    if permission_meta and isinstance(result, dict):
                        result['_permission'] = permission_meta
                    return ok, result
                except Exception as inner_e:
                    return False, {'error': f'Auto-background failed: {e}, execution failed: {inner_e}'}

    def execute(self, name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        debug = False
        try:
            import os
            debug = os.environ.get('AI_TOOL_RUNNER_DEBUG', '').lower() in ('1', 'true', 'yes')
        except Exception:
            debug = False

        # Track permission metadata for injection into result
        permission_meta = None

        # Check permissions if a permission plugin is set
        # Note: askPermission tool itself is always allowed
        if self._permission_plugin is not None and name != 'askPermission':
            try:
                allowed, perm_info = self._permission_plugin.check_permission(
                    name, args, self._permission_context
                )
                # Build permission metadata for result injection
                permission_meta = {
                    'decision': 'allowed' if allowed else 'denied',
                    'reason': perm_info.get('reason', ''),
                    'method': perm_info.get('method', 'unknown'),
                }
                # Record permission check to ledger
                if self._ledger is not None:
                    self._ledger._record('permission-check', {
                        'tool': name,
                        'args': args,
                        'allowed': allowed,
                        'reason': perm_info.get('reason', ''),
                        'method': perm_info.get('method', 'unknown'),
                    })
                if not allowed:
                    if debug:
                        print(f"[ai_tool_runner] permission denied for {name}: {perm_info.get('reason', '')}")
                    return False, {'error': f"Permission denied: {perm_info.get('reason', '')}", '_permission': permission_meta}
                if debug:
                    print(f"[ai_tool_runner] permission granted for {name}: {perm_info.get('reason', '')}")
            except Exception as perm_exc:
                if debug:
                    print(f"[ai_tool_runner] permission check failed for {name}: {perm_exc}")
                # Record permission error to ledger
                if self._ledger is not None:
                    self._ledger._record('permission-error', {
                        'tool': name,
                        'args': args,
                        'error': str(perm_exc),
                    })
                # On permission check failure, deny by default for safety
                return False, {'error': f'Permission check failed: {perm_exc}'}

        # Check for auto-background capability
        if self._auto_background_enabled and self._registry:
            bg_plugin = self._get_plugin_for_tool(name)
            if bg_plugin is not None:
                try:
                    threshold = bg_plugin.get_auto_background_threshold(name)
                    if threshold is not None and threshold > 0:
                        if debug:
                            print(f"[ai_tool_runner] using auto-background for {name} "
                                  f"(threshold={threshold}s)")
                        return self._execute_with_auto_background(
                            name, args, bg_plugin, threshold, permission_meta
                        )
                except Exception as e:
                    if debug:
                        print(f"[ai_tool_runner] auto-background check failed for {name}: {e}")
                    # Fall through to normal execution

        fn = self._map.get(name)
        if not fn:
            if debug:
                print(f"[ai_tool_runner] execute: no executor registered for {name}, attempting generic execution")
            # Check if generic execution is allowed via env var
            if os.environ.get('AI_EXECUTE_TOOLS', '').lower() in ('1', 'true', 'yes'):
                try:
                    ok, res = _generic_executor(name, args, debug=debug)
                    # Inject permission metadata if available
                    if permission_meta and isinstance(res, dict):
                        res['_permission'] = permission_meta
                    return ok, res
                except Exception as exc:
                    if debug:
                        print(f"[ai_tool_runner] generic executor failed for {name}: {exc}")
                    return False, {'error': str(exc)}
            else:
                return False, {'error': f'No executor registered for {name}'}
        try:
            if debug:
                print(f"[ai_tool_runner] execute: invoking {name} with args={args}")
            if fn.__name__ == 'mcp_based_tool':
                result = fn(name, args)
            else:
                result = fn(args)
            # Inject permission metadata if available and result is a dict
            if permission_meta and isinstance(result, dict):
                result['_permission'] = permission_meta
            return True, result
        except Exception as exc:
            if debug:
                print(f"[ai_tool_runner] execute: {name} raised {exc}")
            return False, {'error': str(exc)}


def extract_text_from_parts(response) -> str:
    """Extract text from response parts without triggering warnings for function_call parts.

    This avoids the SDK warning when accessing .text on responses containing function calls.
    """
    text_parts = []
    try:
        for cand in getattr(response, 'candidates', []) or []:
            content = getattr(cand, 'content', None)
            if not content:
                continue
            for part in getattr(content, 'parts', []) or []:
                # Only extract text parts, skip function_call parts
                text = getattr(part, 'text', None)
                if text:
                    text_parts.append(text)
    except Exception:
        pass
    return ''.join(text_parts)


def extract_function_calls(response) -> List[Dict[str, Any]]:
    """Return list of function call infos extracted from a genai response object."""
    out: List[Dict[str, Any]] = []
    try:
        # New SDK provides response.function_calls convenience property
        func_calls = getattr(response, 'function_calls', None)
        if func_calls:
            for fc in func_calls:
                name = getattr(fc, 'name', None)
                # Args may be in function_call.args or directly on fc.args
                fc_inner = getattr(fc, 'function_call', None)
                if fc_inner:
                    args = dict(getattr(fc_inner, 'args', {}) or {})
                else:
                    args = dict(getattr(fc, 'args', {}) or {})
                out.append({'name': name, 'args': args})
            return out

        # Fallback: iterate through candidates/parts
        for cand in getattr(response, 'candidates', []) or []:
            content = getattr(cand, 'content', None)
            if not content:
                continue
            for p in getattr(content, 'parts', []) or []:
                fc = getattr(p, 'function_call', None)
                if fc:
                    out.append({'name': getattr(fc, 'name', None), 'args': dict(getattr(fc, 'args', {}) or {})})
    except Exception:
        pass
    return out


def extract_finish_reason(response) -> str:
    """Extract finish_reason from a genai response object.

    The finish_reason indicates why the model stopped generating. Possible values:
    - STOP: Normal completion
    - MAX_TOKENS: Hit token limit
    - SAFETY: Blocked by safety filters
    - RECITATION: Blocked for copyright reasons
    - MALFORMED_FUNCTION_CALL: Invalid function call generated
    - OTHER: Unknown/unspecified reason
    - BLOCKLIST, PROHIBITED_CONTENT, SPII, LANGUAGE: Various content filters

    Returns:
        The finish_reason as a string, or 'UNKNOWN' if not found.
    """
    try:
        candidates = getattr(response, 'candidates', None)
        if candidates and len(candidates) > 0:
            finish_reason = getattr(candidates[0], 'finish_reason', None)
            if finish_reason is not None:
                # Handle both enum and string representations
                if hasattr(finish_reason, 'name'):
                    return finish_reason.name
                return str(finish_reason)
    except Exception:
        pass
    return 'UNKNOWN'


def make_function_response_part(name: str, response_obj: Any) -> types.Part:
    """Create a Part object representing a function response to feed back to the model.

    Uses types.Part.from_function_response from the google-genai SDK.
    """
    try:
        return types.Part.from_function_response(name=name, response=response_obj)
    except Exception:
        # Fallback: embed JSON string into a text part
        return types.Part.from_text(text=json.dumps(response_obj, ensure_ascii=False))


def register_function_declarations(funcs: Iterable[Callable]) -> List[types.FunctionDeclaration]:
    """Register Python functions as FunctionDeclarations.

    Note: The new google-genai SDK can accept Python functions directly as tools,
    so this function may not be necessary in most cases.
    """
    decls: List[types.FunctionDeclaration] = []
    for f in funcs:
        try:
            # The new SDK can accept functions directly, but if we need declarations
            # we can create them manually or use the SDK's introspection
            decls.append(f)  # Pass function directly, SDK will handle it
        except Exception:
            pass
    return decls


def run_function_call_loop(
    client: 'genai.Client',
    model_name: str,
    initial_parts: List[types.Part],
    declared_tools: Optional[types.Tool] = None,
    executor: Optional[ToolExecutor] = None,
    ledger: Optional[TokenLedger] = None,
    max_turns: Optional[int] = None,
    trace: bool = False,
    system_instruction: Optional[str] = None,
    history: Optional[List[types.Content]] = None
) -> Dict[str, Any]:
    """Run iterative function-call loop with the model.

    Args:
        client: google.genai.Client instance.
        model_name: Model name (e.g., 'gemini-2.5-flash').
        initial_parts: list of types.Part objects forming the user's initial content.
        declared_tools: a types.Tool instance containing FunctionDeclaration(s) (or None).
        executor: a ToolExecutor instance mapping function names to callables.
        ledger: optional TokenLedger for recording events.
        max_turns: maximum iterations to allow.
        trace: if True, return 'trace' key containing per-turn data.
        system_instruction: optional system instruction to guide model behavior.
        history: optional existing conversation history (list of Content objects).
            If provided, the new user message is appended to this history.

    Returns a dict: {'text': final_text, 'function_results': [...], 'diagnostics': {...},
                     'trace': [...], 'history': updated_history}.
    """
    # Normalize initial_parts: accept either Part or Content objects
    user_parts: List[types.Part] = []
    for item in initial_parts:
        try:
            item_parts = getattr(item, 'parts', None)
            if item_parts:
                user_parts.extend(item_parts)
                continue
        except Exception:
            pass
        user_parts.append(item)

    # Initialize conversation history
    conversation: List[types.Content] = list(history) if history else []
    # Add the new user message to conversation
    user_content = types.Content(parts=user_parts, role='user')
    conversation.append(user_content)

    # Parts accumulator for function call responses within this turn
    current_turn_parts: List[types.Part] = list(user_parts)

    function_results: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {}
    trace_list: List[Dict[str, Any]] = []

    turn = 0
    while True:
        # Build contents for the request from full conversation history
        contents = conversation

        # Flatten prompt for token counting (use last user message)
        prompt_text = ''
        try:
            prompt_text = '\n'.join([getattr(p, 'text', str(p)) for p in current_turn_parts if hasattr(p, 'text')])
        except Exception:
            prompt_text = str(current_turn_parts)

        # Count tokens
        try:
            count_info = client.models.count_tokens(model=model_name, contents=prompt_text)
            if ledger is not None:
                ledger._record('prompt-count', {
                    'prompt_tokens': getattr(count_info, 'total_tokens', None)
                })
        except Exception:
            if ledger is not None:
                ledger._record('prompt-count-error', {'error': 'count_tokens failed'})

        # Build config with tools and system instruction
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[declared_tools] if declared_tools is not None else None
        )

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )

        # Extract text from parts directly to avoid SDK warning about function_call parts
        text_out = extract_text_from_parts(response)

        # Record output token usage
        usage = getattr(response, 'usage_metadata', None)
        if usage and ledger is not None:
            ledger._record('response', {
                'prompt_tokens': getattr(usage, 'prompt_token_count', None),
                'output_tokens': getattr(usage, 'candidates_token_count', None),
                'total_tokens': getattr(usage, 'total_token_count', None)
            })

        func_calls = extract_function_calls(response)
        if trace:
            trace_list.append({'turn': turn, 'text': text_out, 'function_calls': func_calls})

        # Add model response to conversation history
        try:
            model_content = response.candidates[0].content
            conversation.append(model_content)
        except (IndexError, AttributeError):
            # Fallback: create Content from extracted text
            if text_out:
                conversation.append(types.Content(parts=[types.Part.from_text(text=text_out)], role='model'))

        # If no function calls, finish
        if not func_calls:
            final_text = text_out if text_out else ''
            if trace:
                trace_list.append({'turn': turn, 'text': final_text, 'function_calls': []})
            return {'text': final_text, 'function_results': function_results, 'diagnostics': diagnostics, 'trace': trace_list, 'turns': turn + 1, 'history': conversation}

        # Execute calls
        for fc in func_calls:
            name = fc.get('name')
            args = fc.get('args') or {}

            ok, res = (False, {'error': 'no executor'})
            if executor is not None:
                ok, res = executor.execute(name, args)

            function_results.append({'name': name, 'args': args, 'ok': ok, 'result': res})

            if ledger is not None:
                try:
                    ledger._record('tool-call', {'function': name, 'args': args, 'result': res})
                except Exception:
                    pass

            # Record detailed result for CLI tools
            if ledger is not None:
                try:
                    if name == 'cli_call' and isinstance(res, dict):
                        ledger._record('tool-result', {
                            'function': name,
                            'ok': ok,
                            'stdout': res.get('stdout'),
                            'stderr': res.get('stderr'),
                            'returncode': res.get('returncode'),
                            'error': res.get('error') if 'error' in res else None
                        })
                    else:
                        ledger._record('tool-result', {'function': name, 'ok': ok})
                except Exception:
                    pass

            # Append function response part for next iteration
            try:
                func_part = make_function_response_part(name, res)
                current_turn_parts.append(func_part)
            except Exception:
                pass

        # Add function responses to conversation as user content
        if current_turn_parts:
            # Filter to only function response parts (not the original user text)
            func_response_parts = [p for p in current_turn_parts if hasattr(p, 'function_response')]
            if func_response_parts:
                conversation.append(types.Content(parts=func_response_parts, role='user'))
            # Reset for next turn
            current_turn_parts = []

        turn += 1
        if max_turns is not None and turn >= max_turns:
            # Use extract_text_from_parts to avoid SDK warning
            final_text = extract_text_from_parts(response)
            diagnostics['note'] = 'max_turns_reached'
            return {'text': final_text, 'function_results': function_results, 'diagnostics': diagnostics, 'trace': trace_list, 'turns': turn, 'history': conversation}

    return {'text': '', 'function_results': function_results, 'diagnostics': diagnostics, 'trace': trace_list, 'turns': turn, 'history': conversation}


def run_single_prompt(
    client: 'genai.Client',
    model_name: str,
    prompt: str,
    ledger_path: pathlib.Path,
    trace: bool = False,
    trace_dir: pathlib.Path | None = None,
    registry: Optional['PluginRegistry'] = None,
    permission_plugin: Optional['PermissionPlugin'] = None,
    permission_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run a single prompt with tools from the provided plugin registry.

    Args:
        client: google.genai.Client instance.
        model_name: Model name (e.g., 'gemini-2.5-flash').
        prompt: The prompt to send to the model.
        ledger_path: Path to write the token ledger.
        trace: If True, include trace data in the result.
        trace_dir: Directory for trace files.
        registry: PluginRegistry with already-enabled plugins. If None, no tools available.
        permission_plugin: Optional PermissionPlugin for access control.
        permission_config: Optional config for initializing permission plugin.

    Returns:
        Dict with 'text', 'summary', 'ledger' keys, and optionally error/diagnostic info.
    """
    ledger = TokenLedger()
    response = None
    use_fc = os.environ.get('AI_USE_CHAT_FUNCTIONS') not in (None, '', '0', 'false', 'False')

    try:
        if use_fc:
            executor = ToolExecutor(ledger=ledger)
            all_tool_decls = []

            # Load tools from the plugin registry
            if registry:
                for name, fn in registry.get_exposed_executors().items():
                    executor.register(name, fn)
                # Get ToolSchemas and convert to SDK declarations
                all_schemas = registry.get_exposed_tool_schemas()
                all_tool_decls = [tool_schema_to_sdk(s) for s in all_schemas]

            # Initialize permission plugin if provided
            perm_plugin = permission_plugin
            if perm_plugin is not None:
                try:
                    perm_plugin.initialize(permission_config)
                    executor.set_permission_plugin(perm_plugin)
                    # Register askPermission tool from permission plugin
                    for name, fn in perm_plugin.get_executors().items():
                        executor.register(name, fn)
                    perm_schemas = perm_plugin.get_tool_schemas()
                    all_tool_decls.extend([tool_schema_to_sdk(s) for s in perm_schemas])
                except Exception as perm_err:
                    # Log warning but continue without permission plugin
                    if ledger:
                        ledger._record('permission-init-error', {'error': str(perm_err)})

            tool_decl = types.Tool(function_declarations=all_tool_decls) if all_tool_decls else None

            # Collect system instructions from plugins
            system_instructions_parts = []
            if registry:
                registry_instructions = registry.get_system_instructions()
                if registry_instructions:
                    system_instructions_parts.append(registry_instructions)
            if perm_plugin:
                perm_instructions = perm_plugin.get_system_instructions()
                if perm_instructions:
                    system_instructions_parts.append(perm_instructions)
            system_instruction = "\n\n".join(system_instructions_parts) if system_instructions_parts else None

            fc_result = run_function_call_loop(
                client,
                model_name,
                [types.Part.from_text(text=prompt)],
                declared_tools=tool_decl,
                executor=executor,
                ledger=ledger,
                max_turns=None,
                trace=trace,
                system_instruction=system_instruction
            )
            response = type('R', (), {})()
            response.text = fc_result.get('text')
            response.candidates = []
            diagnostics_from_runner = fc_result
        else:
            response = ledger.generate_with_accounting(client, model_name, prompt)
    except Exception as exc:
        try:
            summary = ledger.summarize()
        except Exception:
            summary = {}
        try:
            ledger.write_ledger(str(ledger_path))
        except Exception:
            pass
        return {
            "text": None,
            "error": f"Generation failed: {exc}",
            "summary": summary,
            "ledger": str(ledger_path),
            "turns": 0,
        }

    text_val = None
    diagnostics: Dict[str, Any] = {}
    try:
        text_val = getattr(response, "text", None)
    except Exception as exc:
        diagnostics['text_error'] = str(exc)

    if not text_val:
        try:
            candidates = getattr(response, 'candidates', None)
            diagnostics['candidates'] = candidates
        except Exception:
            diagnostics['candidates'] = None
        try:
            diagnostics['usage_metadata'] = getattr(response, 'usage_metadata', None)
        except Exception:
            diagnostics['usage_metadata'] = None

    function_calls = extract_function_calls(response)
    if function_calls:
        diagnostics['function_calls'] = function_calls
        for fn_entry in function_calls:
            try:
                ledger._record('tool-call', {'function': fn_entry['name'], 'args': fn_entry['args']})
            except Exception:
                pass

    try:
        summary = ledger.summarize()
    except Exception:
        summary = {}
    try:
        ledger.write_ledger(str(ledger_path))
    except Exception:
        pass

    result: Dict[str, Any] = {
        "text": text_val,
        "summary": summary,
        "ledger": str(ledger_path),
    }
    # Include turns count from function call loop
    if 'diagnostics_from_runner' in locals() and isinstance(diagnostics_from_runner, dict):
        result['turns'] = diagnostics_from_runner.get('turns', 1)
    else:
        result['turns'] = 1  # Single turn for non-function-call mode
    if diagnostics:
        result['response_diagnostic'] = diagnostics

    if trace and trace_dir is not None:
        try:
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"{ledger_path.stem}.trace.json"
            trace_payload = {
                'prompt': prompt,
                'text': text_val,
                'diagnostic': diagnostics,
                'summary': summary,
            }
            if 'fc_result' in locals() and isinstance(fc_result, dict):
                for k in ['tool_result', 'function_result', 'result', 'output']:
                    if k in fc_result:
                        trace_payload['tool_result'] = fc_result[k]
                        break
                else:
                    trace_payload['fc_result'] = fc_result
            with open(trace_path, 'w', encoding='utf-8') as tf:
                json.dump(trace_payload, tf, ensure_ascii=False, indent=2)
        except Exception as e:
            import sys
            print(f"[Warning] Failed to write trace file: {e}", file=sys.stderr)
    return result


__all__ = [
    'ToolExecutor',
    'extract_text_from_parts',
    'extract_function_calls',
    'make_function_response_part',
    'register_function_declarations',
    'run_function_call_loop',
    'run_single_prompt',
]


def _generic_executor(name: str, args: Dict[str, Any], debug: bool = False) -> Tuple[bool, Any]:
    """Generic fallback executor: attempt to run a CLI command or MCP client based on name/args.

    - If `name` looks like a CLI tool (contains '-cli' or 'confluence'), shell out accordingly.
    - If `name` looks like an MCP client command, attempt to call a MCP client function (placeholder).
    This is intentionally conservative and returns structured errors when not possible.
    """
    # Heuristics for CLI tools
    lname = name.lower() if name else ''
    if 'confluence' in lname or 'confluence-cli' in lname or lname.endswith('_get'):
        # Expect args to include page id; try to construct a reasonable command
        page_id = args.get('page_id') or args.get('page') or args.get('id')
        if not page_id:
            return False, {'error': 'generic_executor: missing page id'}
        cmd = ['confluence-cli', 'get', '--page', str(page_id)]
        if debug:
            print(f"[ai_tool_runner] generic_executor running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            out = proc.stdout or proc.stderr or ''
            return True, {'raw': out}
        except Exception as exc:
            return False, {'error': str(exc)}

    # MCP client placeholder: look for 'mcp' prefix
    if lname.startswith('mcp') or lname.startswith('mcp_'):
        # Placeholder: if you have an MCP client library, call it here.
        return False, {'error': 'MCP client execution not implemented in generic executor'}

    return False, {'error': f'generic_executor: cannot handle function {name}'}
