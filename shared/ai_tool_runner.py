
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING
import subprocess
import os
import pathlib
from google import genai
from google.genai import types
from shared.token_accounting import TokenLedger

if TYPE_CHECKING:
    from shared.plugins.registry import PluginRegistry
    from shared.plugins.permission import PermissionPlugin


class ToolExecutor:
    """Registry mapping tool names to callables.

    Executors should accept a single dict-like argument and return a JSON-serializable result.

    Supports optional permission checking via a PermissionPlugin. When a permission
    plugin is set, all tool executions are checked against the permission policy
    before execution.
    """
    def __init__(self, ledger: Optional[TokenLedger] = None):
        self._map: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._permission_plugin: Optional['PermissionPlugin'] = None
        self._permission_context: Dict[str, Any] = {}
        self._ledger: Optional[TokenLedger] = ledger

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

    def execute(self, name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        debug = False
        try:
            import os
            debug = os.environ.get('AI_TOOL_RUNNER_DEBUG', '').lower() in ('1', 'true', 'yes')
        except Exception:
            debug = False

        # Check permissions if a permission plugin is set
        # Note: askPermission tool itself is always allowed
        if self._permission_plugin is not None and name != 'askPermission':
            try:
                allowed, reason = self._permission_plugin.check_permission(
                    name, args, self._permission_context
                )
                # Record permission check to ledger
                if self._ledger is not None:
                    self._ledger._record('permission-check', {
                        'tool': name,
                        'args': args,
                        'allowed': allowed,
                        'reason': reason,
                    })
                if not allowed:
                    if debug:
                        print(f"[ai_tool_runner] permission denied for {name}: {reason}")
                    return False, {'error': f'Permission denied: {reason}'}
                if debug:
                    print(f"[ai_tool_runner] permission granted for {name}: {reason}")
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

        fn = self._map.get(name)
        if not fn:
            if debug:
                print(f"[ai_tool_runner] execute: no executor registered for {name}, attempting generic execution")
            # Check if generic execution is allowed via env var
            if os.environ.get('AI_EXECUTE_TOOLS', '').lower() in ('1', 'true', 'yes'):
                try:
                    ok, res = _generic_executor(name, args, debug=debug)
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
            # sig = inspect.signature(fn)
            # # If function expects exactly one parameter, pass the dict directly (legacy style)
            # if len(sig.parameters) == 1:
            #     return True, fn(args)
            # # Special case: MCP tools should always receive a single dict argument
            # if name == 'mcp_based_tool':
            #     return True, fn(args)
            # # Otherwise attempt to expand dict into keyword arguments
            # return True, fn(**args)
            if fn.__name__ == 'mcp_based_tool':
                return True, fn(name, args)
            else:
                return True, fn(args)
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
    trace: bool = False
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

    Returns a dict: {'text': final_text, 'function_results': [...], 'diagnostics': {...}, 'trace': [...]}.
    """
    # Normalize initial_parts: accept either Part or Content objects
    parts_accum: List[types.Part] = []
    for item in initial_parts:
        try:
            item_parts = getattr(item, 'parts', None)
            if item_parts:
                parts_accum.extend(item_parts)
                continue
        except Exception:
            pass
        parts_accum.append(item)

    function_results: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {}
    trace_list: List[Dict[str, Any]] = []

    turn = 0
    while True:
        # Build contents for the request
        contents = [types.Content(parts=parts_accum, role='user')]

        # Flatten prompt for token counting
        prompt_text = ''
        try:
            prompt_text = '\n'.join([getattr(p, 'text', str(p)) for p in parts_accum if hasattr(p, 'text')])
        except Exception:
            prompt_text = str(parts_accum)

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

        # Build config with tools
        config = types.GenerateContentConfig(
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

        # If no function calls, finish
        if not func_calls:
            final_text = text_out if text_out else ''
            if trace:
                trace_list.append({'turn': turn, 'text': final_text, 'function_calls': []})
            return {'text': final_text, 'function_results': function_results, 'diagnostics': diagnostics, 'trace': trace_list, 'turns': turn + 1}

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
                parts_accum.append(func_part)
            except Exception:
                pass

        turn += 1
        if max_turns is not None and turn >= max_turns:
            # Use extract_text_from_parts to avoid SDK warning
            final_text = extract_text_from_parts(response)
            diagnostics['note'] = 'max_turns_reached'
            return {'text': final_text, 'function_results': function_results, 'diagnostics': diagnostics, 'trace': trace_list, 'turns': turn}

    return {'text': '', 'function_results': function_results, 'diagnostics': diagnostics, 'trace': trace_list, 'turns': turn}


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
                for name, fn in registry.get_enabled_executors().items():
                    executor.register(name, fn)
                all_tool_decls = registry.get_enabled_declarations()

            # Initialize permission plugin if provided
            perm_plugin = permission_plugin
            if perm_plugin is not None:
                try:
                    perm_plugin.initialize(permission_config)
                    executor.set_permission_plugin(perm_plugin)
                    # Register askPermission tool from permission plugin
                    for name, fn in perm_plugin.get_executors().items():
                        executor.register(name, fn)
                    all_tool_decls.extend(perm_plugin.get_function_declarations())
                except Exception as perm_err:
                    # Log warning but continue without permission plugin
                    if ledger:
                        ledger._record('permission-init-error', {'error': str(perm_err)})

            tool_decl = types.Tool(function_declarations=all_tool_decls) if all_tool_decls else None
            fc_result = run_function_call_loop(
                client,
                model_name,
                [types.Part.from_text(text=prompt)],
                declared_tools=tool_decl,
                executor=executor,
                ledger=ledger,
                max_turns=None,
                trace=trace
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
        except Exception:
            pass
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
