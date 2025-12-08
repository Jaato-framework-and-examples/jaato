import os
import argparse
import json
import pathlib
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from shared import (
    normalize_ca_env_vars,
    PluginRegistry,
    JaatoClient,
    TokenLedger,
)

TEMPLATE_DIR = ROOT / "shared" / "prompt_templates"


def get_templates_for_domain(domain: str) -> Tuple[Dict[str, pathlib.Path], Dict[str, pathlib.Path]]:
    """Dynamically load templates for a domain based on directory convention.

    Structure: prompt_templates/{domain}/{tool_type}/{scenario}.txt
    """
    domain_dir = TEMPLATE_DIR / domain
    if not domain_dir.exists():
        raise SystemExit(f"Unknown domain: {domain}. No directory at {domain_dir}")

    cli_dir = domain_dir / "cli"
    mcp_dir = domain_dir / "mcp"

    cli_templates = {}
    mcp_templates = {}

    if cli_dir.exists():
        for f in cli_dir.glob("*.txt"):
            scenario = f.stem  # Filename without extension is the scenario
            cli_templates[scenario] = f

    if mcp_dir.exists():
        for f in mcp_dir.glob("*.txt"):
            scenario = f.stem
            mcp_templates[scenario] = f

    return cli_templates, mcp_templates


def load_template(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def substitute(template: str, mapping: Dict[str, str]) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace(f"{{{{{k}}}}}", v)
    return out


def build_prompt(scenario: str, tool_type: str, params: Dict[str, Any], domain: str,
                 cli_templates: Dict[str, pathlib.Path], mcp_templates: Dict[str, pathlib.Path]) -> str:
    """Build a prompt from a template and domain-specific parameters.

    Args:
        scenario: The scenario name (e.g., 'list_issues', 'get_page').
        tool_type: Either 'cli' or 'mcp'.
        params: Domain-specific parameters dict from --domain-params JSON.
        domain: The domain name (e.g., 'github', 'confluence').
        cli_templates: Dict mapping scenario names to CLI template paths.
        mcp_templates: Dict mapping scenario names to MCP template paths.

    Returns:
        The prompt string with placeholders substituted.
    """
    if tool_type == "cli":
        template_path = cli_templates[scenario]
    else:
        template_path = mcp_templates[scenario]
    template = load_template(template_path)

    mapping: Dict[str, str] = {}

    if domain == "confluence":
        # Confluence-specific mappings
        if scenario == "get_page":
            mapping["PAGE_ID"] = params.get("page_id") or "UNKNOWN"
        elif scenario == "search":
            mapping["CQL_QUERY"] = params.get("cql_query") or "type=page"
            mapping["LIMIT"] = str(params.get("limit", 10))
            mapping["TOP_N"] = str(params.get("top_n", 5))
        elif scenario == "update_page":
            mapping["PAGE_ID"] = params.get("page_id") or "UNKNOWN"
            mapping["CURRENT_TITLE"] = params.get("current_title") or "Untitled"
            body_file = params.get("current_body_file")
            if body_file:
                body_text = pathlib.Path(body_file).read_text(encoding="utf-8")
            else:
                body_text = "(No body provided)"
            mapping["CURRENT_BODY"] = body_text
            mapping["CHANGE_REQUEST"] = params.get("change_request") or "Revise wording for clarity"
        elif scenario == "list_children":
            mapping["PARENT_PAGE_ID"] = params.get("parent_page_id") or params.get("page_id") or "UNKNOWN"
            mapping["LIMIT"] = str(params.get("limit", 10))
            mapping["NOW_UTC"] = params.get("now_utc") or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    elif domain == "github":
        # GitHub-specific mappings
        mapping["OWNER"] = params.get("owner") or "UNKNOWN"
        mapping["REPO"] = params.get("repo") or "UNKNOWN"
        if scenario == "get_issue":
            mapping["ISSUE_NUMBER"] = str(params.get("issue_number", 1))
        elif scenario == "search_issues":
            mapping["SEARCH_QUERY"] = params.get("search_query") or "is:open"
            mapping["LIMIT"] = str(params.get("limit", 10))
            mapping["TOP_N"] = str(params.get("top_n", 5))
        elif scenario == "list_issues":
            mapping["LIMIT"] = str(params.get("limit", 10))
            mapping["NOW_UTC"] = params.get("now_utc") or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    return substitute(template, mapping)


def print_config(args, domain_params: Dict[str, Any], project_id: str, location: str,
                 model_name: str, scenarios: List[str], all_scenarios: List[str],
                 cli_extra_paths: List[str], submission_timestamp: str) -> None:
    """Print all configuration parameters at startup."""
    print("=" * 70)
    print("  CLI vs MCP Harness Configuration")
    print("=" * 70)
    print()
    print("  Command-line arguments:")
    print(f"    --domain:        {args.domain}")
    print(f"    --env-file:      {args.env_file}")
    print(f"    --model-name:    {args.model_name or '(from env)'}")
    print(f"    --scenarios:     {args.scenarios}")
    print(f"    --runs:          {args.runs}")
    print(f"    --output:        {args.output}")
    print(f"    --verbose:       {args.verbose}")
    print(f"    --trace:         {args.trace}")
    print(f"    --trace-dir:     {args.trace_dir}")
    print(f"    --domain-params: {args.domain_params}")
    print()
    print("  Resolved values:")
    print(f"    PROJECT_ID:      {project_id}")
    print(f"    LOCATION:        {location}")
    print(f"    MODEL_NAME:      {model_name}")
    print(f"    CLI extra paths: {cli_extra_paths or '(none)'}")
    print()
    print("  Domain parameters:")
    if domain_params:
        for k, v in domain_params.items():
            print(f"    {k}: {v}")
    else:
        print("    (none)")
    print()
    print("  Scenarios:")
    print(f"    Available: {all_scenarios}")
    print(f"    Selected:  {scenarios}")
    print()
    print("  Trace organization:")
    print(f"    Timestamp:  {submission_timestamp}")
    print(f"    Output dir: {args.trace_dir}/{submission_timestamp}/{args.domain}/")
    print(f"    Structure:  <output_dir>/<scenario>/<plugin>/run<N>.{{trace.json,jsonl}}")
    print()
    print("=" * 70)
    print()


def extract_function_calls(history) -> List[Dict[str, Any]]:
    """Extract function calls and responses from conversation history.

    Args:
        history: List of Content objects from jaato.get_history().

    Returns:
        List of dicts with function call/response details.
    """
    calls = []
    for content in history:
        role = getattr(content, 'role', 'unknown')
        for part in content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                calls.append({
                    'type': 'function_call',
                    'role': role,
                    'name': fc.name,
                    'args': dict(fc.args) if fc.args else {},
                })
            elif hasattr(part, 'function_response') and part.function_response:
                fr = part.function_response
                # Convert response to JSON-serializable format
                response = fr.response
                if hasattr(response, '__dict__'):
                    response = str(response)
                calls.append({
                    'type': 'function_response',
                    'role': role,
                    'name': fr.name,
                    'response': response,
                })
    return calls


def run_with_ledger(
    jaato: JaatoClient,
    registry: PluginRegistry,
    prompt: str,
    ledger_path: pathlib.Path,
    trace: bool = False,
    trace_dir: pathlib.Path = None
) -> Dict[str, Any]:
    """Run a single prompt with JaatoClient, handling ledger and results.

    Args:
        jaato: Configured JaatoClient instance.
        registry: Plugin registry (already has plugins exposed).
        prompt: The prompt to send.
        ledger_path: Path to write ledger JSONL.
        trace: Whether to write trace files.
        trace_dir: Directory for trace files.

    Returns:
        Dict with 'text', 'summary', 'ledger', 'turns' keys.
    """
    import json as _json

    ledger = TokenLedger()
    jaato.configure_tools(registry, ledger=ledger)

    try:
        response = jaato.send_message(prompt, on_output=lambda s, t, m: None)
        text = response
        error = None
    except Exception as exc:
        text = None
        error = str(exc)

    # Get history for function call tracing
    history = jaato.get_history()
    function_calls = extract_function_calls(history)

    # Get summary and write ledger
    try:
        summary = ledger.summarize()
    except Exception:
        summary = {}

    try:
        ledger.write_ledger(str(ledger_path))
    except Exception:
        pass

    result = {
        "text": text,
        "summary": summary,
        "ledger": str(ledger_path),
        "turns": summary.get('calls', 1),  # Approximate turns from call count
    }

    if error:
        result["error"] = error

    # Write trace if requested
    if trace and trace_dir is not None:
        try:
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"{ledger_path.stem}.trace.json"
            trace_payload = {
                'prompt': prompt,
                'text': text,
                'function_calls': function_calls,
                'summary': summary,
            }
            with open(trace_path, 'w', encoding='utf-8') as tf:
                _json.dump(trace_payload, tf, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return result


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs:
        return {"runs": 0, "avg_prompt_tokens": 0, "avg_output_tokens": 0, "avg_total_tokens": 0, "avg_turns": 0, "errors": 0}
    # Handle missing summary keys gracefully (can happen on generation errors)
    total_prompt = sum(r.get("summary", {}).get("total_prompt_tokens", 0) or 0 for r in runs)
    total_output = sum(r.get("summary", {}).get("total_output_tokens", 0) or 0 for r in runs)
    total = sum(r.get("summary", {}).get("total_tokens", 0) or 0 for r in runs)
    total_turns = sum(r.get("turns", 0) or 0 for r in runs)
    errors = sum(1 for r in runs if r.get("error"))
    return {
        "runs": len(runs),
        "avg_prompt_tokens": total_prompt / len(runs),
        "avg_output_tokens": total_output / len(runs),
        "avg_total_tokens": total / len(runs),
        "avg_turns": total_turns / len(runs),
        "errors": errors,
    }


def print_comparison_table(summary: Dict[str, Any]) -> None:
    """Print a formatted side-by-side comparison table of CLI vs MCP results."""
    meta = summary.get("meta", {})
    scenarios = summary.get("scenarios", {})

    # Header
    print("\n" + "=" * 70)
    print(f"  CLI vs MCP Token Comparison | Model: {meta.get('model', 'N/A')} | Domain: {meta.get('domain', 'N/A')}")
    print("=" * 70)

    if not scenarios:
        print("  No scenarios to display.")
        return

    # Table header
    print(f"\n  {'Metric':<20} {'CLI':>12} {'MCP':>12} {'Diff':>12} {'Winner':>10}")
    print("  " + "-" * 66)

    for scenario_name, data in scenarios.items():
        cli = data.get("cli", {})
        mcp = data.get("mcp", {})

        print(f"\n  [{scenario_name}] (runs: {cli.get('runs', 0)})")
        print("  " + "-" * 66)

        metrics = [
            ("Prompt Tokens", "avg_prompt_tokens"),
            ("Output Tokens", "avg_output_tokens"),
            ("Total Tokens", "avg_total_tokens"),
            ("Avg Turns", "avg_turns"),
        ]

        for label, key in metrics:
            cli_val = cli.get(key, 0)
            mcp_val = mcp.get(key, 0)
            diff = mcp_val - cli_val
            diff_pct = (diff / cli_val * 100) if cli_val > 0 else 0

            # Determine winner (lower is better for tokens and turns)
            if cli_val < mcp_val:
                winner = "CLI"
                diff_str = f"+{diff:.1f}" if key == "avg_turns" else f"+{diff:.0f}"
            elif mcp_val < cli_val:
                winner = "MCP"
                diff_str = f"{diff:.1f}" if key == "avg_turns" else f"{diff:.0f}"
            else:
                winner = "TIE"
                diff_str = "0"

            # Use 1 decimal place for turns, 0 for tokens
            if key == "avg_turns":
                print(f"  {label:<20} {cli_val:>12.1f} {mcp_val:>12.1f} {diff_str:>12} {winner:>10}")
            else:
                print(f"  {label:<20} {cli_val:>12.0f} {mcp_val:>12.0f} {diff_str:>12} {winner:>10}")

    # Summary footer
    print("\n" + "=" * 70)

    # Calculate totals across all scenarios
    total_cli_tokens = sum(s["cli"].get("avg_total_tokens", 0) for s in scenarios.values())
    total_mcp_tokens = sum(s["mcp"].get("avg_total_tokens", 0) for s in scenarios.values())

    if total_cli_tokens < total_mcp_tokens:
        overall_winner = "CLI"
        savings = total_mcp_tokens - total_cli_tokens
        savings_pct = (savings / total_mcp_tokens * 100) if total_mcp_tokens > 0 else 0
    elif total_mcp_tokens < total_cli_tokens:
        overall_winner = "MCP"
        savings = total_cli_tokens - total_mcp_tokens
        savings_pct = (savings / total_cli_tokens * 100) if total_cli_tokens > 0 else 0
    else:
        overall_winner = "TIE"
        savings = 0
        savings_pct = 0

    print(f"  Overall: {overall_winner} wins with {savings:.0f} fewer tokens ({savings_pct:.1f}% savings)")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Harness: Compare token usage for CLI vs MCP across scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GitHub domain
  python cli_mcp_harness.py --domain github --scenarios list_issues \\
    --domain-params '{"owner": "apanoia", "repo": "moon-rendering", "limit": 10}'

  # Confluence domain
  python cli_mcp_harness.py --domain confluence --scenarios get_page \\
    --domain-params '{"page_id": "12345"}'

Domain parameters (passed via --domain-params JSON):
  github:     owner, repo, issue_number, search_query, limit, top_n
  confluence: page_id, cql_query, current_title, current_body_file,
              change_request, parent_page_id, limit, top_n
"""
    )

    # Domain selection
    parser.add_argument("--domain", default="github", help="Tool domain: confluence or github")

    # Common arguments
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model-name", default=None, help="Override MODEL_NAME env")
    parser.add_argument("--scenarios", default="all", help="Comma separated scenario list or 'all'")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", default="cli_vs_mcp/harness_summary.json")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--trace", action="store_true", help="Write per-run trace files")
    parser.add_argument("--trace-dir", default="cli_vs_mcp/traces",
                        help="Base directory for traces (organized as: {trace-dir}/{timestamp}/{domain}/{scenario}/{plugin}/)")
    parser.add_argument("--cli-path", default=None, help="Extra path to CLI binary (e.g., path to gh or confluence-cli)")

    # Domain-specific parameters as JSON
    parser.add_argument("--domain-params", default="{}", help="Domain-specific parameters as JSON string")

    args = parser.parse_args()

    # Parse domain parameters
    try:
        domain_params = json.loads(args.domain_params)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid --domain-params JSON: {e}")

    load_dotenv(args.env_file)
    normalize_ca_env_vars()

    required = ["PROJECT_ID", "LOCATION", "MODEL_NAME", "GOOGLE_APPLICATION_CREDENTIALS"]
    missing = [v for v in required if not os.environ.get(v) and not (v == "MODEL_NAME" and args.model_name)]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")
    project_id = os.environ["PROJECT_ID"]
    location = os.environ["LOCATION"]
    model_name = args.model_name or os.environ["MODEL_NAME"]

    # Initialize JaatoClient for Vertex AI
    jaato = JaatoClient()
    jaato.connect(project_id, location, model_name)

    # Load templates for the selected domain
    domain = args.domain
    cli_templates, mcp_templates = get_templates_for_domain(domain)

    # CLI extra path from argument
    cli_extra_paths = [args.cli_path] if args.cli_path else []

    # Create and discover plugins (pass model_name for requirements checking)
    registry = PluginRegistry(model_name=model_name)
    registry.discover()
    if args.verbose:
        print(f"[Plugins] Available: {registry.list_available()}")

    # Determine available scenarios from templates
    all_scenarios = list(cli_templates.keys())

    if args.scenarios == "all":
        scenarios = all_scenarios
    else:
        scenarios = [s.strip() for s in args.scenarios.split(',') if s.strip()]
        for s in scenarios:
            if s not in all_scenarios:
                raise SystemExit(f"Unknown scenario for domain '{domain}': {s}. Available: {all_scenarios}")

    # Generate timestamp for hierarchical trace organization
    submission_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Print configuration summary
    print_config(args, domain_params, project_id, location, model_name,
                 scenarios, all_scenarios, cli_extra_paths, submission_timestamp)

    # List available models
    if args.verbose:
        try:
            available_models = jaato.list_available_models()
            print(f"  Available models: {', '.join(available_models)}")
            print()
        except Exception as e:
            print(f"  Could not list available models: {e}")
            print()

    # Base directories for traces and output
    trace_base_dir = pathlib.Path(args.trace_dir)
    out_dir = pathlib.Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "scenarios": {},
        "meta": {
            "runs": args.runs,
            "model": model_name,
            "domain": domain,
            "submission_timestamp": submission_timestamp
        }
    }

    for scenario in scenarios:
        if args.verbose:
            print(f"[Scenario] {scenario}")

        # Create hierarchical trace directories: {base}/{timestamp}/{domain}/{scenario}/{plugin}/
        scenario_base_dir = trace_base_dir / submission_timestamp / domain / scenario
        cli_trace_dir = scenario_base_dir / "cli"
        mcp_trace_dir = scenario_base_dir / "mcp"
        cli_trace_dir.mkdir(parents=True, exist_ok=True)
        mcp_trace_dir.mkdir(parents=True, exist_ok=True)

        if args.verbose:
            print(f"  CLI trace directory: {cli_trace_dir}")
            print(f"  MCP trace directory: {mcp_trace_dir}")

        cli_runs: List[Dict[str, Any]] = []
        mcp_runs: List[Dict[str, Any]] = []
        for run_index in range(1, args.runs + 1):
            # CLI run: enable only CLI plugin
            if args.verbose:
                print(f"  CLI run {run_index}/{args.runs}")
            registry.expose_tool('cli', config={'extra_paths': cli_extra_paths} if cli_extra_paths else None)
            cli_prompt = build_prompt(scenario, "cli", domain_params, domain, cli_templates, mcp_templates)
            cli_ledger = cli_trace_dir / f"run{run_index}.jsonl"
            cli_res = run_with_ledger(
                jaato, registry, cli_prompt, cli_ledger,
                trace=args.trace, trace_dir=cli_trace_dir
            )
            cli_runs.append(cli_res)
            registry.unexpose_tool('cli')

            time.sleep(0.3)

            # MCP run: enable only MCP plugin
            if args.verbose:
                print(f"  MCP run {run_index}/{args.runs}")
            registry.expose_tool('mcp')
            mcp_prompt = build_prompt(scenario, "mcp", domain_params, domain, cli_templates, mcp_templates)
            mcp_ledger = mcp_trace_dir / f"run{run_index}.jsonl"
            mcp_res = run_with_ledger(
                jaato, registry, mcp_prompt, mcp_ledger,
                trace=args.trace, trace_dir=mcp_trace_dir
            )
            mcp_runs.append(mcp_res)
            registry.unexpose_tool('mcp')
        summary["scenarios"][scenario] = {
            "cli": aggregate_runs(cli_runs),
            "mcp": aggregate_runs(mcp_runs),
            "cli_details": [{"ledger": r["ledger"], "summary": r["summary"], "turns": r.get("turns", 0)} for r in cli_runs],
            "mcp_details": [{"ledger": r["ledger"], "summary": r["summary"], "turns": r.get("turns", 0)} for r in mcp_runs],
        }

    # Write summary to hierarchical location: {trace_dir}/{timestamp}/{domain}/harness_summary.json
    summary_dir = trace_base_dir / submission_timestamp / domain
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "harness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Also write to the legacy output path if different
    legacy_output = pathlib.Path(args.output)
    if legacy_output.resolve() != summary_path.resolve():
        legacy_output.parent.mkdir(parents=True, exist_ok=True)
        legacy_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.verbose:
        print(f"\nSummary written to {summary_path}")
        if legacy_output.resolve() != summary_path.resolve():
            print(f"Also written to {legacy_output}")
        print_comparison_table(summary)


if __name__ == "__main__":
    main()
