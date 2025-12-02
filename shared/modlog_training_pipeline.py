import os
import json
from typing import List, Dict, Any, TYPE_CHECKING
from google.genai import types
from google.oauth2 import service_account

from .jaato_client import JaatoClient
from .token_accounting import TokenLedger

if TYPE_CHECKING:
    from google import genai

# Prompt template loader
PROMPT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'prompt_templates')

def load_prompt_template(name: str) -> str:
    """Load a prompt template from the shared/prompt_templates directory by filename (without .txt)."""
    fname = os.path.join(PROMPT_TEMPLATE_DIR, f"{name}.txt")
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Prompt template not found: {fname}")
    with open(fname, 'r', encoding='utf-8') as f:
        return f.read()

# Reusable modification log training pipeline utilities.
# This module exposes pure functions so that multiple COBOL sources can be
# processed sequentially by an orchestrating script without spawning separate
# processes.
#
# Public entry points:
#   create_jaato_client(project, location, model_name) -> JaatoClient
#   load_cobol_source(path: str) -> List[str]
#   ai_parse_mod_history(history_region_text: str, jaato: JaatoClient, verbose: bool, ledger) -> List[Dict]
#   identify_code_changes(entry: Dict, full_source_text: str, jaato: JaatoClient, ledger, verbose: bool) -> Dict
#   build_training_pairs(jaato: JaatoClient, entries, source_lines, full_source_text, ledger, verbose) -> List[Dict]
#   write_jsonl(path: str, records: List[Dict]) -> None
# (Orchestrator scripts should call individual functions; consolidated process_source removed.)
#
# Each function requires a TokenLedger instance passed in (from shared.token_accounting)
# for consistent token accounting across a batch.

CHANGE_CLASS_MAP = {
    'bug': 'bug_fix',
    'abend': 'bug_fix',
    'fix': 'bug_fix',
    'performance': 'performance',
    'tuning': 'performance',
    'security': 'security',
    'mask': 'security',
    'encrypt': 'security',
    'enhancement': 'enhancement',
    'multi-currency': 'enhancement',
    'refactor': 'refactor',
}


def classify_description(desc: str) -> str:
    lower = desc.lower()
    for key, label in CHANGE_CLASS_MAP.items():
        if key in lower:
            return label
    return 'other'


def create_jaato_client(
    project: str | None = None,
    location: str = 'global',
    model_name: str = 'gemini-2.5-flash'
) -> JaatoClient:
    """Create and connect a JaatoClient for Vertex AI.

    Args:
        project: GCP project ID (defaults to PROJECT_ID env var).
        location: Vertex AI region (defaults to LOCATION env var or 'global').
        model_name: Model name (defaults to MODEL_NAME env var or 'gemini-2.5-flash').

    Returns:
        Connected JaatoClient instance.
    """
    project_id = project or os.environ.get('PROJECT_ID')
    loc = location or os.environ.get('LOCATION', 'global')
    model = os.environ.get('MODEL_NAME', model_name)
    if not project_id:
        raise RuntimeError('PROJECT_ID required (pass argument or set env var)')
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not creds_path:
        raise RuntimeError('GOOGLE_APPLICATION_CREDENTIALS required (env var)')
    if not os.path.isfile(creds_path):
        raise FileNotFoundError(f'Service account key file not found: {creds_path}')

    if os.environ.get('VERBOSE', '1') not in ('0', 'false', 'False'):
        creds = service_account.Credentials.from_service_account_file(creds_path)
        print(f"[auth] SA: {creds.service_account_email}")

    jaato = JaatoClient()
    jaato.connect(project_id, loc, model)
    return jaato


# Backwards compatibility alias
def init_vertex(project: str | None = None, location: str = 'global', model_name: str = 'gemini-2.5-flash'):
    """Deprecated: Use create_jaato_client() instead."""
    jaato = create_jaato_client(project, location, model_name)
    # Return tuple for backwards compatibility with existing callers
    return jaato, jaato.model_name


def load_cobol_source(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().splitlines()


def ai_parse_mod_history(history_region_text: str, jaato: JaatoClient, verbose: bool, ledger: TokenLedger) -> List[Dict]:
    """Parse modification history from COBOL source using AI.

    Args:
        history_region_text: The history region text from the COBOL source.
        jaato: Connected JaatoClient instance.
        verbose: Whether to print debug information.
        ledger: TokenLedger for accounting.

    Returns:
        List of parsed modification entries.
    """
    template = load_prompt_template("parse_mod_history_prompt")
    prompt = template.replace("{history_region_text}", history_region_text)
    raw = jaato.generate(prompt, ledger).strip()
    data: List[Dict] = []
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find('[')
        end = raw.rfind(']')
        if start != -1 and end != -1 and end > start:
            snippet = raw[start:end+1]
            try:
                data = json.loads(snippet)
            except Exception:
                if verbose:
                    print('[ai-parse] bracket heuristic failed')
        elif verbose:
            print('[ai-parse] no JSON brackets found')

    normalized: List[Dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        desc = item.get('description', '') or ''
        if not desc.strip():
            continue
        date_val = item.get('date') or ''
        prog = item.get('programmer') or None
        chosen_delim = item.get('delimiter') if isinstance(item.get('delimiter'), str) else None
        normalized.append({
            'date': str(date_val).strip(),
            'programmer': prog if prog else None,
            'description': desc.strip(),
            'change_type': classify_description(desc),
            'delimiter': chosen_delim,
        })
    # Return parsed & normalized entries
    return normalized


def identify_code_changes(entry: Dict, full_source_text: str, jaato: JaatoClient, ledger: TokenLedger, verbose: bool) -> Dict[str, Any]:
    """Identify concrete COBOL code changes for one history entry.

    If AI_USE_CHAT_FUNCTIONS is truthy, use function calling with the heuristic tool.
    Supports iterative multi-turn tool calls up to AI_FC_MAX_TURNS (default 2).

    Args:
        entry: The modification history entry.
        full_source_text: The full COBOL source code.
        jaato: Connected JaatoClient instance.
        ledger: TokenLedger for accounting.
        verbose: Whether to print debug information.

    Returns:
        Dict with 'explanation' and optional 'lines'.
    """
    use_fc = os.environ.get('AI_USE_CHAT_FUNCTIONS') not in (None, '', '0', 'false', 'False')

    if not use_fc:
        # Simple prompt path (no function calling)
        template = load_prompt_template("identify_code_changes_prompt")
        prompt = template
        placeholder_map = {
            "{entry['date']}": entry.get('date', ''),
            "{entry['programmer']}": entry.get('programmer', ''),
            "{entry['change_type']}": entry.get('change_type', ''),
            "{entry['description']}": entry.get('description', ''),
            "{full_source_text}": full_source_text,
        }
        for ph, val in placeholder_map.items():
            prompt = prompt.replace(ph, str(val))
        raw = jaato.generate(prompt, ledger).strip()
        return {"explanation": raw}

    # Function-calling path using JaatoClient with custom tools
    try:
        from .change_tools import changed_lines_tool, set_current_source
    except Exception as exc:
        if verbose:
            print(f"[identify_code_changes] tool import failed: {exc}; falling back.")
        return {"explanation": "Tool import failed; no function calling executed."}

    # Make full source available to tool without sending to model
    set_current_source(full_source_text)

    # Prepare tool declaration
    changed_decl = types.FunctionDeclaration.from_func(changed_lines_tool)

    # Configure JaatoClient with custom tool
    jaato.configure_custom_tools(
        declarations=[changed_decl],
        executors={
            'changed_lines_tool': lambda args: changed_lines_tool(
                description=entry.get('description', ''),
                delimiter=entry.get('delimiter')
            )
        },
        ledger=ledger
    )

    # Build prompt
    template = load_prompt_template("identify_code_changes_function_call_prompt")
    prompt = template
    placeholder_map = {
        "{entry['date']}": entry.get('date', ''),
        "{entry['programmer']}": entry.get('programmer', ''),
        "{entry['change_type']}": entry.get('change_type', ''),
        "{entry['description']}": entry.get('description', ''),
    }
    for ph, val in placeholder_map.items():
        prompt = prompt.replace(ph, str(val))

    # Send message with function calling
    response_text = jaato.send_message(prompt)

    final_text = (response_text or '').strip()
    return {"explanation": final_text if final_text else "No explanation text returned."}


def build_training_pairs(jaato: JaatoClient, entries: List[Dict], source_lines: List[str], full_source_text: str, ledger: TokenLedger, verbose: bool) -> List[Dict]:
    """Build training pairs from modification entries.

    Args:
        jaato: Connected JaatoClient instance.
        entries: List of modification history entries.
        source_lines: Source code as list of lines.
        full_source_text: Source code as full text.
        ledger: TokenLedger for accounting.
        verbose: Whether to print debug information.

    Returns:
        List of training pair dicts.
    """
    pairs = []
    for e in entries:
        change_analysis = identify_code_changes(e, full_source_text, jaato, ledger, verbose)
        explanation = change_analysis.get('explanation') or "No explanation returned."
        changed_items = heuristic_changed_lines(e, source_lines)
        # Append code excerpt section to assistant message if any lines found
        if changed_items:
            excerpt = '\n'.join(f"{ci['line_number']:>5}: {ci['code_line']}" for ci in changed_items)
            explanation = explanation + "\n\nLikely affected lines (heuristic):\n" + excerpt
        user_instruction = (
            f"Identify the concrete COBOL code changed by the logged modification ({e['change_type']}) from {e['date']} "
            f"by {e['programmer']}: {e['description']}"
        )
        pairs.append({
            'messages': [
                {'role': 'user', 'content': user_instruction},
                {'role': 'assistant', 'content': explanation},
            ],
            'metadata': {
                'date': e['date'],
                'programmer': e['programmer'],
                'change_type': e['change_type'],
                'description': e['description'],
                'delimiter': e.get('delimiter'),
                'changed_item_count': len(changed_items),
                'changed_items': changed_items,
                'heuristic': True,
            }
        })
    return pairs


def write_jsonl(path: str, records: List[Dict]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def prepare_history_region(source_lines: List[str], max_chars: int = 0) -> str:
    ws_index = next((i for i, l in enumerate(source_lines) if 'WORKING STORAGE SECTION' in l.upper()), None)
    history_region_lines = source_lines[:ws_index] if ws_index is not None else source_lines
    text = '\n'.join(history_region_lines)
    if max_chars and max_chars > 0:
        return text[:max_chars]
    return text

def write_raw_entries(path: str, entries: List[Dict], verbose: bool = False) -> None:
    if verbose:
        print(f"[early] Writing {len(entries)} raw entries to {path}")
    with open(path, 'w', encoding='utf-8') as ef:
        for e in entries:
            ef.write(json.dumps({
                'date': e.get('date'),
                'programmer': e.get('programmer'),
                'change_type': e.get('change_type'),
                'description': e.get('description'),
                'delimiter': e.get('delimiter'),
            }, ensure_ascii=False) + '\n')

def validate_entry_schema(entries: List[Dict]) -> List[str]:
    required = {'date','programmer','change_type','description','delimiter'}
    problems: List[str] = []
    for idx, e in enumerate(entries, start=1):
        if not isinstance(e, dict):
            problems.append(f'Entry {idx} not a dict')
            continue
        missing = required - set(e.keys())
        if missing:
            problems.append(f'Entry {idx} missing keys: {sorted(missing)}')
        desc = e.get('description')
        if not desc or not str(desc).strip():
            problems.append(f'Entry {idx} empty description')
        ct = e.get('change_type')
        if ct not in ('bug_fix','performance','security','enhancement','refactor','other'):
            problems.append(f'Entry {idx} invalid change_type: {ct}')
    return problems

def heuristic_changed_lines(entry: Dict[str, Any], source_lines: List[str], limit: int = 12) -> List[Dict[str, Any]]:
    desc = entry.get('description', '') or ''
    delim = entry.get('delimiter') or ''
    raw_tokens: List[str] = []
    for part in (desc + ' ' + delim).replace('/', ' ').replace(',', ' ').replace('.', ' ').split():
        cleaned = ''.join(ch for ch in part if ch.isalnum() or ch in ('_', '-')).strip()
        if cleaned:
            raw_tokens.append(cleaned)
    stop = {
        'the','and','for','with','from','this','that','into','will','have','been','being','were','shall','used','also','more','less','data','code','logic','process','change','changes','updated','adjusted','modify','modified','fix','fixed','performance','security','enhancement','other'
    }
    candidates: List[str] = []
    for t in raw_tokens:
        tl = t.lower()
        if tl in stop:
            continue
        if len(t) >= 4 or ('-' in t) or ('_' in t):
            candidates.append(t.upper())
    seen = set()
    filtered: List[str] = []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            filtered.append(t)
    matched: List[Dict[str, Any]] = []
    if not filtered:
        return matched
    for idx, line in enumerate(source_lines, start=1):
        uline = line.upper()
        score = sum(1 for token in filtered if token in uline)
        if score:
            matched.append({'line_number': idx, 'code_line': line, 'match_score': score})
    matched.sort(key=lambda d: (-d['match_score'], d['line_number']))
    final: List[Dict[str, Any]] = []
    used = set()
    for m in matched:
        if m['line_number'] in used:
            continue
        final.append({'line_number': m['line_number'], 'code_line': m['code_line']})
        used.add(m['line_number'])
        if len(final) >= limit:
            break
    return final

def format_user_instruction(entry: Dict[str, Any]) -> str:
    return (
        f"Identify the concrete COBOL code changed by the logged modification ({entry['change_type']}) from {entry['date']} "
        f"by {entry['programmer']}: {entry['description']}"
    )

def make_training_pair(entry: Dict[str, Any], explanation: str, changed_items: List[Dict[str, Any]] | None, heuristic_used: bool) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        'date': entry.get('date'),
        'programmer': entry.get('programmer'),
        'change_type': entry.get('change_type'),
        'description': entry.get('description'),
        'delimiter': entry.get('delimiter'),
        'changed_item_count': len(changed_items) if changed_items else 0,
        'heuristic': heuristic_used,
    }
    if changed_items:
        meta['changed_items'] = changed_items
    return {
        'messages': [
            {'role': 'user', 'content': format_user_instruction(entry)},
            {'role': 'assistant', 'content': explanation},
        ],
        'metadata': meta,
    }


__all__ = [
    'create_jaato_client',
    'init_vertex',  # Deprecated, use create_jaato_client
    'load_cobol_source',
    'ai_parse_mod_history',
    'identify_code_changes',
    'build_training_pairs',
    'write_jsonl',
    'prepare_history_region',
    'write_raw_entries',
    'validate_entry_schema',
    'heuristic_changed_lines',
    'format_user_instruction',
    'make_training_pair',
]
