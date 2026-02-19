from typing import Dict, Any, List

_CURRENT_SOURCE_TEXT: str | None = None

def set_current_source(source_text: str) -> None:
    """Set the source text used by changed_lines_tool. Avoid passing large text through model args."""
    global _CURRENT_SOURCE_TEXT
    _CURRENT_SOURCE_TEXT = source_text or ''

def changed_lines_tool(description: str, delimiter: str | None) -> Dict[str, Any]:
    """Heuristic line selector based on description and optional delimiter.
    Uses globally stored source text set via set_current_source().
    Returns {'lines': [{line_number, code_line} ...]} limited to 12.
    """
    if _CURRENT_SOURCE_TEXT is None:
        return {'lines': []}
    merged = (description or '') + ' ' + (delimiter or '')
    raw_tokens: List[str] = []
    for part in merged.replace('/', ' ').replace(',', ' ').replace('.', ' ').split():
        cleaned = ''.join(ch for ch in part if ch.isalnum() or ch in ('_', '-')).strip()
        if cleaned:
            raw_tokens.append(cleaned)
    stop = {
        'the','and','for','with','from','this','that','into','will','have','been','being','were','shall','used','also','more','less','data','code','logic','process','change','changes','updated','adjusted','modify','modified','fix','fixed','performance','security','enhancement','other'
    }
    candidate_tokens: List[str] = []
    for t in raw_tokens:
        if t.lower() in stop:
            continue
        if len(t) >= 4 or ('-' in t) or ('_' in t):
            candidate_tokens.append(t.upper())
    seen = set()
    filtered_tokens: List[str] = []
    for t in candidate_tokens:
        if t not in seen:
            seen.add(t)
            filtered_tokens.append(t)
    source_lines = _CURRENT_SOURCE_TEXT.split('\n')
    matched: List[Dict[str, Any]] = []
    if filtered_tokens:
        for idx, line in enumerate(source_lines, start=1):
            uline = line.upper()
            score = sum(1 for tok in filtered_tokens if tok in uline)
            if score:
                matched.append({'line_number': idx, 'code_line': line, 'match_score': score})
    matched.sort(key=lambda d: (-d['match_score'], d['line_number']))
    final: List[Dict[str, Any]] = []
    used = set()
    for m in matched:
        ln = m['line_number']
        if ln in used:
            continue
        final.append({'line_number': ln, 'code_line': m['code_line']})
        used.add(ln)
        if len(final) >= 12:
            break
    return {'lines': final}

__all__ = ["changed_lines_tool", "set_current_source"]
