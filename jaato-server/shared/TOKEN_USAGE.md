# Vertex AI Token Usage & Accounting Guide

## 1. Field Definitions
| Field | Source | Meaning |
|-------|--------|---------|
| `prompt_tokens` | `response.usage_metadata.prompt_token_count` | Tokens consumed from your provided prompt after Vertex AI normalizes/serializes it (includes system-added formatting around your text). |
| `output_tokens` | `response.usage_metadata.candidates_token_count` | Tokens in the returned (surfaced) candidate(s). Some SDK versions report only the primary candidate here. |
| `total_tokens` | `response.usage_metadata.total_token_count` | Billable total: prompt + all generated candidate tokens + internal/system/safety/reasoning tokens that Vertex AI accounts for. Often > `prompt_tokens + output_tokens`. |
| `pre-count total_tokens` | `model.count_tokens(prompt).total_tokens` | Estimated tokens for the prompt alone prior to generation (planning/budgeting). |

## 2. Why `total_tokens` Can Be Larger Than Prompt + Output
Difference = `total_tokens - (prompt_tokens + output_tokens)` captures *unexposed* or *additional* token consumption. Common sources:
1. Multiple internal candidates (reranking or safety filtering) not all surfaced.
2. Safety / moderation / grounding passes that tokenize context and generated text.
3. System / hidden instructions prepended internally to guide the model.
4. Internal reasoning or planning steps (chain-of-thought style expansions not returned).
5. Truncation: you receive only part of a longer generated candidate; full length counted in billing.
6. SDK semantics: `candidates_token_count` may represent only the first candidate or accepted tokens rather than all generated sequences.

Example: Prompt=22, Output=1745, Total=3306 ⇒ Extra=1539 (≈46.5%). Those 1539 tokens are attributable to one or more sources above.

## 3. Validation Steps to Diagnose the Gap
Use these techniques to understand which factor dominates:
- **Candidate Inspection**:
  ```python
  print(len(response.candidates))
  for i, c in enumerate(response.candidates):
      print(f"Candidate {i} text snippet:", c.content.parts[0].text[:300])
  ```
- **Request More Candidates Explicitly**:
  ```python
  response = model.generate_content(
      PROMPT,
      generation_config={"candidate_count": 3}
  )
  ```
  Compare combined visible lengths vs `total_token_count`.
- **Minimal Prompt Test**: Use a very short prompt ("Hi") and check ratio; large proportional overhead suggests fixed system/safety tokens.
- **Pre-count vs Prompt**: If `count_tokens(prompt)` ≈ `prompt_token_count` consistently, extra tokens arise post-prompt.
- **Raw Metadata Dump**:
  ```python
  print(response.usage_metadata.__dict__)
  ```
  (Adjust if object structure differs.)
- **REST API Cross-Check**: Call Vertex AI REST endpoint directly; inspect the full JSON for any additional usage fields not surfaced by the SDK.

## 4. Instrumentation Enhancements
### 4.1 Current Script Diagnostics (Implemented)
The active `simple-connectivity-test.py` now performs:
- Candidate count logging (`len(response.candidates)`).
- 500‑character preview per candidate (joins all textual parts).
- Raw `usage_metadata` structure dump (via `__dict__` fallback to attribute enumeration).
- Pre-count estimation (`model.count_tokens`) versus actual prompt tokens.

### 4.2 Internal / Unattributed Token Computation
You can compute hidden/internal tokens per call:
```python
internal_tokens = usage.total_token_count - (usage.prompt_token_count + usage.candidates_token_count)
internal_pct = round(internal_tokens * 100 / usage.total_token_count, 2) if usage.total_token_count else 0.0
print("Internal tokens:", internal_tokens, "(", internal_pct, "%)")
```
If `internal_pct` is persistently high (>50%), inspect candidate generation count, moderation filters, or reasoning phases.

### 4.3 Persisting Diagnostics
Append detailed events to a ledger for later anomaly review:
```python
event = {
    "prompt_tokens": usage.prompt_token_count,
    "output_tokens": usage.candidates_token_count,
    "total_tokens": usage.total_token_count,
    "internal_tokens": internal_tokens,
    "internal_pct": internal_pct,
    "candidates": len(getattr(response, "candidates", []) or []),
}
with open("token_diagnostics.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(event) + "\n")
```

### 4.4 Multi-Candidate Reconciliation
When requesting multiple candidates (`generation_config={"candidate_count": N}`), approximate visible tokens per candidate and compare:
```python
for i, cand in enumerate(response.candidates):
    text_parts = [p.text for p in getattr(cand, "content", {}).parts if hasattr(p, "text")]
    approx_visible = sum(len(tp.split()) for tp in text_parts)
    print(f"Candidate {i} approx visible tokens: {approx_visible}")
```
If `total_token_count` >> sum of all visible candidate approximations + prompt, internal processes are dominant.
Add richer logging and analysis to your script:
```python
# Rough heuristic tokenizer (NOT exact Gemini tokenizer)
import re

def rough_tokenize(text: str) -> int:
    return len(re.findall(r"\w+|[\.!?,;]", text))

visible_tokens = rough_tokenize(response.text)
print("Approx visible tokens:", visible_tokens)
print("Prompt tokens:", usage.prompt_token_count)
print("Output tokens (reported):", usage.candidates_token_count)
print("Total tokens (reported):", usage.total_token_count)
print("Unattributed (internal) tokens:", usage.total_token_count - (usage.prompt_token_count + usage.candidates_token_count))
```
Persist events for historical tracking:
```python
import json, time
with open("token_events_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({"ts": time.time(), "usage": {
        "prompt": usage.prompt_token_count,
        "output": usage.candidates_token_count,
        "total": usage.total_token_count,
        "internal": usage.total_token_count - (usage.prompt_token_count + usage.candidates_token_count)
    }}) + "\n")
```

## 5. Cost Estimation Patterns
Use published Gemini pricing (replace placeholders):
```python
PROMPT_RATE = 0.000125   # USD per prompt token (example placeholder)
OUTPUT_RATE = 0.000375   # USD per output token (example placeholder)

prompt_cost = usage.prompt_token_count * PROMPT_RATE
output_cost = usage.candidates_token_count * OUTPUT_RATE
reported_total_cost = usage.total_token_count * (PROMPT_RATE + OUTPUT_RATE) / 2  # If blended rate scenario
```
If internal tokens are billed at same rate as output tokens, adjust formula accordingly:
```python
internal_tokens = usage.total_token_count - (usage.prompt_token_count + usage.candidates_token_count)
internal_cost = internal_tokens * OUTPUT_RATE  # assumption
```
Always verify current pricing tiers; some models have separate context vs generation rates, or a unified rate.

## 6. Recommended Practices
- **Budget with `total_token_count`**, not just visible output.
- **Track internal overhead percentage**: `internal_pct = internal_tokens / total_token_count`.
- **Optimize prompts**: Slim redundant instructions; internal overhead will still exist but prompt size reductions can cascade.
- **Batch related queries** if model supports multi-turn context reuse (reduces repeated system overhead).
- **Alerting**: Raise a warning if internal overhead > threshold (e.g., 50%).

## 7. Example Overhead Summary Function
```python
def overhead_summary(usage):
    prompt = usage.prompt_token_count
    output = usage.candidates_token_count
    total = usage.total_token_count
    internal = total - (prompt + output)
    return {
        "prompt": prompt,
        "output": output,
        "total": total,
        "internal": internal,
        "internal_pct": round(internal * 100 / total, 2) if total else 0.0
    }
```

## 8. Common Misinterpretations
| Misinterpretation | Correction |
|-------------------|------------|
| `candidates_token_count` includes all generated content | May only include surfaced/primary candidate. |
| `total_token_count` = prompt + output always | Can include hidden/system/reasoning tokens. |
| Internal tokens are errors | They are normal for moderation, reasoning, or multi-candidate generation. |
| Large overhead means wasted cost | It may reflect safety & quality processes; reduce only if overhead consistently high without benefit. |

## 9. Next Enhancements
Potential improvements you can implement next:
- CSV/JSON full session ledger (prompt, output, total, internal, latency).
- Rolling averages & trend analysis (moving window) for internal overhead.
- Automatic per-request cost estimation & budget alerts.
- Multi-candidate reconciliation: sum token lengths of all returned candidates vs reported totals.

## 10. Quick Checklist
- [ ] Log raw usage metadata.
- [ ] Store per-call token ledger.
- [ ] Compute internal overhead percentage.
- [ ] Estimate costs with current pricing.
- [ ] Periodically review anomalies (>60% internal overhead). 

---
Generated to document current understanding of Vertex AI Gemini token accounting and provide actionable instrumentation strategies. Updated with candidate & raw metadata diagnostics (Section 4) on: 2025-11-18.
