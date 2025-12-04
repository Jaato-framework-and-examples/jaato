# GC Plugin Benchmark

Automated benchmark for comparing GC plugin efficiency, with focus on **context quality** - how well important information is retained after garbage collection.

## Approach: Fact Retention Testing

The benchmark embeds trackable "facts" (names, dates, numbers, decisions) in conversations, runs GC plugins, then tests if the model can still recall those facts from the post-GC context.

This provides an objective, measurable quality metric:
- **Retention Rate** = facts recalled / total facts
- Works consistently across all GC strategies
- Real LLM calls test actual summarization quality

## Quick Start

```bash
# Full benchmark (all plugins, all scenarios, quality testing)
.venv/bin/python gc-benchmark/run_benchmark.py --env-file .env

# Fast mode (tokens only, skip quality testing)
.venv/bin/python gc-benchmark/run_benchmark.py --no-quality

# Specific plugins only
.venv/bin/python gc-benchmark/run_benchmark.py --plugins gc_truncate,gc_hybrid
```

## How It Works

1. **Scenarios**: Pre-built conversations with embedded "facts" at various positions
2. **GC Execution**: Each plugin runs on each scenario (timed)
3. **Quality Testing**: LLM asked fact-recall questions using post-GC context
4. **Metrics**: Retention rate, tokens freed, compression ratio, timing

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--plugins` | all | Comma-separated list: `gc_truncate,gc_summarize,gc_hybrid` |
| `--scenarios` | all | Comma-separated list or `all` |
| `--output` | `gc_results.json` | JSON output path |
| `--threshold` | 80.0 | GC trigger threshold percentage |
| `--preserve-turns` | 5 | Number of recent turns to preserve |
| `--no-quality` | false | Skip fact retention testing (faster) |
| `--env-file` | `.env` | Environment file path |
| `--verbose` | false | Verbose output during benchmark |

## Scenarios

| Scenario | Turns | Facts | Description |
|----------|-------|-------|-------------|
| `short_conversation` | 10 | 5 | Basic test, few facts |
| `long_conversation` | 50 | 15 | Extended conversation, facts throughout |
| `tool_heavy` | 30 | 8 | Many function calls interspersed |
| `fact_dense` | 25 | 12 | High fact density in early turns (stress test) |

## Understanding Results

### Console Output

```
================================================================================
  GC Plugin Benchmark Results
================================================================================
  Model: gemini-2.5-flash
  Scenarios: 4
  Plugins: gc_truncate, gc_summarize, gc_hybrid
  Duration: 45.2s

--------------------------------------------------------------------------------
Scenario             Plugin          Tokens Freed   Retention   Time (ms)
--------------------------------------------------------------------------------
short_conversation   gc_truncate            1200        40%         2.1
short_conversation   gc_summarize            950        85%       312.5
short_conversation   gc_hybrid              1100        75%       298.3
...

--------------------------------------------------------------------------------
  Plugin Summary
--------------------------------------------------------------------------------
Plugin               Avg Tokens Freed  Avg Retention  Avg Time (ms)
--------------------------------------------------------------------------------
gc_truncate                     4850           32%            3.7
gc_summarize                    3575           82%          417.8
gc_hybrid                       4150           76%          382.4

================================================================================
  WINNER: gc_summarize (score: 0.84)
================================================================================
```

### Metrics Explained

- **Retention Rate**: Percentage of embedded facts the model can recall after GC
- **Tokens Freed**: Context space recovered by GC
- **Compression Ratio**: `tokens_after / tokens_before`
- **By Category**: Retention breakdown by fact type (entity, number, date, decision)
- **By Position**: Retention by where facts appeared (early, middle, late)

## Adding Custom Scenarios

Create a JSON file in `scenarios/`:

```json
{
  "name": "my_scenario",
  "description": "Custom test scenario",
  "turns": [
    {"role": "user", "text": "I'm working on Project Alpha with a budget of $50,000."},
    {"role": "model", "text": "I understand. How can I help with Project Alpha?"},
    ...
  ],
  "facts": [
    {
      "id": "project_name",
      "category": "entity",
      "turn_index": 0,
      "text": "Project Alpha",
      "question": "What is the project name?",
      "answer": "Project Alpha"
    },
    {
      "id": "budget",
      "category": "number",
      "turn_index": 0,
      "text": "$50,000",
      "question": "What is the budget?",
      "answer": "$50,000"
    }
  ]
}
```

### Fact Categories

- `entity`: Named people, places, organizations
- `number`: Budgets, counts, measurements
- `date`: Deadlines, start dates, events
- `decision`: Choices made in conversation

## Cost Considerations

LLM calls per full benchmark run:
- **Summarization**: 2 plugins × 4 scenarios = 8 calls
- **Quality testing**: 3 plugins × 4 scenarios × 10 facts = 120 calls
- **Total**: ~128 LLM calls

Use `--no-quality` for fast iteration (8 calls only).

## Environment Variables

Required in `.env`:
```
PROJECT_ID=your-gcp-project
LOCATION=us-central1
MODEL_NAME=gemini-2.5-flash  # optional, defaults to gemini-2.5-flash
```
