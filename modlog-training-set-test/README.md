# Modification Log Training Set Test

Minimal prototype that:
1. Parses a COBOL source file's modification history comment block.
2. Classifies each entry (bug_fix / performance / security / enhancement / refactor / other).
3. Calls a Vertex AI Gemini model to generate an explanatory response per entry.
4. Produces a JSONL file (`training_data.jsonl`) suitable for instruction-style fine‑tuning or LoRA.

## Files
- `sample_cobol.cbl` – Sample COBOL program with a synthetic modification history.
- `generate_training_set.py` – Main script.

## Requirements
Use (or extend) existing environment setup. Additional dependencies: `vertexai` (implied by google-cloud-aiplatform if already installed). If isolated, create a `requirements.txt` like:
```
google-cloud-aiplatform
vertexai
```

## Environment Variables
- `GOOGLE_APPLICATION_CREDENTIALS` – Path to service account key JSON.
- `PROJECT_ID` – GCP project.
- `LOCATION` – Vertex AI region (default `global`).
- `MODEL_NAME` – Gemini model (default `gemini-2.5-flash`).
- `VERBOSE` – Set to `0` for quiet output.
- `REQUESTS_CA_BUNDLE` (optional) – Path to a combined CA bundle (e.g. certifi + corporate root). If present, it is normalized and validated by `load_env_file`.
- `SSL_CERT_FILE` (optional) – Alternate env var honored by some libraries for custom CA bundle. If set it is also normalized/validated.

### .env File Fallback
If an environment variable is not set, the loader attempts to load KEY=VALUE pairs from a `.env` file. Resolution order:
1. Explicit path provided via `--env-file` (resolved relative to current working directory if not absolute).
2. If the path is not found and the value is exactly `.env`, it will look in the directory of the running script (script‑local `.env`).
3. If still not found a notice is printed and execution continues.

Lines beginning with `#` are ignored. Variables already present in the process environment are not overwritten.

Example `.env` file:
```
PROJECT_ID=your-project-id
LOCATION=global
MODEL_NAME=gemini-2.5-flash
GOOGLE_APPLICATION_CREDENTIALS=C:/path/to/service_account.json
REQUESTS_CA_BUNDLE=C:/path/to/combined_corp_cert_bundle.pem
```

Custom file path: `--env-file path/to/vars.env`

## Usage
```bash
# From project root (ensure venv active & creds set)
export PROJECT_ID=your-project
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
python modlog-training-set-test/generate_training_set.py \
  --source modlog-training-set-test/sample_cobol.cbl \
  --out modlog-training-set-test/training_data.jsonl \
  --env-file modlog-training-set-test/.env
```
Quiet mode:
```bash
VERBOSE=0 python modlog-training-set-test/generate_training_set.py --source modlog-training-set-test/sample_cobol.cbl
```

### Early Raw Entry Output & Streaming Pair Generation
The script now supports immediate writing of parsed modification log entries and streaming generation of training pairs (line‑by‑line) to avoid holding large datasets in memory.

Flag:
* `--entries-out <file>` – Writes raw normalized log entries **immediately after parsing** (before any per‑entry explanation calls). Each line contains: `date`, `programmer`, `change_type`, `description`, `delimiter`.

Streaming pairs: training examples are appended incrementally to the `--out` file as they are generated; a progress message appears every 5 entries in verbose mode.

Example using both outputs:
```bash
python modlog-training-set-test/generate_training_set.py \
  --source modlog-training-set-test/sample_cobol.cbl \
  --out modlog-training-set-test/training_data.jsonl \
  --entries-out modlog-training-set-test/modlog_entries.jsonl \
  --env-file modlog-training-set-test/.env
```

Result:
* `modlog_entries.jsonl` written early after history parse.
* `training_data.jsonl` populated progressively.

### Stage Control (Single Script)
The single `generate_training_set.py` script can now enable/disable pipeline stages:

| Flag | Description | Requirements |
|------|-------------|--------------|
| `--parse-only` | Parse history and write entries; skip explanation generation. | Must supply `--entries-out` |
| `--explain-only` | Skip parsing; load entries from `--entries-in` and generate pairs. | Must supply `--entries-in` |
| `--entries-out <file>` | Write raw entries immediately after parse. | Parse stage active |
| `--entries-in <file>` | Use pre-parsed entries JSONL instead of parsing. | `--explain-only` |
| `--stream` | Stream pair writing with periodic progress. | Not used with `--heuristic-lines` |
| `--heuristic-lines` | Add heuristic changed_items and affected line excerpt. | Batch mode (disables `--stream`) |
| `--validate-entries` | Validate entry schema (keys/change_type/description) before generating pairs. | Optional (parse or explain stage) |
| `--append` | Append pairs to existing output file instead of overwriting. | Explanation stage (ignored with `--heuristic-lines`) |
| `--mode <name>` | Unified shortcut selecting a preset combination of flags (see below). | Optional |

Examples:

Parse only (produce entries list):
```bash
python modlog-training-set-test/generate_training_set.py \
  --source modlog-training-set-test/sample_cobol.cbl \
  --parse-only \
  --entries-out modlog-training-set-test/modlog_entries.jsonl \
  --env-file modlog-training-set-test/.env
```

Explain only (reuse previously parsed entries):
```bash
python modlog-training-set-test/generate_training_set.py \
  --source modlog-training-set-test/sample_cobol.cbl \
  --explain-only \
  --entries-in modlog-training-set-test/modlog_entries.jsonl \
  --out modlog-training-set-test/training_data.jsonl \
  --env-file modlog-training-set-test/.env
```

Full pipeline with heuristic line mapping:
```bash
python modlog-training-set-test/generate_training_set.py \
  --source modlog-training-set-test/sample_cobol.cbl \
  --out modlog-training-set-test/training_data.jsonl \
  --entries-out modlog-training-set-test/modlog_entries.jsonl \
  --heuristic-lines \
  --env-file modlog-training-set-test/.env
```

Appending additional pairs (reuse pre-parsed entries from a prior run):
```bash
python modlog-training-set-test/generate_training_set.py \
  --source modlog-training-set-test/sample_cobol.cbl \
  --explain-only \
  --entries-in modlog-training-set-test/modlog_entries.jsonl \
  --out modlog-training-set-test/training_data.jsonl \
  --append \
  --stream \
  --env-file modlog-training-set-test/.env
```

Schema validation example (fail fast on malformed entries):
```bash
python modlog-training-set-test/generate_training_set.py \
  --source modlog-training-set-test/sample_cobol.cbl \
  --explain-only \
  --entries-in modlog-training-set-test/modlog_entries.jsonl \
  --validate-entries \
  --out modlog-training-set-test/training_data.jsonl \
  --env-file modlog-training-set-test/.env
```

Streaming (default) example already shown above; omit `--heuristic-lines` to keep memory usage minimal.

### Unified `--mode` Argument
To simplify flag combinations you can use one of the predefined modes:

| Mode | Equivalent Flags / Behavior |
|------|-----------------------------|
| `parse` | `--parse-only --entries-out modlog_entries.jsonl` |
| `full` | Full pipeline (parse + explain) |
| `full-stream` | Full pipeline with `--stream` |
| `full-heuristic` | Full pipeline with `--heuristic-lines` (disables streaming) |
| `full-append` | Full pipeline appending pairs (`--append`) |
| `explain` | `--explain-only --entries-in modlog_entries.jsonl` |
| `explain-stream` | `--explain-only --stream` |
| `explain-validate` | `--explain-only --validate-entries` |

If `--mode` is supplied it overrides any explicit stage flags passed. Missing `--entries-out` / `--entries-in` paths default to `modlog_entries.jsonl` when required by the chosen mode.

Example using mode:
```bash
python modlog-training-set-test/generate_training_set.py \
  --source modlog-training-set-test/sample_cobol.cbl \
  --out modlog-training-set-test/training_data.jsonl \
  --env-file modlog-training-set-test/.env \
  --mode full-stream
```

### VS Code Interactive Debug Configuration
An interactive debug configuration (`ModLog: Interactive Unified`) prompts for:

- Source COBOL file
- Output training data file
- Entries in/out files
- Max history characters
- Mode (pick list)
- .env file path

During launch VS Code shows input boxes or a dropdown (for `mode`). Press Enter to accept defaults. The configuration injects `--mode` and related arguments so you avoid manual flag typing for common workflows.

To add or edit the configuration see `.vscode/launch.json`. Remove older individual interactive configs if present; the unified one covers all scenarios.

## Output Format (JSONL)
Each line:
```json
{
  "messages": [
    {"role": "user", "content": "Explain the significance ..."},
    {"role": "assistant", "content": "<model explanation>"}
  ],
  "metadata": {
    "date": "2023-08-19",
    "programmer": "MLEE",
    "change_type": "bug_fix",
    "description": "Fixed S0C7 abend on numeric MOVE (bug 4172)"
  }
}

  ## Token Accounting
  This script now records token usage (prompt/output/total/internal) for:
  - AI modification history extraction
  - Each generated training example explanation

  Implementation: `TokenLedger` in `shared/token_accounting.py` shared across scripts.
  Ledger file: `token_events_ledger.jsonl` (override with `LEDGER_PATH`).

  Example summary line (verbose mode):
  ```
  [ledger] Calls=7 prompt=180 output=2450 total=4100 -> token_events_ledger.jsonl
  ```

  Disable verbose printing with `VERBOSE=0` (ledger still written).

  ## Environment Loading
  Environment variables are loaded from a `.env` file using `python-dotenv`:
  ```python
  from dotenv import load_dotenv
  load_dotenv('.env')
  ```
  For CA bundle handling, use `shared.ssl_helper`:
  ```python
  from shared.ssl_helper import active_cert_bundle
  bundle = active_cert_bundle(verbose=True)  # returns active CA bundle path if set
  ```

  ## Two-Stage Parsing & Change Identification
  Pipeline now consists of:
  1. History Extraction (LLM) limited to region above `WORKING STORAGE SECTION`.
  2. Per-Entry Change Identification (LLM) locating concrete COBOL code elements impacted.

  Each training record includes an assistant explanation. Optional heuristic line linkage (`--heuristic-lines`) enriches metadata with `changed_items` (top 12 inferred source lines) and appends an excerpt to the assistant content.

  Delimiter field:
  - History extraction returns a `delimiter` token when identifiable (e.g. `OVER-I`, `INCIKPDB-I/F`, `MACRO-I/E`).
  - Use this for grouping, filtering, or correlating with source line markers; stored in both early raw entry JSONL and training pair metadata.

  AI Retry Configuration:
  If you receive 429 / ResourceExhausted errors from Vertex AI, set in `.env`:
  ```
  AI_RETRY_ATTEMPTS=5
  AI_RETRY_BASE_DELAY=1.0
  AI_RETRY_MAX_DELAY=30.0
  AI_RETRY_LOG_SILENT=0
  ```
  Tweaking these controls exponential backoff for both history extraction and explanation generation stages.
```

## Extensibility Ideas
- Add fallback LLM parsing if regex misses entries.
- Generate additional variants (e.g., code completion tasks).
- Aggregate multiple COBOL files in a directory.
- Add cost & token accounting similar to previous test harness.

## Disclaimer
This is a minimal prototype for experimentation; does not include retries, advanced error handling, or dataset balancing.
