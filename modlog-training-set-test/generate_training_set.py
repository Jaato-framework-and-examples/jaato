import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
from typing import List, Dict, Any
from shared.token_accounting import TokenLedger
from dotenv import load_dotenv
from shared.modlog_training_pipeline import (
    init_vertex,
    ai_parse_mod_history,
    identify_code_changes,
    load_cobol_source,
    prepare_history_region,
    write_raw_entries,
    validate_entry_schema,
    heuristic_changed_lines,
    make_training_pair,
    format_user_instruction,
)

"""Simple script:
1. Parse modification history from a COBOL source file.
2. Classify each modification (bug_fix, performance, security, enhancement, other).
3. Generate instruction/response training pairs via Gemini.
4. Write JSONL file ready for post-training / LoRA style fine-tuning.

Environment variables expected:
  GOOGLE_APPLICATION_CREDENTIALS  -> path to SA key
  PROJECT_ID                      -> GCP project
  LOCATION                        -> Vertex AI region (default global)
  MODEL_NAME                      -> Gemini model (default gemini-2.5-flash)

Run:
  python generate_training_set.py --source sample_cobol.cbl --out training_data.jsonl

Quiet mode:
  Set VERBOSE=0 to reduce console output.
"""

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate training JSONL from COBOL modification log')
    parser.add_argument('--source', required=True, help='Path to COBOL source file')
    parser.add_argument('--out', default='training_data.jsonl', help='Output JSONL file for training pairs')
    parser.add_argument('--entries-out', default=None, help='JSONL file to write raw parsed log entries (required for --parse-only)')
    parser.add_argument('--entries-in', default=None, help='JSONL file containing pre-parsed entries (skips parse stage; used with --explain-only)')
    parser.add_argument('--env-file', default='.env', help='Path to .env file with KEY=VALUE lines')
    parser.add_argument('--mode', choices=[
        'parse', 'full', 'full-stream', 'full-heuristic', 'full-append',
        'explain', 'explain-stream', 'explain-validate'
    ], help='Unified mode selector; overrides individual flags if set')
    parser.add_argument('--parse-only', action='store_true', help='Run only the history parse stage and write entries-out')
    parser.add_argument('--explain-only', action='store_true', help='Skip parse; read entries from --entries-in and generate pairs')
    parser.add_argument('--heuristic-lines', action='store_true', help='Add heuristic changed_items to metadata (non-stream, batch mode)')
    parser.add_argument('--stream', action='store_true', help='Stream writing of pairs as they are generated (ignored if --heuristic-lines)')
    parser.add_argument('--validate-entries', action='store_true', help='Validate loaded or parsed entries schema before explanation stage')
    parser.add_argument('--append', action='store_true', help='Append training pairs to existing --out file instead of overwriting')
    parser.add_argument('--max-history-chars', type=int, default=0, help='Max chars from history region passed to model (0 = no limit)')
    # AI parsing is now the default and only method.
    args = parser.parse_args()

    # If unified --mode is provided, map to legacy flags.
    if args.mode:
        # Warn if user also passed explicit flags (non-default)
        conflicting = []
        if args.parse_only: conflicting.append('--parse-only')
        if args.explain_only: conflicting.append('--explain-only')
        if args.heuristic_lines: conflicting.append('--heuristic-lines')
        if args.stream: conflicting.append('--stream')
        if args.validate_entries: conflicting.append('--validate-entries')
        if args.append: conflicting.append('--append')
        if conflicting:
            print('[warn] --mode overrides explicit flags:', ', '.join(conflicting))
        # Reset all flags
        args.parse_only = False
        args.explain_only = False
        args.heuristic_lines = False
        args.stream = False
        args.validate_entries = False
        args.append = False
        # Apply mapping
        if args.mode == 'parse':
            args.parse_only = True
        elif args.mode == 'full':
            pass  # default full pipeline
        elif args.mode == 'full-stream':
            args.stream = True
        elif args.mode == 'full-heuristic':
            args.heuristic_lines = True
        elif args.mode == 'full-append':
            args.append = True
        elif args.mode == 'explain':
            args.explain_only = True
        elif args.mode == 'explain-stream':
            args.explain_only = True
            args.stream = True
        elif args.mode == 'explain-validate':
            args.explain_only = True
            args.validate_entries = True
        # Mode-specific defaults for entries-out/in if omitted
        if args.parse_only and not args.entries_out:
            args.entries_out = 'modlog_entries.jsonl'
        if args.explain_only and not args.entries_in:
            args.entries_in = 'modlog_entries.jsonl'
        print(f'[mode] Applied mode "{args.mode}" -> parse_only={args.parse_only} explain_only={args.explain_only} stream={args.stream} heuristic={args.heuristic_lines} validate={args.validate_entries} append={args.append}')

    verbose = os.environ.get('VERBOSE', '1') not in ('0', 'false', 'False')

    load_dotenv(args.env_file)

    # Validate stage flags
    if args.parse_only and args.explain_only:
        raise SystemExit('Cannot use --parse-only and --explain-only together.')
    if args.explain_only and not args.entries_in:
        raise SystemExit('--explain-only requires --entries-in <file>.')
    if args.parse_only and not args.entries_out:
        raise SystemExit('--parse-only requires --entries-out <file>.')
    if args.heuristic_lines and args.stream:
        if verbose:
            print('[warn] --heuristic-lines overrides --stream (batch mode required).')
    if args.append and args.parse_only:
        raise SystemExit('--append has no effect with --parse-only.')
    if args.append and args.out == '-':
        raise SystemExit('--append requires a file path in --out, not stdin placeholder.')

    if verbose:
        print('[1/7] Initializing Vertex AI client...')
    client, model_name = init_vertex()
    ledger = TokenLedger()

    # Stage 1: parse or load entries
    if args.explain_only:
        if verbose:
            print('[2/7] Loading entries from file (skip parse stage)...')
        with open(args.entries_in, 'r', encoding='utf-8') as f:
            entries = [json.loads(line) for line in f if line.strip()]
        if verbose:
            print(f"[loaded] {len(entries)} entries from {args.entries_in}")
        # We still need source for heuristic or context
        lines = load_cobol_source(args.source)
        source_text = '\n'.join(lines)
    else:
        if verbose:
            print('[2/7] Loading COBOL source...')
        lines = load_cobol_source(args.source)
        source_text = '\n'.join(lines)
        history_region_text = prepare_history_region(lines, args.max_history_chars)
        if verbose:
            print('[3/7] Parsing modification history (model call)...')
        entries = ai_parse_mod_history(history_region_text, client, model_name, verbose, ledger)
        if not entries:
            print('No modification entries identified; aborting.')
            return
        if args.entries_out:
            write_raw_entries(args.entries_out, entries, verbose)
        if args.parse_only:
            if verbose:
                print('[parse-only] Skipping explanation stage.')
            ledger_path = ledger.write_ledger() or 'N/A'
            if verbose:
                summary = ledger.summarize()
                print(f"[ledger] Calls={summary['calls']} prompt={summary['total_prompt_tokens']} output={summary['total_output_tokens']} total={summary['total_tokens']} -> {ledger_path}")
                print(f'✅ Done. Parsed {len(entries)} entries.')
            else:
                print(f'Parsed {len(entries)} entries.')
            return

    # Stage 2: explanation generation (if not parse-only)
    if verbose:
        print('[4/7] Generating training pairs...')

    # Optional schema validation
    if args.validate_entries:
        if verbose:
            print('[validation] Checking entry schema...')
        problems = validate_entry_schema(entries)
        if problems:
            for p in problems[:50]:
                print('[validation][error]', p)
            raise SystemExit(f'Validation failed with {len(problems)} problems.')
        elif verbose:
            print(f'[validation] {len(entries)} entries passed schema checks.')

    # heuristic_changed_lines now provided by shared module

    pair_count = 0
    file_mode = 'a' if args.append and not args.heuristic_lines else 'w'
    if args.append and verbose and not args.heuristic_lines:
        print(f'[append] Writing pairs in append mode to {args.out}')

    if args.heuristic_lines:
        # Batch mode collecting all pairs to support heuristic lines
        records = []
        for entry in entries:
            change = identify_code_changes(entry, source_text, client, model_name, ledger, verbose)
            explanation = change.get('explanation') or 'No explanation returned.'
            changed_items = heuristic_changed_lines(entry, lines)
            if changed_items:
                excerpt = '\n'.join(f"{ci['line_number']:>5}: {ci['code_line']}" for ci in changed_items)
                explanation += '\n\nLikely affected lines (heuristic):\n' + excerpt
            records.append(make_training_pair(entry, explanation, changed_items, heuristic_used=True))
        with open(args.out, 'w', encoding='utf-8') as pf:
            for r in records:
                pf.write(json.dumps(r, ensure_ascii=False) + '\n')
        pair_count = len(records)
    else:
        # Streaming or batch without heuristics
        with open(args.out, file_mode, encoding='utf-8') as pf:
            for idx, entry in enumerate(entries, start=1):
                change = identify_code_changes(entry, source_text, client, model_name, ledger, verbose)
                explanation = change.get('explanation') or 'No explanation returned.'
                record = make_training_pair(entry, explanation, None, heuristic_used=False)
                pf.write(json.dumps(record, ensure_ascii=False) + '\n')
                pair_count += 1
                if args.stream and verbose and pair_count % 5 == 0:
                    print(f'  [progress] {pair_count} pairs written...')
    if verbose:
        print(f'[5/7] Generated {pair_count} pairs -> {args.out}')
    ledger_path = ledger.write_ledger() or 'N/A'
    if verbose:
        summary = ledger.summarize()
        print(f"[ledger] Calls={summary['calls']} prompt={summary['total_prompt_tokens']} output={summary['total_output_tokens']} total={summary['total_tokens']} -> {ledger_path}")

    if verbose:
        final_msg = f"✅ Done. Wrote {pair_count} examples to {args.out}"
        if args.entries_out and not args.explain_only:
            final_msg += f"; entries written to {args.entries_out}"
        if args.explain_only:
            final_msg += f" (entries loaded from {args.entries_in})"
        if args.heuristic_lines:
            final_msg += " (heuristic line mapping enabled)"
        print(f'[6/7] {final_msg}')
    else:
        base_msg = f'Wrote {pair_count} examples.'
        if args.entries_out and not args.explain_only:
            base_msg += f' Entries: {len(entries)}'
        print(base_msg)
    if verbose:
        print('[7/7] Complete.')


if __name__ == '__main__':
    main()
