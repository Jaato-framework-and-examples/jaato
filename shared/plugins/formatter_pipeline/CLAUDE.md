# CLAUDE.md — formatter_pipeline

## Buffering Models

Formatters use two distinct buffering strategies:

### Line-by-line (table_formatter)
Classifies each line individually as table/non-table. Requires `_line_buffer` for incomplete lines (no trailing `\n`), because without it lines are never classified and the table is never detected as complete.

### Buffer-level regex (code_block_formatter, mermaid_formatter)
Searches the entire accumulated buffer for the closing pattern `\n\`\`\``. The closing fence is detected as soon as all three backticks arrive, regardless of trailing newline. No `_line_buffer` needed.

## Turn Feedback

Formatters may implement the optional `get_turn_feedback() -> Optional[str]` method. At turn-end, after `flush()`, the pipeline's `collect_turn_feedback()` iterates all formatters, collects non-None feedback, and stores it as `_pending_feedback`.

The server retrieves this via `get_pending_feedback()` (which clears it) and stores it on the session. On the next `send_message()`, the feedback is prepended as a `<hidden>` block so the model can self-correct.

`get_turn_feedback()` is one-shot: implementations should return and clear their stored feedback.

## Optional Methods

Formatters may implement these without being part of the formal Protocol:

| Method | Purpose | Called by |
|--------|---------|-----------|
| `wire_dependencies(tool_registry)` | Inject tool plugin deps | FormatterRegistry |
| `get_system_instructions()` | Model instructions | `pipeline.get_system_instructions()` |
| `get_turn_feedback()` | Error feedback for model | `pipeline.collect_turn_feedback()` |
