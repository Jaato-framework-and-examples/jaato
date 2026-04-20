# Project Backlog

Tracked backlog items. Each entry links to a design doc or implementation
plan. Promote to a feature branch / ticket when work is ready to start.

> Some older backlog items live in the jaato memory store as
> `project_backlog_*.md` (e.g. `project_backlog_fork_replay.md`,
> `project_backlog_conversation_fork.md`). New items are tracked here;
> migrate to the memory store if/when that becomes the canonical location.

## Open

### Rewind-with-hint for truncated tool calls

- **Design**: [docs/design/rewind-with-hint.md](design/rewind-with-hint.md)
- **Status**: Design drafted, not scheduled.
- **Summary**: Detect when an LLM emits a tool call with empty/truncated
  arguments (typically because `max_tokens` cut off mid-serialization of a
  large `content` parameter), rewind the last assistant message to keep its
  narration but drop the failed `tool_use`, and inject a synthetic user-role
  hint naming the specific tool and suggesting a chunked-write strategy.
- **Why it matters**: Replaces the current workaround of baking large-payload
  guidance into every session's system prompt. Reactive, targeted, bounded.
- **Entry points**: `jaato-server/shared/jaato_session.py:3962` (hook after
  `_add_model_response_to_history`); new `shared/rewind.py` module for the
  detector.
- **Open questions**: Cache invalidation cost on Anthropic provider; detector
  scope v1 conservatism; budget-reset semantics.
