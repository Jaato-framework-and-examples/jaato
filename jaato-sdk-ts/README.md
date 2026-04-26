# @jaato/sdk

TypeScript / JavaScript SDK for [jaato-server](https://github.com/Jaato-framework-and-examples/jaato).

Mirrors the Python [`jaato-sdk`](../jaato-sdk/) method-for-method, with
identical noun naming (camelCase per JS convention) so cross-language
parity is enforced by construction.

**Status: pre-alpha (Phase 2).** This package currently ships only the
codegen-generated event/request types. The `JaatoClient` class that
wraps the WS protocol arrives in Phase 3 — see
[`project_backlog_sdk_feature_parity.md`](../docs/) for the full plan.

## Wire-protocol types

The full event/request type surface is generated from the Python
side's pydantic models:

```
jaato-sdk/jaato_sdk/events.py  (source of truth, pydantic)
            │
            ▼
scripts/codegen_ts_events.py    (uses pydantic.TypeAdapter to emit JSON Schema,
                                 then pipes through json-schema-to-typescript)
            │
            ▼
jaato-sdk-ts/src/events.ts     (generated, committed)
```

CI fails any PR that touches `events.py` without re-running codegen and
committing the regenerated `events.ts`.

## Regenerating

From the repo root:

```bash
.venv/bin/python scripts/codegen_ts_events.py
```

Or from `jaato-sdk-ts/`:

```bash
npm run codegen
```

## Verifying

The CI staleness gate uses:

```bash
.venv/bin/python scripts/codegen_ts_events.py --check
```

which exits non-zero (with a unified diff) if the committed
`events.ts` is stale relative to a fresh regeneration.

## Importing

```typescript
import {
  EventType,
  JaatoEvent,
  SendMessageRequest,
  AgentOutputEvent,
  // ...
} from "@jaato/sdk";

function handle(event: JaatoEvent): void {
  switch (event.type) {
    case EventType.AGENT_OUTPUT:
      // event narrowed to AgentOutputEvent
      console.log(event.text);
      break;
    case EventType.PERMISSION_REQUESTED:
      // ...
      break;
  }
}
```

## License

BUSL-1.1 (matches jaato-server / jaato-sdk).
