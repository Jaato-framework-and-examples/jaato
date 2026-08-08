# SDK subscribe-API perf baseline

Reference numbers for the typed event subscription dispatcher. These
are **directional baselines** captured on the dev machine on
2026-04-27 — your absolute numbers will vary, but the relative ratios
should not.

## Workload

Both SDKs run the same synthetic workload (`test_subscribe_load.py`
and `client.load.test.ts`):

- 10,000 events
- Single shared payload (no per-event allocation)
- Bypass the WS/IPC transport — drives `_dispatch` / `_dispatchEvent`
  directly. Measures dispatch cost in isolation.

## How to run

| SDK | Command | Default behavior |
|---|---|---|
| Python | `pytest -m load jaato-sdk/jaato_sdk/tests/` | Skipped from regular `pytest` runs (filtered out by `addopts = "-m 'not load'"` in `pyproject.toml`). |
| TS | `npm run test:load` (in `jaato-sdk-ts/`) | Separate suite from `npm test`. |

## Reference numbers (2026-04-27)

### Python (CPython 3.12)

| Test | Result | Acceptance gate |
|---|---|---|
| Baseline (catchall ×1) | ~500k events/s | informational |
| 100 typed handlers | per-call ratio ≈ 0.26× baseline | **<3× per-call regression** |
| 10k subscribe/unsub churn | bucket empty after churn | **no leak** |
| Async fire-and-forget | ~23% of sync rate | **≥5% of sync** |

### TypeScript (Node 22 / V8)

| Test | Result | Acceptance gate |
|---|---|---|
| Baseline (catchall ×1) | ~3M events/s | informational |
| 100 typed handlers | per-call ratio < 1× baseline (amortized fixed cost) | **<3× per-call regression** |
| 10k subscribe/unsub churn | bucket empty after churn | **no leak** |
| Async fire-and-forget | ~3-5% of sync rate | **≥1% of sync** |

## How to interpret

- **Per-call ratio < 1×**: more handlers can actually be *cheaper* per
  invocation because the snapshot+iteration fixed cost amortizes. We
  still cap at 3× to catch true regressions in the dispatch loop.
- **Async ratio low**: Promise/Task allocation per event is the
  dominant cost when handlers do nothing. Real workloads (I/O-bound)
  see a much smaller relative gap because the I/O dwarfs scheduling.
  Don't conclude "async is too slow" — conclude "don't use async
  handlers for things that are already sync".
- **Cross-language comparison**: TS is ~6× faster than Python on the
  raw catchall path because V8 optimizes hot loops aggressively. This
  is expected and not actionable.

## When to update this file

Update the reference numbers when:

1. The dispatcher implementation changes (snapshot strategy, bucket
   structure, async scheduling).
2. A major Python or Node version bump shifts the baseline.
3. CI runners change (currently dev-machine numbers; tighten gates if
   we move to dedicated runners).

Do **not** chase day-to-day noise — the asserted ratios are
intentionally generous so that ordinary jitter doesn't fail the gate.
