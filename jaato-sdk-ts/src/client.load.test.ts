// Load / perf tests for the subscribe API.
//
// Not part of the default ``npm test`` run.  Execute with::
//
//     npm run test:load
//
// Baselines and ratios live in docs/sdk-perf-baseline.md.  These
// tests assert *ratios* between configurations (stable across
// machines), not absolute timings.
//
// Mirror of jaato-sdk/jaato_sdk/tests/test_subscribe_load.py — keeping
// the cross-language perf contract identical so a regression in one
// SDK shows up in the other under the same workload.

import { strict as assert } from "node:assert";
import { describe, test } from "node:test";

import { JaatoClient } from "./client.js";
import { EventTypeValue, type EventType, type JaatoEvent } from "./events.js";

const N_EVENTS = 10_000;
const N_HANDLERS = 100;

function makeOutput(): JaatoEvent {
  // One reused payload — _dispatch never mutates it.
  return {
    type: EventTypeValue.AGENT_OUTPUT,
    timestamp: new Date().toISOString(),
    agent_id: "main",
    source: "model",
    text: "x",
    mode: "append",
  } as unknown as JaatoEvent;
}

/**
 * Drive `events` synthetic dispatches through the client's internal
 * dispatcher and return events-per-second.  We bypass the WS transport
 * entirely — this measures dispatch cost, not network or framing.
 */
function measureThroughput(client: JaatoClient, events: number): number {
  const payload = makeOutput();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const dispatch = (client as any)._dispatchEvent.bind(client) as (
    e: JaatoEvent,
  ) => void;
  const start = process.hrtime.bigint();
  for (let i = 0; i < events; i++) {
    dispatch(payload);
  }
  const elapsedNs = Number(process.hrtime.bigint() - start);
  const elapsedSec = elapsedNs / 1e9;
  return elapsedSec > 0 ? events / elapsedSec : Infinity;
}

function newClient(): JaatoClient {
  // Never connected — we drive _dispatchEvent directly.
  return new JaatoClient({ url: "ws://localhost:0" });
}

describe("JaatoClient subscribe API — load", () => {
  test("baseline: catchall x1", () => {
    const c = newClient();
    let counter = 0;
    c.subscribeAll(() => {
      counter += 1;
    });

    const rate = measureThroughput(c, N_EVENTS);

    assert.equal(counter, N_EVENTS);
    // eslint-disable-next-line no-console
    console.log(`\n[baseline] catchall x1: ${rate.toFixed(0).padStart(12)} events/s`);
  });

  test("100 typed handlers: per-call cost stays bounded", () => {
    // Baseline (1 handler, throughput is per-invocation).
    const cBaseline = newClient();
    cBaseline.subscribeAll(() => undefined);
    const baselineInvocations = measureThroughput(cBaseline, N_EVENTS) * 1;

    // 100 typed handlers all on the same event type.
    const c = newClient();
    for (let i = 0; i < N_HANDLERS; i++) {
      c.subscribe(EventTypeValue.AGENT_OUTPUT, () => undefined);
    }
    const typedEventRate = measureThroughput(c, N_EVENTS);
    const typedInvocations = typedEventRate * N_HANDLERS;

    const perCallRatio =
      typedInvocations > 0 ? baselineInvocations / typedInvocations : Infinity;

    // eslint-disable-next-line no-console
    console.log(
      `\n[100-typed]  baseline_invocations=${baselineInvocations
        .toFixed(0)
        .padStart(12)}/s  100-typed_invocations=${typedInvocations
        .toFixed(0)
        .padStart(12)}/s  per-call=${perCallRatio.toFixed(2)}x`,
    );

    assert.ok(
      perCallRatio < 3.0,
      `Per-handler-invocation cost is ${perCallRatio.toFixed(2)}x worse with ` +
        "100 handlers than with 1; expected <3x. Investigate dispatch overhead.",
    );
  });

  test("subscribe/unsubscribe churn: no leak in handler buckets", () => {
    const c = newClient();
    const cycles = 10_000;

    for (let i = 0; i < cycles; i++) {
      const unsub = c.subscribe(EventTypeValue.AGENT_OUTPUT, () => undefined);
      unsub();
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const typedHandlers = (c as any)._typedHandlers as Map<EventType, unknown[]>;
    const bucket = typedHandlers.get(EventTypeValue.AGENT_OUTPUT) ?? [];
    assert.equal(
      bucket.length,
      0,
      `Leak: ${bucket.length} entries remain after ${cycles} sub/unsub cycles`,
    );
  });

  test("async handler throughput: scheduling overhead bounded", async () => {
    const cSync = newClient();
    cSync.subscribe(EventTypeValue.AGENT_OUTPUT, () => undefined);
    const syncRate = measureThroughput(cSync, N_EVENTS);

    const cAsync = newClient();
    cAsync.subscribe(EventTypeValue.AGENT_OUTPUT, async () => undefined);
    const asyncRate = measureThroughput(cAsync, N_EVENTS);

    const ratio = syncRate > 0 ? asyncRate / syncRate : 0;
    // eslint-disable-next-line no-console
    console.log(
      `\n[async]      sync=${syncRate.toFixed(0).padStart(12)}/s  ` +
        `async=${asyncRate.toFixed(0).padStart(12)}/s  ratio=${ratio.toFixed(2)}x`,
    );

    // Async handlers create a Promise + .catch chain per dispatch.
    // Empirical baseline on V8 ~3% of sync rate — this is the inherent
    // cost of allocating + scheduling a microtask per event.  We flag
    // only catastrophic regressions (sub-1% of sync) so day-to-day noise
    // doesn't fail the gate.  See docs/sdk-perf-baseline.md.
    assert.ok(
      ratio >= 0.01,
      `Async dispatch dropped to ${(ratio * 100).toFixed(1)}% of sync ` +
        "throughput; promise overhead has regressed catastrophically.",
    );

    // Wait for any pending tasks to clear before exiting the test.
    await new Promise<void>((r) => setImmediate(r));
  });
});
