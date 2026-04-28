// Type helpers for the typed `subscribe` / `subscribeOnce` /
// `subscribeMany` API on JaatoClient.
//
// These build on the generated JaatoEvent discriminated union from
// ./events.ts.  Pure type math — no runtime values — so this file is
// safe to hand-edit (it's not part of the codegen pipeline).

import type { EventType, JaatoEvent } from "./events.js";

/**
 * Narrow a JaatoEvent down to the concrete interface whose `type`
 * field equals `T`.
 *
 * @example
 *   type Perm = EventByType<'permission.requested'>;
 *   //   ^? PermissionRequestedEvent
 */
export type EventByType<T extends EventType> = Extract<JaatoEvent, { type: T }>;

/**
 * Sync or async handler for one specific event type.
 *
 * Async handlers are dispatched fire-and-forget by the client; their
 * completion is independent of the next event's delivery.
 */
export type EventHandler<T extends EventType> = (
  event: EventByType<T>,
) => void | Promise<void>;

/**
 * Catchall handler — receives every event regardless of type.
 */
export type CatchallEventHandler = (event: JaatoEvent) => void | Promise<void>;

/**
 * Argument shape for {@link JaatoClient.subscribeMany}: a partial
 * mapping from event type to its typed handler. TypeScript will
 * narrow each value to the matching event interface.
 *
 * @example
 *   client.subscribeMany({
 *     'permission.requested': (e) => console.log(e.request_id),
 *     'agent.completed':      (e) => console.log(e.token_count),
 *   });
 */
export type SubscribeManyMap = {
  [K in EventType]?: EventHandler<K>;
};

/**
 * Idempotent unsubscribe returned by every `subscribe*` method.
 *
 * Calling it more than once is a no-op. Calling it during dispatch
 * only takes effect for the next event — the in-flight handler list
 * is snapshotted at the start of dispatch.
 */
export type Unsubscribe = () => void;
