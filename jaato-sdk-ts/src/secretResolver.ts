/**
 * `app://` secret resolution, application side (#1226).
 *
 * A jaato daemon serving several people from one root process holds no
 * per-user credential: a workspace `.env` carries a REFERENCE
 * (`GH_TOKEN=app://github`), and at every session spawn the daemon asks the
 * application that OWNS the workspace to resolve it, over the same bind
 * channel #1074's `ticket.bind` / `ticket.revoke` ride.  This is the daemon ->
 * application direction (`secret.resolve`), the one verb that runs that way.
 *
 * This is the SDK-side HOOK an application registers to answer it.  It owns
 * only the wire: it subscribes to `secret.resolve`, hands each request to a
 * caller-supplied resolver, and sends back a correlated
 * `secret.resolve.result`.  It contains NO credential policy — minting a
 * GitHub App user token from a stored refresh token, keying a binding by
 * `(sub, workspace)`, revoking at the provider — all of that is the
 * application's own (the BFF, #1227).  Keeping the two apart is what lets the
 * daemon evolve the transport without the credential logic, and vice versa.
 *
 * The daemon carries its own deadline: an application that never registers a
 * responder, or whose resolver is slow, simply does not answer, and the daemon
 * drops the reference (or refuses the bootstrap for `app://name?required`)
 * exactly as an explicit refusal would.  So there is nothing here to fail
 * loudly about — not answering IS a valid, handled outcome.
 */
import { EventTypeValue, type SecretResolveRequest } from "./events.js";
import { JaatoClient } from "./client.js";
import type { Unsubscribe } from "./event-typing.js";

/** What the application decided about one `app://` reference. */
export interface SecretResolveOutcome {
  /**
   * `ok` (with `value`) / `not_found` / `denied` / `error`.  Defaults to `ok`
   * when a `value` is returned and `not_found` when it is not, so a resolver
   * can return `{ value }` for the common case and `{}` to decline.
   */
  status?: "ok" | "not_found" | "denied" | "error";
  /** The resolved secret; present only for an `ok` outcome. */
  value?: string | null;
  /**
   * ISO-8601 UTC instant the value stops being valid, when known (a GitHub
   * App user token lasts ~8h).  The daemon schedules a re-resolution a margin
   * before it, so the session never holds a dead token.
   */
  expiresAt?: string | null;
  /** Human-readable elaboration for a non-`ok` outcome.  Never the secret. */
  detail?: string | null;
}

/**
 * Resolves one `app://` reference for one of this application's users.
 *
 * `user` is the unqualified identity (the `user` half of the workspace owner
 * `app:user` — the daemon already knows which application it asked).
 * `workspace` is the absolute workspace path, for a per-workspace binding.
 * `name` is the reference name (`github` in `app://github`).
 */
export type SecretResolveHandler = (request: {
  user: string;
  workspace: string;
  name: string;
}) => SecretResolveOutcome | Promise<SecretResolveOutcome>;

/**
 * Answers the daemon's `secret.resolve` requests over a bind-channel client.
 *
 * Construct with the same `JaatoClient` the application authenticated its bind
 * channel with (the app-credential connection), then {@link start}.  Each
 * `secret.resolve` is handed to the resolver and its outcome sent back
 * correlated by `request_id`; a resolver that throws answers `error` rather
 * than leaving the daemon to wait out its deadline.
 */
export class SecretResolveResponder {
  private _unsubscribe: Unsubscribe | null = null;

  constructor(
    private readonly _client: JaatoClient,
    private readonly _handler: SecretResolveHandler,
  ) {}

  /**
   * Begin answering `secret.resolve` requests.  Idempotent; returns an
   * unsubscribe that also clears the internal handle so {@link start} may be
   * called again.
   */
  start(): Unsubscribe {
    if (this._unsubscribe) return this._unsubscribe;
    const off = this._client.subscribe(
      EventTypeValue.SECRET_RESOLVE_REQUEST,
      (event) => {
        // Fire-and-forget: the reply is sent from within the async handler,
        // and a failure there becomes an `error` result, never an unhandled
        // rejection that could take the process down.
        void this._answer(event as SecretResolveRequest);
      },
    );
    this._unsubscribe = () => {
      off();
      this._unsubscribe = null;
    };
    return this._unsubscribe;
  }

  /** Stop answering.  Safe to call when not started. */
  stop(): void {
    this._unsubscribe?.();
  }

  private async _answer(request: SecretResolveRequest): Promise<void> {
    let outcome: SecretResolveOutcome;
    try {
      outcome = await this._handler({
        user: request.user ?? "",
        workspace: request.workspace ?? "",
        name: request.name ?? "",
      });
    } catch (err) {
      outcome = { status: "error", detail: (err as Error).message };
    }
    const status =
      outcome.status ??
      (outcome.value != null ? "ok" : "not_found");
    const result: Record<string, unknown> = {
      type: EventTypeValue.SECRET_RESOLVE_RESULT,
      request_id: request.request_id,
      status,
    };
    if (status === "ok") result.value = outcome.value ?? null;
    if (outcome.expiresAt != null) result.expires_at = outcome.expiresAt;
    if (outcome.detail != null) result.detail = outcome.detail;
    try {
      await this._client.sendRawEvent(result);
    } catch {
      // The channel dropped between request and reply; the daemon's deadline
      // handles it (the reference is dropped), so there is nothing to do here.
    }
  }
}
