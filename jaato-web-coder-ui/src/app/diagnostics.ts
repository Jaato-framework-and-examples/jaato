/**
 * The rail's Diagnostics section (#1294): a self-service, per-session
 * confinement and runtime view -- what the daemon's record claims about
 * this session, and a live re-check of whether it is actually true right
 * now.
 *
 * The incident this answers (#1253) was a CACHED ``sandbox_mode`` field
 * read as "confined" when it was not.  So the daemon's ``session.diagnostics``
 * verb (protocol 1.25) always measures freshly on the runner -- there is
 * no separate "just show me the record" call -- and this module keeps the
 * two kinds of field the answer carries visually distinct (see
 * ``DiagnosticsState`` in ``store/types.ts``): the RECORD fields are the
 * daemon's own bookkeeping, stamped at spawn and never re-measured here;
 * ``probe`` is what was just found on the runner's threads, at the moment
 * of THIS call.
 *
 * ACCESS is the daemon's decision, never this module's: a caller the
 * workspace-owner gate refuses gets ``category: "not_owner"`` and nothing
 * else -- the same #1283 gate the Memories section reads.
 *
 * There is deliberately no export, download, "copy as JSON" or any other
 * persistence surface here -- this is the live-view tier only (#1294's own
 * scope line). What is rendered is exactly what the last answer carried,
 * and nothing is written to disk, to the browser's storage, or to the
 * clipboard by this module.
 *
 * WHEN IT IS ASKED: only when the section is first opened for a session
 * (a live confinement re-probe is not something to run unasked for every
 * session on every daemon), and thereafter only the explicit "Re-check
 * now" action -- there is no background poll, unlike the Memories
 * section's write-triggered refresh, because nothing else in the wire
 * protocol signals that confinement facts may have changed.
 */
import { MIN_DIAGNOSTICS_PROTOCOL, isProtocolCompatible } from "@jaato/sdk";
import { useJaato } from "@/store/store";
import { getClient, isConnected } from "@/sdk/connection";
import type { DiagnosticsGrants, DiagnosticsProbe } from "@/store/types";

type Patch = Parameters<ReturnType<typeof useJaato.getState>["patchDiagnostics"]>[0];
const patch = (p: Patch) => useJaato.getState().patchDiagnostics(p);

/** Whether a daemon speaking ``protocol`` serves the diagnostics verb. */
export function servesDiagnostics(protocol: string | null | undefined): boolean {
  return !!protocol && isProtocolCompatible(protocol, MIN_DIAGNOSTICS_PROTOCOL);
}

/** What a refusal category means to the person who opened the section. */
export function diagnosticsRefusalText(category: string | null | undefined, error: string | null | undefined): string {
  switch (category) {
    case "not_owner": return "Only the owner of this workspace can view its diagnostics.";
    case "no_session": return "No session is attached.";
    case "runner_unreachable": return `The session's runner did not answer${error ? `: ${error}` : "."}`;
    default: return error || "The daemon refused the request.";
  }
}

/**
 * The section header's value: ``enforced`` / ``not confined`` / ``2 divergent
 * threads``, or ``null`` before anything has been checked -- an unasked
 * probe is not the same as a clean one.
 */
export function diagnosticsSummary(d: Pick<
  import("@/store/types").DiagnosticsState,
  "status" | "checkedAt" | "probe"
>): string | null {
  if (d.status === "error") return "refused";
  if (d.checkedAt == null || !d.probe) return null;
  const p = d.probe;
  if (!p.ok) return "could not check";
  if (p.scan && p.scan.divergent > 0) return `${p.scan.divergent} divergent`;
  return p.enforced ? "enforced" : p.confined ? "complain mode" : "not confined";
}

/**
 * The collapsed "AppArmor grants" line (#1326): what ``//child`` may exec,
 * and who decided.  ``scoped`` means only what the fragments name;
 * ``unscoped`` means every in-PATH binary.  ``not recorded`` is its own
 * answer, never read as either.
 */
export function grantsSummary(g: DiagnosticsGrants): string {
  const declared = g.declared_by === ""
    ? " (declared by: unknown)"
    : g.declared_by ? ` (declared by ${g.declared_by})` : "";
  if (!g.recorded) {
    const req = g.requested_fragments;
    const scope = req == null ? "unscoped" : `scoped — ${req.length} requested`;
    return `exec: ${scope}${declared} · not recorded`;
  }
  const n = g.fragments?.length ?? 0;
  const frags = `${n} fragment${n === 1 ? "" : "s"}`;
  const missing = g.missing_fragments?.length ? `, ${g.missing_fragments.length} missing` : "";
  if (g.exec_scope === "scoped") return `exec: scoped — ${frags}${missing}${declared}`;
  return `exec: unscoped (all PATH binaries) — ${frags}${missing}`;
}

let generation = 0;

/**
 * Ask the daemon now.  Keeps the last answer while asking, so a re-check
 * never blanks a populated panel, and a failed ask reports itself rather
 * than being read as "not confined".
 */
export async function refreshDiagnostics(): Promise<void> {
  if (!isConnected()) return;
  const sessionId = useJaato.getState().sessionId;
  if (!sessionId) return;
  const client = getClient();
  if (!servesDiagnostics(client.serverProtocolVersion)) {
    patch({ status: "unsupported", error: null });
    return;
  }
  const gen = ++generation;
  const current = () => gen === generation && useJaato.getState().sessionId === sessionId;
  patch({ status: "loading" });
  try {
    const answer = await client.getDiagnostics();
    if (!current()) return;
    if (answer.ok === false) {
      patch({
        status: "error",
        error: diagnosticsRefusalText(answer.category ?? null, answer.error ?? null),
        category: answer.category ?? null,
      });
      return;
    }
    patch({
      status: "loaded",
      error: null,
      category: null,
      runnerIdentity: (answer.runner_identity as unknown as Record<string, unknown> | null) ?? null,
      confinementId: answer.confinement_id ?? "",
      sandboxMode: answer.sandbox_mode ?? null,
      consumption: (answer.consumption as unknown as Record<string, unknown> | null) ?? null,
      notebookBoundaryKind: answer.notebook_boundary_kind ?? null,
      protocolVersion: answer.protocol_version ?? "",
      serverVersion: answer.server_version ?? "",
      probe: (answer.probe as unknown as DiagnosticsProbe | null) ?? null,
      apparmorGrants: (answer.apparmor_grants as unknown as DiagnosticsGrants | null) ?? null,
      checkedAt: Date.now(),
    });
  } catch (err) {
    if (!current()) return;
    patch({ status: "error", error: err instanceof Error ? err.message : String(err) });
  }
}
