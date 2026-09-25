/**
 * The rail's Diagnostics section (#1294): the attached session's own
 * confinement and runtime facts, self-diagnosed on demand.
 *
 * The data and every action live in ``app/diagnostics.ts``; this component
 * draws the store slice.  It renders two things, kept visually apart on
 * purpose:
 *
 *   - the SESSION RECORD -- runner identity, the AppArmor profile the
 *     record claims, ``sandbox_mode``, spend, the notebook boundary, the
 *     protocol/build -- in the neutral "record" tone, because these are
 *     CACHED facts the daemon stamped at spawn and never re-measures for
 *     this call;
 *   - the LIVE RE-CHECK -- what ``probe_confinement_now`` measured on the
 *     runner's own threads at the moment of the last answer -- in its own
 *     bordered block with a distinct heading and a timestamp, so a reader
 *     can never mistake "what the record claims" for "what was just
 *     measured".  A probe that could not determine an answer says so in
 *     words, never a guessed yes or no.
 *
 * "Re-check now" is the one action here, and it is the ONLY action: there
 * is no export, no download, and no "copy as JSON" -- this panel is the
 * live-view tier of #1294 and nothing else.
 */
import { useEffect } from "react";
import { useJaato } from "@/store/store";
import type { DiagnosticsThread } from "@/store/types";
import { refreshDiagnostics } from "@/app/diagnostics";

function since(ms: number | null): string {
  if (ms == null) return "";
  const s = Math.max(0, Math.round((Date.now() - ms) / 1000));
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  return `${m}m ago`;
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  if (value == null || value === "") return null;
  return (
    <div className="flex gap-2 text-[12px]">
      <span className="w-[104px] shrink-0 text-text-muted">{label}</span>
      <span className="min-w-0 break-words font-mono">{value}</span>
    </div>
  );
}

function ThreadList({ title, threads, tone }: { title: string; threads: DiagnosticsThread[]; tone: string }) {
  if (threads.length === 0) return null;
  return (
    <div className="mt-1">
      <p className={`m-0 text-[11px] uppercase tracking-[0.08em] ${tone}`}>{title} ({threads.length})</p>
      <ul className="m-0 pl-0 list-none">
        {threads.map((t) => (
          <li key={t.tid} className="font-mono text-[11px] text-text-muted">
            tid={t.tid} {t.name ? `name=${t.name}` : "(unknown thread)"} {t.label ? `label=${t.label}` : t.reason ? `(${t.reason})` : ""}
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * The live re-check block.  ``probe === null`` means either nothing has
 * been checked yet (before the first answer) or this session has no
 * runner to probe (the record fields still render above it) -- the two
 * are told apart by whether ``checkedAt`` is set at all.
 */
function ProbeBlock({ probe, checkedAt }: { probe: import("@/store/types").DiagnosticsProbe | null; checkedAt: number | null }) {
  if (checkedAt == null) {
    return <p className="m-0 py-1 text-[12px] text-text-muted">Not checked yet -- click Re-check now.</p>;
  }
  if (!probe) {
    return <p className="m-0 py-1 text-[12px] text-text-muted">This session has no runner subprocess to probe -- it runs in-process, which is not the confined-runner posture this check reports on.</p>;
  }
  if (!probe.ok) {
    return (
      <p className="m-0 py-1 text-[12px] text-warning" role="status">
        Could not determine confinement: {probe.error || "the probe did not answer."}
      </p>
    );
  }
  const scan = probe.scan;
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <span className={`inline-block w-2 h-2 rounded-full ${probe.enforced ? "bg-success" : probe.confined ? "bg-warning" : "bg-error"}`} aria-hidden="true" />
        <span className="text-[13px]">
          {probe.enforced ? "Enforced" : probe.confined ? "Confined, not enforced (complain mode)" : "Not confined"}
        </span>
      </div>
      <Row label="expected profile" value={probe.expected_profile || "(none declared)"} />
      <Row label="kernel reports" value={probe.current_profile ? `${probe.current_profile}${probe.current_mode ? ` (${probe.current_mode})` : ""}` : "(unlabelled)"} />
      {scan && (
        <>
          <Row label="threads" value={`${scan.scanned} scanned, ${scan.matched} matched, ${scan.divergent} divergent, ${scan.unreadable} unreadable, ${scan.gone} gone`} />
          <Row label="route" value={`${scan.route}${scan.uniform ? "" : " -- NOT uniform"}`} />
          <ThreadList title="Divergent threads" threads={scan.divergent_threads} tone="text-error" />
          <ThreadList title="Unreadable threads" threads={scan.unreadable_threads} tone="text-warning" />
        </>
      )}
    </div>
  );
}

export function DiagnosticsPanel() {
  const d = useJaato((s) => s.diagnostics);
  const sessionId = useJaato((s) => s.sessionId);
  useEffect(() => {
    if (sessionId && useJaato.getState().diagnostics.status === "idle") void refreshDiagnostics();
  }, [sessionId]);
  if (!sessionId) return <p className="m-0 px-3.5 py-2 text-[13px] text-text-muted">No session attached.</p>;
  if (d.status === "unsupported") {
    return <p className="m-0 px-3.5 py-2 text-[13px] text-text-muted">This daemon is older than protocol 1.25 and cannot report session diagnostics.</p>;
  }
  return (
    <div className="px-3.5 py-2 text-[13px] flex flex-col gap-2.5">
      <div className="flex items-center gap-2">
        <span className="font-mono text-[11px] text-text-muted">{checkedLine(d.checkedAt)}</span>
        <span className="flex-1" />
        <button type="button" onClick={() => { void refreshDiagnostics(); }} disabled={d.status === "loading"} className="btn btn-sm btn-steel" aria-label="Re-check now">
          {d.status === "loading" ? "Checking…" : "Re-check now"}
        </button>
      </div>
      {d.status === "error" && <p className="m-0 py-1 text-error" role="alert">{d.error}</p>}
      {d.status === "loaded" && (
        <>
          <section aria-label="Session record">
            <p className="kicker kicker-muted m-0 mb-1">Session record (as tracked by the daemon)</p>
            <div className="flex flex-col gap-1">
              <Row label="confinement" value={d.confinementId || "(none requested)"} />
              <Row label="sandbox mode" value={d.sandboxMode ?? "(none)"} />
              <Row label="runner" value={runnerLine(d.runnerIdentity)} />
              <Row label="notebook boundary" value={d.notebookBoundaryKind ?? "(no notebook plugin)"} />
              <Row label="protocol" value={d.protocolVersion} />
              <Row label="server" value={d.serverVersion} />
              {d.consumption && <SpendRow consumption={d.consumption} />}
            </div>
          </section>
          <section aria-label="Live confinement check" className="border hairline p-2">
            <p className="kicker kicker-muted m-0 mb-1">Live re-check (just measured)</p>
            <ProbeBlock probe={d.probe} checkedAt={d.checkedAt} />
          </section>
        </>
      )}
    </div>
  );
}

function checkedLine(checkedAt: number | null): string {
  return checkedAt == null ? "never checked" : `checked ${since(checkedAt)}`;
}

function runnerLine(identity: Record<string, unknown> | null): string {
  if (!identity) return "(none -- in-process)";
  const pid = identity.runner_pid;
  const pool = identity.pool_served ? "pool-served" : "cold-spawned";
  const stale = identity.stale ? " (stale record)" : "";
  const cascade = identity.cascade_driver_id ? `, cascade ${identity.cascade_driver_id}` : "";
  return `pid ${pid ?? "?"}, ${pool}${cascade}${stale}`;
}

function consumptionUsd(consumption: Record<string, unknown> | null): number | null {
  if (!consumption) return null;
  const usd = consumption["usd"] ?? (consumption["totals"] as Record<string, unknown> | undefined)?.["usd"];
  return usd != null ? Number(usd) : null;
}

/**
 * The "spend" row.  When the aspect names a plain ``usd`` figure that is
 * the whole answer, exactly as before.  When it does not (an older daemon,
 * a differently-shaped aspect), the raw dict used to be dumped straight
 * into the row as ``JSON.stringify(...).slice(0, 80)`` -- a plain-language
 * line is above the fold instead, and the JSON sits behind a
 * ``<details>`` disclosure for whoever wants it (#1304 §5: raw JSON goes
 * behind a disclosure, not into the row itself).
 */
function SpendRow({ consumption }: { consumption: Record<string, unknown> }) {
  const usd = consumptionUsd(consumption);
  if (usd != null) return <Row label="spend" value={`$${usd.toFixed(4)}`} />;
  return (
    <div className="flex gap-2 text-[12px]">
      <span className="w-[104px] shrink-0 text-text-muted">spend</span>
      <details className="min-w-0">
        <summary className="cursor-pointer text-text-muted">reported in a shape this panel does not summarise</summary>
        <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-[11px] text-text-muted">{JSON.stringify(consumption, null, 2)}</pre>
      </details>
    </div>
  );
}
