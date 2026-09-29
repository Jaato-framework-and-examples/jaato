/**
 * The rail's Proposals section: reference CLAIMS agents proposed with
 * ``proposeReference``, and -- for the workspace owner -- Promote / Dismiss.
 * When the workspace has sub-bundles, Promote has a selector naming where
 * the entry goes (the catalog root by default); an indexed bundle is
 * reconciled by the daemon and the notice says if its index was not.
 *
 * The data and every action live in ``app/referenceClaims.ts``; this
 * component draws the store slice.  What it shows, per claim:
 *
 *   • the proposed name and id, its tags, and where the document is (a
 *     workspace path, or "inline");
 *   • who proposed it and whether a person approved the call at the prompt
 *     (``origin.witnessed_by``) -- **unreviewed** until promoted, because a
 *     claim is model-written and nobody has looked at it;
 *   • the daemon's ``problems``: why a promotion would be refused right now,
 *     shown BEFORE the button, which is then disabled;
 *   • the typed ``links`` it declares, a ``supersedes`` marked because
 *     promoting it reroutes requests for the target, and ``warnings`` (an
 *     edge this workspace's catalog cannot place) that do not block Promote;
 *   • expanding shows the description and any inline content.
 *
 * Every model-written string is rendered as React text, never as markup.
 * Promote / Dismiss are drawn only when the daemon says this connection may
 * curate (``may_curate``); the daemon enforces the same gate.  Every control
 * is a real ``button`` whose accessible name contains its visible word, and
 * none is hover-gated.
 */
import { useEffect, useState } from "react";
import { useJaato } from "@/store/store";
import type { ReferenceBundleOption, ReferenceClaimRow } from "@/store/types";
import {
  describeClaimLink,
  describeClaimOrigin,
  isRoutingLink,
  dismissReferenceClaim,
  promoteReferenceClaim,
  refreshReferenceClaims,
  toggleReferenceClaim,
} from "@/app/referenceClaims";

function when(ts: unknown): string {
  if (typeof ts !== "string" || !ts) return "";
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? ts : d.toISOString().slice(0, 16).replace("T", " ");
}

export function ProposalRow({ row, mayCurate, bundles = [] }: {
  row: ReferenceClaimRow; mayCurate: boolean; bundles?: ReferenceBundleOption[];
}) {
  const expanded = useJaato((s) => s.referenceClaims.expanded === row.claim_id);
  const busy = useJaato((s) => s.referenceClaims.busy[row.claim_id]);
  const [confirmDismiss, setConfirmDismiss] = useState(false);
  const [bundle, setBundle] = useState("");
  const title = row.name || row.id;
  const blocked = row.problems.length > 0;
  const witnessed = !!row.origin?.witnessed_by;
  return (
    <div className="border-t hairline" data-testid="proposal-row" data-claim-id={row.claim_id}>
      <div className="flex gap-2 py-2 items-start">
        <span className={`pt-0.5 ${witnessed ? "text-steel" : "text-warning"}`} title={witnessed ? "approved at the prompt" : "not approved at a prompt"} aria-hidden="true">{witnessed ? "◐" : "○"}</span>
        <div className="min-w-0 flex-1">
          <span className="block text-[13px]">{title}</span>
          <span className="block font-mono text-[11px] text-text-muted truncate">
            <span className="text-warning uppercase tracking-[0.08em] mr-1.5">unreviewed</span>
            <span className="mr-1.5">{row.id}</span>
            {row.tags.map((t) => `#${t}`).join(" ")}
          </span>
          <span className="block font-mono text-[11px] text-text-muted truncate">{row.type === "local" && row.path ? row.path : "inline"}</span>
        </div>
        <button
          type="button"
          onClick={() => toggleReferenceClaim(row.claim_id)}
          aria-expanded={expanded}
          aria-label={`${expanded ? "Hide" : "Show"} proposal ${title}`}
          className="btn btn-sm btn-quiet self-center shrink-0"
        >
          {expanded ? "Hide" : "Show"}
        </button>
      </div>
      {blocked && (
        <ul className="m-0 pl-6 pb-1.5 list-none text-[12px] text-error" aria-label={`Why ${title} cannot be promoted`}>
          {row.problems.map((p) => <li key={p}>{p}</li>)}
        </ul>
      )}
      {!!row.links?.length && (
        <ul className="m-0 pl-6 pb-1.5 list-none font-mono text-[11px] text-text-muted" aria-label={`Links proposed by ${title}`} data-testid="proposal-links">
          {row.links.map((l) => (
            <li key={`${l.rel}:${l.to}`} className={isRoutingLink(l) ? "text-warning" : undefined}>
              {describeClaimLink(l)}{isRoutingLink(l) ? " (requests for it will get this one)" : ""}
            </li>
          ))}
        </ul>
      )}
      {!!row.warnings?.length && (
        <ul className="m-0 pl-6 pb-1.5 list-none text-[12px] text-warning" aria-label={`Warnings for ${title}`}>
          {row.warnings.map((w) => <li key={w}>{w}</li>)}
        </ul>
      )}
      {expanded && (
        <div className="pl-6 pb-2 flex flex-col gap-1.5">
          {row.description && <p className="m-0 text-[12px]">{row.description}</p>}
          {row.type === "inline" && row.content !== undefined && (
            <pre className="m-0 whitespace-pre-wrap font-mono text-[12px]" data-testid="proposal-content">{row.content}</pre>
          )}
          <p className="m-0 font-mono text-[11px] text-text-muted">
            {[describeClaimOrigin(row.origin), when(row.origin?.at) ? `proposed ${when(row.origin?.at)}` : "", `claim ${row.claim_id}`]
              .filter(Boolean).join(" · ")}
          </p>
        </div>
      )}
      {mayCurate && (
        <div className="pl-6 pb-2 flex flex-wrap gap-1.5" aria-label={`Curate proposal ${row.claim_id}`}>
          {bundles.length > 0 && (
            <select
              value={bundle}
              onChange={(e) => setBundle(e.target.value)}
              disabled={!!busy || blocked}
              aria-label={`Promote proposal ${title} into`}
              className="font-mono text-[11px] bg-transparent border hairline px-1"
            >
              <option value="">catalog</option>
              {bundles.map((b) => (
                <option key={b.name} value={b.name}>{b.indexed ? `${b.name} (indexed)` : b.name}</option>
              ))}
            </select>
          )}
          <button type="button" disabled={!!busy || blocked} onClick={() => { void promoteReferenceClaim(row.claim_id, bundle); }} className="btn btn-sm btn-steel" aria-label={`Promote proposal ${title}`}>Promote</button>
          {confirmDismiss ? (
            <>
              <button type="button" disabled={!!busy} onClick={() => { setConfirmDismiss(false); void dismissReferenceClaim(row.claim_id); }} className="btn btn-sm btn-quiet text-error" aria-label={`Confirm dismiss proposal ${title}`}>Confirm dismiss</button>
              <button type="button" onClick={() => setConfirmDismiss(false)} className="btn btn-sm btn-quiet">Keep</button>
            </>
          ) : (
            <button type="button" disabled={!!busy} onClick={() => setConfirmDismiss(true)} className="btn btn-sm btn-quiet" aria-label={`Dismiss proposal ${title}`}>Dismiss</button>
          )}
          {busy && <span className="font-mono text-[11px] text-text-muted self-center">{busy}…</span>}
        </div>
      )}
    </div>
  );
}

/** The section body.  Opening it asks when nothing has been asked yet. */
export function ProposalsPanel() {
  const r = useJaato((s) => s.referenceClaims);
  const sessionId = useJaato((s) => s.sessionId);
  useEffect(() => {
    if (sessionId && useJaato.getState().referenceClaims.status === "idle") void refreshReferenceClaims();
  }, [sessionId]);
  const mayCurate = r.mayCurate === true;
  if (!sessionId) return <p className="m-0 px-3.5 py-2 text-[13px] text-text-muted">No session attached.</p>;
  if (r.status === "unsupported") {
    return <p className="m-0 px-3.5 py-2 text-[13px] text-text-muted">This daemon is older than protocol 1.33 and cannot list reference proposals.</p>;
  }
  return (
    <div className="px-3.5 py-2 text-[13px]">
      <div className="flex items-center gap-2 pb-1.5">
        <span className="text-[12px] text-text-muted">References agents proposed, waiting for review.</span>
        <span className="flex-1" />
        <button type="button" onClick={() => { void refreshReferenceClaims(); }} className="btn btn-sm btn-quiet" aria-label="Refresh proposals">Refresh</button>
      </div>
      {r.status === "error" && <p className="m-0 py-1 text-error" role="alert">Could not read the proposals: {r.error}</p>}
      {r.notice && <p className={`m-0 py-1 ${r.notice.error ? "text-error" : r.notice.warning ? "text-warning" : "text-text-muted"}`} role="status">{r.notice.text}</p>}
      {r.mayCurate === false && r.rows.length > 0 && (
        <p className="m-0 py-1 text-[11px] text-text-muted">Only the owner of this workspace can promote or dismiss these.</p>
      )}
      {r.status === "loading" && r.rows.length === 0 ? (
        <p className="m-0 py-2 text-text-muted">Loading…</p>
      ) : r.rows.length === 0 ? (
        <p className="m-0 py-2 text-text-muted">{r.status === "loaded" ? "Nothing proposed." : ""}</p>
      ) : (
        r.rows.map((row) => <ProposalRow key={row.claim_id} row={row} mayCurate={mayCurate} bundles={r.bundles} />)
      )}
      {r.unreadable.length > 0 && (
        <p className="m-0 py-1 text-[11px] text-warning">
          {r.unreadable.length} file{r.unreadable.length === 1 ? "" : "s"} in .jaato/references-claims could not be shown: {r.unreadable.join(", ")}
        </p>
      )}
    </div>
  );
}
