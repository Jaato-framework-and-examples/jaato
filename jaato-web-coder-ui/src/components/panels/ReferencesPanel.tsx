/**
 * The rail's References section: the workspace reference catalog, each
 * reference's typed links both ways, and -- for the workspace owner -- an
 * editor for one reference's links.
 *
 * The data and every action live in ``app/referenceCatalog.ts``; this
 * component draws the store slice.  What it shows, per reference:
 *
 *   • its name, id and the sub-bundle it lives in (nothing for the root);
 *   • its declared links, a dangling one (a target the catalog does not
 *     hold) marked, a ``supersedes`` marked because it reroutes requests for
 *     the target;
 *   • expanding shows the description and the links pointing AT it;
 *   • the owner gets Edit links: each link a row (target, relation, note,
 *     Remove), an Add link row, and Save / Cancel.  A save sends the whole
 *     list and the daemon writes only the ``links`` key.
 *
 * A reference whose id is declared in two files is marked and gets no
 * editor: the daemon refuses to guess which file to change.  Catalog text
 * (name, description, a note) is rendered as React text, never as markup.
 * Every control is a real ``button`` / ``input`` / ``select`` with an
 * accessible name that contains its visible word, none hover-gated.
 */
import { useEffect } from "react";
import { useJaato } from "@/store/store";
import type { ReferenceCatalogRow, ReferenceClaimLink } from "@/store/types";
import {
  LINK_RELS,
  REL_EFFECT,
  cancelEditingLinks,
  refreshReferenceCatalog,
  saveDraftLinks,
  setDraftLinks,
  startEditingLinks,
  toggleReference,
} from "@/app/referenceCatalog";
import { describeClaimLink, isRoutingLink } from "@/app/referenceClaims";

function LinksEditor({ row }: { row: ReferenceCatalogRow }) {
  const draft = useJaato((s) => s.referenceCatalog.editing);
  const busy = useJaato((s) => !!s.referenceCatalog.busy[row.id]);
  if (!draft || draft.id !== row.id) return null;
  const links = draft.links;
  const title = row.name || row.id;
  const update = (i: number, change: Partial<ReferenceClaimLink>) =>
    setDraftLinks(links.map((l, j) => (j === i ? { ...l, ...change } : l)));
  return (
    <div className="pl-6 pb-2 flex flex-col gap-1.5" aria-label={`Edit links of ${title}`} data-testid="links-editor">
      {links.length === 0 && <p className="m-0 text-[12px] text-text-muted">No links. Saving removes every declared link.</p>}
      {links.map((l, i) => (
        <div key={i} className="flex flex-wrap gap-1 items-center" data-testid="link-draft-row">
          <select
            value={l.rel}
            onChange={(e) => update(i, { rel: e.target.value })}
            aria-label={`Relation of link ${i + 1}`}
            title={REL_EFFECT[l.rel as keyof typeof REL_EFFECT] ?? ""}
            className="font-mono text-[11px] bg-transparent border hairline px-1"
          >
            {LINK_RELS.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
          <input
            value={l.to}
            onChange={(e) => update(i, { to: e.target.value })}
            aria-label={`Target of link ${i + 1}`}
            placeholder="reference id"
            className="font-mono text-[11px] bg-transparent border hairline px-1 w-[9rem]"
          />
          <input
            value={l.note ?? ""}
            onChange={(e) => update(i, { note: e.target.value })}
            aria-label={`Note of link ${i + 1}`}
            placeholder="note (optional)"
            className="text-[11px] bg-transparent border hairline px-1 flex-1 min-w-[6rem]"
          />
          <button type="button" onClick={() => setDraftLinks(links.filter((_, j) => j !== i))} className="btn btn-sm btn-quiet" aria-label={`Remove link ${i + 1}`}>Remove</button>
        </div>
      ))}
      <div className="flex flex-wrap gap-1.5">
        <button type="button" onClick={() => setDraftLinks([...links, { to: "", rel: "elaborates" }])} className="btn btn-sm btn-quiet" aria-label={`Add link to ${title}`}>Add link</button>
        <span className="flex-1" />
        <button type="button" disabled={busy} onClick={() => { void saveDraftLinks(); }} className="btn btn-sm btn-steel" aria-label={`Save links of ${title}`}>Save</button>
        <button type="button" disabled={busy} onClick={cancelEditingLinks} className="btn btn-sm btn-quiet" aria-label={`Cancel editing ${title}`}>Cancel</button>
        {busy && <span className="font-mono text-[11px] text-text-muted self-center">saving…</span>}
      </div>
    </div>
  );
}

export function ReferenceRow({ row, mayCurate }: { row: ReferenceCatalogRow; mayCurate: boolean }) {
  const expanded = useJaato((s) => s.referenceCatalog.expanded === row.id);
  const editing = useJaato((s) => s.referenceCatalog.editing?.id === row.id);
  const anotherEditing = useJaato((s) => !!s.referenceCatalog.editing && s.referenceCatalog.editing.id !== row.id);
  const title = row.name || row.id;
  return (
    <div className="border-t hairline" data-testid="reference-row" data-reference-id={row.id}>
      <div className="flex gap-2 py-2 items-start">
        <div className="min-w-0 flex-1">
          <span className="block text-[13px]">{title}</span>
          <span className="block font-mono text-[11px] text-text-muted truncate">
            <span className="mr-1.5">{row.id}</span>
            {row.bundle && <span className="mr-1.5">bundle {row.bundle}</span>}
            {row.duplicate_id && <span className="text-error">id declared in two files</span>}
          </span>
        </div>
        <button
          type="button"
          onClick={() => toggleReference(row.id)}
          aria-expanded={expanded}
          aria-label={`${expanded ? "Hide" : "Show"} reference ${title}`}
          className="btn btn-sm btn-quiet self-center shrink-0"
        >
          {expanded ? "Hide" : "Show"}
        </button>
      </div>
      {!editing && row.links.length > 0 && (
        <ul className="m-0 pl-6 pb-1.5 list-none font-mono text-[11px] text-text-muted" aria-label={`Links of ${title}`} data-testid="reference-links">
          {row.links.map((l) => (
            <li key={`${l.rel}:${l.to}`} className={l.dangling ? "text-error" : isRoutingLink(l) ? "text-warning" : undefined}>
              {describeClaimLink(l)}
              {l.dangling ? " (not in this catalog)" : isRoutingLink(l) ? " (requests for it get this one)" : ""}
            </li>
          ))}
        </ul>
      )}
      {expanded && !editing && (
        <div className="pl-6 pb-2 flex flex-col gap-1.5">
          {row.description && <p className="m-0 text-[12px]">{row.description}</p>}
          <p className="m-0 font-mono text-[11px] text-text-muted" aria-label={`Links to ${title}`}>
            {row.linked_from.length
              ? `linked from ${row.linked_from.map((l) => `${l.from} (${l.rel.replace(/-/g, " ")})`).join(", ")}`
              : "nothing links to it"}
          </p>
          <p className="m-0 font-mono text-[11px] text-text-muted">{row.file}</p>
        </div>
      )}
      {editing && <LinksEditor row={row} />}
      {mayCurate && !editing && !row.duplicate_id && (
        <div className="pl-6 pb-2 flex gap-1.5">
          <button type="button" disabled={anotherEditing} onClick={() => startEditingLinks(row)} className="btn btn-sm btn-quiet" aria-label={`Edit links of ${title}`}>Edit links</button>
        </div>
      )}
    </div>
  );
}

/** The section body.  Opening it asks when nothing has been asked yet. */
export function ReferencesPanel() {
  const r = useJaato((s) => s.referenceCatalog);
  const sessionId = useJaato((s) => s.sessionId);
  useEffect(() => {
    if (sessionId && useJaato.getState().referenceCatalog.status === "idle") void refreshReferenceCatalog();
  }, [sessionId]);
  if (!sessionId) return <p className="m-0 px-3.5 py-2 text-[13px] text-text-muted">No session attached.</p>;
  if (r.status === "unsupported") {
    return <p className="m-0 px-3.5 py-2 text-[13px] text-text-muted">This daemon is older than protocol 1.33 and cannot list the reference catalog.</p>;
  }
  const mayCurate = r.mayCurate === true;
  return (
    <div className="px-3.5 py-2 text-[13px]">
      <div className="flex items-center gap-2 pb-1.5">
        <span className="text-[12px] text-text-muted">This workspace's reference catalog and how its entries link.</span>
        <span className="flex-1" />
        <button type="button" onClick={() => { void refreshReferenceCatalog(); }} className="btn btn-sm btn-quiet" aria-label="Refresh references">Refresh</button>
      </div>
      {r.status === "error" && <p className="m-0 py-1 text-error" role="alert">Could not read the catalog: {r.error}</p>}
      {r.notice && <p className={`m-0 py-1 ${r.notice.error ? "text-error" : r.notice.warning ? "text-warning" : "text-text-muted"}`} role="status">{r.notice.text}</p>}
      {r.mayCurate === false && r.rows.length > 0 && (
        <p className="m-0 py-1 text-[11px] text-text-muted">Only the owner of this workspace can change these links.</p>
      )}
      {r.status === "loading" && r.rows.length === 0 ? (
        <p className="m-0 py-2 text-text-muted">Loading…</p>
      ) : r.rows.length === 0 ? (
        <p className="m-0 py-2 text-text-muted">{r.status === "loaded" ? "The catalog is empty." : ""}</p>
      ) : (
        r.rows.map((row) => <ReferenceRow key={row.file} row={row} mayCurate={mayCurate} />)
      )}
      {r.unreadable.length > 0 && (
        <p className="m-0 py-1 text-[11px] text-warning">
          {r.unreadable.length} file{r.unreadable.length === 1 ? "" : "s"} in .jaato/references could not be shown: {r.unreadable.join(", ")}
        </p>
      )}
    </div>
  );
}
