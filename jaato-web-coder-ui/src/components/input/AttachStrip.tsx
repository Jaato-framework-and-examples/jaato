/**
 * The attachment strip: an "Attach files" control, the folder the next
 * files go into, and one chip per attached file with its state — queued
 * (○, waiting for a workspace), staging (● pulsing, with an indeterminate
 * progress bar), staged (✓) or failed (✗, the reason in the title).
 * Rendered above the composer during a session and on the session picker
 * before one exists; both feed ``app/staging.ts`` and read the store's
 * ``uploads``.
 *
 * The strip shows only the ACTIVE context's uploads (#1250): an upload
 * carries the session (or picker) it was staged into, and the strip filters
 * to ``uploadScope(state)``, so a file attached in session A does not follow
 * you into an unrelated session B.
 *
 * ``×`` drops a chip from the strip only — a file already staged stays in
 * the workspace (the Files panel still lists it), one still queued is
 * forgotten before any bytes are sent.
 */
import { useRef, useState } from "react";
import { attachFiles, discardUpload } from "@/app/staging";
import { formatSize } from "@/protocol/attachments";
import { uploadScope, useJaato } from "@/store/store";
import type { StagedUpload } from "@/store/types";

const GLYPH: Record<StagedUpload["status"], { text: string; cls: string }> = {
  queued: { text: "○", cls: "text-text-muted" },
  staging: { text: "●", cls: "text-primary pulse" },
  staged: { text: "✓", cls: "text-success" },
  failed: { text: "✗", cls: "text-error" },
};

export interface AttachStripProps {
  /** One line under the controls saying where the files go and when. */
  hint: string;
  /** Show the strip's frame even with no chips (the picker's dropzone); the composer shows it only once something is attached. */
  always?: boolean;
}

export function AttachStrip({ hint, always }: AttachStripProps) {
  // Only the active context's uploads (#1250): a file attached in session A
  // must not show above session B's composer.  Both selectors return stable
  // references, so the filter runs on state change, not every render.
  const allUploads = useJaato((s) => s.uploads);
  const sessionId = useJaato((s) => s.sessionId);
  const uploads = allUploads.filter((u) => u.scope === uploadScope({ sessionId }));
  const [folder, setFolder] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  if (!always && uploads.length === 0) {
    return (
      <input ref={inputRef} type="file" multiple className="hidden" aria-label="Attach files" data-attach-input onChange={(e) => { attachFiles(Array.from(e.target.files ?? []), folder); e.target.value = ""; }} />
    );
  }
  return (
    <div className="plate plate-ground mb-2 text-[13px]" role="group" aria-label="Attached files">
      <input ref={inputRef} type="file" multiple className="hidden" aria-label="Attach files" data-attach-input onChange={(e) => { attachFiles(Array.from(e.target.files ?? []), folder); e.target.value = ""; }} />
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 px-3 py-2 border-b hairline">
        <button type="button" onClick={() => inputRef.current?.click()} className="btn btn-sm btn-steel">Attach files</button>
        <label className="flex items-center gap-2 text-text-muted">
          <span className="kicker kicker-muted">into</span>
          <input value={folder} onChange={(e) => setFolder(e.target.value)} placeholder="workspace root" aria-label="Folder for the next files" className="input input-mono text-[12px] py-0.5 w-40" />
        </label>
        <span className="text-[12px] text-text-muted">{hint}</span>
      </div>
      {uploads.length > 0 && (
        <ul className="flex flex-wrap gap-1.5 px-3 py-2" aria-label="Attachments">
          {uploads.map((u) => (
            <li key={u.id} data-status={u.status} title={u.error ?? u.path} className={`flex flex-col gap-0.5 border hairline px-2 py-0.5 font-mono text-[12px] ${u.status === "failed" ? "border-error" : ""}`}>
              <div className="flex items-center gap-1.5">
                <span className={GLYPH[u.status].cls} aria-label={u.status}>{GLYPH[u.status].text}</span>
                <span className="max-w-[18rem] truncate">{u.path}</span>
                <span className="text-text-muted">{formatSize(u.size)}</span>
                {u.status === "failed" && <span className="text-error truncate max-w-[16rem]">{u.error}</span>}
                <button type="button" onClick={() => discardUpload(u.id)} aria-label={`Remove ${u.path}`} className="text-text-muted hover:text-error px-0.5">×</button>
              </div>
              {u.status === "staging" && (
                // No byte-level progress from the transport (the batch's
                // frames are handed over in one go), so an indeterminate
                // sweep rather than a faked percentage.  It ends when the
                // status leaves "staging" -- ✓ staged, ✗ failed, or the
                // #1248 staging timeout, which resolves to failed.
                <div className="bar bar-indeterminate w-full" role="progressbar" aria-label="Staging" aria-valuetext="staging" data-testid="staging-bar"><span /></div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Files carried by a drop or a paste, or none — a text paste is not an attachment. */
export function filesFromTransfer(dt: DataTransfer | null | undefined): File[] {
  if (!dt) return [];
  const out: File[] = [];
  for (const item of Array.from(dt.items ?? [])) {
    if (item.kind !== "file") continue;
    const f = item.getAsFile();
    if (f) out.push(f);
  }
  if (!out.length && dt.files?.length) out.push(...Array.from(dt.files));
  return out;
}
