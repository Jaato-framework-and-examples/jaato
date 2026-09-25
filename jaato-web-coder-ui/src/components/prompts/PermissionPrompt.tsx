/**
 * Inline permission plate (design frame 05), regrouped per jaato/#1304 §2
 * (the final rollout item from the closed issue #1304): a full-width,
 * warning-edged plate carrying a risk tag and a plain-language question,
 * the server's pre-formatted ``prompt_lines`` (a diff for file edits, at
 * full measure) or the raw arguments when none were sent, any analyzer
 * warnings, then the response options -- **Allow** and **Deny** as the
 * two primary buttons (unchanged from before this regroup: same markup,
 * same accessible name, so nothing that already types "y" / "n" or
 * clicks them broke), an **"Allow for…"** dropdown for the scoped grants
 * (turn / idle / always / all, plus the destructive ``never``), a
 * **note** field for the ``c`` / ``yc`` comment variants (collapsed until
 * asked for), and a **hidden** group (``once`` / edit) behind a "More"
 * link. ``protocol/permissionCard.ts`` holds every pure grouping /
 * wording decision, so this component is display only.
 *
 * **Not shown: cwd / confinement / network.** No wire field carries them
 * -- a tool call's arguments never include the workspace root or the
 * AppArmor posture, and inventing a plausible-looking line here would be
 * exactly the kind of unverifiable claim this codebase's own conventions
 * warn against (see CLAUDE.md's "A cut neighbourhood says so" and
 * "Never Confined Is Not Never Served" sections on never fabricating a
 * number nobody measured). If a future server change puts that
 * information on ``PermissionRequestedEvent``, it renders here; until
 * then this card says only what the daemon told it.
 */
import { useEffect, useRef, useState } from "react";
import type { PendingPermission, PermissionOption } from "@/store/types";
import { useJaato } from "@/store/store";
import { DiffLines } from "@/components/output/JMarkup";
import { Plate } from "@/components/layout/Plate";
import { resolveToolArgs } from "@/protocol/toolIds";
import { resolveToolClass } from "@/protocol/toolClass";
import { execTitle } from "@/protocol/toolPreview";
import { groupPermissionOptions, plainQuestion, riskTag } from "@/protocol/permissionCard";

const TONE_CLASS: Record<"muted" | "warning" | "steel", string> = {
  muted: "text-text-muted",
  warning: "text-warning",
  steel: "text-steel",
};

function optionTitle(o: PermissionOption): string | undefined {
  return o.description ?? o.action ?? undefined;
}

/** The primary Allow/Deny button -- byte-identical markup to the
 * pre-regroup card, so an existing "yes y" / "no n" assertion still
 * matches. */
function PrimaryButton({ o, danger, onClick }: { o: PermissionOption; danger?: boolean; onClick: () => void }) {
  return (
    <button type="button" onClick={onClick} title={optionTitle(o)} className={`btn ${danger ? "btn-danger" : "btn-primary"}`}>
      {o.label ?? o.key} <span className="key">{o.key}</span>
    </button>
  );
}

/** The "Allow for…" dropdown: the scoped grants plus the destructive
 * ``never``, in the same outside-click / Escape pattern as
 * ``PermissionsPlate`` -- the trigger sits OUTSIDE the popover, so the
 * outside-click listener must exempt it or a click that opens it also
 * closes it on the same gesture (that defect, and why a plain
 * ``fireEvent.click`` in a test cannot see it, is recorded on
 * ``PermissionsPlate.test.tsx``). */
function AllowForDropdown({
  durations, destructive, onPick,
}: {
  durations: PermissionOption[];
  destructive: PermissionOption | null;
  onPick: (key: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") { e.preventDefault(); setOpen(false); } };
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (ref.current?.contains(t)) return;
      if (triggerRef.current?.contains(t)) return;
      setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("mousedown", onDown);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("mousedown", onDown);
    };
  }, [open]);

  if (durations.length === 0 && !destructive) return null;

  return (
    <div className="relative">
      <button
        ref={triggerRef}
        type="button"
        className="btn"
        aria-expanded={open}
        aria-haspopup="true"
        onClick={() => setOpen((v) => !v)}
      >
        Allow for… <span className="key">▾</span>
      </button>
      {open && (
        <div ref={ref} className="absolute bottom-full mb-1.5 right-0 z-30 w-[15rem]">
          <Plate edge="steel" className="bg-surface shadow-lg p-1.5 flex flex-col gap-0.5" role="menu" aria-label="Allow for a scope">
            {durations.map((o) => (
              <button
                key={o.key}
                type="button"
                role="menuitem"
                title={optionTitle(o)}
                className="btn justify-between"
                onClick={() => { setOpen(false); onPick(o.key); }}
              >
                <span>{o.label ?? o.key}</span>
                <span className="key">{o.key}</span>
              </button>
            ))}
            {destructive && (
              <button
                type="button"
                role="menuitem"
                title={optionTitle(destructive)}
                className="btn btn-danger justify-between"
                onClick={() => { setOpen(false); onPick(destructive.key); }}
              >
                <span>{destructive.label ?? destructive.key}</span>
                <span className="key">{destructive.key}</span>
              </button>
            )}
          </Plate>
        </div>
      )}
    </div>
  );
}

export function PermissionPrompt({ p, onRespond }: { p: PendingPermission; onRespond: (key: string) => void }) {
  const focus = useJaato((s) => s.focusPermission);
  const toolIdNames = useJaato((s) => s.toolIdNames);
  const [noteOpen, setNoteOpen] = useState(false);
  const [note, setNote] = useState("");
  const [moreOpen, setMoreOpen] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const inField = t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT");
      if (e.key === "Tab" && !inField) {
        e.preventDefault();
        const n = p.options.length || 1;
        focus(p.requestId, (p.focus + (e.shiftKey ? -1 : 1) + n) % n);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [p, focus]);

  const displayName = toolIdNames[p.toolName] ?? p.toolName;
  const argEntries = Object.entries(resolveToolArgs(p.toolArgs ?? {}, toolIdNames));
  const level = p.warningLevel ?? "warning";
  const rawOptions = p.options.length ? p.options : [{ key: "y", label: "yes" }, { key: "n", label: "no" }];
  const grouped = groupPermissionOptions(rawOptions);
  const cls = resolveToolClass(displayName, p.toolClass);
  const tag = riskTag(cls);
  const commandLine = cls === "exec" ? execTitle(displayName, resolveToolArgs(p.toolArgs ?? {}, toolIdNames)) : null;
  // The command line already shows the effect for an exec-class ask --
  // rendering the raw args grid too would duplicate that same text.
  const hasEffect = p.promptLines.length > 0 || (argEntries.length > 0 && !commandLine);

  const respond = (key: string) => onRespond(key);
  const respondAllow = () => {
    if (noteOpen && note.trim() && grouped.allowComment) { onRespond(`yc:${note.trim()}`); return; }
    if (grouped.allow) respond(grouped.allow.key);
  };
  const respondDeny = () => {
    if (noteOpen && note.trim() && grouped.denyComment) { onRespond(`c:${note.trim()}`); return; }
    if (grouped.deny) respond(grouped.deny.key);
  };

  return (
    <Plate edge="warning" className="my-3" role="group" aria-label={`Permission request for ${p.toolName}`}>
      <div className="px-3.5 py-2.5 flex items-center gap-3 border-b hairline">
        <span className="text-warning">⚠</span>
        <span className="kicker text-warning">Permission requested for</span>
        <span className="font-mono text-sm">{p.toolName}</span>
        <span className={`kicker ${TONE_CLASS[tag.tone]}`} title="This tool's coarse risk class">{tag.label}</span>
        <span className="flex-1" />
        <span className="font-mono text-[11px] text-text-muted">agent {p.agentId}</span>
      </div>
      <p className="px-3.5 pt-2 pb-0 m-0 text-[13px]">{plainQuestion(cls)}</p>
      {commandLine && (
        <div className="px-3.5 pt-1.5"><code className="code-block px-2 py-1 text-[13px]">{commandLine}</code></div>
      )}
      {hasEffect && (
        p.promptLines.length > 0 ? (
          p.formatHint === "diff" ? <DiffLines lines={p.promptLines} /> : <pre className="code-block whitespace-pre-wrap px-3.5 py-2.5 max-h-[40vh] overflow-auto">{p.promptLines.join("\n")}</pre>
        ) : (
          <dl className="px-3.5 py-2.5 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-[13px] font-mono max-h-[40vh] overflow-auto m-0">
            {argEntries.map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-text-muted">{k}</dt>
                <dd className="whitespace-pre-wrap break-words m-0">{typeof v === "string" ? v : JSON.stringify(v, null, 2)}</dd>
              </div>
            ))}
          </dl>
        )
      )}
      {p.warnings && (
        <div className={`px-3.5 py-2 text-[13px] whitespace-pre-wrap border-t hairline ${level === "error" ? "text-error" : level === "info" ? "text-text-muted" : "text-warning"}`}>{p.warnings}</div>
      )}
      <div className="px-3.5 py-3 flex flex-wrap items-center gap-2.5 border-t hairline">
        {grouped.allow && <PrimaryButton o={grouped.allow} onClick={respondAllow} />}
        <AllowForDropdown durations={grouped.durations} destructive={grouped.destructive} onPick={respond} />
        {(grouped.allowComment || grouped.denyComment) && (
          <button type="button" className="link text-[12px]" onClick={() => setNoteOpen((v) => !v)}>
            {noteOpen ? "Cancel note" : "+ Add a note"}
          </button>
        )}
        {grouped.hidden.length > 0 && (
          <button type="button" className="link text-[12px]" onClick={() => setMoreOpen((v) => !v)}>
            {moreOpen ? "Fewer options" : "More…"}
          </button>
        )}
        {grouped.other.map((o) => (
          <button key={o.key} type="button" className="btn btn-sm" title={optionTitle(o)} onClick={() => respond(o.key)}>
            {o.label ?? o.key} <span className="key">{o.key}</span>
          </button>
        ))}
        <span className="flex-1" />
        {grouped.deny && <PrimaryButton o={grouped.deny} danger onClick={respondDeny} />}
      </div>
      {moreOpen && grouped.hidden.length > 0 && (
        <div className="px-3.5 pb-3 flex flex-wrap items-center gap-2.5 border-t hairline pt-2.5">
          {grouped.hidden.map((o) => (
            <button key={o.key} type="button" className="btn btn-sm btn-quiet" title={optionTitle(o)} onClick={() => respond(o.key)}>
              {o.label ?? o.key} <span className="key">{o.key}</span>
            </button>
          ))}
        </div>
      )}
      {noteOpen && (
        <div className="px-3.5 pb-3 border-t hairline pt-2.5">
          <label className="block text-[11px] text-text-muted mb-1" htmlFor={`note-${p.requestId}`}>
            Feedback the model will see with your decision
          </label>
          <textarea
            id={`note-${p.requestId}`}
            className="w-full text-[13px] font-mono px-2 py-1.5 border hairline rounded-none bg-transparent"
            rows={2}
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Why -- the model reads this back"
          />
        </div>
      )}
    </Plate>
  );
}
