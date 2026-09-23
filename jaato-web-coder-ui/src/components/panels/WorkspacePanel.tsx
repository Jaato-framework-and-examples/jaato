/**
 * Files created / modified this session, as a compact tree (the TUI's
 * Ctrl+W panel), with the two per-entry actions that panel binds to keys:
 *
 * - **hide** (TUI ``h``): drop the entry from THIS view.  Client-side only —
 *   a directory hides everything under it, "show hidden" brings the set
 *   back dimmed with an ``H`` marker so an entry can be unhidden, and the
 *   set is dropped with the session.
 * - **ignore** (TUI ``i``): toggle the entry's line in the workspace's
 *   ``.gitignore``.  The TUI writes the file itself; here it goes through
 *   the daemon's ``workspace.ignore`` verb, whose answer is rendered as the
 *   notice under the header.  The daemon's monitor stops reporting the
 *   entry from then on, but does not remove what the panel already shows —
 *   pair it with hide for that.
 *
 * - **reset** (TUI ``Delete``, ``workspace_clear``): empty the panel so it
 *   shows only files that change from now on -- including a file it already
 *   listed, if the agent touches it again.  **show everything** drops the
 *   reset.  Per viewer; it changes what this panel shows and nothing on the
 *   daemon.  The reset survives a reconnect because the daemon numbers every
 *   change (``store/workspaceView.ts``), and is dropped, with a notice, when
 *   it cannot be honoured.
 *
 * - **collapse** (TUI Left/Right): the arrow in front of a directory folds
 *   it to one line carrying how many files it holds.  Every directory
 *   starts expanded and a reset expands them all again, as in the TUI.
 *
 * Entry ids match the TUI's: a directory is its path with a trailing ``/``,
 * a file is its workspace-relative path.  The section header (``Files``
 * and the count) is the rail's; this is the body.
 */
import { useMemo } from "react";
import { useJaato } from "@/store/store";
import { toggleWorkspaceIgnore } from "@/app/actions";
import { visibleFiles } from "@/store/workspaceView";

interface Node { name: string; path: string; change?: string; children: Map<string, Node> }

function build(files: Record<string, string>): Node {
  const root: Node = { name: "", path: "", children: new Map() };
  for (const [p, change] of Object.entries(files)) {
    const parts = p.split("/").filter(Boolean);
    let cur = root;
    parts.forEach((part, i) => {
      let n = cur.children.get(part);
      if (!n) { n = { name: part, path: parts.slice(0, i + 1).join("/"), children: new Map() }; cur.children.set(part, n); }
      if (i === parts.length - 1) n.change = change;
      cur = n;
    });
  }
  return root;
}

/** The TUI's hide-set identifier: directories carry a trailing slash. */
export function entryId(n: Pick<Node, "path"> & { children: { size: number } }): string {
  return n.children.size ? `${n.path}/` : n.path;
}

/** True when ``id`` is hidden directly or lives under a hidden directory. */
export function isHidden(id: string, hidden: readonly string[]): boolean {
  return hidden.some((h) => h === id || (h.endsWith("/") && id.startsWith(h)));
}

/** Files under a directory node, however deep. */
export function countFiles(n: Pick<Node, "children">): number {
  let total = 0;
  for (const c of n.children.values()) total += c.children.size ? countFiles(c) : 1;
  return total;
}

/** Files (not directories) that the hide set removes from view. */
export function countHiddenFiles(files: Record<string, string>, hidden: readonly string[]): number {
  return Object.keys(files).filter((p) => isHidden(p, hidden)).length;
}

const CHANGE_CLS: Record<string, string> = { created: "text-success", added: "text-success", modified: "text-warning", deleted: "text-error" };

function RowActions({ id, hidden, ignored }: { id: string; hidden: boolean; ignored: boolean | undefined }) {
  const toggleHidden = useJaato((s) => s.toggleWorkspaceHidden);
  // Always drawn, dimmed until the row is hovered or a button focused: a
  // touch screen has no hover, so an action that only appears on hover is
  // unreachable there -- the tablet the panel was first tried on could not
  // hide or ignore anything.
  const cls = "font-heading uppercase tracking-[0.08em] text-[10px] px-1 text-text-muted hover:text-steel opacity-60 group-hover:opacity-100 focus:opacity-100";
  return (
    <span className="ml-auto shrink-0 flex gap-1">
      <button type="button" className={cls} onClick={() => toggleHidden(id)} aria-label={`${hidden ? "Unhide" : "Hide"} ${id}`} title={hidden ? "Show this entry again" : "Hide this entry from the panel (this session only)"}>
        {hidden ? "unhide" : "hide"}
      </button>
      <button type="button" className={cls} onClick={() => { toggleWorkspaceIgnore(id).catch(() => undefined); }} aria-label={`${ignored ? "Remove" : "Add"} ${id} ${ignored ? "from" : "to"} .gitignore`} title={ignored ? "Remove this entry from the workspace .gitignore" : "Add this entry to the workspace .gitignore"}>
        {ignored ? "unignore" : "ignore"}
      </button>
    </span>
  );
}

interface TreeProps { node: Node; depth: number; hidden: readonly string[]; showHidden: boolean; ignored: Record<string, boolean>; collapsed: readonly string[] }

function DirToggle({ node, id, folded }: { node: Node; id: string; folded: boolean }) {
  const toggle = useJaato((s) => s.toggleWorkspaceCollapsed);
  return (
    <button type="button" className="text-text-muted hover:text-steel text-left" onClick={() => toggle(id)} aria-expanded={!folded} aria-label={`${folded ? "Expand" : "Collapse"} ${id}`}>
      {folded ? "▸" : "▾"} {node.name}/{folded && <span className="ml-1.5 text-[10px]">({countFiles(node)})</span>}
    </button>
  );
}

function Tree({ node, depth, hidden, showHidden, ignored, collapsed }: TreeProps) {
  const entries = [...node.children.values()].sort((a, b) => (a.children.size ? 0 : 1) - (b.children.size ? 0 : 1) || a.name.localeCompare(b.name));
  return (
    <ul className="list-none m-0 p-0">
      {entries.map((n) => {
        const id = entryId(n);
        const isDir = n.children.size > 0;
        const hid = isHidden(id, hidden);
        if (hid && !showHidden) return null;
        const folded = isDir && collapsed.includes(id);
        const label = isDir
          ? <DirToggle node={n} id={id} folded={folded} />
          : <span className={CHANGE_CLS[n.change ?? ""] ?? ""} title={n.change}>{n.change === "deleted" ? "−" : n.change === "created" || n.change === "added" ? "+" : "~"} {n.name}</span>;
        return (
          <li key={n.path} className="font-mono text-[11.5px] leading-[1.8]">
            <div className={`group flex items-center gap-1 pr-1 ${hid ? "opacity-50" : ""}`} style={{ paddingLeft: depth * 12 }} data-hidden={hid || undefined}>
              {hid && <span className="text-text-muted" title="Hidden">H</span>}
              {label}
              {ignored[id] && <span className="text-text-muted text-[10px]" title="In .gitignore">i</span>}
              <RowActions id={id} hidden={hid} ignored={ignored[id]} />
            </div>
            {isDir && !folded && <Tree node={n} depth={depth + 1} hidden={hidden} showHidden={showHidden} ignored={ignored} collapsed={collapsed} />}
          </li>
        );
      })}
    </ul>
  );
}

/** The Files panel's files after its reset point, if one is set. */
export function useVisibleWorkspaceFiles(): Record<string, string> {
  const files = useJaato((s) => s.workspaceFiles);
  const seqs = useJaato((s) => s.workspaceSeqs);
  const reset = useJaato((s) => s.workspaceReset);
  return useMemo(() => visibleFiles({ files, seqs, reset }), [files, seqs, reset]);
}

function ResetBar({ total, isReset }: { total: number; isReset: boolean }) {
  const reset = useJaato((s) => s.resetWorkspaceView);
  const showAll = useJaato((s) => s.showAllWorkspace);
  const all = useJaato((s) => Object.keys(s.workspaceFiles).length);
  if (!isReset && total === 0) return null;
  return (
    <div className="mb-2 text-[11px] text-text-muted flex items-center gap-2 flex-wrap">
      {isReset && <span>Showing changes since the reset{all > total ? ` (${all - total} earlier not shown)` : ""}.</span>}
      {isReset && <button type="button" className="link" onClick={showAll}>show everything</button>}
      {total > 0 && (
        <button type="button" className="link" onClick={reset} title="Empty the panel; only files that change from now on will appear, including ones already listed">
          {isReset ? "reset again" : "reset"}
        </button>
      )}
    </div>
  );
}

export function WorkspacePanel() {
  const files = useVisibleWorkspaceFiles();
  const isReset = useJaato((s) => s.workspaceReset !== null);
  const hidden = useJaato((s) => s.workspaceHidden);
  const showHidden = useJaato((s) => s.workspaceShowHidden);
  const toggleShowHidden = useJaato((s) => s.toggleWorkspaceShowHidden);
  const collapsed = useJaato((s) => s.workspaceCollapsed);
  const ignored = useJaato((s) => s.workspaceIgnored);
  const notice = useJaato((s) => s.workspaceNotice);
  const tree = useMemo(() => build(files), [files]);
  const total = Object.keys(files).length;
  const hiddenCount = useMemo(() => countHiddenFiles(files, hidden), [files, hidden]);
  return (
    <div className="px-3.5 py-3">
      {notice && (
        <div role="status" className={`text-[11px] mb-2 ${notice.error ? "text-error" : "text-text-muted"}`}>{notice.text}</div>
      )}
      <ResetBar total={total} isReset={isReset} />
      {total === 0 ? <div className="text-xs text-text-muted italic">{isReset ? "No files changed since the reset." : "No files changed yet."}</div> : <Tree node={tree} depth={0} hidden={hidden} showHidden={showHidden} ignored={ignored} collapsed={collapsed} />}
      {hiddenCount > 0 && (
        <div className="mt-2 text-[11px] text-text-muted flex items-center gap-2">
          <span>{hiddenCount} hidden</span>
          <button type="button" className="link" onClick={toggleShowHidden}>{showHidden ? "hide hidden" : "show hidden"}</button>
        </div>
      )}
      {total > 0 && <div className="mt-2.5 text-xs text-text-muted"><span className="font-heading uppercase tracking-[0.08em] text-[10px]">hide</span> drops an entry from the panel; <span className="font-heading uppercase tracking-[0.08em] text-[10px]">ignore</span> adds it to <span className="font-mono">.gitignore</span>.</div>}
    </div>
  );
}
