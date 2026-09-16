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
 * Entry ids match the TUI's: a directory is its path with a trailing ``/``,
 * a file is its workspace-relative path.
 */
import { useMemo } from "react";
import { useJaato } from "@/store/store";
import { toggleWorkspaceIgnore } from "@/app/actions";

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

/** Files (not directories) that the hide set removes from view. */
export function countHiddenFiles(files: Record<string, string>, hidden: readonly string[]): number {
  return Object.keys(files).filter((p) => isHidden(p, hidden)).length;
}

const CHANGE_CLS: Record<string, string> = { created: "text-success", added: "text-success", modified: "text-warning", deleted: "text-error" };

function RowActions({ id, hidden, ignored }: { id: string; hidden: boolean; ignored: boolean | undefined }) {
  const toggleHidden = useJaato((s) => s.toggleWorkspaceHidden);
  const cls = "text-[10px] px-1 rounded hover:bg-surface-2 text-text-muted hover:text-text opacity-0 group-hover:opacity-100 focus:opacity-100";
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

function Tree({ node, depth, hidden, showHidden, ignored }: { node: Node; depth: number; hidden: readonly string[]; showHidden: boolean; ignored: Record<string, boolean> }) {
  const entries = [...node.children.values()].sort((a, b) => (a.children.size ? 0 : 1) - (b.children.size ? 0 : 1) || a.name.localeCompare(b.name));
  return (
    <ul>
      {entries.map((n) => {
        const id = entryId(n);
        const isDir = n.children.size > 0;
        const hid = isHidden(id, hidden);
        if (hid && !showHidden) return null;
        const label = isDir
          ? <span className="text-text-muted">▾ {n.name}/</span>
          : <span className={CHANGE_CLS[n.change ?? ""] ?? ""} title={n.change}>{n.change === "deleted" ? "−" : n.change === "created" || n.change === "added" ? "+" : "~"} {n.name}</span>;
        return (
          <li key={n.path} className="font-mono text-xs leading-5">
            <div className={`group flex items-center gap-1 pr-1 ${hid ? "opacity-50" : ""}`} style={{ paddingLeft: depth * 12 }} data-hidden={hid || undefined}>
              {hid && <span className="text-text-muted" title="Hidden">H</span>}
              {label}
              {ignored[id] && <span className="text-text-muted text-[10px]" title="In .gitignore">i</span>}
              <RowActions id={id} hidden={hid} ignored={ignored[id]} />
            </div>
            {isDir && <Tree node={n} depth={depth + 1} hidden={hidden} showHidden={showHidden} ignored={ignored} />}
          </li>
        );
      })}
    </ul>
  );
}

export function WorkspacePanel() {
  const files = useJaato((s) => s.workspaceFiles);
  const hidden = useJaato((s) => s.workspaceHidden);
  const showHidden = useJaato((s) => s.workspaceShowHidden);
  const toggleShowHidden = useJaato((s) => s.toggleWorkspaceShowHidden);
  const ignored = useJaato((s) => s.workspaceIgnored);
  const notice = useJaato((s) => s.workspaceNotice);
  const tree = useMemo(() => build(files), [files]);
  const total = Object.keys(files).length;
  const hiddenCount = useMemo(() => countHiddenFiles(files, hidden), [files, hidden]);
  const visible = total - hiddenCount;
  return (
    <div className="p-3">
      <div className="flex items-baseline justify-between mb-2 text-[13px]">
        <span className="font-semibold">Workspace changes</span>
        <span className="text-xs text-text-muted">{showHidden ? total : visible}</span>
      </div>
      {notice && (
        <div role="status" className={`text-[11px] mb-2 ${notice.error ? "text-error" : "text-text-muted"}`}>{notice.text}</div>
      )}
      {total === 0 ? <div className="text-xs text-text-muted italic">No files changed yet.</div> : <Tree node={tree} depth={0} hidden={hidden} showHidden={showHidden} ignored={ignored} />}
      {hiddenCount > 0 && (
        <div className="mt-2 text-[11px] text-text-muted flex items-center gap-2">
          <span>{hiddenCount} hidden</span>
          <button type="button" className="underline hover:text-text" onClick={toggleShowHidden}>{showHidden ? "hide hidden" : "show hidden"}</button>
        </div>
      )}
      {total > 0 && <div className="mt-2 text-[10px] text-text-muted">Hover an entry: <b>hide</b> drops it from this view · <b>ignore</b> toggles its line in .gitignore.</div>}
    </div>
  );
}
