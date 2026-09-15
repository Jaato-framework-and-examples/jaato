/** Files created / modified this session, as a compact tree (TUI Ctrl+W panel). */
import { useMemo } from "react";
import { useJaato } from "@/store/store";

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

const CHANGE_CLS: Record<string, string> = { created: "text-success", added: "text-success", modified: "text-warning", deleted: "text-error" };

function Tree({ node, depth }: { node: Node; depth: number }) {
  const entries = [...node.children.values()].sort((a, b) => (a.children.size ? 0 : 1) - (b.children.size ? 0 : 1) || a.name.localeCompare(b.name));
  return (
    <ul>
      {entries.map((n) => (
        <li key={n.path} style={{ paddingLeft: depth * 12 }} className="font-mono text-xs leading-5">
          {n.children.size ? <span className="text-text-muted">▾ {n.name}/</span> : (
            <span className={CHANGE_CLS[n.change ?? ""] ?? ""} title={n.change}>{n.change === "deleted" ? "−" : n.change === "created" || n.change === "added" ? "+" : "~"} {n.name}</span>
          )}
          {n.children.size > 0 && <Tree node={n} depth={depth + 1} />}
        </li>
      ))}
    </ul>
  );
}

export function WorkspacePanel() {
  const files = useJaato((s) => s.workspaceFiles);
  const tree = useMemo(() => build(files), [files]);
  const count = Object.keys(files).length;
  return (
    <div className="p-3">
      <div className="flex items-baseline justify-between mb-2 text-[13px]"><span className="font-semibold">Workspace changes</span><span className="text-xs text-text-muted">{count}</span></div>
      {count === 0 ? <div className="text-xs text-text-muted italic">No files changed yet.</div> : <Tree node={tree} depth={0} />}
    </div>
  );
}
