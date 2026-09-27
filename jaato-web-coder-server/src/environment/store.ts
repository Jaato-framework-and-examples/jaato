/**
 * Declined proposals, remembered per user and workspace
 * (docs/design/web-coder-environment-bootstrap.md §9).
 *
 * A proposal the user dismissed ("Node 22 detected. Bind it?" → Not now) is
 * not shown again for that ``(OIDC sub, workspace)``.  Nothing secret lives
 * here, but the file names a user's workspaces, so it is written 0600 and
 * atomically, like the other stores.  A missing or unreadable file is an
 * empty store: forgetting a decline shows a chip again, which is harmless.
 */
import { randomBytes } from "node:crypto";
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { isToolId, type ToolId } from "./catalog.js";

type Declines = Record<string, Record<string, ToolId[]>>;

export class FileEnvironmentStore {
  private readonly _path: string;
  private _declines: Declines;

  constructor(path: string) {
    this._path = path;
    this._declines = FileEnvironmentStore._load(path);
  }

  private static _load(path: string): Declines {
    try {
      const raw = JSON.parse(readFileSync(path, "utf8")) as { declines?: unknown };
      const out: Declines = {};
      const d = raw?.declines;
      if (!d || typeof d !== "object") return out;
      for (const [sub, byWs] of Object.entries(d as Record<string, unknown>)) {
        if (!byWs || typeof byWs !== "object") continue;
        out[sub] = {};
        for (const [ws, tools] of Object.entries(byWs as Record<string, unknown>)) {
          if (Array.isArray(tools)) out[sub]![ws] = tools.filter(isToolId);
        }
      }
      return out;
    } catch { return {}; }
  }

  declined(sub: string, workspace: string): ToolId[] {
    return [...(this._declines[sub]?.[workspace] ?? [])];
  }

  decline(sub: string, workspace: string, tool: ToolId): void {
    const byWs = (this._declines[sub] ??= {});
    const list = (byWs[workspace] ??= []);
    if (!list.includes(tool)) { list.push(tool); this._save(); }
  }

  /** Forget a decline: binding the tool later means the user changed their mind. */
  undecline(sub: string, workspace: string, tool: ToolId): void {
    const list = this._declines[sub]?.[workspace];
    if (!list || !list.includes(tool)) return;
    this._declines[sub]![workspace] = list.filter((t) => t !== tool);
    this._save();
  }

  private _save(): void {
    mkdirSync(dirname(this._path), { recursive: true, mode: 0o700 });
    const tmp = join(dirname(this._path), `.${randomBytes(6).toString("hex")}.tmp`);
    writeFileSync(tmp, `${JSON.stringify({ declines: this._declines }, null, 2)}\n`, { mode: 0o600 });
    renameSync(tmp, this._path);
  }
}
