/**
 * #1247: a file the user hid stays hidden across its own later changes.
 *
 * The Files panel's hidden set (``workspaceHidden``) stores a path id
 * (``entryId``) and the render re-derives that id from the file's key in
 * ``files`` on every change.  If the key is not canonical, a later report of
 * the same file under a different form (a leading ``./``) produces a
 * different id, the hidden match is lost, and the file reappears -- with the
 * divergent key lingering as a second row.  ``canonicalPath`` at the reducer
 * boundary keeps the key stable, so these drive the real reducer +
 * ``build`` / ``entryId`` / ``isHidden`` and assert the file stays hidden.
 */
import { describe, expect, it } from "vitest";
import { applyChanged, canonicalPath, visibleFiles, type WorkspaceNumbering } from "./workspaceView";
import { build, entryId, isHidden } from "@/components/panels/WorkspacePanel";

const empty: WorkspaceNumbering = { files: {}, seqs: {}, epoch: null, seq: 0, reset: null };

function changed(cur: WorkspaceNumbering, seq: number, changes: [string, string][], epoch = "e1") {
  return applyChanged(cur, { changes: changes.map(([path, status]) => ({ path, status })), seq, epoch });
}

/** The file-entry ids the panel would list, given the current files. */
function fileIds(files: Record<string, string>): string[] {
  const out: string[] = [];
  const walk = (node: ReturnType<typeof build>) => {
    for (const c of node.children.values()) {
      if (c.children.size) walk(c);
      else out.push(entryId(c));
    }
  };
  walk(build(files));
  return out;
}

/** The file ids the panel actually SHOWS (hidden ones dropped). */
function shownIds(files: Record<string, string>, hidden: readonly string[]): string[] {
  return fileIds(files).filter((id) => !isHidden(id, hidden));
}

describe("canonicalPath", () => {
  it("is a no-op for the relative paths the daemon actually sends", () => {
    expect(canonicalPath("logs/session.log")).toBe("logs/session.log");
    expect(canonicalPath("session.log")).toBe("session.log");
  });
  it("collapses the forms that break an exact-string match", () => {
    expect(canonicalPath("./logs/session.log")).toBe("logs/session.log");
    expect(canonicalPath("logs//session.log")).toBe("logs/session.log");
    expect(canonicalPath("logs/session.log/")).toBe("logs/session.log");
    expect(canonicalPath("logs/./session.log")).toBe("logs/session.log");
  });
  it("preserves a single leading slash for sandbox absolute keys", () => {
    expect(canonicalPath("/srv/corpus/x.txt")).toBe("/srv/corpus/x.txt");
    expect(canonicalPath("//srv/x")).toBe("/srv/x");
  });
});

describe("a hidden file stays hidden across its changes (#1247)", () => {
  it("stays hidden when re-reported under the same path", () => {
    let n = changed(empty, 1, [["logs/session.log", "created"]]);
    const hidden = [entryId({ path: "logs/session.log", children: { size: 0 } })];
    n = changed(n, 2, [["logs/session.log", "modified"]]);
    expect(shownIds(visibleFiles(n), hidden)).toEqual([]);
  });

  it("stays hidden when the same file is later reported with a leading ./", () => {
    // Hide it from its clean report...
    let n = changed(empty, 1, [["logs/session.log", "created"]]);
    const ids = fileIds(n.files);
    expect(ids).toEqual(["logs/session.log"]);
    const hidden = ids.slice(0, 1);
    // ...then the same file changes again, reported under a ./-prefixed form.
    n = changed(n, 2, [["./logs/session.log", "modified"]]);
    // One row, canonical, and still hidden -- no lingering ./ duplicate.
    expect(fileIds(n.files)).toEqual(["logs/session.log"]);
    expect(shownIds(visibleFiles(n), hidden)).toEqual([]);
  });

  it("leaves an unrelated file visible", () => {
    let n = changed(empty, 1, [["logs/session.log", "created"], ["src/main.ts", "modified"]]);
    const hidden = ["logs/session.log"];
    n = changed(n, 2, [["./logs/session.log", "modified"], ["src/main.ts", "modified"]]);
    expect(shownIds(visibleFiles(n), hidden)).toEqual(["src/main.ts"]);
  });

  it("unhide brings it back", () => {
    let n = changed(empty, 1, [["logs/session.log", "created"]]);
    n = changed(n, 2, [["./logs/session.log", "modified"]]);
    // Removing the id from the hidden set (the toggle) shows it again.
    expect(shownIds(visibleFiles(n), [])).toEqual(["logs/session.log"]);
  });
});
