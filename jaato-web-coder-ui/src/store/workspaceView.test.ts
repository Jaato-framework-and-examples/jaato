/**
 * The Files panel's reset point (#1189).  The load-bearing cases are the two
 * the issue names: a reset must survive a reconnect's snapshot, and a reset
 * taken against a monitor that no longer exists must be DROPPED rather than
 * compared -- a restarted counter compared against an old mark hides every
 * new change and leaves the panel empty with nothing saying why.
 */
import { describe, expect, it } from "vitest";
import { applyChanged, applySnapshot, markReset, visibleFiles, type WorkspaceNumbering } from "./workspaceView";

const empty: WorkspaceNumbering = { files: {}, seqs: {}, epoch: null, seq: 0, reset: null };

function changed(cur: WorkspaceNumbering, seq: number, changes: [string, string][], epoch = "e1") {
  return applyChanged(cur, { changes: changes.map(([path, status]) => ({ path, status })), seq, epoch });
}

describe("reset shows only what changes afterwards", () => {
  it("empties the panel, and a file that changes again comes back", () => {
    let n: WorkspaceNumbering = changed(empty, 1, [["a.py", "created"], ["b.py", "modified"]]);
    n = { ...n, reset: markReset(n) };
    expect(visibleFiles(n)).toEqual({});
    // b.py was already listed; touching it again is exactly what the reset is for.
    n = changed(n, 2, [["b.py", "modified"], ["c.py", "created"]]);
    expect(visibleFiles(n)).toEqual({ "b.py": "modified", "c.py": "created" });
  });

  it("keeps the full list, so showing everything brings it all back", () => {
    let n: WorkspaceNumbering = changed(empty, 1, [["a.py", "created"]]);
    n = { ...n, reset: markReset(n) };
    n = changed(n, 2, [["c.py", "created"]]);
    expect(visibleFiles({ ...n, reset: null })).toEqual({ "a.py": "created", "c.py": "created" });
  });
});

describe("a reconnect's snapshot", () => {
  it("keeps the reset when the snapshot is from the same monitor", () => {
    let n: WorkspaceNumbering = changed(empty, 1, [["a.py", "created"]]);
    n = { ...n, reset: markReset(n) };
    // While disconnected, the agent wrote c.py; the snapshot replaces the list wholesale.
    const out = applySnapshot(n, {
      files: [{ path: "a.py", status: "created" }, { path: "c.py", status: "created" }],
      seq: 2, epoch: "e1", seqs: { "a.py": 1, "c.py": 2 },
    });
    expect(out.voided).toBeNull();
    expect(out.reset).toEqual({ epoch: "e1", seq: 1 });
    expect(visibleFiles(out)).toEqual({ "c.py": "created" });
  });

  it("drops the reset, and says so, when the monitor was rebuilt (a session reload)", () => {
    let n: WorkspaceNumbering = changed(empty, 500, [["a.py", "created"]]);
    n = { ...n, reset: markReset(n) };
    // The new monitor counts from 0: compared against 500, this would hide everything.
    const out = applySnapshot(n, {
      files: [{ path: "a.py", status: "created" }, { path: "c.py", status: "created" }],
      seq: 3, epoch: "e2", seqs: { "c.py": 3 },
    });
    expect(out.reset).toBeNull();
    expect(out.voided).toMatch(/reloaded/);
    expect(Object.keys(visibleFiles(out)).sort()).toEqual(["a.py", "c.py"]);
  });

  it("drops the reset against a daemon that numbers nothing (before protocol 1.19)", () => {
    let n = applyChanged(empty, { changes: [{ path: "a.py", status: "created" }] });
    n = { ...n, reset: markReset(n) };
    // Between snapshots, local numbering still works...
    n = applyChanged(n, { changes: [{ path: "c.py", status: "created" }] });
    expect(visibleFiles(n)).toEqual({ "c.py": "created" });
    // ...and a snapshot, which says nothing about when, ends the reset out loud.
    const out = applySnapshot(n, { files: [{ path: "a.py", status: "created" }, { path: "c.py", status: "created" }] });
    expect(out.reset).toBeNull();
    expect(out.voided).toMatch(/1\.19/);
  });

  it("reads a path the snapshot gives no number for as older than any mark", () => {
    const n: WorkspaceNumbering = { ...changed(empty, 4, [["x.py", "created"]]), reset: { epoch: "e1", seq: 4 } };
    const out = applySnapshot(n, { files: [{ path: "restored.py", status: "modified" }], seq: 4, epoch: "e1", seqs: {} });
    expect(visibleFiles(out)).toEqual({});
  });
});

it("a changed event from a new monitor drops the reset too", () => {
  let n: WorkspaceNumbering = changed(empty, 500, [["a.py", "created"]]);
  n = { ...n, reset: markReset(n) };
  const out = changed(n, 1, [["c.py", "created"]], "e2");
  expect(out.reset).toBeNull();
  expect(out.voided).toMatch(/reloaded/);
});

it("the control: with no reset, the panel shows everything and nothing is voided", () => {
  const out = applySnapshot(changed(empty, 1, [["a.py", "created"]]), {
    files: [{ path: "a.py", status: "created" }], seq: 1, epoch: "e2", seqs: { "a.py": 1 },
  });
  expect(out.voided).toBeNull();
  expect(visibleFiles(out)).toEqual({ "a.py": "created" });
});
