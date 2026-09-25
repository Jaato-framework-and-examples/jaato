import { describe, expect, it } from "vitest";
import { groupPermissionOptions, plainQuestion, riskTag } from "./permissionCard";
import type { PermissionOption } from "@/store/types";

const FULL_OPTIONS: PermissionOption[] = [
  { key: "y", label: "yes" },
  { key: "n", label: "no" },
  { key: "a", label: "always" },
  { key: "t", label: "turn" },
  { key: "i", label: "idle" },
  { key: "once", label: "once" },
  { key: "never", label: "never" },
  { key: "all", label: "all" },
  { key: "c", label: "deny-comment" },
  { key: "yc", label: "allow-comment" },
];

describe("groupPermissionOptions", () => {
  it("sorts the full DEFAULT_PERMISSION_OPTIONS vocabulary into its five groups", () => {
    const g = groupPermissionOptions(FULL_OPTIONS);
    expect(g.allow?.key).toBe("y");
    expect(g.deny?.key).toBe("n");
    expect(g.durations.map((o) => o.key)).toEqual(["t", "i", "a", "all"]);
    expect(g.destructive?.key).toBe("never");
    expect(g.allowComment?.key).toBe("yc");
    expect(g.denyComment?.key).toBe("c");
    expect(g.hidden.map((o) => o.key).sort()).toEqual(["once"]);
    expect(g.other).toEqual([]);
  });

  it("orders the durations t/i/a/all regardless of wire order", () => {
    const shuffled: PermissionOption[] = [
      { key: "all", label: "all" },
      { key: "t", label: "turn" },
      { key: "a", label: "always" },
      { key: "i", label: "idle" },
    ];
    expect(groupPermissionOptions(shuffled).durations.map((o) => o.key)).toEqual(["t", "i", "a", "all"]);
  });

  it("degrades to just the two primary buttons for a minimal option set", () => {
    const g = groupPermissionOptions([{ key: "y", label: "yes" }, { key: "n", label: "no" }]);
    expect(g.allow?.key).toBe("y");
    expect(g.deny?.key).toBe("n");
    expect(g.durations).toEqual([]);
    expect(g.destructive).toBeNull();
    expect(g.allowComment).toBeNull();
    expect(g.denyComment).toBeNull();
    expect(g.hidden).toEqual([]);
    expect(g.other).toEqual([]);
  });

  it("never drops a key the vocabulary has no opinion about -- it lands in other", () => {
    const g = groupPermissionOptions([
      { key: "y", label: "yes" },
      { key: "n", label: "no" },
      { key: "x", label: "a future option kind" },
    ]);
    expect(g.other.map((o) => o.key)).toEqual(["x"]);
  });

  it("classes edit (e) as hidden alongside once", () => {
    const g = groupPermissionOptions([
      { key: "y", label: "yes" }, { key: "n", label: "no" },
      { key: "once", label: "once" }, { key: "e", label: "edit" },
    ]);
    expect(g.hidden.map((o) => o.key).sort()).toEqual(["e", "once"]);
  });
});

describe("plainQuestion / riskTag", () => {
  it("has an entry for every tool class, and none of them are the raw class name", () => {
    for (const cls of ["housekeeping", "write", "exec", "read", "agent", "other"] as const) {
      expect(plainQuestion(cls)).not.toBe(cls);
      expect(plainQuestion(cls).length).toBeGreaterThan(0);
      expect(riskTag(cls).label).not.toBe(cls);
    }
  });

  it("marks write and exec as the elevated tones, and housekeeping/read/other as muted", () => {
    expect(riskTag("write").tone).toBe("warning");
    expect(riskTag("exec").tone).toBe("warning");
    expect(riskTag("housekeeping").tone).toBe("muted");
    expect(riskTag("read").tone).toBe("muted");
    expect(riskTag("other").tone).toBe("muted");
    expect(riskTag("agent").tone).toBe("steel");
  });
});
