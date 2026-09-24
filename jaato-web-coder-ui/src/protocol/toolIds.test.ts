import { describe, expect, it } from "vitest";
import { resolveToolArgs, resolveToolIds, toolIdMappings } from "./toolIds";

const NAMES = { c_bbc5e661: "system", t_a3f2b1c0: "readFile" };

describe("resolveToolIds", () => {
  it("shows a category id by the name a person knows -- the reported case", () => {
    expect(resolveToolArgs({ category_id: "c_bbc5e661" }, NAMES)).toEqual({ category_id: "system" });
  });

  it("resolves inside arrays and nested objects", () => {
    expect(resolveToolIds({ tool_names: ["t_a3f2b1c0", "c_bbc5e661"], nested: { id: "t_a3f2b1c0" } }, NAMES))
      .toEqual({ tool_names: ["readFile", "system"], nested: { id: "readFile" } });
  });

  it("leaves an id the map does not name, and every other value, as it is", () => {
    expect(resolveToolArgs({ category_id: "c_00000000", path: "src/a.py", n: 3, flag: true }, NAMES))
      .toEqual({ category_id: "c_00000000", path: "src/a.py", n: 3, flag: true });
  });

  it("does not resolve a value that merely contains an id, or an inherited key", () => {
    expect(resolveToolIds("see c_bbc5e661", NAMES)).toBe("see c_bbc5e661");
    expect(resolveToolIds("toString", NAMES)).toBe("toString");
  });
});

it("toolIdMappings keeps string -> string entries only", () => {
  expect(toolIdMappings({ c_1: "system", bad: 3 })).toEqual({ c_1: "system" });
  expect(toolIdMappings(null)).toBeNull();
  expect(toolIdMappings(["x"])).toBeNull();
});
