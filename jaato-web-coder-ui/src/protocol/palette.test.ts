import { describe, expect, it } from "vitest";
import { filterGroups, flattenGroups, groupCommands } from "./palette";

const COMMANDS = [
  { name: "help", description: "Show available commands" },
  { name: "tools", description: "Manage tools" },
  { name: "tools list", description: "List all tools" },
  { name: "tools enable", description: "Enable a tool" },
  { name: "permissions status", description: "Show permission status" },
  { name: "permissions whitelist", description: "Whitelist tools" },
  { name: "context", description: "Show context window usage" },
];

describe("groupCommands: by the name's first word, the command list's own area", () => {
  it("groups multi-word commands under their shared first word", () => {
    const groups = groupCommands(COMMANDS);
    const tools = groups.find((g) => g.area === "tools");
    expect(tools?.items.map((c) => c.name).sort()).toEqual(["tools", "tools enable", "tools list"].sort());
  });
  it("puts the bare command first within its area, then sorts the rest by name", () => {
    const groups = groupCommands(COMMANDS);
    const tools = groups.find((g) => g.area === "tools")!;
    expect(tools.items[0]!.name).toBe("tools");
    expect(tools.items.slice(1).map((c) => c.name)).toEqual(["tools enable", "tools list"]);
  });
  it("a bare command with no siblings is its own area", () => {
    const groups = groupCommands(COMMANDS);
    expect(groups.find((g) => g.area === "help")?.items.map((c) => c.name)).toEqual(["help"]);
    expect(groups.find((g) => g.area === "context")?.items.map((c) => c.name)).toEqual(["context"]);
  });
  it("areas sort alphabetically, so the render order is reproducible", () => {
    const groups = groupCommands(COMMANDS);
    expect(groups.map((g) => g.area)).toEqual(["context", "help", "permissions", "tools"]);
  });
  it("a command with no name is dropped, never a group of its own", () => {
    expect(groupCommands([{ name: "" }, { name: "help" }]).map((g) => g.area)).toEqual(["help"]);
  });
});

describe("filterGroups: a plain, case-insensitive substring test", () => {
  it("an empty query returns every group unchanged", () => {
    expect(filterGroups(groupCommands(COMMANDS), "")).toEqual(groupCommands(COMMANDS));
  });
  it("matches the name", () => {
    const groups = filterGroups(groupCommands(COMMANDS), "tools li");
    expect(flattenGroups(groups).map((c) => c.name)).toEqual(["tools list"]);
  });
  it("matches the description too", () => {
    const groups = filterGroups(groupCommands(COMMANDS), "whitelist");
    expect(flattenGroups(groups).map((c) => c.name)).toEqual(["permissions whitelist"]);
  });
  it("is case-insensitive", () => {
    expect(flattenGroups(filterGroups(groupCommands(COMMANDS), "TOOLS")).length).toBeGreaterThan(0);
  });
  it("drops a group left with no matches rather than rendering it empty", () => {
    const groups = filterGroups(groupCommands(COMMANDS), "tools li");
    expect(groups.find((g) => g.area === "permissions")).toBeUndefined();
  });
  it("no match at all is an empty list, not every group", () => {
    expect(filterGroups(groupCommands(COMMANDS), "xyzzy")).toEqual([]);
  });
});

describe("flattenGroups: the order arrow-key navigation counts over", () => {
  it("matches the groups' own render order", () => {
    const groups = groupCommands(COMMANDS);
    const flat = flattenGroups(groups);
    expect(flat.map((c) => c.name)).toEqual([
      "context", "help",
      "permissions status", "permissions whitelist",
      "tools", "tools enable", "tools list",
    ]);
  });
});
