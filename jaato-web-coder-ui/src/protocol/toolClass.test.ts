import { describe, expect, it } from "vitest";
import { classifyTool, resolveToolClass } from "./toolClass";

describe("classifyTool", () => {
  it("classes the issue's own housekeeping list", () => {
    for (const n of ["createPlan", "startPlan", "setStepStatus", "completePlan", "list_tools", "get_tool_schemas", "listReferences", "list_subagent_profiles", "subscribeToEvents"]) {
      expect(classifyTool(n)).toBe("housekeeping");
    }
  });

  it("classes file_edit as write", () => {
    expect(classifyTool("writeNewFile")).toBe("write");
    expect(classifyTool("updateFile")).toBe("write");
    expect(classifyTool("removeFile")).toBe("write");
  });

  it("classes a subprocess/shell/notebook call as exec", () => {
    expect(classifyTool("cli_based_tool")).toBe("exec");
    expect(classifyTool("shell_input")).toBe("exec");
    expect(classifyTool("notebook_execute")).toBe("exec");
  });

  it("classes a look-only call as read", () => {
    expect(classifyTool("readFile")).toBe("read");
    expect(classifyTool("glob_files")).toBe("read");
  });

  it("classes subagent/session verbs as agent", () => {
    expect(classifyTool("spawn_subagent")).toBe("agent");
    expect(classifyTool("send_to_session")).toBe("agent");
  });

  it("falls back to other for anything it has no opinion about, never a guess", () => {
    expect(classifyTool("mcp__server__tool")).toBe("other");
    expect(classifyTool("store_memory")).toBe("other");
    expect(classifyTool("call_service")).toBe("other");
  });
});

describe("resolveToolClass (jaato/#1304 phase 3)", () => {
  it("prefers the daemon's own tool_class over the client table", () => {
    // The daemon's answer for a name the client table does NOT have an
    // opinion about -- proves the table is not consulted at all when the
    // server reported something.
    expect(resolveToolClass("store_memory", "write")).toBe("write");
  });

  it("falls back to the client table when the daemon reported nothing", () => {
    expect(resolveToolClass("createPlan", null)).toBe("housekeeping");
    expect(resolveToolClass("createPlan", undefined)).toBe("housekeeping");
  });

  it("falls back rather than trusting a value outside the closed vocabulary", () => {
    expect(resolveToolClass("createPlan", "not-a-real-class")).toBe("housekeeping");
  });
});
