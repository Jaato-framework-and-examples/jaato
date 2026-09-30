import { describe, expect, it } from "vitest";
import { isUnknownToolCall, misfireLabel } from "./toolMisfire";

describe("isUnknownToolCall", () => {
  it("recognises the daemon's refusal of a name it has no executor for", () => {
    expect(isUnknownToolCall({ status: "error", errorMessage: "No executor registered for t_7ab5", output: "" })).toBe(true);
    // the runner's cli-only executor quotes the name
    expect(isUnknownToolCall({ status: "error", errorMessage: "No executor registered for 'foo'", output: "" })).toBe(true);
  });
  it("leaves a real tool's failure, a running call and a success alone", () => {
    expect(isUnknownToolCall({ status: "error", errorMessage: "Permission denied", output: "" })).toBe(false);
    expect(isUnknownToolCall({ status: "running", errorMessage: "No executor registered for x", output: "" })).toBe(false);
    expect(isUnknownToolCall({ status: "success", errorMessage: null, output: "No executor registered for x" })).toBe(false);
  });
  it("caps a long name in the caption", () => {
    expect(misfireLabel("x".repeat(40))).toBe(`called a tool that does not exist: ${"x".repeat(31)}…`);
  });
});
