import { describe, expect, it } from "vitest";
import { faultFromError } from "./sessionFault";

describe("faultFromError: recoverable === false is the one signal", () => {
  it("a non-recoverable error is a fault, naming its error_type and message", () => {
    expect(faultFromError("RunnerBootstrapFailed", "AppArmor mismatch", false)).toEqual({
      errorType: "RunnerBootstrapFailed", message: "AppArmor mismatch",
    });
  });
  it("recoverable: true is not a fault", () => {
    expect(faultFromError("SessionError", "Session not found: x", true)).toBeNull();
  });
  it("an absent recoverable field defaults to true on the wire, so it is not a fault either", () => {
    expect(faultFromError("SomeError", "message", undefined)).toBeNull();
  });
  it("a falsy-but-not-boolean recoverable (0, \"\") is not the wire's false and is not a fault", () => {
    expect(faultFromError("X", "m", 0)).toBeNull();
    expect(faultFromError("X", "m", "")).toBeNull();
  });
  it("a missing error_type falls back to a name, never blank", () => {
    expect(faultFromError(undefined, "boom", false)).toEqual({ errorType: "SessionError", message: "boom" });
  });
  it("a non-string message reads as empty rather than throwing or stringifying junk", () => {
    expect(faultFromError("X", undefined, false)).toEqual({ errorType: "X", message: "" });
  });
});
