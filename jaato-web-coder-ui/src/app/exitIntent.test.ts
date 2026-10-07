import { describe, expect, it } from "vitest";
import { consumeExited, markExited } from "./exitIntent";

function memoryStorage(): Storage {
  const m = new Map<string, string>();
  return {
    get length() { return m.size; },
    clear: () => m.clear(),
    getItem: (k) => m.get(k) ?? null,
    key: (i) => [...m.keys()][i] ?? null,
    removeItem: (k) => { m.delete(k); },
    setItem: (k, v) => { m.set(k, v); },
  };
}

describe("exit intent", () => {
  it("is consumed once: the connect screen skips one auto-connect, a reload starts clean", () => {
    const s = memoryStorage();
    expect(consumeExited(s)).toBe(false);
    markExited(s);
    expect(consumeExited(s)).toBe(true);
    expect(consumeExited(s)).toBe(false);
  });

  it("tolerates a missing or refusing storage", () => {
    expect(consumeExited(null)).toBe(false);
    markExited(null);
    const refusing = { ...memoryStorage(), getItem: () => { throw new Error("denied"); }, setItem: () => { throw new Error("denied"); } } as Storage;
    markExited(refusing);
    expect(consumeExited(refusing)).toBe(false);
  });
});
