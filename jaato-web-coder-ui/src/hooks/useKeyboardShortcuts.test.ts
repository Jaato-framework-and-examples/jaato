/**
 * #1304 §7: Ctrl+P / Ctrl+B / Ctrl+T / Ctrl+A / Ctrl+O used to be bound
 * directly in this hook and are gone -- a browser shortcut (print,
 * bookmarks, a tab Chrome will not hand a page, select-all, open-file)
 * outranks an app binding on the same key.  Asserted at the source rather
 * than by driving a real browser dialog in e2e: a headless Chromium's
 * ``window.print()`` is a no-op, so an e2e case pressing bare Ctrl+P could
 * not tell "fell through to the browser" from "the app silently ate it".
 * This is the one place that can.
 */
import { cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const stop = vi.fn(async () => undefined);
vi.mock("@/sdk/connection", () => ({
  getClient: () => ({ stop }),
  isConnected: () => true,
}));

import { useKeyboardShortcuts } from "./useKeyboardShortcuts";
import { useJaato } from "@/store/store";

function press(key: string, opts: Partial<KeyboardEventInit> = {}): boolean {
  const ev = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true, ...opts });
  window.dispatchEvent(ev);
  return ev.defaultPrevented;
}

beforeEach(() => {
  useJaato.getState().resetSessionState();
  stop.mockClear();
});
afterEach(cleanup);

describe("useKeyboardShortcuts: the direct bindings are gone", () => {
  it.each(["p", "b", "t", "a", "o"])("Ctrl+%s is not prevented and changes nothing in the store", (key) => {
    renderHook(() => useKeyboardShortcuts());
    const before = useJaato.getState();
    const prevented = press(key, { ctrlKey: true });
    expect(prevented).toBe(false);
    const after = useJaato.getState();
    expect(after.paletteOpen).toBe(before.paletteOpen);
    expect(after.ui).toEqual(before.ui);
  });

  it("Cmd+P (macOS) is likewise untouched", () => {
    renderHook(() => useKeyboardShortcuts());
    const prevented = press("p", { metaKey: true });
    expect(prevented).toBe(false);
    expect(useJaato.getState().paletteOpen).toBe(false);
  });
});

describe("useKeyboardShortcuts: what replaced them", () => {
  it("Ctrl+K opens the command palette, and IS prevented", () => {
    renderHook(() => useKeyboardShortcuts());
    expect(useJaato.getState().paletteOpen).toBe(false);
    const prevented = press("k", { ctrlKey: true });
    expect(prevented).toBe(true);
    expect(useJaato.getState().paletteOpen).toBe(true);
  });

  it("Cmd+K (macOS) opens it too", () => {
    renderHook(() => useKeyboardShortcuts());
    press("k", { metaKey: true });
    expect(useJaato.getState().paletteOpen).toBe(true);
  });

  it("Ctrl+Shift+W (or Alt+W) still opens the Files rail panel -- Ctrl+W alone is the browser's close-tab", () => {
    renderHook(() => useKeyboardShortcuts());
    expect(useJaato.getState().ui.activePanel).toBeNull();
    press("w", { altKey: true });
    expect(useJaato.getState().ui.activePanel).toBe("files");
  });
});
