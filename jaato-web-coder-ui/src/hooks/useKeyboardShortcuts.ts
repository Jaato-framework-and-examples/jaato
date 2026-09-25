/**
 * Global shortcuts, kept to the TUI's defaults where a browser allows
 * them (Ctrl+W closes a tab in most browsers, so the workspace panel
 * also answers to Alt+W).
 *
 * Ctrl+P / Ctrl+B / Ctrl+T / Ctrl+A / Ctrl+O used to be bound directly
 * here (#1304 §5 briefly gave Ctrl+P / Ctrl+B a direct binding to the
 * icon rail's panel select) and are gone now (#1304 §7): each is a
 * browser shortcut first (print, bookmarks, a new tab Chrome will not
 * let a page intercept, select-all, open-file), so a bare Ctrl+P now
 * reaches Chrome's own Print exactly as it should. What replaced them is
 * one binding, Ctrl/⌘+K, which opens the command palette -- also the
 * leader ``app/leaderKeys.ts`` documents, so "K then P" still opens
 * Plan, from inside the palette rather than from a direct listener here.
 */
import { useEffect } from "react";
import { useJaato } from "@/store/store";
import { isBusy } from "@/store/phase";
import { getClient, isConnected } from "@/sdk/connection";

export function useKeyboardShortcuts(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      const mod = e.ctrlKey || e.metaKey;
      const st = useJaato.getState();
      if (mod && k === "k") { e.preventDefault(); st.setPaletteOpen(true); return; }
      if ((mod || e.altKey) && k === "w" && (e.altKey || e.shiftKey)) { e.preventDefault(); st.setActivePanel("files"); return; }
      if (mod && k === "c" && !(e.target as HTMLElement | null)?.matches?.("textarea,input") && !window.getSelection()?.toString()) {
        // Ctrl+C with nothing selected = stop, like the TUI (copy still works with a selection).
        if (isConnected() && isBusy(st, st.selectedAgentId)) { e.preventDefault(); getClient().stop().catch(() => undefined); }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
