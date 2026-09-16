/**
 * Global shortcuts, kept to the TUI's defaults where a browser allows
 * them (Ctrl+W closes a tab in most browsers, so the workspace panel
 * also answers to Alt+W).
 */
import { useEffect } from "react";
import { useJaato } from "@/store/store";
import { getClient, isConnected } from "@/sdk/connection";

export function useKeyboardShortcuts(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      const mod = e.ctrlKey || e.metaKey;
      const st = useJaato.getState();
      if (mod && k === "p") { e.preventDefault(); st.toggleUi("showPlan"); }
      else if (mod && k === "b") { e.preventDefault(); st.toggleUi("showBudget"); }
      else if ((mod || e.altKey) && k === "w" && (e.altKey || e.shiftKey)) { e.preventDefault(); st.toggleUi("showWorkspace"); }
      else if (mod && k === "t") {
        e.preventDefault();
        const next = !st.ui.showTools;
        st.toggleUi("showTools");
        st.setAllToolsExpanded(st.selectedAgentId, next);
      }
      else if (mod && k === "a" && !(e.target as HTMLElement | null)?.matches?.("textarea,input")) {
        e.preventDefault();
        const i = st.agentOrder.indexOf(st.selectedAgentId);
        st.selectAgent(st.agentOrder[(i + 1) % st.agentOrder.length] ?? st.selectedAgentId);
      }
      else if (mod && k === "c" && !(e.target as HTMLElement | null)?.matches?.("textarea,input") && !window.getSelection()?.toString()) {
        // Ctrl+C with nothing selected = stop, like the TUI (copy still works with a selection).
        if (isConnected() && st.processing[st.selectedAgentId]) { e.preventDefault(); getClient().stop().catch(() => undefined); }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
