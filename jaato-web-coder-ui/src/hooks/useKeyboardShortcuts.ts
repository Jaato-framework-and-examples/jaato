/**
 * Global shortcuts, kept to the TUI's defaults where a browser allows
 * them (Ctrl+W closes a tab in most browsers, so the workspace panel
 * also answers to Alt+W).
 *
 * Ctrl+P / Ctrl+B / Alt+W now select the icon rail's panel (#1304 §5)
 * rather than toggling a removed ``ui.show*`` flag -- pressing the same
 * shortcut again while that panel is already open CLOSES it (the rail's
 * own single-panel toggle), which is what "toggle" meant before.  The
 * full leader-key redesign (#1304 §7) is a later PR; this only keeps the
 * three pre-existing bindings pointed at their new target.
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
      if (mod && k === "p") { e.preventDefault(); st.setActivePanel("plan"); }
      else if (mod && k === "b") { e.preventDefault(); st.setActivePanel("budget"); }
      else if ((mod || e.altKey) && k === "w" && (e.altKey || e.shiftKey)) { e.preventDefault(); st.setActivePanel("files"); }
      else if (mod && k === "t") {
        e.preventDefault();
        // The TUI's Ctrl+T: every tool block in the chat expands or
        // collapses at once, and new blocks follow the setting.
        st.setToolsExpanded(!st.ui.showTools);
      }
      else if (mod && k === "a" && !(e.target as HTMLElement | null)?.matches?.("textarea,input")) {
        e.preventDefault();
        const i = st.agentOrder.indexOf(st.selectedAgentId);
        st.selectAgent(st.agentOrder[(i + 1) % st.agentOrder.length] ?? st.selectedAgentId);
      }
      else if (mod && k === "c" && !(e.target as HTMLElement | null)?.matches?.("textarea,input") && !window.getSelection()?.toString()) {
        // Ctrl+C with nothing selected = stop, like the TUI (copy still works with a selection).
        if (isConnected() && isBusy(st, st.selectedAgentId)) { e.preventDefault(); getClient().stop().catch(() => undefined); }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
