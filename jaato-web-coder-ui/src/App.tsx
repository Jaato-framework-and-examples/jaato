import { useEffect } from "react";
import { useJaato } from "@/store/store";
import { useLoadNotes } from "@/app/useNote";
import { applyTheme, loadThemePreference } from "@/theme/themes";
import { ConnectScreen } from "@/screens/ConnectScreen";
import { WorkspaceScreen } from "@/screens/WorkspaceScreen";
import { SessionScreen } from "@/screens/SessionScreen";
// Side-effect import: installs the store subscription that asks the daemon
// to bootstrap the jaato-sdk skill into each workspace this client serves
// (#1263), on workspace selection and on session start.
import "@/app/bootstrapSkill";
import "@/app/memories";

export default function App() {
  const screen = useJaato((s) => s.screen);
  const theme = useJaato((s) => s.ui.theme);
  const setTheme = useJaato((s) => s.setTheme);
  useEffect(() => { setTheme(loadThemePreference()); }, [setTheme]);
  useEffect(() => { applyTheme(theme); }, [theme]);
  // Loaded once for the whole page: the picker, the rail and the exit prompt
  // all read one set, keyed by session id, and it is not session state --
  // a note outlives the session being attached, detached or swapped.
  useLoadNotes();
  if (screen === "connect") return <ConnectScreen />;
  if (screen === "workspaces") return <WorkspaceScreen />;
  return <SessionScreen />;
}
