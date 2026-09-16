import { useEffect } from "react";
import { useJaato } from "@/store/store";
import { applyTheme, loadThemePreference } from "@/theme/themes";
import { ConnectScreen } from "@/screens/ConnectScreen";
import { WorkspaceScreen } from "@/screens/WorkspaceScreen";
import { SessionScreen } from "@/screens/SessionScreen";

export default function App() {
  const screen = useJaato((s) => s.screen);
  const theme = useJaato((s) => s.ui.theme);
  const setTheme = useJaato((s) => s.setTheme);
  useEffect(() => { setTheme(loadThemePreference()); }, [setTheme]);
  useEffect(() => { applyTheme(theme); }, [theme]);
  if (screen === "connect") return <ConnectScreen />;
  if (screen === "workspaces") return <WorkspaceScreen />;
  return <SessionScreen />;
}
