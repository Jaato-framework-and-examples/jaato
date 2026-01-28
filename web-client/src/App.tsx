import { useWebSocket } from '@/hooks/useWebSocket';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';
import { Header } from '@/components/layout/Header';
import { Sidebar } from '@/components/layout/Sidebar';
import { MainLayout } from '@/components/layout/MainLayout';
import { StatusBar } from '@/components/layout/StatusBar';
import { PermissionModal } from '@/components/modals/PermissionModal';
import { WorkspaceScreen } from '@/components/workspace/WorkspaceScreen';
import { useUIStore } from '@/stores/ui';
import { usePermissionStore } from '@/stores/permissions';
import { useWorkspaceStore } from '@/stores/workspace';

// Get WebSocket URL from environment, or derive from current hostname
function getWebSocketUrl(): string {
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL;
  }
  // Use same hostname as the page, default port 8080
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.hostname || 'localhost';
  const port = import.meta.env.VITE_WS_PORT || '8080';
  return `${protocol}//${host}:${port}`;
}

const WS_URL = getWebSocketUrl();

function App() {
  const { status } = useWebSocket(WS_URL);
  const { sidebarOpen } = useUIStore();
  const { pendingRequest } = usePermissionStore();
  const { workspaceMode, configStatus, selectedWorkspace } = useWorkspaceStore();

  useKeyboardShortcuts();

  // In workspace mode, show workspace screen until a workspace is selected and configured
  const showWorkspaceScreen =
    workspaceMode && (!selectedWorkspace || !configStatus?.configured);

  if (showWorkspaceScreen) {
    return <WorkspaceScreen />;
  }

  return (
    <div className="flex h-screen flex-col bg-base text-base">
      {/* Header */}
      <Header connectionStatus={status} />

      {/* Main content area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        {sidebarOpen && <Sidebar />}

        {/* Main content */}
        <MainLayout />
      </div>

      {/* Status bar */}
      <StatusBar />

      {/* Permission modal */}
      {pendingRequest && <PermissionModal request={pendingRequest} />}
    </div>
  );
}

export default App;
