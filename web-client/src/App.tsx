import { useWebSocket } from '@/hooks/useWebSocket';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';
import { Header } from '@/components/layout/Header';
import { Sidebar } from '@/components/layout/Sidebar';
import { MainLayout } from '@/components/layout/MainLayout';
import { StatusBar } from '@/components/layout/StatusBar';
import { PermissionModal } from '@/components/modals/PermissionModal';
import { useUIStore } from '@/stores/ui';
import { usePermissionStore } from '@/stores/permissions';

// Get WebSocket URL from environment or default
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8080';

function App() {
  const { status } = useWebSocket(WS_URL);
  const { sidebarOpen } = useUIStore();
  const { pendingRequest } = usePermissionStore();

  useKeyboardShortcuts();

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
