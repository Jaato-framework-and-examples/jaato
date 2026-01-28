import { PlanPanel } from '@/components/plan/PlanPanel';
import { OutputPane } from '@/components/output/OutputPane';
import { InputArea } from '@/components/input/InputArea';
import { useUIStore } from '@/stores/ui';
import { useAgentStore } from '@/stores/agents';

export function MainLayout() {
  const { planPanelOpen } = useUIStore();
  const { selectedAgentId } = useAgentStore();

  return (
    <main className="flex flex-1 flex-col overflow-hidden">
      {/* Plan panel (sticky top) */}
      {planPanelOpen && <PlanPanel />}

      {/* Output pane (scrollable) */}
      <OutputPane agentId={selectedAgentId} />

      {/* Input area (sticky bottom) */}
      <InputArea />
    </main>
  );
}
