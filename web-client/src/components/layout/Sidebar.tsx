import { useAgentStore } from '@/stores/agents';

export function Sidebar() {
  const { agents, selectedAgentId, selectAgent } = useAgentStore();

  const agentList = Array.from(agents.values());

  return (
    <aside className="flex w-64 flex-col border-r border-border bg-surface">
      {/* Agents section */}
      <div className="flex-1 overflow-y-auto p-4">
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted">
          Agents
        </h3>

        {agentList.length === 0 ? (
          <p className="text-sm text-muted">No active agents</p>
        ) : (
          <ul className="space-y-1">
            {agentList.map((agent) => (
              <li key={agent.id}>
                <button
                  onClick={() => selectAgent(agent.id)}
                  className={`flex w-full items-center gap-2 rounded px-3 py-2 text-left text-sm transition-colors ${
                    selectedAgentId === agent.id
                      ? 'bg-primary/10 text-primary'
                      : 'hover:bg-base'
                  }`}
                >
                  {/* Status indicator */}
                  <span
                    className={`h-2 w-2 rounded-full ${
                      agent.status === 'active'
                        ? 'bg-success animate-pulse'
                        : agent.status === 'error'
                          ? 'bg-error'
                          : 'bg-muted'
                    }`}
                  />

                  {/* Agent info */}
                  <div className="flex-1 truncate">
                    <div className="font-medium">{agent.name}</div>
                    {agent.model && (
                      <div className="text-xs text-muted">{agent.model}</div>
                    )}
                  </div>

                  {/* Type badge */}
                  {agent.type === 'subagent' && (
                    <span className="rounded bg-secondary/20 px-1.5 py-0.5 text-xs text-secondary">
                      sub
                    </span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Tools section (placeholder) */}
      <div className="border-t border-border p-4">
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted">
          Tools
        </h3>
        <p className="text-sm text-muted">Tool tree coming soon</p>
      </div>
    </aside>
  );
}
