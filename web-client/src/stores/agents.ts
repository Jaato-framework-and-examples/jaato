/**
 * Multi-agent state management
 */

import { create } from 'zustand';

export type AgentStatus = 'idle' | 'active' | 'done' | 'error';
export type OutputMode = 'write' | 'append';

export interface OutputLine {
  id: string;
  source: string;
  text: string;
  timestamp: string;
}

export interface Agent {
  id: string;
  name: string;
  type: 'main' | 'subagent';
  model?: string;
  status: AgentStatus;
  output: OutputLine[];
  createdAt: string;
}

interface AgentStore {
  // State
  agents: Map<string, Agent>;
  selectedAgentId: string;
  pendingOutput: Map<string, OutputLine[]>; // For events before agent creation

  // Actions
  createAgent: (
    id: string,
    name: string,
    type: 'main' | 'subagent',
    model?: string
  ) => void;
  removeAgent: (id: string) => void;
  updateStatus: (id: string, status: AgentStatus) => void;
  appendOutput: (
    agentId: string,
    source: string,
    text: string,
    mode: OutputMode
  ) => void;
  selectAgent: (id: string) => void;
  clearOutput: (agentId: string) => void;
  reset: () => void;
}

let lineIdCounter = 0;
function generateLineId(): string {
  return `line-${++lineIdCounter}`;
}

export const useAgentStore = create<AgentStore>((set) => ({
  agents: new Map(),
  selectedAgentId: 'main',
  pendingOutput: new Map(),

  createAgent: (id, name, type, model) => {
    set((state) => {
      const agents = new Map(state.agents);
      const pendingOutput = state.pendingOutput.get(id) || [];

      agents.set(id, {
        id,
        name,
        type,
        model,
        status: 'idle',
        output: pendingOutput,
        createdAt: new Date().toISOString(),
      });

      // Clear pending output for this agent
      const newPendingOutput = new Map(state.pendingOutput);
      newPendingOutput.delete(id);

      // Auto-select main agent
      const selectedAgentId = type === 'main' ? id : state.selectedAgentId;

      return { agents, pendingOutput: newPendingOutput, selectedAgentId };
    });
  },

  removeAgent: (id) => {
    set((state) => {
      const agents = new Map(state.agents);
      agents.delete(id);

      // Select another agent if current was removed
      let selectedAgentId = state.selectedAgentId;
      if (selectedAgentId === id) {
        const remaining = Array.from(agents.keys());
        selectedAgentId = remaining[0] || 'main';
      }

      return { agents, selectedAgentId };
    });
  },

  updateStatus: (id, status) => {
    set((state) => {
      const agents = new Map(state.agents);
      const agent = agents.get(id);
      if (agent) {
        agents.set(id, { ...agent, status });
      }
      return { agents };
    });
  },

  appendOutput: (agentId, source, text, mode) => {
    set((state) => {
      const agents = new Map(state.agents);
      const agent = agents.get(agentId);

      if (agent) {
        const output = [...agent.output];

        if (mode === 'append' && output.length > 0) {
          // Append to last line from same source
          const lastLine = output[output.length - 1];
          if (lastLine.source === source) {
            output[output.length - 1] = {
              ...lastLine,
              text: lastLine.text + text,
            };
          } else {
            output.push({
              id: generateLineId(),
              source,
              text,
              timestamp: new Date().toISOString(),
            });
          }
        } else {
          output.push({
            id: generateLineId(),
            source,
            text,
            timestamp: new Date().toISOString(),
          });
        }

        agents.set(agentId, { ...agent, output });
        return { agents };
      } else {
        // Queue for later if agent doesn't exist yet
        const pendingOutput = new Map(state.pendingOutput);
        const pending = pendingOutput.get(agentId) || [];
        pending.push({
          id: generateLineId(),
          source,
          text,
          timestamp: new Date().toISOString(),
        });
        pendingOutput.set(agentId, pending);
        return { pendingOutput };
      }
    });
  },

  selectAgent: (id) => {
    set({ selectedAgentId: id });
  },

  clearOutput: (agentId) => {
    set((state) => {
      const agents = new Map(state.agents);
      const agent = agents.get(agentId);
      if (agent) {
        agents.set(agentId, { ...agent, output: [] });
      }
      return { agents };
    });
  },

  reset: () => {
    set({
      agents: new Map(),
      selectedAgentId: 'main',
      pendingOutput: new Map(),
    });
  },
}));

// Selectors
export const useSelectedAgent = () => {
  const agents = useAgentStore((state) => state.agents);
  const selectedId = useAgentStore((state) => state.selectedAgentId);
  return agents.get(selectedId);
};

export const useAgentOutput = (agentId: string) => {
  const agents = useAgentStore((state) => state.agents);
  return agents.get(agentId)?.output || [];
};
