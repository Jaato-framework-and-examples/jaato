/**
 * Plan/todo state management
 */

import { create } from 'zustand';
import type { PlanStep } from '@/types/events';

interface PlanStore {
  // State
  steps: PlanStep[];
  expanded: boolean;
  agentPlans: Map<string, PlanStep[]>; // Per-agent plans

  // Actions
  updatePlan: (agentId: string, steps: PlanStep[]) => void;
  clearPlan: (agentId: string) => void;
  toggleExpanded: () => void;
  setExpanded: (expanded: boolean) => void;
  reset: () => void;
}

export const usePlanStore = create<PlanStore>((set, get) => ({
  steps: [],
  expanded: false,
  agentPlans: new Map(),

  updatePlan: (agentId, steps) => {
    set((state) => {
      const agentPlans = new Map(state.agentPlans);
      agentPlans.set(agentId, steps);

      // Use main agent's plan as default display
      const mainPlan = agentPlans.get('main') || steps;

      return { agentPlans, steps: mainPlan };
    });
  },

  clearPlan: (agentId) => {
    set((state) => {
      const agentPlans = new Map(state.agentPlans);
      agentPlans.delete(agentId);

      // Update displayed steps
      const mainPlan = agentPlans.get('main') || [];

      return { agentPlans, steps: mainPlan };
    });
  },

  toggleExpanded: () => {
    set((state) => ({ expanded: !state.expanded }));
  },

  setExpanded: (expanded) => {
    set({ expanded });
  },

  reset: () => {
    set({ steps: [], expanded: false, agentPlans: new Map() });
  },
}));

// Selectors
export const usePlanProgress = () => {
  const steps = usePlanStore((state) => state.steps);
  const completed = steps.filter((s) => s.status === 'completed').length;
  const inProgress = steps.filter((s) => s.status === 'in_progress').length;
  return {
    total: steps.length,
    completed,
    inProgress,
    pending: steps.length - completed - inProgress,
  };
};
