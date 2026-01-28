/**
 * Token usage and context state management
 */

import { create } from 'zustand';

interface ContextUsage {
  totalTokens: number;
  promptTokens: number;
  outputTokens: number;
  contextLimit: number;
  percentUsed: number;
  tokensRemaining: number;
  gcThreshold: number;
  gcStrategy?: string;
}

interface ContextStore extends ContextUsage {
  // Actions
  update: (usage: Partial<ContextUsage>) => void;
  reset: () => void;
}

const initialState: ContextUsage = {
  totalTokens: 0,
  promptTokens: 0,
  outputTokens: 0,
  contextLimit: 0,
  percentUsed: 0,
  tokensRemaining: 0,
  gcThreshold: 80,
  gcStrategy: undefined,
};

export const useContextStore = create<ContextStore>((set) => ({
  ...initialState,

  update: (usage) => {
    set((state) => ({ ...state, ...usage }));
  },

  reset: () => {
    set(initialState);
  },
}));

// Selectors
export const useTokenUsageDisplay = () => {
  const totalTokens = useContextStore((state) => state.totalTokens);
  const contextLimit = useContextStore((state) => state.contextLimit);
  const percentUsed = useContextStore((state) => state.percentUsed);
  const gcThreshold = useContextStore((state) => state.gcThreshold);

  const formatTokens = (n: number): string => {
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(0)}k`;
    return n.toString();
  };

  return {
    used: formatTokens(totalTokens),
    limit: formatTokens(contextLimit),
    percent: percentUsed.toFixed(0),
    nearThreshold: percentUsed >= gcThreshold - 10,
    overThreshold: percentUsed >= gcThreshold,
  };
};
