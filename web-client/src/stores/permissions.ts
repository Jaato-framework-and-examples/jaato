/**
 * Permission request state management
 */

import { create } from 'zustand';

export interface ResponseOption {
  key: string;
  label: string;
  action: string;
}

export interface PermissionRequest {
  requestId: string;
  agentId: string;
  toolName: string;
  toolArgs: Record<string, unknown>;
  promptLines: string[];
  responseOptions: ResponseOption[];
  formatHint?: 'diff' | 'default';
  warnings?: string;
  warningLevel?: 'info' | 'warning' | 'error';
  receivedAt: string;
}

export interface PermissionDecision {
  requestId: string;
  toolName: string;
  response: string;
  granted: boolean;
  timestamp: string;
}

interface PermissionStore {
  // State
  pendingRequest: PermissionRequest | null;
  history: PermissionDecision[];

  // Actions
  setRequest: (request: PermissionRequest) => void;
  clearRequest: () => void;
  addDecision: (decision: PermissionDecision) => void;
  reset: () => void;
}

export const usePermissionStore = create<PermissionStore>((set) => ({
  pendingRequest: null,
  history: [],

  setRequest: (request) => {
    set({ pendingRequest: request });
  },

  clearRequest: () => {
    set({ pendingRequest: null });
  },

  addDecision: (decision) => {
    set((state) => ({
      history: [...state.history, decision],
      pendingRequest: null,
    }));
  },

  reset: () => {
    set({ pendingRequest: null, history: [] });
  },
}));
