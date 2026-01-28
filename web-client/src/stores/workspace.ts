/**
 * Workspace state management
 */

import { create } from 'zustand';
import { useConnectionStore } from './connection';
import { createTimestamp } from '@/lib/events';
import type {
  Workspace,
  WorkspaceListEvent,
  WorkspaceCreatedEvent,
  ConfigStatusEvent,
  ConfigUpdatedEvent,
  ServerEvent,
} from '@/types/events';

interface ConfigStatus {
  workspace: string;
  configured: boolean;
  provider?: string;
  model?: string;
  availableProviders: string[];
  missingFields: string[];
}

interface WorkspaceStore {
  // State
  workspaceMode: boolean;
  workspaces: Workspace[];
  selectedWorkspace: string | null;
  configStatus: ConfigStatus | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  setWorkspaceMode: (enabled: boolean) => void;
  setWorkspaces: (workspaces: Workspace[]) => void;
  addWorkspace: (workspace: Workspace) => void;
  setSelectedWorkspace: (name: string | null) => void;
  setConfigStatus: (status: ConfigStatus | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  // API actions (send requests to server)
  requestWorkspaceList: () => void;
  createWorkspace: (name: string) => void;
  selectWorkspace: (name: string) => void;
  updateConfig: (provider: string, model?: string, apiKey?: string) => void;

  // Event handler
  handleEvent: (event: ServerEvent) => void;
}

export const useWorkspaceStore = create<WorkspaceStore>((set, get) => ({
  // Initial state
  workspaceMode: false,
  workspaces: [],
  selectedWorkspace: null,
  configStatus: null,
  isLoading: false,
  error: null,

  setWorkspaceMode: (enabled) => set({ workspaceMode: enabled }),

  setWorkspaces: (workspaces) => set({ workspaces, isLoading: false }),

  addWorkspace: (workspace) =>
    set((state) => ({
      workspaces: [...state.workspaces, workspace],
    })),

  setSelectedWorkspace: (name) => set({ selectedWorkspace: name }),

  setConfigStatus: (status) => set({ configStatus: status, isLoading: false }),

  setLoading: (loading) => set({ isLoading: loading }),

  setError: (error) => set({ error, isLoading: false }),

  requestWorkspaceList: () => {
    const { send } = useConnectionStore.getState();
    set({ isLoading: true, error: null });
    send({
      type: 'workspace.list',
      timestamp: createTimestamp(),
    });
  },

  createWorkspace: (name) => {
    const { send } = useConnectionStore.getState();
    set({ isLoading: true, error: null });
    send({
      type: 'workspace.create',
      name,
      timestamp: createTimestamp(),
    });
  },

  selectWorkspace: (name) => {
    const { send } = useConnectionStore.getState();
    set({ isLoading: true, error: null, selectedWorkspace: name });
    send({
      type: 'workspace.select',
      name,
      timestamp: createTimestamp(),
    });
  },

  updateConfig: (provider, model, apiKey) => {
    const { send } = useConnectionStore.getState();
    set({ isLoading: true, error: null });
    send({
      type: 'config.update',
      provider,
      model,
      api_key: apiKey,
      timestamp: createTimestamp(),
    });
  },

  handleEvent: (event) => {
    switch (event.type) {
      case 'workspace.list_response': {
        const wsEvent = event as WorkspaceListEvent;
        set({ workspaces: wsEvent.workspaces, isLoading: false });
        break;
      }
      case 'workspace.created': {
        const createEvent = event as WorkspaceCreatedEvent;
        set((state) => ({
          workspaces: [...state.workspaces, createEvent.workspace],
          isLoading: false,
        }));
        break;
      }
      case 'config.status': {
        const statusEvent = event as ConfigStatusEvent;
        set({
          configStatus: {
            workspace: statusEvent.workspace,
            configured: statusEvent.configured,
            provider: statusEvent.provider,
            model: statusEvent.model,
            availableProviders: statusEvent.available_providers,
            missingFields: statusEvent.missing_fields,
          },
          isLoading: false,
        });
        break;
      }
      case 'config.updated': {
        const updateEvent = event as ConfigUpdatedEvent;
        if (updateEvent.success) {
          // Refresh config status after successful update
          const { selectedWorkspace } = get();
          if (selectedWorkspace) {
            set({
              configStatus: {
                workspace: updateEvent.workspace,
                configured: true,
                provider: updateEvent.provider,
                model: updateEvent.model,
                availableProviders: get().configStatus?.availableProviders || [],
                missingFields: [],
              },
              isLoading: false,
            });
          }
        }
        break;
      }
      case 'error': {
        // Handle workspace-related errors
        const errorEvent = event as { message?: string };
        set({ error: errorEvent.message || 'Unknown error', isLoading: false });
        break;
      }
    }
  },
}));
