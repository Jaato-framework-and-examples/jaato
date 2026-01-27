/**
 * WebSocket connection state management
 */

import { create } from 'zustand';
import ReconnectingWebSocket from 'reconnecting-websocket';
import { parseServerEvent, serializeClientRequest } from '@/lib/events';
import type { ServerEvent, ClientRequest } from '@/types/events';

export type ConnectionStatus =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting';

interface ServerInfo {
  version: string;
  clientId: string;
  protocolVersion: string;
}

interface ConnectionStore {
  // State
  status: ConnectionStatus;
  clientId: string | null;
  serverInfo: ServerInfo | null;
  error: string | null;

  // Internal
  _ws: ReconnectingWebSocket | null;
  _eventHandlers: Set<(event: ServerEvent) => void>;

  // Actions
  connect: (url: string) => void;
  disconnect: () => void;
  send: (request: ClientRequest) => void;
  subscribe: (handler: (event: ServerEvent) => void) => () => void;

  // Internal actions
  _setStatus: (status: ConnectionStatus) => void;
  _setServerInfo: (info: ServerInfo) => void;
  _setError: (error: string | null) => void;
}

export const useConnectionStore = create<ConnectionStore>((set, get) => ({
  // Initial state
  status: 'disconnected',
  clientId: null,
  serverInfo: null,
  error: null,
  _ws: null,
  _eventHandlers: new Set(),

  connect: (url: string) => {
    const { _ws, status } = get();

    // Already connected or connecting
    if (_ws && (status === 'connected' || status === 'connecting')) {
      return;
    }

    set({ status: 'connecting', error: null });

    const ws = new ReconnectingWebSocket(url, [], {
      maxRetries: Infinity,
      reconnectionDelayGrowFactor: 2,
      maxReconnectionDelay: 30000,
      minReconnectionDelay: 1000,
    });

    ws.onopen = () => {
      set({ status: 'connected', error: null });
    };

    ws.onclose = () => {
      const currentStatus = get().status;
      if (currentStatus !== 'disconnected') {
        set({ status: 'reconnecting' });
      }
    };

    ws.onerror = () => {
      set({ error: 'Connection error' });
    };

    ws.onmessage = (event) => {
      const serverEvent = parseServerEvent(event.data);
      if (!serverEvent) return;

      // Handle connected event specially
      if (serverEvent.type === 'connected') {
        const info = serverEvent.server_info;
        set({
          clientId: info.client_id,
          serverInfo: {
            version: info.version,
            clientId: info.client_id,
            protocolVersion: info.protocol_version,
          },
        });
      }

      // Notify all subscribers
      const handlers = get()._eventHandlers;
      handlers.forEach((handler) => handler(serverEvent));
    };

    set({ _ws: ws });
  },

  disconnect: () => {
    const { _ws } = get();
    if (_ws) {
      _ws.close();
    }
    set({
      status: 'disconnected',
      clientId: null,
      serverInfo: null,
      _ws: null,
    });
  },

  send: (request: ClientRequest) => {
    const { _ws, status } = get();
    if (_ws && status === 'connected') {
      _ws.send(serializeClientRequest(request));
    } else {
      console.warn('Cannot send: not connected');
    }
  },

  subscribe: (handler: (event: ServerEvent) => void) => {
    const handlers = get()._eventHandlers;
    handlers.add(handler);
    return () => {
      handlers.delete(handler);
    };
  },

  _setStatus: (status) => set({ status }),
  _setServerInfo: (info) => set({ serverInfo: info, clientId: info.clientId }),
  _setError: (error) => set({ error }),
}));
