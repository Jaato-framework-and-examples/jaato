/**
 * WebSocket connection hook with event routing
 */

import { useEffect, useCallback } from 'react';
import { useConnectionStore } from '@/stores/connection';
import { useAgentStore } from '@/stores/agents';
import { usePermissionStore } from '@/stores/permissions';
import { usePlanStore } from '@/stores/plan';
import { useContextStore } from '@/stores/context';
import { useWorkspaceStore } from '@/stores/workspace';
import type { ServerEvent } from '@/types/events';
import { createClientConfigRequest } from '@/lib/protocol';

const DEFAULT_WS_URL = 'ws://localhost:8080';

export function useWebSocket(url: string = DEFAULT_WS_URL) {
  const { connect, disconnect, status, send, subscribe } = useConnectionStore();

  // Route events to appropriate stores
  const routeEvent = useCallback((event: ServerEvent) => {
    console.log('[WS Event]', event.type, event);

    switch (event.type) {
      // Connection events - handle workspace_mode
      case 'connected':
        console.log('[WS] Connected event, workspace_mode:', event.server_info.workspace_mode);
        if (event.server_info.workspace_mode) {
          console.log('[WS] Setting workspace mode to true');
          useWorkspaceStore.getState().setWorkspaceMode(true);
          // Request workspace list when in workspace mode
          useWorkspaceStore.getState().requestWorkspaceList();
        }
        break;

      // Workspace events
      case 'workspace.list_response':
      case 'workspace.created':
      case 'config.status':
      case 'config.updated':
        useWorkspaceStore.getState().handleEvent(event);
        break;

      // Agent events
      case 'agent.created':
        useAgentStore.getState().createAgent(
          event.agent_id,
          event.agent_name,
          event.agent_type,
          event.model
        );
        break;

      case 'agent.output':
        useAgentStore.getState().appendOutput(
          event.agent_id,
          event.source,
          event.text,
          event.mode
        );
        break;

      case 'agent.status_changed':
        useAgentStore.getState().updateStatus(event.agent_id, event.status);
        break;

      // Permission events
      case 'permission.requested':
        usePermissionStore.getState().setRequest({
          requestId: event.request_id,
          agentId: event.agent_id,
          toolName: event.tool_name,
          toolArgs: event.tool_args,
          promptLines: event.prompt_lines,
          responseOptions: event.response_options,
          formatHint: event.format_hint,
          warnings: event.warnings,
          warningLevel: event.warning_level,
          receivedAt: new Date().toISOString(),
        });
        break;

      case 'permission.resolved':
        usePermissionStore.getState().addDecision({
          requestId: event.request_id,
          toolName: event.tool_name,
          response: event.method || 'unknown',
          granted: event.granted,
          timestamp: new Date().toISOString(),
        });
        break;

      // Plan events
      case 'plan.updated':
        usePlanStore.getState().updatePlan(event.agent_id, event.steps);
        break;

      case 'plan.cleared':
        usePlanStore.getState().clearPlan(event.agent_id);
        break;

      // Context events
      case 'context.updated':
        useContextStore.getState().update({
          totalTokens: event.total_tokens,
          promptTokens: event.prompt_tokens,
          outputTokens: event.output_tokens,
          contextLimit: event.context_limit,
          percentUsed: event.percent_used,
          tokensRemaining: event.tokens_remaining,
          gcThreshold: event.gc_threshold,
          gcStrategy: event.gc_strategy,
        });
        break;

      // Error handling
      case 'error':
        console.error('Server error:', event.message);
        break;

      case 'system_message':
        console.log(`[${event.level}] ${event.message}`);
        break;

      default:
        // Log unhandled events for debugging
        if (import.meta.env.DEV) {
          console.log('Unhandled event:', event.type, event);
        }
    }
  }, []);

  // Connect on mount
  useEffect(() => {
    connect(url);

    const unsubscribe = subscribe(routeEvent);

    return () => {
      unsubscribe();
    };
  }, [url, connect, subscribe, routeEvent]);

  // Send client config after connection
  useEffect(() => {
    if (status === 'connected') {
      send(createClientConfigRequest(
        window.location.pathname,
        {},
        window.innerWidth
      ));
    }
  }, [status, send]);

  return {
    status,
    disconnect,
    send,
  };
}
