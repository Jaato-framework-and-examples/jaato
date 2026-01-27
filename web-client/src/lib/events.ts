/**
 * Event parsing and serialization utilities
 */

import type { ServerEvent, ClientRequest } from '@/types/events';

/**
 * Parse a JSON string into a typed ServerEvent
 */
export function parseServerEvent(json: string): ServerEvent | null {
  try {
    const event = JSON.parse(json);
    if (!event.type) {
      console.warn('Received event without type:', event);
      return null;
    }
    return event as ServerEvent;
  } catch (error) {
    console.error('Failed to parse server event:', error, json);
    return null;
  }
}

/**
 * Serialize a client request to JSON
 */
export function serializeClientRequest(request: ClientRequest): string {
  return JSON.stringify(request);
}

/**
 * Create a timestamp in ISO 8601 format
 */
export function createTimestamp(): string {
  return new Date().toISOString();
}

/**
 * Type guard functions for event discrimination
 */
export function isAgentOutputEvent(
  event: ServerEvent
): event is import('@/types/events').AgentOutputEvent {
  return event.type === 'agent.output';
}

export function isPermissionRequestedEvent(
  event: ServerEvent
): event is import('@/types/events').PermissionRequestedEvent {
  return event.type === 'permission.requested';
}

export function isToolCallStartEvent(
  event: ServerEvent
): event is import('@/types/events').ToolCallStartEvent {
  return event.type === 'tool.call_start';
}

export function isToolCallEndEvent(
  event: ServerEvent
): event is import('@/types/events').ToolCallEndEvent {
  return event.type === 'tool.call_end';
}

export function isPlanUpdatedEvent(
  event: ServerEvent
): event is import('@/types/events').PlanUpdatedEvent {
  return event.type === 'plan.updated';
}

export function isContextUpdatedEvent(
  event: ServerEvent
): event is import('@/types/events').ContextUpdatedEvent {
  return event.type === 'context.updated';
}
