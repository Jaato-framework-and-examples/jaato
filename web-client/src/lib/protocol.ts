/**
 * Client request builders
 */

import type {
  SendMessageRequest,
  PermissionResponseRequest,
  ClarificationResponseRequest,
  StopRequest,
  CommandRequest,
  ClientConfigRequest,
} from '@/types/events';
import { createTimestamp } from './events';

export interface Attachment {
  type: 'file' | 'image';
  path?: string;
  data?: string;
  mime_type?: string;
}

/**
 * Create a message send request
 */
export function createSendMessageRequest(
  text: string,
  attachments: Attachment[] = []
): SendMessageRequest {
  return {
    type: 'message.send',
    text,
    attachments,
    timestamp: createTimestamp(),
  };
}

/**
 * Create a permission response request
 */
export function createPermissionResponse(
  requestId: string,
  response: string
): PermissionResponseRequest {
  return {
    type: 'permission.response',
    request_id: requestId,
    response,
    timestamp: createTimestamp(),
  };
}

/**
 * Create a clarification response request
 */
export function createClarificationResponse(
  requestId: string,
  answer: string
): ClarificationResponseRequest {
  return {
    type: 'clarification.response',
    request_id: requestId,
    answer,
    timestamp: createTimestamp(),
  };
}

/**
 * Create a stop request
 */
export function createStopRequest(agentId?: string): StopRequest {
  return {
    type: 'stop',
    agent_id: agentId,
    timestamp: createTimestamp(),
  };
}

/**
 * Create a command request
 */
export function createCommandRequest(
  name: string,
  args: string[] = []
): CommandRequest {
  return {
    type: 'command',
    name,
    args,
    timestamp: createTimestamp(),
  };
}

/**
 * Create a client config request
 */
export function createClientConfigRequest(
  cwd: string,
  env: Record<string, string> = {},
  terminalWidth: number = 120
): ClientConfigRequest {
  return {
    type: 'client.config',
    cwd,
    env,
    terminal_width: terminalWidth,
    timestamp: createTimestamp(),
  };
}
