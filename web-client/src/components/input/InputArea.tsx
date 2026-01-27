import { useState, useRef, useCallback, KeyboardEvent } from 'react';
import { useConnectionStore } from '@/stores/connection';
import { useAgentStore } from '@/stores/agents';
import { useUIStore } from '@/stores/ui';
import { usePermissionStore } from '@/stores/permissions';
import { createSendMessageRequest } from '@/lib/protocol';

export function InputArea() {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { send, status } = useConnectionStore();
  const { appendOutput, selectedAgentId } = useAgentStore();
  const { setInputFocused } = useUIStore();
  const { pendingRequest } = usePermissionStore();

  const isConnected = status === 'connected';
  const isDisabled = !isConnected || pendingRequest !== null;

  const handleSubmit = useCallback(() => {
    if (!message.trim() || isDisabled) return;

    // Add user message to output
    appendOutput(selectedAgentId, 'user', message, 'write');

    // Send to server
    send(createSendMessageRequest(message));

    // Clear input
    setMessage('');

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [message, isDisabled, appendOutput, selectedAgentId, send]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleInput = () => {
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  };

  return (
    <div className="border-t border-border bg-surface p-4">
      <div className="mx-auto max-w-4xl">
        <div className="flex gap-3">
          {/* Text input */}
          <div className="relative flex-1">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              onInput={handleInput}
              onFocus={() => setInputFocused(true)}
              onBlur={() => setInputFocused(false)}
              placeholder={
                isDisabled
                  ? pendingRequest
                    ? 'Waiting for permission response...'
                    : 'Connecting...'
                  : 'Type a message... (Enter to send, Shift+Enter for new line)'
              }
              disabled={isDisabled}
              rows={1}
              data-input-prompt
              className={`w-full resize-none rounded-lg border border-border bg-base px-4 py-3 text-sm transition-colors focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary ${
                isDisabled ? 'cursor-not-allowed opacity-50' : ''
              }`}
            />
          </div>

          {/* Send button */}
          <button
            onClick={handleSubmit}
            disabled={isDisabled || !message.trim()}
            className={`flex h-12 w-12 items-center justify-center rounded-lg transition-colors ${
              isDisabled || !message.trim()
                ? 'cursor-not-allowed bg-base text-muted'
                : 'bg-primary text-white hover:bg-primary/90'
            }`}
            aria-label="Send message"
          >
            <svg
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          </button>
        </div>

        {/* Hints */}
        <div className="mt-2 flex items-center justify-between text-xs text-muted">
          <span>
            <kbd className="rounded bg-base px-1">Enter</kbd> to send,{' '}
            <kbd className="rounded bg-base px-1">Shift+Enter</kbd> for new line
          </span>
          <span>
            <kbd className="rounded bg-base px-1">Ctrl+P</kbd> toggle plan
          </span>
        </div>
      </div>
    </div>
  );
}
