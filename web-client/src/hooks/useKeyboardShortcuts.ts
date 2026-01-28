/**
 * Global keyboard shortcuts hook
 */

import { useEffect, useCallback } from 'react';
import { useUIStore } from '@/stores/ui';
import { usePermissionStore } from '@/stores/permissions';
import { useConnectionStore } from '@/stores/connection';
import { createStopRequest, createPermissionResponse } from '@/lib/protocol';

interface ShortcutHandlers {
  onSubmit?: () => void;
  onStop?: () => void;
}

export function useKeyboardShortcuts(handlers: ShortcutHandlers = {}) {
  const { togglePlanPanel, toggleSidebar, inputFocused } = useUIStore();
  const { pendingRequest } = usePermissionStore();
  const { send, status } = useConnectionStore();

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      const { key, ctrlKey, metaKey } = event;
      const modKey = ctrlKey || metaKey;

      // Global shortcuts (work regardless of focus)
      if (key === 'Escape') {
        // Stop generation
        if (status === 'connected') {
          send(createStopRequest());
          handlers.onStop?.();
        }
        event.preventDefault();
        return;
      }

      // Permission modal shortcuts (when modal is open)
      if (pendingRequest) {
        const option = pendingRequest.responseOptions.find(
          (opt) => opt.key.toLowerCase() === key.toLowerCase()
        );
        if (option) {
          send(createPermissionResponse(pendingRequest.requestId, option.key));
          event.preventDefault();
          return;
        }
      }

      // Shortcuts that don't work when input is focused
      if (inputFocused) {
        return;
      }

      // Toggle shortcuts
      if (modKey && key === 'p') {
        togglePlanPanel();
        event.preventDefault();
        return;
      }

      if (modKey && key === 'b') {
        toggleSidebar();
        event.preventDefault();
        return;
      }

      // Focus input with '/'
      if (key === '/') {
        const input = document.querySelector<HTMLTextAreaElement>(
          '[data-input-prompt]'
        );
        if (input) {
          input.focus();
          event.preventDefault();
        }
        return;
      }
    },
    [
      status,
      send,
      pendingRequest,
      inputFocused,
      togglePlanPanel,
      toggleSidebar,
      handlers,
    ]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);
}
