import { useCallback, useEffect } from 'react';
import { useConnectionStore } from '@/stores/connection';
import { createPermissionResponse } from '@/lib/protocol';
import type { PermissionRequest } from '@/stores/permissions';

interface PermissionModalProps {
  request: PermissionRequest;
}

export function PermissionModal({ request }: PermissionModalProps) {
  const { send } = useConnectionStore();

  const handleResponse = useCallback(
    (responseKey: string) => {
      send(createPermissionResponse(request.requestId, responseKey));
    },
    [send, request.requestId]
  );

  // Keyboard shortcuts for responses
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const option = request.responseOptions.find(
        (opt) => opt.key.toLowerCase() === e.key.toLowerCase()
      );
      if (option) {
        e.preventDefault();
        handleResponse(option.key);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [request.responseOptions, handleResponse]);

  const isDiff = request.formatHint === 'diff';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="max-h-[90vh] w-full max-w-2xl overflow-hidden rounded-lg border border-border bg-surface shadow-xl">
        {/* Header */}
        <div className="border-b border-border px-6 py-4">
          <h2 className="text-lg font-semibold">Permission Required</h2>
          <p className="mt-1 text-sm text-muted">
            Tool: <span className="font-medium text-primary">{request.toolName}</span>
          </p>
        </div>

        {/* Content */}
        <div className="max-h-96 overflow-y-auto px-6 py-4">
          {/* Prompt lines (may contain diff) */}
          <pre
            className={`overflow-x-auto rounded-lg bg-base p-4 text-sm ${
              isDiff ? 'font-mono' : ''
            }`}
          >
            {request.promptLines.map((line, i) => {
              if (!isDiff) {
                return <div key={i}>{line || '\u00A0'}</div>;
              }

              // Diff coloring
              const isAdd = line.startsWith('+') && !line.startsWith('+++');
              const isRemove = line.startsWith('-') && !line.startsWith('---');
              const lineClass = isAdd
                ? 'diff-line-add'
                : isRemove
                  ? 'diff-line-remove'
                  : '';

              return (
                <div key={i} className={lineClass}>
                  {line || '\u00A0'}
                </div>
              );
            })}
          </pre>

          {/* Warnings */}
          {request.warnings && (
            <div
              className={`mt-4 flex items-start gap-2 rounded-lg p-3 ${
                request.warningLevel === 'error'
                  ? 'bg-error/10 text-error'
                  : request.warningLevel === 'warning'
                    ? 'bg-warning/10 text-warning'
                    : 'bg-primary/10 text-primary'
              }`}
            >
              <svg
                className="mt-0.5 h-5 w-5 flex-shrink-0"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
              <span className="text-sm">{request.warnings}</span>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex flex-wrap gap-2 border-t border-border px-6 py-4">
          {request.responseOptions.map((option) => {
            const isAllow = option.action === 'allow';
            const isDeny = option.action === 'deny';

            return (
              <button
                key={option.key}
                onClick={() => handleResponse(option.key)}
                className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                  isAllow
                    ? 'bg-success text-white hover:bg-success/90'
                    : isDeny
                      ? 'bg-error text-white hover:bg-error/90'
                      : 'bg-base hover:bg-border'
                }`}
              >
                <kbd className="rounded bg-black/20 px-1.5 py-0.5 text-xs">
                  {option.key}
                </kbd>
                <span>{option.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
