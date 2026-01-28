import { useTokenUsageDisplay } from '@/stores/context';
import { useSelectedAgent } from '@/stores/agents';

export function StatusBar() {
  const usage = useTokenUsageDisplay();
  const selectedAgent = useSelectedAgent();

  return (
    <footer className="flex h-8 items-center justify-between border-t border-border bg-surface px-4 text-xs">
      {/* Left section - Agent info */}
      <div className="flex items-center gap-3">
        {selectedAgent && (
          <>
            <span className="text-muted">Agent:</span>
            <span className="font-medium">{selectedAgent.name}</span>
            {selectedAgent.model && (
              <span className="text-muted">({selectedAgent.model})</span>
            )}
          </>
        )}
      </div>

      {/* Right section - Token usage */}
      <div className="flex items-center gap-3">
        {/* Token bar */}
        <div className="flex items-center gap-2">
          <span className="text-muted">Context:</span>
          <div className="h-2 w-24 overflow-hidden rounded-full bg-base">
            <div
              className={`h-full transition-all ${
                usage.overThreshold
                  ? 'bg-error'
                  : usage.nearThreshold
                    ? 'bg-warning'
                    : 'bg-primary'
              }`}
              style={{ width: `${Math.min(100, parseFloat(usage.percent))}%` }}
            />
          </div>
          <span
            className={
              usage.overThreshold
                ? 'text-error'
                : usage.nearThreshold
                  ? 'text-warning'
                  : ''
            }
          >
            {usage.percent}%
          </span>
        </div>

        {/* Token count */}
        <span className="text-muted">
          {usage.used} / {usage.limit}
        </span>

        {/* Keyboard hint */}
        <span className="text-muted">
          <kbd className="rounded bg-base px-1">Esc</kbd> to stop
        </span>
      </div>
    </footer>
  );
}
