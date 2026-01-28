import { usePlanStore, usePlanProgress } from '@/stores/plan';
import { useUIStore } from '@/stores/ui';
import type { PlanStep } from '@/types/events';

const statusSymbols: Record<PlanStep['status'], string> = {
  pending: '\u25CB',      // ○
  in_progress: '\u25D0',  // ◐
  completed: '\u25CF',    // ●
  failed: '\u2717',       // ✗
  blocked: '\u23F3',      // ⏳
};

const statusColors: Record<PlanStep['status'], string> = {
  pending: 'text-muted',
  in_progress: 'text-warning',
  completed: 'text-success',
  failed: 'text-error',
  blocked: 'text-warning',
};

export function PlanPanel() {
  const { steps, expanded, toggleExpanded } = usePlanStore();
  const { togglePlanPanel } = useUIStore();
  const progress = usePlanProgress();

  if (steps.length === 0) {
    return null;
  }

  return (
    <div className="border-b border-border bg-surface">
      {/* Compact header */}
      <div className="flex items-center justify-between px-4 py-2">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium">Plan</span>
          <span className="text-sm text-muted">
            {progress.completed}/{progress.total}
          </span>

          {/* Progress symbols */}
          <div className="flex gap-1">
            {steps.map((step, i) => (
              <span
                key={i}
                className={`text-sm ${statusColors[step.status]}`}
                title={step.content}
              >
                {statusSymbols[step.status]}
              </span>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={toggleExpanded}
            className="rounded p-1 hover:bg-base"
            aria-label={expanded ? 'Collapse plan' : 'Expand plan'}
          >
            <svg
              className={`h-4 w-4 transition-transform ${expanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </button>

          <button
            onClick={togglePlanPanel}
            className="rounded p-1 hover:bg-base"
            aria-label="Hide plan panel"
          >
            <svg
              className="h-4 w-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Expanded view */}
      {expanded && (
        <div className="border-t border-border px-4 py-3">
          <ul className="space-y-2">
            {steps.map((step, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className={`mt-0.5 ${statusColors[step.status]}`}>
                  {statusSymbols[step.status]}
                </span>
                <span
                  className={`text-sm ${
                    step.status === 'completed' ? 'text-muted line-through' : ''
                  } ${step.status === 'in_progress' ? 'font-medium' : ''}`}
                >
                  {step.status === 'in_progress' && step.active_form
                    ? step.active_form
                    : step.content}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
