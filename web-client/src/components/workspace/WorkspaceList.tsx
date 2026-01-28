/**
 * WorkspaceList - Displays available workspaces for selection
 */

import type { Workspace } from '@/types/events';

interface WorkspaceListProps {
  workspaces: Workspace[];
  selectedWorkspace: string | null;
  onSelect: (name: string) => void;
  isLoading: boolean;
}

export function WorkspaceList({
  workspaces,
  selectedWorkspace,
  onSelect,
  isLoading,
}: WorkspaceListProps) {
  if (workspaces.length === 0 && !isLoading) {
    return (
      <div className="text-center py-8 text-muted">
        <p>No workspaces found.</p>
        <p className="text-sm mt-2">Create a new workspace to get started.</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {workspaces.map((workspace) => (
        <button
          key={workspace.name}
          onClick={() => onSelect(workspace.name)}
          disabled={isLoading}
          className={`
            w-full text-left p-4 rounded-lg border transition-colors
            ${
              selectedWorkspace === workspace.name
                ? 'border-primary bg-primary/10'
                : 'border-border hover:border-primary/50 hover:bg-surface'
            }
            ${isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          `}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {/* Status indicator */}
              <div
                className={`w-3 h-3 rounded-full ${
                  workspace.configured ? 'bg-success' : 'bg-warning'
                }`}
                title={workspace.configured ? 'Configured' : 'Not configured'}
              />
              <div>
                <h3 className="font-medium text-base">{workspace.name}</h3>
                {workspace.configured && workspace.provider && (
                  <p className="text-sm text-muted">
                    {workspace.provider}
                    {workspace.model && ` / ${workspace.model}`}
                  </p>
                )}
                {!workspace.configured && (
                  <p className="text-sm text-warning">Needs configuration</p>
                )}
              </div>
            </div>
            {workspace.last_accessed && (
              <span className="text-xs text-muted">
                Last used: {new Date(workspace.last_accessed).toLocaleDateString()}
              </span>
            )}
          </div>
        </button>
      ))}
    </div>
  );
}
