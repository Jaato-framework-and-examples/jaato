/**
 * WorkspaceScreen - Main workspace selection and configuration screen
 */

import { useState, useEffect } from 'react';
import { useWorkspaceStore } from '@/stores/workspace';
import { useConnectionStore } from '@/stores/connection';
import { WorkspaceList } from './WorkspaceList';
import { CreateWorkspace } from './CreateWorkspace';
import { ConfigureWorkspace } from './ConfigureWorkspace';
import { Button } from '@/components/common/Button';
import { Spinner } from '@/components/common/Spinner';

type ScreenView = 'list' | 'create' | 'configure';

export function WorkspaceScreen() {
  const [view, setView] = useState<ScreenView>('list');

  const {
    workspaces,
    selectedWorkspace,
    configStatus,
    isLoading,
    error,
    requestWorkspaceList,
    createWorkspace,
    selectWorkspace,
    updateConfig,
    setSelectedWorkspace,
  } = useWorkspaceStore();

  const { status: connectionStatus } = useConnectionStore();

  // Request workspace list on mount when connected
  useEffect(() => {
    if (connectionStatus === 'connected') {
      requestWorkspaceList();
    }
  }, [connectionStatus, requestWorkspaceList]);

  // When a workspace is selected and config status arrives, decide next step
  useEffect(() => {
    if (configStatus && selectedWorkspace) {
      if (!configStatus.configured) {
        // Needs configuration
        setView('configure');
      }
      // If configured, parent component will handle transition to chat
    }
  }, [configStatus, selectedWorkspace]);

  const handleSelectWorkspace = (name: string) => {
    selectWorkspace(name);
  };

  const handleCreateWorkspace = (name: string) => {
    createWorkspace(name);
    // After creation, auto-select and go to configure
    setSelectedWorkspace(name);
    setView('configure');
  };

  const handleSaveConfig = (provider: string, model?: string, apiKey?: string) => {
    updateConfig(provider, model, apiKey);
  };

  const handleBack = () => {
    setView('list');
    setSelectedWorkspace(null);
  };

  // Show loading state while connecting
  if (connectionStatus === 'connecting' || connectionStatus === 'reconnecting') {
    return (
      <div className="flex h-screen items-center justify-center bg-base">
        <div className="text-center">
          <Spinner size="lg" />
          <p className="mt-4 text-muted">
            {connectionStatus === 'connecting' ? 'Connecting to server...' : 'Reconnecting...'}
          </p>
        </div>
      </div>
    );
  }

  // Show error if disconnected
  if (connectionStatus === 'disconnected') {
    return (
      <div className="flex h-screen items-center justify-center bg-base">
        <div className="text-center max-w-md">
          <div className="text-4xl mb-4">&#x26A0;</div>
          <h2 className="text-xl font-semibold mb-2">Connection Failed</h2>
          <p className="text-muted">
            Unable to connect to the server. Please check that the server is running and try again.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen items-center justify-center bg-base p-4">
      <div className="w-full max-w-lg">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold mb-2">Jaato</h1>
          <p className="text-muted">Select a workspace to begin</p>
        </div>

        {/* Error display */}
        {error && (
          <div className="mb-4 p-4 rounded-lg bg-error/10 border border-error/30 text-error">
            {error}
          </div>
        )}

        {/* Main content card */}
        <div className="bg-surface rounded-xl border border-border p-6 shadow-lg">
          {view === 'list' && (
            <>
              {isLoading && workspaces.length === 0 ? (
                <div className="flex justify-center py-8">
                  <Spinner />
                </div>
              ) : (
                <>
                  <WorkspaceList
                    workspaces={workspaces}
                    selectedWorkspace={selectedWorkspace}
                    onSelect={handleSelectWorkspace}
                    isLoading={isLoading}
                  />

                  <div className="mt-6 pt-4 border-t border-border">
                    <Button
                      variant="ghost"
                      onClick={() => setView('create')}
                      disabled={isLoading}
                      className="w-full"
                    >
                      + Create New Workspace
                    </Button>
                  </div>
                </>
              )}
            </>
          )}

          {view === 'create' && (
            <CreateWorkspace
              onCreate={handleCreateWorkspace}
              onCancel={() => setView('list')}
              isLoading={isLoading}
              existingNames={workspaces.map((w) => w.name)}
            />
          )}

          {view === 'configure' && configStatus && (
            <ConfigureWorkspace
              configStatus={configStatus}
              onSave={handleSaveConfig}
              onBack={handleBack}
              isLoading={isLoading}
            />
          )}
        </div>

        {/* Footer */}
        <p className="text-center text-sm text-muted mt-6">
          Workspaces contain project-specific configurations and history.
        </p>
      </div>
    </div>
  );
}
