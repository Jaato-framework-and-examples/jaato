/**
 * CreateWorkspace - Form to create a new workspace
 */

import { useState } from 'react';
import { Button } from '@/components/common/Button';

interface CreateWorkspaceProps {
  onCreate: (name: string) => void;
  onCancel: () => void;
  isLoading: boolean;
  existingNames: string[];
}

export function CreateWorkspace({
  onCreate,
  onCancel,
  isLoading,
  existingNames,
}: CreateWorkspaceProps) {
  const [name, setName] = useState('');
  const [error, setError] = useState<string | null>(null);

  const validateName = (value: string): string | null => {
    if (!value.trim()) {
      return 'Name is required';
    }
    if (value.includes('/') || value.includes('\\')) {
      return 'Name cannot contain slashes';
    }
    if (value.startsWith('.')) {
      return 'Name cannot start with a dot';
    }
    if (existingNames.includes(value)) {
      return 'A workspace with this name already exists';
    }
    return null;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const validationError = validateName(name);
    if (validationError) {
      setError(validationError);
      return;
    }
    onCreate(name.trim());
  };

  const handleNameChange = (value: string) => {
    setName(value);
    if (error) {
      setError(validateName(value));
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="workspace-name" className="block text-sm font-medium mb-2">
          Workspace Name
        </label>
        <input
          id="workspace-name"
          type="text"
          value={name}
          onChange={(e) => handleNameChange(e.target.value)}
          placeholder="my-project"
          disabled={isLoading}
          autoFocus
          className={`
            w-full px-4 py-2 rounded-lg border bg-base text-base
            focus:outline-none focus:ring-2 focus:ring-primary/50
            ${error ? 'border-error' : 'border-border'}
            ${isLoading ? 'opacity-50' : ''}
          `}
        />
        {error && <p className="text-sm text-error mt-1">{error}</p>}
        <p className="text-sm text-muted mt-1">
          This will create a new directory under the workspace root.
        </p>
      </div>

      <div className="flex gap-3 justify-end">
        <Button variant="ghost" onClick={onCancel} disabled={isLoading}>
          Cancel
        </Button>
        <Button type="submit" disabled={isLoading || !name.trim()}>
          {isLoading ? 'Creating...' : 'Create Workspace'}
        </Button>
      </div>
    </form>
  );
}
