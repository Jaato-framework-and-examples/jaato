/**
 * ConfigureWorkspace - Provider and model configuration form
 */

import { useState, useEffect } from 'react';
import { Button } from '@/components/common/Button';

interface ConfigStatus {
  workspace: string;
  configured: boolean;
  provider?: string;
  model?: string;
  availableProviders: string[];
  missingFields: string[];
}

interface ConfigureWorkspaceProps {
  configStatus: ConfigStatus;
  onSave: (provider: string, model?: string, apiKey?: string) => void;
  onBack: () => void;
  isLoading: boolean;
}

// Provider-specific model suggestions
const PROVIDER_MODELS: Record<string, string[]> = {
  anthropic: ['claude-sonnet-4-20250514', 'claude-opus-4-20250514', 'claude-3-5-haiku-20241022'],
  google: ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash'],
  github: ['gpt-4o', 'gpt-4o-mini', 'o1-preview', 'o1-mini'],
  ollama: ['qwen3:32b', 'llama3.1:70b', 'mistral:latest'],
  antigravity: ['antigravity-gemini-3-pro', 'antigravity-claude-sonnet-4-5'],
  claude_cli: ['claude-sonnet-4-20250514'],
};

// Providers that need API keys
const PROVIDERS_NEED_KEY = ['anthropic', 'github'];

// Providers that use OAuth
const PROVIDERS_OAUTH = ['antigravity'];

export function ConfigureWorkspace({
  configStatus,
  onSave,
  onBack,
  isLoading,
}: ConfigureWorkspaceProps) {
  const [provider, setProvider] = useState(configStatus.provider || '');
  const [model, setModel] = useState(configStatus.model || '');
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);

  // Update model when provider changes
  useEffect(() => {
    if (provider && PROVIDER_MODELS[provider]) {
      // Set default model for provider if not already set
      if (!model || !PROVIDER_MODELS[provider].includes(model)) {
        setModel(PROVIDER_MODELS[provider][0] || '');
      }
    }
  }, [provider, model]);

  const needsApiKey = PROVIDERS_NEED_KEY.includes(provider);
  const usesOAuth = PROVIDERS_OAUTH.includes(provider);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(provider, model || undefined, apiKey || undefined);
  };

  const canSubmit = provider && (!needsApiKey || apiKey);

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="text-center mb-6">
        <h2 className="text-xl font-semibold">Configure {configStatus.workspace}</h2>
        <p className="text-muted text-sm mt-1">
          Set up your AI provider and model for this workspace.
        </p>
      </div>

      {/* Provider Selection */}
      <div>
        <label htmlFor="provider" className="block text-sm font-medium mb-2">
          Provider
        </label>
        <select
          id="provider"
          value={provider}
          onChange={(e) => setProvider(e.target.value)}
          disabled={isLoading}
          className="w-full px-4 py-2 rounded-lg border border-border bg-base text-base focus:outline-none focus:ring-2 focus:ring-primary/50"
        >
          <option value="">Select a provider...</option>
          {configStatus.availableProviders.map((p) => (
            <option key={p} value={p}>
              {p.charAt(0).toUpperCase() + p.slice(1).replace('_', ' ')}
            </option>
          ))}
        </select>
      </div>

      {/* Model Selection */}
      {provider && (
        <div>
          <label htmlFor="model" className="block text-sm font-medium mb-2">
            Model
          </label>
          <select
            id="model"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={isLoading}
            className="w-full px-4 py-2 rounded-lg border border-border bg-base text-base focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="">Select a model...</option>
            {(PROVIDER_MODELS[provider] || []).map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <p className="text-sm text-muted mt-1">
            Or enter a custom model name in the .env file.
          </p>
        </div>
      )}

      {/* API Key (for providers that need it) */}
      {needsApiKey && (
        <div>
          <label htmlFor="api-key" className="block text-sm font-medium mb-2">
            API Key
          </label>
          <div className="relative">
            <input
              id="api-key"
              type={showApiKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={`Enter your ${provider} API key`}
              disabled={isLoading}
              className="w-full px-4 py-2 pr-20 rounded-lg border border-border bg-base text-base focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
            <button
              type="button"
              onClick={() => setShowApiKey(!showApiKey)}
              className="absolute right-2 top-1/2 -translate-y-1/2 px-2 py-1 text-sm text-muted hover:text-base"
            >
              {showApiKey ? 'Hide' : 'Show'}
            </button>
          </div>
          <p className="text-sm text-muted mt-1">
            Your API key will be stored in the workspace's .env file.
          </p>
        </div>
      )}

      {/* OAuth notice */}
      {usesOAuth && (
        <div className="p-4 rounded-lg bg-surface border border-border">
          <p className="text-sm">
            <strong>{provider}</strong> uses OAuth for authentication.
            After saving, use the <code className="px-1 bg-base rounded">auth login</code> command
            to complete the authentication flow.
          </p>
        </div>
      )}

      {/* Missing fields warning */}
      {configStatus.missingFields.length > 0 && (
        <div className="p-4 rounded-lg bg-warning/10 border border-warning/30">
          <p className="text-sm text-warning">
            Missing configuration: {configStatus.missingFields.join(', ')}
          </p>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3 justify-between pt-4">
        <Button variant="ghost" onClick={onBack} disabled={isLoading}>
          Back
        </Button>
        <Button type="submit" disabled={isLoading || !canSubmit}>
          {isLoading ? 'Saving...' : 'Save Configuration'}
        </Button>
      </div>
    </form>
  );
}
