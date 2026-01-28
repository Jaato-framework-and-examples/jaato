/**
 * UI state management
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Theme = 'dark' | 'light' | 'system';

interface UIStore {
  // State
  theme: Theme;
  sidebarOpen: boolean;
  planPanelOpen: boolean;
  inputFocused: boolean;

  // Actions
  setTheme: (theme: Theme) => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  togglePlanPanel: () => void;
  setPlanPanelOpen: (open: boolean) => void;
  setInputFocused: (focused: boolean) => void;
}

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      theme: 'dark',
      sidebarOpen: true,
      planPanelOpen: true,
      inputFocused: false,

      setTheme: (theme) => {
        set({ theme });
        // Apply theme to document
        applyTheme(theme);
      },

      toggleSidebar: () => {
        set((state) => ({ sidebarOpen: !state.sidebarOpen }));
      },

      setSidebarOpen: (open) => {
        set({ sidebarOpen: open });
      },

      togglePlanPanel: () => {
        set((state) => ({ planPanelOpen: !state.planPanelOpen }));
      },

      setPlanPanelOpen: (open) => {
        set({ planPanelOpen: open });
      },

      setInputFocused: (focused) => {
        set({ inputFocused: focused });
      },
    }),
    {
      name: 'jaato-ui-preferences',
      partialize: (state) => ({
        theme: state.theme,
        sidebarOpen: state.sidebarOpen,
        planPanelOpen: state.planPanelOpen,
      }),
    }
  )
);

// Theme application helper
function applyTheme(theme: Theme) {
  const root = document.documentElement;

  if (theme === 'system') {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
  } else {
    root.setAttribute('data-theme', theme);
  }
}

// Initialize theme on load
if (typeof window !== 'undefined') {
  const stored = localStorage.getItem('jaato-ui-preferences');
  if (stored) {
    try {
      const { state } = JSON.parse(stored);
      if (state?.theme) {
        applyTheme(state.theme);
      }
    } catch {
      applyTheme('dark');
    }
  }

  // Listen for system theme changes
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    const currentTheme = useUIStore.getState().theme;
    if (currentTheme === 'system') {
      applyTheme('system');
    }
  });
}
