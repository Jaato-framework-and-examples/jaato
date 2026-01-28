/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Semantic colors via CSS variables
        primary: 'var(--color-primary)',
        secondary: 'var(--color-secondary)',
        success: 'var(--color-success)',
        warning: 'var(--color-warning)',
        error: 'var(--color-error)',
        surface: 'var(--color-surface)',
        border: 'var(--color-border)',
        muted: 'var(--color-text-muted)',
      },
      backgroundColor: {
        base: 'var(--color-bg)',
        surface: 'var(--color-surface)',
      },
      textColor: {
        base: 'var(--color-text)',
        muted: 'var(--color-text-muted)',
      },
      borderColor: {
        DEFAULT: 'var(--color-border)',
      },
    },
  },
  plugins: [],
};
