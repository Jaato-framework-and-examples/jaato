/**
 * Themes are the TUI's own ``jaato-tui/themes/*.json`` files, imported
 * at build time so both clients share one palette definition.  The
 * eleven base colours become CSS custom properties on ``<html>``;
 * everything in ``theme.css`` (including Pygments token roles) is
 * expressed in terms of them.
 */
import dark from "../../../jaato-tui/themes/dark.json";
import light from "../../../jaato-tui/themes/light.json";
import highContrast from "../../../jaato-tui/themes/high-contrast.json";
import dracula from "../../../jaato-tui/themes/dracula.json";
import latte from "../../../jaato-tui/themes/latte.json";
import mocha from "../../../jaato-tui/themes/mocha.json";

export interface ThemeColors {
  primary: string;
  secondary: string;
  accent?: string;
  success: string;
  warning: string;
  error: string;
  muted: string;
  background: string;
  surface: string;
  text: string;
  text_muted: string;
}

export interface ThemeFile {
  name: string;
  description?: string;
  colors: ThemeColors;
}

const files = [dark, light, highContrast, dracula, latte, mocha] as unknown as ThemeFile[];

export const THEMES: Record<string, ThemeFile> = Object.fromEntries(files.map((t) => [t.name, t]));
export const THEME_NAMES = Object.keys(THEMES);

function luminance(hex: string): number {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!m) return 0;
  const [r, g, b] = [m[1], m[2], m[3]].map((h) => parseInt(h!, 16) / 255);
  return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!;
}

export function isDarkTheme(name: string): boolean {
  const t = THEMES[name];
  return t ? luminance(t.colors.background) < 0.5 : true;
}

export function applyTheme(name: string): void {
  const t = THEMES[name] ?? THEMES.dark!;
  const root = document.documentElement;
  const c = t.colors;
  const vars: Record<string, string> = {
    "--c-primary": c.primary,
    "--c-secondary": c.secondary,
    "--c-accent": c.accent ?? c.secondary,
    "--c-success": c.success,
    "--c-warning": c.warning,
    "--c-error": c.error,
    "--c-muted": c.muted,
    "--c-bg": c.background,
    "--c-surface": c.surface,
    "--c-text": c.text,
    "--c-text-muted": c.text_muted,
  };
  for (const [k, v] of Object.entries(vars)) root.style.setProperty(k, v);
  root.dataset.theme = t.name;
  root.style.colorScheme = isDarkTheme(t.name) ? "dark" : "light";
}

const STORAGE_KEY = "jaato.theme";

export function loadThemePreference(): string {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v && THEMES[v]) return v;
  } catch {
    /* storage unavailable */
  }
  return "dark";
}

export function saveThemePreference(name: string): void {
  try {
    localStorage.setItem(STORAGE_KEY, name);
  } catch {
    /* storage unavailable */
  }
}
