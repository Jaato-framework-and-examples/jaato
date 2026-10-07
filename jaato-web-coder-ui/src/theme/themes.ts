/**
 * Themes are the TUI's own ``jaato-tui/themes/*.json`` files, imported
 * at build time so both clients share one palette definition.  The
 * eleven base colours become CSS custom properties on ``<html>``;
 * everything in ``theme.css`` (including Pygments token roles) is
 * expressed in terms of them.
 *
 * The redesign adds a twelfth, ``steel``: the INTERFACE accent that
 * carries the chrome (kickers, primary buttons, the selected tab, the
 * context bar) while the theme's own ``primary`` / ``success`` /
 * ``warning`` / ``error`` shrink to state glyphs.  It is not in the
 * theme files because the TUI has no equivalent role; it is derived here
 * from the theme's ground -- full steel on a light one, a lighter steel
 * on a dark one -- so a theme swap stays a theme swap.
 *
 * ``WEB_OVERRIDES`` is the one place the web client departs from a theme
 * file: the light theme's ground is the design's paper (``#f2f2f3`` /
 * white plates / near-black ink) rather than the terminal's ``#f5f5f5``
 * / pure black, and the dark theme's plates are the design's ``#202223``.
 * State colours are never overridden -- they are what makes a light
 * theme the TUI's light theme.
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

/** The interface accent per ground (see the module docstring). */
export const STEEL = { light: "#3F6690", dark: "#94bce3" } as const;

/** Web-only departures from the theme files; see the module docstring. */
const WEB_OVERRIDES: Record<string, Partial<ThemeColors>> = {
  light: { background: "#f2f2f3", surface: "#ffffff", text: "#1d1f20", text_muted: "#5d5d60", muted: "#7a7a7d" },
  dark: { surface: "#202223" },
};

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

/** The colours the web client paints for ``name``: the theme file plus the web overrides and the derived steel. */
export function resolveThemeColors(name: string): ThemeColors & { steel: string } {
  const t = THEMES[name] ?? THEMES.light!;
  const c = { ...t.colors, ...(WEB_OVERRIDES[t.name] ?? {}) };
  return { ...c, steel: isDarkTheme(t.name) ? STEEL.dark : STEEL.light };
}

export function applyTheme(name: string): void {
  const t = THEMES[name] ?? THEMES.light!;
  const root = document.documentElement;
  const c = resolveThemeColors(t.name);
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
    "--c-steel": c.steel,
  };
  for (const [k, v] of Object.entries(vars)) root.style.setProperty(k, v);
  root.dataset.theme = t.name;
  root.style.colorScheme = isDarkTheme(t.name) ? "dark" : "light";
}

const STORAGE_KEY = "jaato.theme";

/** The redesign is drawn on the light face first; ``theme dark`` (and the others) remain one command away. */
export const DEFAULT_THEME = "light";

export function loadThemePreference(): string {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v && THEMES[v]) return v;
  } catch {
    /* storage unavailable */
  }
  return DEFAULT_THEME;
}

export function saveThemePreference(name: string): void {
  try {
    localStorage.setItem(STORAGE_KEY, name);
  } catch {
    /* storage unavailable */
  }
}
