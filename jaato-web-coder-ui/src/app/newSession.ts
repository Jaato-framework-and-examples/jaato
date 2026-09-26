/**
 * The session picker's New session column (design 2a), as pure data:
 * which base profiles there are, what model each one leaves the session
 * with, and what ``session.new`` is asked for on Start.
 *
 * Three model cases, decided by the base profile alone:
 *
 *  • ``default`` -- the workspace ``.env`` / global basic profile, which
 *    defines no model of its own: a provider and a model are REQUIRED here,
 *    and there is no inherit/override choice to make.
 *  • a profile that defines a model -- INHERIT it (the default) or
 *    OVERRIDE it for this session only; the profile file is not touched.
 *  • a profile that defines none -- the same required pick as ``default``,
 *    but worded as the profile's gap rather than the picker's question.
 *
 * The override reaches the daemon as ``session.new --model/--provider``
 * (protocol 1.26), applied to a copy of the resolved profile, so the base
 * profile keeps its plugins, persona and limits and only the binding
 * changes.  ``default`` with a picked model sends no ``--profile`` at all.
 */
import type { ProfileInfo } from "@/store/types";

/** The name the picker gives the no-profile base. */
export const DEFAULT_PROFILE = "default";

export interface BaseProfile {
  name: string;
  description: string;
  provider: string;
  model: string;
  /** The no-profile base (``session.new`` without ``--profile``). */
  isDefault: boolean;
  /**
   * ``default`` on a daemon with NO workspace mode: there is no workspace
   * config to read, and the daemon's own ``.env`` usually binds a model, so
   * a pick is optional there rather than required.  Never set in workspace
   * mode, where the design's rule holds -- ``default`` requires a pick.
   */
  envFallback?: boolean;
}

export type ModelMode = "inherit" | "override";

export interface ModelChoice {
  provider: string;
  model: string;
}

/**
 * The base profiles, ``default`` first.  A daemon that lists a real
 * profile file named ``default`` supplies that entry (with whatever model
 * it defines) in place of the synthesised one.
 */
export function baseProfiles(profiles: ProfileInfo[], opts: { envFallback?: boolean } = {}): BaseProfile[] {
  const toBase = (p: ProfileInfo): BaseProfile => ({
    name: p.name,
    description: String(p.description ?? ""),
    provider: String(p.provider ?? ""),
    model: String(p.model ?? ""),
    isDefault: p.name === DEFAULT_PROFILE,
  });
  const listed = profiles.filter((p) => p.name).map(toBase);
  const own = listed.find((p) => p.isDefault);
  const rest = listed.filter((p) => !p.isDefault).sort((a, b) => a.name.localeCompare(b.name));
  const def: BaseProfile = own ?? {
    name: DEFAULT_PROFILE,
    description: opts.envFallback ? "Basic profile: the daemon's .env provider and model." : "Basic profile. Defines no model.",
    provider: "", model: "", isDefault: true, envFallback: opts.envFallback === true,
  };
  return [def, ...rest];
}

/** Does this base profile bind a model of its own? */
export function definesModel(p: BaseProfile): boolean {
  return Boolean(p.model);
}

/** ``provider / model``, or the model alone when no provider is named. */
export function formatBinding(provider: string, model: string): string {
  return [provider, model].filter(Boolean).join(" / ");
}

/** The model tag under a profile: ``model: you select`` / ``model: p / m`` / ``model: not defined``. */
export function modelTag(p: BaseProfile): string {
  if (definesModel(p)) return `model: ${formatBinding(p.provider, p.model)}`;
  if (p.envFallback) return "model: daemon .env, or you select";
  return p.isDefault ? "model: you select" : "model: not defined";
}

/** The binding an ``envFallback`` default resolves to with nothing picked: whatever the daemon's .env says. */
export const ENV_BINDING: ModelChoice = Object.freeze({ provider: "", model: "" });

/**
 * The model the session will run on, or ``null`` while one is still
 * missing -- which is exactly when Start is disabled.  ``ENV_BINDING``
 * means "the daemon decides from its .env" (``envFallback`` only).
 */
export function resolveModel(base: BaseProfile, mode: ModelMode, pick: ModelChoice): ModelChoice | null {
  if (definesModel(base) && mode === "inherit") return { provider: base.provider, model: base.model };
  if (pick.provider && pick.model.trim()) return { provider: pick.provider, model: pick.model.trim() };
  if (base.envFallback) return ENV_BINDING;
  return null;
}

/** The ``Model · …`` status line, and whether it is still a warning. */
export function modelStatus(base: BaseProfile, mode: ModelMode, pick: ModelChoice): { text: string; warn: boolean } {
  if (definesModel(base)) {
    return mode === "inherit"
      ? { text: `inherited from ${base.name}`, warn: false }
      : { text: `overriding ${base.model}`, warn: false };
  }
  if (base.envFallback) return { text: resolveModel(base, mode, pick) === ENV_BINDING ? "from the daemon's .env" : "selected", warn: false };
  if (base.isDefault) return { text: "selected", warn: false };
  const filled = resolveModel(base, mode, pick) !== null;
  return { text: `${base.name} defines none, select one`, warn: !filled };
}

/**
 * What Start asks ``session.new`` for.  An inherited model sends nothing
 * but the profile -- the daemon resolves the profile's own binding, so
 * nothing the picker read can go stale between listing and start.
 */
export function startRequest(base: BaseProfile, mode: ModelMode, pick: ModelChoice): { profile: string | null; model?: ModelChoice } | null {
  const resolved = resolveModel(base, mode, pick);
  if (!resolved) return null;
  const profile = base.isDefault && !definesModel(base) ? null : base.name;
  if (definesModel(base) && mode === "inherit") return { profile };
  if (resolved === ENV_BINDING) return { profile };
  return { profile, model: resolved };
}

/**
 * The model names the picker can suggest for a provider.  There is no
 * catalog verb, so the suggestions are what this client has SEEN bound to
 * that provider -- profiles, sessions, the workspace ``.env`` -- and the
 * field stays free text for a model nobody has used yet.
 */
export function knownModels(provider: string, seen: ModelChoice[]): string[] {
  const out = new Set<string>();
  for (const s of seen) if (s.model && (!provider || s.provider === provider)) out.add(s.model);
  return [...out].sort();
}

/** The providers to offer: the daemon's configurable ones, plus any seen bound in a profile or session. */
export function knownProviders(available: string[], seen: ModelChoice[]): string[] {
  const out = new Set(available.filter(Boolean));
  for (const s of seen) if (s.provider) out.add(s.provider);
  return [...out].sort();
}

/** The Start summary: ``profile · provider / model · N files``, or the reason there is none. */
export function startSummary(base: BaseProfile, resolved: ModelChoice | null, files: number): string {
  if (!resolved) return "Select a model to start";
  const parts = [base.name, resolved === ENV_BINDING ? "daemon .env" : formatBinding(resolved.provider, resolved.model)];
  if (files > 0) parts.push(`${files} ${files === 1 ? "file" : "files"}`);
  return parts.join(" · ");
}
