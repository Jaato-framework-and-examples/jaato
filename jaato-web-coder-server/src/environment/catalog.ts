/**
 * What the environment bootstrap knows how to install
 * (docs/design/web-coder-environment-bootstrap.md §4, §5).
 *
 * Two tables, both deliberately small and closed:
 *
 * - {@link TOOLCHAINS}: the toolchains this server can install with mise AND
 *   link into ``<ws>/.home/.local/bin`` so a confined session can run them.
 *   An operator's allow-list (``environment.tools``) may name only these.
 *   Rust is absent on purpose: rustup keeps its own ``~/.rustup`` /
 *   ``~/.cargo`` homes outside mise's data directory, and its proxies need
 *   ``RUSTUP_HOME`` at run time, which nothing in a session sets.  Java is
 *   absent until #806 (language servers never reaped) is fixed.
 * - {@link LANGUAGE_SERVERS}: one server per toolchain, and how to install
 *   it without writing outside ``<ws>/.home``.  Every server is enabled only
 *   when the operator PINS its version (``environment.lsp.<name>``); an
 *   unpinned server is never installed.
 *
 * ``python`` is a pseudo-toolchain: every managed workspace already has
 * Python, so binding it installs only basedpyright.
 *
 * Every path here is relative to the workspace and lands under ``.home``,
 * where the confinement template already grants exec
 * (``.home/.local/bin/*`` and ``.home/.local/share/**\/bin/*``, #1273/#1274).
 */

/** A toolchain id the page and the routes use. */
export type ToolId = "python" | "node" | "go" | "bun";

/** A language-server id, as ``environment.lsp`` pins it. */
export type ServerId = "basedpyright" | "typescript-language-server" | "gopls";

export interface ToolchainSpec {
  id: ToolId;
  /** What the page shows. */
  label: string;
  /** mise's name for it; ``null`` for the Python pseudo-toolchain (nothing to install). */
  mise: string | null;
  /** The ``.tool-versions`` / ``mise.toml`` names that pin it. */
  toolVersionsNames: string[];
  /** The language server bound with it, if any. */
  server: ServerId | null;
  /** Names a ``command not found`` for which suggests this toolchain (phase 4's mid-session chip). */
  commands: string[];
}

export const TOOLCHAINS: Record<ToolId, ToolchainSpec> = {
  python: { id: "python", label: "Python (language server only)", mise: null, toolVersionsNames: [], server: "basedpyright", commands: ["basedpyright", "basedpyright-langserver", "pyright"] },
  node: { id: "node", label: "Node.js", mise: "node", toolVersionsNames: ["node", "nodejs"], server: "typescript-language-server", commands: ["node", "npm", "npx", "corepack", "tsc"] },
  go: { id: "go", label: "Go", mise: "go", toolVersionsNames: ["go", "golang"], server: "gopls", commands: ["go", "gofmt", "gopls"] },
  bun: { id: "bun", label: "Bun", mise: "bun", toolVersionsNames: ["bun"], server: null, commands: ["bun", "bunx"] },
};

export const TOOL_IDS = Object.keys(TOOLCHAINS) as ToolId[];

export function isToolId(v: unknown): v is ToolId {
  return typeof v === "string" && (TOOL_IDS as string[]).includes(v);
}

export interface ServerSpec {
  id: ServerId;
  /** The ``.lsp.json`` key and ``languageId``. */
  language: string;
  /** The toolchain it needs installed first. */
  needs: ToolId;
}

export const LANGUAGE_SERVERS: Record<ServerId, ServerSpec> = {
  basedpyright: { id: "basedpyright", language: "python", needs: "python" },
  "typescript-language-server": { id: "typescript-language-server", language: "typescript", needs: "node" },
  gopls: { id: "gopls", language: "go", needs: "go" },
};

/** Where a language server this bootstrap installed lives, under the workspace. */
export const LSP_DIR = ".home/.local/share/jaato-lsp";
/** Where linked binaries go: on every command's ``PATH`` already (#1273). */
export const LOCAL_BIN = ".home/.local/bin";
/** mise's directories, under the workspace HOME. */
export const MISE_DATA_DIR = ".home/.local/share/mise";
export const MISE_CONFIG_DIR = ".home/.config/mise";
export const MISE_CACHE_DIR = ".home/.cache/mise";
export const MISE_STATE_DIR = ".home/.local/state/mise";
/** The managed mise global config: the bindings, under the workspace HOME, never in the repository. */
export const MISE_CONFIG_PATH = `${MISE_CONFIG_DIR}/config.toml`;

/** The files the bootstrap owns, and their marker ids. */
export const ENVIRONMENT_MANIFEST_PATH = ".jaato/environment.json";
export const ENVIRONMENT_INSTRUCTIONS_PATH = ".jaato/instructions/45-environment.md";
export const LSP_CONFIG_PATH = ".lsp.json";
export const MARKER_TOOLCHAINS = "toolchains";
export const MARKER_ENVIRONMENT = "environment";
export const MARKER_LSP = "lsp";

/**
 * The allowed version (from the operator's list) a detected pin maps to, or
 * ``null``.  A pin matches an allowed entry when it equals it or extends it at
 * a component boundary: ``22.11.0`` matches ``22`` and ``22.11``; ``220`` does
 * not match ``22``.  The most specific allowed entry wins.  A leading ``v``,
 * ``>=`` / ``^`` / ``~`` and whitespace are ignored on the pin side.
 */
export function matchAllowedVersion(pin: string, allowed: string[]): string | null {
  const p = pin.trim().replace(/^(?:>=|\^|~|=|v)+/, "").trim();
  if (!p) return null;
  let best: string | null = null;
  for (const a of allowed) {
    const ok = p === a || p.startsWith(`${a}.`);
    if (ok && (best === null || a.length > best.length)) best = a;
  }
  return best;
}
