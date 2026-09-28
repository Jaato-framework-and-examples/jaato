/**
 * The toolchains the operator may allow (docs/design/web-coder-environment-bootstrap.md §4, §5).
 *
 * This server holds POLICY only: which toolchains and versions a workspace
 * may bind, and which language server versions are pinned.  It never
 * installs, detects or writes anything in a workspace; the web coder's
 * toolchains plugin does that, in the session's runner
 * (``plugin/jaato_web_coder_toolchains``), reading the policy from the offer
 * the page stages.  The plugin's ``catalog.py`` is the full table (mise
 * names, commands, install recipes); this one keeps what configuration
 * checking and the page need: the ids, their labels, and which server each
 * brings.  ``test/environment.test.ts`` checks the two agree on the ids.
 *
 * ``python`` is a pseudo-toolchain: binding it installs only basedpyright.
 * Rust is absent on purpose (rustup keeps its homes outside mise).
 */

/** A toolchain id the page and the plugin use. */
export type ToolId = "python" | "node" | "go" | "bun" | "java" | "maven" | "gradle";

/** A language-server id, as ``environment.lsp`` pins it. */
export type ServerId = "basedpyright" | "typescript-language-server" | "gopls" | "jdtls";

export interface ToolchainSpec {
  id: ToolId;
  /** What the page shows. */
  label: string;
  /** Whether mise installs it (``false`` only for the python pseudo-toolchain). */
  installable: boolean;
  /** The language server bound with it, if any. */
  server: ServerId | null;
}

export const TOOLCHAINS: Record<ToolId, ToolchainSpec> = {
  python: { id: "python", label: "Python (language server only)", installable: false, server: "basedpyright" },
  node: { id: "node", label: "Node.js", installable: true, server: "typescript-language-server" },
  go: { id: "go", label: "Go", installable: true, server: "gopls" },
  bun: { id: "bun", label: "Bun", installable: true, server: null },
  java: { id: "java", label: "Java", installable: true, server: "jdtls" },
  maven: { id: "maven", label: "Maven", installable: true, server: null },
  gradle: { id: "gradle", label: "Gradle", installable: true, server: null },
};

export const TOOL_IDS = Object.keys(TOOLCHAINS) as ToolId[];

export function isToolId(v: unknown): v is ToolId {
  return typeof v === "string" && (TOOL_IDS as string[]).includes(v);
}

export const SERVER_IDS: ServerId[] = ["basedpyright", "typescript-language-server", "gopls", "jdtls"];

/** Where the page stages the offer; the plugin reads it there. */
export const TOOLCHAIN_OFFER_PATH = ".jaato/toolchain-offer.json";
export const MARKER_TOOLCHAIN_OFFER = "toolchain-offer";
