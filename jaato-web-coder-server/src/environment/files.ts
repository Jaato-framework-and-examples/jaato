/**
 * The files an environment binding writes, rendered from the manifest
 * (docs/design/web-coder-environment-bootstrap.md §6).
 *
 * The manifest (``.jaato/environment.json``) is the record of what is bound
 * in a workspace.  It belongs to the WORKSPACE, not to a user: a second
 * person the workspace is shared with sees the same toolchains, which is
 * what the runner sees.  Every other file is derived from it:
 *
 * | File | Marker | Read by |
 * |---|---|---|
 * | ``.jaato/environment.json`` | ``"_jaato_managed"`` key | ``get_environment(aspect="runtime")`` (#1346) |
 * | ``.home/.config/mise/config.toml`` | ``#`` comment | mise, for the person who opens a shell |
 * | ``.lsp.json`` | ``"_jaato_managed"`` key | the ``lsp`` plugin (it reads only ``languageServers``) |
 * | ``.jaato/instructions/45-environment.md`` | HTML comment | every session's system prompt |
 *
 * All four are *generated* managed files: refreshed whenever their content
 * differs, and never written over a copy whose marker the user removed.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { ManagedFile } from "../managed-files.js";
import {
  ENVIRONMENT_INSTRUCTIONS_PATH,
  ENVIRONMENT_MANIFEST_PATH,
  LOCAL_BIN,
  LSP_CONFIG_PATH,
  MARKER_ENVIRONMENT,
  MARKER_LSP,
  MARKER_TOOLCHAINS,
  MISE_CONFIG_PATH,
  TOOLCHAINS,
  isToolId,
  type ServerId,
  type ToolId,
} from "./catalog.js";
import type { InstalledServer } from "./installer.js";

export const MANIFEST_VERSION = 1;

/** One bound toolchain, as the manifest records it. */
export interface BoundToolchain {
  tool: ToolId;
  version: string;
  installDir: string | null;
  bin: string[];
  server: InstalledServer | null;
  /** Files the server install put in ``.home/.local/bin`` that are not links (gopls), removed on unbind. */
  serverBin: string[];
  boundAt: string;
}

export interface Manifest {
  toolchains: BoundToolchain[];
}

/** The manifest on disk, or an empty one.  Read whatever its marker says: it is the record. */
export function readManifest(workspace: string): Manifest {
  let raw: unknown;
  try { raw = JSON.parse(readFileSync(join(workspace, ENVIRONMENT_MANIFEST_PATH), "utf8")); } catch { return { toolchains: [] }; }
  const list = (raw as { toolchains?: unknown })?.toolchains;
  if (!Array.isArray(list)) return { toolchains: [] };
  const toolchains = list.filter((t): t is BoundToolchain => !!t && typeof t === "object" && isToolId((t as BoundToolchain).tool) && typeof (t as BoundToolchain).version === "string")
    .map((t) => ({ ...t, bin: Array.isArray(t.bin) ? t.bin.filter((b) => typeof b === "string") : [], serverBin: Array.isArray(t.serverBin) ? t.serverBin.filter((b) => typeof b === "string") : [], server: t.server ?? null, installDir: t.installDir ?? null }));
  return { toolchains };
}

export function manifestFile(m: Manifest): ManagedFile {
  const body = JSON.stringify({
    note: "Toolchains bound to this workspace by the web coder. Edit them from the web coder, not here.",
    toolchains: [...m.toolchains].sort((a, b) => a.tool.localeCompare(b.tool)),
  });
  return { relativePath: ENVIRONMENT_MANIFEST_PATH, markerId: MARKER_ENVIRONMENT, version: MANIFEST_VERSION, body, format: "json", generated: true };
}

/** TOML string, the one quoting rule this file needs. */
function tomlString(v: string): string {
  return `"${v.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

export function miseConfigFile(m: Manifest): ManagedFile {
  const lines = ["# Toolchains bound by the web coder; bind or unbind them there.", "[tools]"];
  for (const t of [...m.toolchains].sort((a, b) => a.tool.localeCompare(b.tool))) {
    const mise = TOOLCHAINS[t.tool].mise;
    if (mise) lines.push(`${mise} = ${tomlString(t.version)}`);
  }
  return { relativePath: MISE_CONFIG_PATH, markerId: MARKER_TOOLCHAINS, version: MANIFEST_VERSION, body: `${lines.join("\n")}\n`, format: "hash", generated: true };
}

/** The ``.lsp.json`` for the bound servers, or ``null`` when none is bound (the file is then removed). */
export function lspConfigFile(m: Manifest): ManagedFile | null {
  const servers: Record<string, { command: string; args: string[]; languageId: string }> = {};
  for (const t of m.toolchains) {
    if (t.server) servers[t.server.language] = { command: t.server.command, args: t.server.args, languageId: t.server.language };
  }
  if (Object.keys(servers).length === 0) return null;
  return { relativePath: LSP_CONFIG_PATH, markerId: MARKER_LSP, version: MANIFEST_VERSION, body: JSON.stringify({ languageServers: servers }), format: "json", generated: true };
}

export function lspConfigIdentity(): ManagedFile {
  return { relativePath: LSP_CONFIG_PATH, markerId: MARKER_LSP, version: MANIFEST_VERSION, body: "{}", format: "json", generated: true };
}

/** The instruction file: what is installed and where more comes from.  ``null`` when nothing is bound. */
export function environmentInstructionsFile(m: Manifest): ManagedFile | null {
  if (m.toolchains.length === 0) return null;
  const rows = [...m.toolchains].sort((a, b) => a.tool.localeCompare(b.tool)).map((t) => {
    const spec = TOOLCHAINS[t.tool];
    const bins = t.bin.length ? ` (\`${t.bin.join("`, `")}\`)` : "";
    const server = t.server ? `; language server: ${t.server.id} ${t.server.version}` : "";
    const version = t.tool === "python" ? "" : ` ${t.version}`;
    return `- ${spec.label}${version}${bins}${server}`;
  });
  const body = [
    "# Toolchains in this workspace",
    "",
    "The user bound these toolchains to this workspace. Their binaries are",
    `linked into \`~/${LOCAL_BIN.slice(".home/".length)}\`, which is on every command's \`PATH\`:`,
    "",
    ...rows,
    "",
    "Do not install another version of these with a system package manager or",
    "a download. If a toolchain you need is missing, say so: the user binds",
    "toolchains from the web coder. For what can run right now, including",
    "anything changed since this prompt was written, call",
    "`get_environment(aspect=\"runtime\")`.",
    "",
  ].join("\n");
  return { relativePath: ENVIRONMENT_INSTRUCTIONS_PATH, markerId: MARKER_ENVIRONMENT, version: MANIFEST_VERSION, body, generated: true };
}

export function environmentInstructionsIdentity(): ManagedFile {
  return { relativePath: ENVIRONMENT_INSTRUCTIONS_PATH, markerId: MARKER_ENVIRONMENT, version: MANIFEST_VERSION, body: "", generated: true };
}

/** The server a toolchain brings, when the operator pinned it. */
export function pinnedServer(tool: ToolId, pins: Partial<Record<ServerId, string>>): { id: ServerId; version: string } | null {
  const id = TOOLCHAINS[tool].server;
  if (!id) return null;
  const version = pins[id];
  return version ? { id, version } : null;
}
