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
 * | ``.jaato/apparmor-fragments/jaato-environment.rules`` | ``#`` comment | the daemon, when it renders a session's AppArmor profile |
 * | ``.jaato/toolchain-offer.json`` | ``"_jaato_managed"`` key | the ``toolchain_offer`` enrichment plugin, in the runner |
 *
 * The first five follow the bindings; the offer also follows the operator's
 * allow-list, so it is rewritten on every status read too.  All are *generated* managed files: refreshed whenever their content
 * differs, and never written over a copy whose marker the user removed.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { ManagedFile } from "../managed-files.js";
import {
  APPARMOR_FRAGMENT_PATH,
  ENVIRONMENT_INSTRUCTIONS_PATH,
  ENVIRONMENT_MANIFEST_PATH,
  LOCAL_BIN,
  LSP_CONFIG_PATH,
  MARKER_APPARMOR,
  MARKER_ENVIRONMENT,
  MARKER_LSP,
  MARKER_TOOLCHAIN_OFFER,
  MARKER_TOOLCHAINS,
  MISE_CONFIG_PATH,
  TOOLCHAIN_OFFER_PATH,
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

/** A path AppArmor would read as a glob or could not quote; such a directory gets no rule (and a note). */
const APPARMOR_UNSAFE = /["*?[\]{}^\\\n\r]/;

/**
 * The bootstrap's AppArmor fragment: what the confinement template does not
 * already grant a bound toolchain.  The template gives ``.home`` ``rwkl``
 * and ``ix`` on ``.local/share/**\/bin/*``, which is enough for a static
 * binary (Go) or one that loads nothing of its own (Node).  A JDK loads its
 * own shared objects (``libjli.so``, ``libjvm.so``), which needs ``m``, and
 * forks through ``lib/jspawnhelper``, which is not under a ``bin/``.  Every
 * rule names a directory the bootstrap installed, never the workspace at
 * large.  ``bin/*`` is repeated here so a profile that scopes its fragments
 * (and so gets no broad exec) can name ``jaato-environment`` and run the
 * toolchains.  ``null`` when nothing is bound.
 *
 * ``notes`` receives one line per directory that could not be expressed.
 */
export function apparmorFragmentFile(m: Manifest, workspace: string, notes: string[] = []): ManagedFile | null {
  const dirs = new Map<string, boolean>();
  for (const t of m.toolchains) {
    if (t.installDir) dirs.set(t.installDir, (dirs.get(t.installDir) ?? false) || t.tool === "java");
    if (t.server?.runtimeDir) dirs.set(t.server.runtimeDir, true);
  }
  if (dirs.size === 0) return null;
  const lines = [
    "# Grants for the toolchains the web coder bound to this workspace; bind or unbind them there.",
    "# Profiles that scope apparmor_fragments name this one: jaato-environment.",
  ];
  for (const [rel, jdk] of [...dirs.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    const abs = join(workspace, rel);
    if (APPARMOR_UNSAFE.test(abs)) { notes.push(`no AppArmor grant for ${rel}: its path has characters AppArmor reads as a pattern`); continue; }
    lines.push(`"${abs}/bin/*" ix,`, `"${abs}/**/*.so" m,`, `"${abs}/**/*.so.*" m,`);
    if (jdk) lines.push(`"${abs}/lib/jspawnhelper" ix,`);
  }
  return { relativePath: APPARMOR_FRAGMENT_PATH, markerId: MARKER_APPARMOR, version: MANIFEST_VERSION, body: `${lines.join("\n")}\n`, format: "hash", generated: true };
}

export function apparmorFragmentIdentity(): ManagedFile {
  return { relativePath: APPARMOR_FRAGMENT_PATH, markerId: MARKER_APPARMOR, version: MANIFEST_VERSION, body: "", format: "hash", generated: true };
}

/** Version of the offer's shape; the plugin gives no hint for one it does not know. */
export const TOOLCHAIN_OFFER_SCHEMA = 1;

/**
 * The offer: every toolchain the operator allows, its versions, the commands
 * that suggest it (from {@link TOOLCHAINS}, the one command table), and the
 * version bound here or ``null``.  An empty ``allowed`` gives an offer with
 * no toolchains, which the plugin reads as "nothing to hint".
 *
 * The backend does not write this file: it returns it in the environment
 * status and the page stages it into the workspace through the daemon
 * (``StageFilesRequest``), because this process may run as an account that
 * cannot write the workspace.  Structured data only: the file is in the
 * workspace, where an unconfined model can write, so the plugin validates
 * every field and builds the sentence itself.
 */
export function toolchainOfferFile(allowed: Array<{ tool: ToolId; label: string; versions: string[] }>, m: Manifest): ManagedFile {
  const bound = new Map(m.toolchains.map((t) => [t.tool, t.version]));
  const body = JSON.stringify({
    schema: TOOLCHAIN_OFFER_SCHEMA,
    toolchains: allowed.map((a) => ({
      tool: a.tool, label: a.label, versions: a.versions, commands: TOOLCHAINS[a.tool].commands, bound: bound.get(a.tool) ?? null,
    })),
  });
  return { relativePath: TOOLCHAIN_OFFER_PATH, markerId: MARKER_TOOLCHAIN_OFFER, version: MANIFEST_VERSION, body, format: "json", generated: true };
}
