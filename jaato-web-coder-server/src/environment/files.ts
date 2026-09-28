/**
 * The one file this server produces for a workspace: the toolchain offer.
 *
 * ``.jaato/toolchain-offer.json`` is the operator's policy, read by the web
 * coder's toolchains plugin in the session's runner.  This server does not
 * write it (it may run as an account that cannot write the workspace): it
 * returns the content in the environment status and the PAGE stages it
 * through the daemon (``StageFilesRequest``).  The plugin validates every
 * field, and the plugin's AppArmor fragment denies a confined session
 * writing the file.
 *
 * Schema 2 (the plugin's ``offer.py`` is the reader):
 *
 * ```json
 * {"schema": 2,
 *  "toolchains": [{"tool": "java", "label": "Java", "versions": ["21"]}],
 *  "servers": {"jdtls": {"version": "1.40.0", "java": "21", "max_heap": "1G"}},
 *  "install": {"timeout_seconds": 900, "paranoid": false}}
 * ```
 */
import type { ManagedFile } from "../managed-files.js";
import { MARKER_TOOLCHAIN_OFFER, TOOLCHAIN_OFFER_PATH, type ServerId, type ToolId } from "./catalog.js";

export const TOOLCHAIN_OFFER_SCHEMA = 2;
const OFFER_VERSION = 2;

/** One server's settings as the offer carries them; ``version`` always present. */
export type ServerSettings = Record<string, string>;

export interface OfferPolicy {
  allowed: Array<{ tool: ToolId; label: string; versions: string[] }>;
  servers: Partial<Record<ServerId, ServerSettings>>;
  installTimeoutSeconds: number;
  paranoid: boolean;
}

/** The offer as a managed JSON file.  An empty ``allowed`` gives an offer with no toolchains, which the plugin reads as "nothing to bind, nothing to hint". */
export function toolchainOfferFile(p: OfferPolicy): ManagedFile {
  const body = JSON.stringify({
    schema: TOOLCHAIN_OFFER_SCHEMA,
    toolchains: p.allowed.map((a) => ({ tool: a.tool, label: a.label, versions: a.versions })),
    servers: p.servers,
    install: { timeout_seconds: p.installTimeoutSeconds, paranoid: p.paranoid },
  });
  return { relativePath: TOOLCHAIN_OFFER_PATH, markerId: MARKER_TOOLCHAIN_OFFER, version: OFFER_VERSION, body, format: "json", generated: true };
}
