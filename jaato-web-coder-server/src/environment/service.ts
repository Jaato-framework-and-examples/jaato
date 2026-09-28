/**
 * The environment bootstrap's policy (docs/design/web-coder-environment-bootstrap.md).
 *
 * This server holds the operator's allow-list and the language-server pins,
 * and the per-user "not now" answers to proposals.  It touches no workspace:
 * it may run as an account that cannot read or write the workspaces (a web
 * coder beside a root daemon).  Everything that acts on a workspace is the
 * web coder's toolchains plugin, in the session's runner:
 *
 * | Who | Does |
 * |---|---|
 * | this service | answers the status: the allow-list, the declines, and the offer's content |
 * | the page | stages the offer (``StageFilesRequest``), sends ``toolchain bind|unbind|scan`` (a user command), reads ``.jaato/environment.json`` (``workspace.file.fetch``) |
 * | the plugin | installs into ``<ws>/.home``, writes ``.lsp.json`` / the mise config / the manifest, proposes from the repositories' markers |
 *
 * Every route asks the daemon, over the bind channel, whether the signed-in
 * user owns the workspace (an empty ``workspace.app_write``, protocol 1.30).
 * A daemon that cannot answer refuses the request.
 */
import { managedContent } from "../managed-files.js";
import { SERVER_IDS, TOOLCHAINS, TOOL_IDS, type ServerId, type ToolId } from "./catalog.js";
import { toolchainOfferFile, type ServerSettings } from "./files.js";
import type { FileEnvironmentStore } from "./store.js";

/** Whether ``user`` owns ``workspace`` on the daemon; ``null`` when the daemon could not be asked. */
export interface WorkspaceOwnership {
  owns(user: string, workspace: string): Promise<boolean | null>;
}

export class EnvironmentError extends Error {
  override name = "EnvironmentError";
  constructor(message: string, readonly status: number = 400) {
    super(message);
  }
}

export interface EnvironmentOptions {
  /** The operator's allow-list: toolchain -> allowed versions. */
  tools: Partial<Record<ToolId, string[]>>;
  /** Pinned language servers and their own settings (``version`` always set). */
  servers: Partial<Record<ServerId, ServerSettings>>;
  paranoid: boolean;
  installTimeoutSeconds: number;
  store: FileEnvironmentStore;
  ownership: WorkspaceOwnership;
  log?: (msg: string) => void;
}

/** A toolchain the operator allows, with the server it brings when pinned. */
export interface AllowedTool {
  tool: ToolId;
  label: string;
  versions: string[];
  server: { id: ServerId; version: string } | null;
}

export interface EnvironmentStatus {
  workspace: string;
  allowed: AllowedTool[];
  /** Proposals this user answered "not now" for, in this workspace. */
  declined: ToolId[];
  /**
   * ``.jaato/toolchain-offer.json`` for the workspace: the page stages it
   * through the daemon.  The plugin reads it as the policy for binds and
   * hints; this server never writes it.
   */
  offer: { path: string; content: string };
}

export class EnvironmentService {
  private readonly o: EnvironmentOptions;
  private readonly _log: (msg: string) => void;

  constructor(opts: EnvironmentOptions) {
    this.o = opts;
    this._log = opts.log ?? (() => undefined);
  }

  /** The toolchains the operator allows, in catalog order; python iff basedpyright is pinned. */
  allowedTools(): AllowedTool[] {
    const out: AllowedTool[] = [];
    for (const tool of TOOL_IDS) {
      const spec = TOOLCHAINS[tool];
      const sid = spec.server;
      const pin = sid ? this.o.servers[sid]?.version : undefined;
      const server = sid && pin ? { id: sid, version: pin } : null;
      if (!spec.installable) {
        if (server) out.push({ tool, label: spec.label, versions: ["system"], server });
        continue;
      }
      const versions = this.o.tools[tool] ?? [];
      if (versions.length) out.push({ tool, label: spec.label, versions: [...versions], server });
    }
    return out;
  }

  /** The offer's content, from the policy alone (identical for every workspace). */
  offerContent(): { path: string; content: string } {
    const allowed = this.allowedTools();
    const servers: Partial<Record<ServerId, ServerSettings>> = {};
    // Only a server whose toolchain is allowed: the plugin installs nothing else.
    for (const sid of SERVER_IDS) {
      const s = this.o.servers[sid];
      if (s && allowed.some((a) => a.server?.id === sid)) servers[sid] = { ...s };
    }
    const file = toolchainOfferFile({
      allowed: allowed.map((a) => ({ tool: a.tool, label: a.label, versions: a.versions })),
      servers, installTimeoutSeconds: this.o.installTimeoutSeconds, paranoid: this.o.paranoid,
    });
    return { path: file.relativePath, content: managedContent(file) };
  }

  /** Ask the daemon whether ``user`` owns ``workspace``; nothing here reads the filesystem. */
  private async _authorize(user: string, workspace: string): Promise<void> {
    if (typeof workspace !== "string" || !workspace || !workspace.startsWith("/")) throw new EnvironmentError("workspace must be an absolute path");
    const owns = await this.o.ownership.owns(user, workspace);
    if (owns === null) throw new EnvironmentError("the daemon could not confirm you own this workspace; try again", 503);
    if (!owns) throw new EnvironmentError("no such workspace for this user", 404);
  }

  async status(sub: string, user: string, workspace: string): Promise<EnvironmentStatus> {
    await this._authorize(user, workspace);
    return {
      workspace,
      allowed: this.allowedTools(),
      declined: this.o.store.declined(sub, workspace),
      offer: this.offerContent(),
    };
  }

  /** "Not now" for a proposal: not shown again to this user in this workspace. */
  async decline(sub: string, user: string, workspace: string, tool: ToolId): Promise<void> {
    await this._authorize(user, workspace);
    this.o.store.decline(sub, workspace, tool);
  }

  /** The opposite: a toolchain the user binds after all is proposed again when unbound. */
  async undecline(sub: string, user: string, workspace: string, tool: ToolId): Promise<void> {
    await this._authorize(user, workspace);
    this.o.store.undecline(sub, workspace, tool);
    this._log(`environment: ${tool} undeclined for ${workspace}`);
  }
}
