/**
 * Type declarations for the package's `exports` entry (`bin/jaato-web-coder-ui.js`).
 * Hand-written: the launcher is plain JavaScript so that `npx @jaato/web-coder-ui`
 * installs nothing but the bundle, and this file is what lets a TypeScript host
 * (`jaato-web-coder-server`) import it.
 */
import type { IncomingMessage, Server, ServerResponse } from "node:http";

/** The bundle this package ships (`<package>/dist`). */
export declare const DIST_DIR: string;

/** What `GET /config.json` answers; see `src/app/launcherConfig.ts` for the page's reading of it. */
export interface StaticConfig {
  daemon?: string;
  token?: string;
  ticketUrl?: string;
  loginUrl?: string;
  /** Where the sign-in backend keeps the user's provider API keys; absent = no combobox, a plain key field. */
  credentialsUrl?: string;
  autoConnect?: boolean;
}

export interface StaticOptions {
  /** Directory holding `index.html` and `assets/`. */
  root: string;
  config: StaticConfig;
  /** `Host` header values accepted (lower-case `host:port`); `null` accepts any. */
  allowedHosts: Set<string> | null;
}

export type StaticHandler = (req: IncomingMessage, res: ServerResponse) => Promise<void>;

export declare function createStaticHandler(opts: StaticOptions): StaticHandler;
export declare function createStaticServer(opts: StaticOptions): Server;

export interface LauncherOptions {
  daemon: string;
  token?: string;
  tokenFile?: string;
  noToken: boolean;
  host: string;
  port: number;
  open: boolean;
  root: string;
  help: boolean;
  version: boolean;
}
export declare function parseArgs(argv: string[]): LauncherOptions;
export declare function isLoopbackHost(host: string): boolean;
export declare function resolveToken(opts: Partial<LauncherOptions>, io?: { readFile?: (p: string) => string; exists?: (p: string) => boolean }): { token: string | undefined; source: string };

export interface BuildInfo { ui: string; sdk: string; protocolMin: string; commit: string; builtAt: string }
export declare function readBuildInfo(root?: string): BuildInfo | null;

export declare function main(opts: LauncherOptions): Promise<Server | null>;
