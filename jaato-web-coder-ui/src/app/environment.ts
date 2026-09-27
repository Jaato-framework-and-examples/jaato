/**
 * The page's side of the sign-in backend's environment bootstrap (#1344):
 * the toolchains, language servers and repository-guidance pointer a
 * workspace has.
 *
 * ``jaato-web-coder-server`` installs a toolchain only when the workspace
 * owner accepts it; this module is how the page asks.  ``config.json`` names
 * ``environmentUrl`` only when the backend has an ``environment:`` block;
 * without it nothing here is called and no chip appears -- the opt-in shape
 * ``app/github.ts`` has.
 *
 * Detection only PROPOSES.  Two moments raise a proposal:
 *
 * - **clone time**: once a New workspace's repositories are checked out, the
 *   page asks the backend to scan them (``refresh``), which also writes the
 *   repository-guidance pointer, and shows what it found;
 * - **mid-session**: a command that failed with ``<name>: command not found``
 *   for a name a known toolchain provides ({@link notFoundCommand},
 *   {@link toolForCommand}) raises a chip in the rail.
 *
 * Accepting calls ``bind``; "Not now" calls ``decline``, which the backend
 * remembers per user and workspace.
 */
import { SignInRequiredError } from "./tickets";

export type ToolId = "python" | "node" | "go" | "bun";

export interface AllowedTool {
  tool: ToolId;
  label: string;
  versions: string[];
  server: { id: string; version: string } | null;
}

export interface Proposal {
  tool: ToolId;
  label: string;
  version: string;
  pin: string | null;
  pinAllowed: boolean;
  source: string;
}

export interface BoundToolchain {
  tool: ToolId;
  version: string;
  bin: string[];
  boundAt: string;
  server: { id: string; version: string } | null;
}

export interface EnvironmentJob {
  id: string;
  tool: ToolId;
  version: string;
  status: "running" | "done" | "failed" | "cancelled";
  log: string[];
  error?: string;
  notes: string[];
  startedAt: string;
  finishedAt?: string;
}

export interface EnvironmentStatus {
  workspace: string;
  toolchains: BoundToolchain[];
  allowed: AllowedTool[];
  proposals: Proposal[];
  declined: ToolId[];
  job: EnvironmentJob | null;
  guidance: string[];
}

export interface EnvironmentApi {
  status(workspace: string): Promise<EnvironmentStatus>;
  /** Re-scan the workspace and rewrite the repository-guidance pointer; the page calls it after a clone. */
  refresh(workspace: string): Promise<EnvironmentStatus>;
  bind(workspace: string, tool: ToolId, version: string): Promise<EnvironmentJob>;
  unbind(workspace: string, tool: ToolId): Promise<{ status: EnvironmentStatus; notes: string[] }>;
  decline(workspace: string, tool: ToolId): Promise<void>;
  job(id: string): Promise<EnvironmentJob>;
  cancel(id: string): Promise<EnvironmentJob>;
}

function subUrl(base: string, leaf: string): string {
  const [path = base, query = ""] = base.split(/\?(.*)/s, 2);
  return `${path.replace(/\/$/, "")}/${leaf}${query ? `?${query}` : ""}`;
}

async function failure(res: Response, what: string): Promise<Error> {
  if (res.status === 401) return new SignInRequiredError("./auth/login");
  let detail = "";
  try { detail = String(((await res.json()) as { error?: unknown }).error ?? ""); } catch { /* not JSON */ }
  return new Error(`${what}: ${detail || `HTTP ${res.status}`}`);
}

export function environmentApi(url: string, fetchImpl: typeof fetch = fetch): EnvironmentApi {
  const common: RequestInit = { credentials: "same-origin", cache: "no-store" };
  const getInit = { ...common, headers: { Accept: "application/json" } };
  const post = (leaf: string, body: unknown) => fetchImpl(subUrl(url, leaf), {
    ...common, method: "POST", headers: { Accept: "application/json", "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  return {
    async status(workspace) {
      const res = await fetchImpl(`${url}${url.includes("?") ? "&" : "?"}workspace=${encodeURIComponent(workspace)}`, getInit);
      if (!res.ok) throw await failure(res, "Reading the workspace's toolchains failed");
      return await res.json() as EnvironmentStatus;
    },
    async refresh(workspace) {
      const res = await post("refresh", { workspace });
      if (!res.ok) throw await failure(res, "Scanning the workspace failed");
      return ((await res.json()) as { status: EnvironmentStatus }).status;
    },
    async bind(workspace, tool, version) {
      const res = await post("bind", { workspace, tool, version });
      if (!res.ok) throw await failure(res, `Binding ${tool} ${version} failed`);
      return ((await res.json()) as { job: EnvironmentJob }).job;
    },
    async unbind(workspace, tool) {
      const res = await post("unbind", { workspace, tool });
      if (!res.ok) throw await failure(res, `Unbinding ${tool} failed`);
      return await res.json() as { status: EnvironmentStatus; notes: string[] };
    },
    async decline(workspace, tool) {
      const res = await post("decline", { workspace, tool });
      if (!res.ok) throw await failure(res, `Declining ${tool} failed`);
    },
    async job(id) {
      const res = await fetchImpl(subUrl(url, `jobs/${encodeURIComponent(id)}`), getInit);
      if (!res.ok) throw await failure(res, "Reading the install's progress failed");
      return ((await res.json()) as { job: EnvironmentJob }).job;
    },
    async cancel(id) {
      const res = await post(`jobs/${encodeURIComponent(id)}/cancel`, {});
      if (!res.ok) throw await failure(res, "Cancelling the install failed");
      return ((await res.json()) as { job: EnvironmentJob }).job;
    },
  };
}

/**
 * Which command a shell said it could not find, or ``null``.  Recognises the
 * shapes bash, dash and ``env`` print: ``bash: line 1: go: command not
 * found``, ``sh: 1: node: not found``, ``/usr/bin/env: 'node': No such file
 * or directory``.  Only a bare command name counts -- a path that is missing
 * is a different problem.
 */
export function notFoundCommand(text: string): string | null {
  const patterns = [
    /(?:^|\n)[^\n:]*:\s*(?:line \d+:\s*)?([A-Za-z0-9._+-]+): command not found/,
    /(?:^|\n)(?:sh|dash|bash):\s*\d+:\s*([A-Za-z0-9._+-]+): not found/,
    /(?:^|\n)\/usr\/bin\/env:\s*['‘]?([A-Za-z0-9._+-]+)['’]?: No such file or directory/,
  ];
  for (const re of patterns) {
    const m = re.exec(text);
    if (m) return m[1]!;
  }
  return null;
}

/** The toolchain that provides ``command``, among those this page knows. */
export const COMMAND_TOOLS: Record<string, ToolId> = {
  node: "node", npm: "node", npx: "node", corepack: "node", tsc: "node",
  go: "go", gofmt: "go", gopls: "go",
  bun: "bun", bunx: "bun",
  basedpyright: "python", "basedpyright-langserver": "python", pyright: "python",
};

export function toolForCommand(command: string): ToolId | null {
  return COMMAND_TOOLS[command] ?? null;
}

/** The chip text for a proposal. */
export function proposalText(p: Proposal): string {
  const what = p.tool === "python" ? `${p.label}` : `${p.label} ${p.version}`;
  if (p.pin && !p.pinAllowed) return `${p.label} detected (${p.source} pins ${p.pin}, which this server does not offer). Bind ${p.version} instead?`;
  return `${what} detected (from ${p.source}). Bind it?`;
}
