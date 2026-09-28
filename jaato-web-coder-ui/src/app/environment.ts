/**
 * The page's side of the environment bootstrap (#1344): a workspace's
 * toolchains and language servers.
 *
 * Three parties, and this module talks to the first:
 *
 * | Who | What the page asks of it |
 * |---|---|
 * | the sign-in backend (``environmentUrl``) | the operator's allow-list, this user's "not now" answers, and the offer's content |
 * | the daemon | stage the offer (``app/toolchainOffer.ts``), fetch ``.jaato/environment.json`` (``app/toolchains.ts``) |
 * | the ``web_coder_toolchains`` plugin, in the session's runner | ``toolchain bind|unbind|scan|cancel``, a user command the page sends |
 *
 * The backend touches no workspace: it may run as an account that cannot.
 * ``config.json`` names ``environmentUrl`` only when it has an
 * ``environment:`` block; without it nothing here is called and no chip
 * appears, the opt-in shape ``app/github.ts`` has.
 */
import { SignInRequiredError } from "./tickets";

export type ToolId = "python" | "node" | "go" | "bun" | "java" | "maven" | "gradle";

export interface AllowedTool {
  tool: ToolId;
  label: string;
  versions: string[];
  server: { id: string; version: string } | null;
}

/** What the backend answers. */
export interface EnvironmentStatus {
  workspace: string;
  allowed: AllowedTool[];
  declined: ToolId[];
  /** ``.jaato/toolchain-offer.json``: the page stages it.  Absent from an older backend. */
  offer?: { path: string; content: string };
}

export interface EnvironmentApi {
  status(workspace: string): Promise<EnvironmentStatus>;
  decline(workspace: string, tool: ToolId): Promise<void>;
  undecline(workspace: string, tool: ToolId): Promise<void>;
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

/** ``onStatus`` sees every status the backend returns: the page stages the offer from it. */
export function environmentApi(url: string, fetchImpl: typeof fetch = fetch, onStatus?: (st: EnvironmentStatus) => void): EnvironmentApi {
  const common: RequestInit = { credentials: "same-origin", cache: "no-store" };
  const post = (leaf: string, body: unknown) => fetchImpl(subUrl(url, leaf), {
    ...common, method: "POST", headers: { Accept: "application/json", "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  return {
    async status(workspace) {
      const res = await fetchImpl(`${url}${url.includes("?") ? "&" : "?"}workspace=${encodeURIComponent(workspace)}`, { ...common, headers: { Accept: "application/json" } });
      if (!res.ok) throw await failure(res, "Reading the workspace's toolchains failed");
      const st = await res.json() as EnvironmentStatus;
      onStatus?.(st);
      return st;
    },
    async decline(workspace, tool) {
      const res = await post("decline", { workspace, tool });
      if (!res.ok) throw await failure(res, `Declining ${tool} failed`);
    },
    async undecline(workspace, tool) {
      const res = await post("undecline", { workspace, tool });
      if (!res.ok) throw await failure(res, `Proposing ${tool} again failed`);
    },
  };
}
