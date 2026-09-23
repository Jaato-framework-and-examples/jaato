/**
 * The "Connect GitHub" settings surface.
 *
 * Lists the accounts a user has connected, offers connect / set-default /
 * disconnect, and — the load-bearing control — never shows or fetches a token:
 * connect is a plain navigation to the server-side OAuth start, and there is
 * no reveal route on the API at all.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { GitHubConnect } from "./GitHubConnect";

const ACCOUNTS = [
  { id: "a", login: "alice", name: "Alice Example", noreplyEmail: "1+alice@users.noreply.github.com", installations: [{ id: 1, account: "acme" }], isDefault: true, createdAt: "t", updatedAt: "t" },
  { id: "b", login: "alice-work", name: null, noreplyEmail: "2+alice-work@users.noreply.github.com", installations: [], isDefault: false, createdAt: "t", updatedAt: "t" },
];

function stubFetch(handler: (url: string, init?: RequestInit) => unknown) {
  const calls: Array<{ url: string; method?: string; body?: string }> = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url: String(url), method: init?.method, body: init?.body as string | undefined });
    return new Response(JSON.stringify(handler(String(url), init) ?? {}), { status: 200, headers: { "content-type": "application/json" } });
  }));
  return calls;
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("GitHubConnect", () => {
  it("lists connected accounts, marks the default, and offers Connect as a link to the OAuth start", async () => {
    stubFetch((url) => (url.endsWith("/accounts") ? { accounts: ACCOUNTS } : {}));
    render(<GitHubConnect githubUrl="./api/github" githubLoginUrl="./auth/github/login" onError={() => {}} />);
    expect(await screen.findByText("@alice (default)")).toBeInTheDocument();
    expect(screen.getByText("@alice-work")).toBeInTheDocument();
    const connect = screen.getByRole("link", { name: /connect a github account/i });
    expect(connect).toHaveAttribute("href", "./auth/github/login");
  });

  it("set-default posts the id and refreshes; onChanged fires for the workspace picker to relist", async () => {
    const calls = stubFetch((url) => {
      if (url.endsWith("/accounts")) return { accounts: ACCOUNTS };
      if (url.endsWith("/default")) return { accounts: [{ ...ACCOUNTS[1], isDefault: true }, { ...ACCOUNTS[0], isDefault: false }] };
      return {};
    });
    const onChanged = vi.fn();
    render(<GitHubConnect githubUrl="./api/github" githubLoginUrl={null} onError={() => {}} onChanged={onChanged} />);
    await screen.findByText("@alice-work");
    fireEvent.click(screen.getByRole("button", { name: /make @alice-work the default/i }));
    await waitFor(() => {
      const call = calls.find((c) => c.url.endsWith("/default"));
      expect(JSON.parse(call!.body!)).toEqual({ id: "b" });
    });
    expect(onChanged).toHaveBeenCalled();
  });

  it("disconnect asks first, then posts the id to /disconnect", async () => {
    const calls = stubFetch((url) => {
      if (url.endsWith("/accounts")) return { accounts: ACCOUNTS };
      if (url.endsWith("/disconnect")) return { disconnected: "alice", accounts: [ACCOUNTS[1]] };
      return {};
    });
    const onNotice = vi.fn();
    render(<GitHubConnect githubUrl="./api/github" githubLoginUrl={null} onError={() => {}} onNotice={onNotice} />);
    await screen.findByText("@alice (default)");
    // The first click only reveals the confirmation — nothing is posted yet.
    fireEvent.click(screen.getByRole("button", { name: /^disconnect @alice$/i }));
    expect(calls.some((c) => c.url.endsWith("/disconnect"))).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: /confirm disconnect @alice/i }));
    await waitFor(() => {
      const call = calls.find((c) => c.url.endsWith("/disconnect"));
      expect(JSON.parse(call!.body!)).toEqual({ id: "a" });
    });
    expect(onNotice).toHaveBeenCalledWith(expect.stringContaining("@alice"));
  });

  it("never fetches a reveal/token route — the token stays on the server", async () => {
    const calls = stubFetch((url) => (url.endsWith("/accounts") ? { accounts: ACCOUNTS } : {}));
    render(<GitHubConnect githubUrl="./api/github" githubLoginUrl="./auth/github/login" onError={() => {}} />);
    await screen.findByText("@alice (default)");
    expect(calls.every((c) => !/reveal|token|secret/.test(c.url))).toBe(true);
  });

  it("says so when no account is connected", async () => {
    stubFetch((url) => (url.endsWith("/accounts") ? { accounts: [] } : {}));
    render(<GitHubConnect githubUrl="./api/github" githubLoginUrl="./auth/github/login" onError={() => {}} />);
    expect(await screen.findByText(/no github accounts connected/i)).toBeInTheDocument();
  });
});
