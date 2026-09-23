/**
 * The workspace GitHub-account dropdown.
 *
 * It reflects the current binding for the workspace, writes a new one through
 * the backend on change, and — the load-bearing control — never fetches or
 * renders a token value: there is no reveal on the GitHub API, and a session
 * acts as the bound account through the daemon, not through anything the
 * browser holds.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { GitHubAccountPicker } from "./GitHubAccountPicker";

const ACCOUNTS = [
  { id: "a", login: "alice", name: null, noreplyEmail: "1+alice@users.noreply.github.com", installations: [], isDefault: true, createdAt: "t", updatedAt: "t" },
  { id: "b", login: "alice-work", name: null, noreplyEmail: "2+alice-work@users.noreply.github.com", installations: [], isDefault: false, createdAt: "t", updatedAt: "t" },
];

function stubFetch(handler: (url: string, init?: RequestInit) => unknown) {
  const calls: Array<{ url: string; method?: string; body?: string }> = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url: String(url), method: init?.method, body: init?.body as string | undefined });
    const body = handler(String(url), init);
    return new Response(JSON.stringify(body ?? {}), { status: 200, headers: { "content-type": "application/json" } });
  }));
  return calls;
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("GitHubAccountPicker", () => {
  it("renders nothing when the backend has no github block", () => {
    const { container } = render(<GitHubAccountPicker githubUrl={null} workspace="ws1" onError={() => {}} />);
    expect(container.firstChild).toBeNull();
  });

  it("reflects the binding for the workspace and lists the accounts", async () => {
    stubFetch((url) => {
      if (url.endsWith("/accounts")) return { accounts: ACCOUNTS };
      if (url.endsWith("/bindings")) return { bindings: [{ workspace: "ws1", accountId: "b" }] };
      return {};
    });
    render(<GitHubAccountPicker githubUrl="./api/github" workspace="ws1" onError={() => {}} />);
    const select = await screen.findByLabelText<HTMLSelectElement>("GitHub account");
    await waitFor(() => expect(select.value).toBe("b"));
    expect(screen.getByRole("option", { name: "@alice (default)" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "@alice-work" })).toBeInTheDocument();
  });

  it("binds the chosen account through POST /bind on change", async () => {
    const calls = stubFetch((url) => {
      if (url.endsWith("/accounts")) return { accounts: ACCOUNTS };
      if (url.endsWith("/bindings")) return { bindings: [] };
      if (url.endsWith("/bind")) return { binding: "set", envWritten: true, gitconfigSeeded: true, reloaded: 0 };
      return {};
    });
    render(<GitHubAccountPicker githubUrl="./api/github" workspace="ws1" onError={() => {}} />);
    const select = await screen.findByLabelText<HTMLSelectElement>("GitHub account");
    await waitFor(() => expect(select).not.toBeDisabled());
    fireEvent.change(select, { target: { value: "a" } });
    await waitFor(() => {
      const bind = calls.find((c) => c.url.endsWith("/bind"));
      expect(bind).toBeTruthy();
      expect(JSON.parse(bind!.body!)).toEqual({ workspace: "ws1", account_id: "a" });
    });
  });

  it("clears the binding through account_id: null when — none — is chosen", async () => {
    const calls = stubFetch((url) => {
      if (url.endsWith("/accounts")) return { accounts: ACCOUNTS };
      if (url.endsWith("/bindings")) return { bindings: [{ workspace: "ws1", accountId: "a" }] };
      if (url.endsWith("/bind")) return { binding: "cleared", envWritten: true, gitconfigSeeded: false, reloaded: 0 };
      return {};
    });
    render(<GitHubAccountPicker githubUrl="./api/github" workspace="ws1" onError={() => {}} />);
    const select = await screen.findByLabelText<HTMLSelectElement>("GitHub account");
    await waitFor(() => expect(select.value).toBe("a"));
    fireEvent.change(select, { target: { value: "" } });
    await waitFor(() => {
      const bind = calls.find((c) => c.url.endsWith("/bind"));
      expect(JSON.parse(bind!.body!)).toEqual({ workspace: "ws1", account_id: null });
    });
  });

  it("never fetches a reveal route — no token ever reaches the browser", async () => {
    const calls = stubFetch((url) => {
      if (url.endsWith("/accounts")) return { accounts: ACCOUNTS };
      if (url.endsWith("/bindings")) return { bindings: [] };
      if (url.endsWith("/bind")) return { binding: "set", envWritten: true, gitconfigSeeded: true, reloaded: 0 };
      return {};
    });
    render(<GitHubAccountPicker githubUrl="./api/github" workspace="ws1" onError={() => {}} />);
    const select = await screen.findByLabelText<HTMLSelectElement>("GitHub account");
    await waitFor(() => expect(select).not.toBeDisabled());
    fireEvent.change(select, { target: { value: "a" } });
    await waitFor(() => expect(calls.some((c) => c.url.endsWith("/bind"))).toBe(true));
    expect(calls.every((c) => !/reveal|token|secret/.test(c.url))).toBe(true);
  });

  it("points at Connect GitHub when no account is connected", async () => {
    stubFetch((url) => {
      if (url.endsWith("/accounts")) return { accounts: [] };
      if (url.endsWith("/bindings")) return { bindings: [] };
      return {};
    });
    render(<GitHubAccountPicker githubUrl="./api/github" workspace="ws1" onError={() => {}} />);
    expect(await screen.findByText(/none connected/i)).toBeInTheDocument();
    expect(screen.queryByRole("combobox")).toBeNull();
  });
});
