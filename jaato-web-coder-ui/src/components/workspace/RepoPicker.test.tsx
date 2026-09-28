/**
 * The repo picker's two jobs beyond listing: autocomplete a repository the
 * App's installations do not cover (GitHub search), and say -- when one is
 * picked -- that the workspace's sessions will be able to clone it but not
 * push, branch or open a pull request there.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { useState } from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { RepoPicker, mergeHits, type PickedRepo } from "./RepoPicker";
import { appReach } from "@/app/github";

const LISTING = {
  account: { id: "a", login: "alice" },
  repos: [{ fullName: "alice/dots", private: false, defaultBranch: "main" }],
  installedOn: ["alice"],
  truncated: false,
  installUrl: "https://github.com/apps/jaato-web-coder/installations/new",
};

function stubFetch() {
  const calls: string[] = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    calls.push(String(url));
    const u = String(url);
    const body = u.includes("/repos") ? LISTING
      : u.includes("/search") ? { repos: [{ fullName: "Jaato-framework-and-examples/jaato", private: false, defaultBranch: "main", appCanWrite: false }] }
      : u.includes("/branches") ? { repo: "x", branches: ["main"] } : {};
    return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
  }));
  return calls;
}

function Harness() {
  const [picked, setPicked] = useState<PickedRepo[]>([]);
  return <RepoPicker githubUrl="./api/github" workspace="w" picked={picked} onChange={setPicked} />;
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("appReach", () => {
  it("says nothing without coverage data, or when the listing is incomplete", () => {
    expect(appReach(null, "a/b")).toBeNull();
    expect(appReach({ ...LISTING, truncated: true }, "x/y")).toBeNull();
    expect(appReach({ account: LISTING.account, repos: [] }, "x/y")).toBeNull();
  });
  it("tells an uninstalled owner from a repository left out of an installation", () => {
    expect(appReach(LISTING, "ALICE/dots")).toEqual({ canWrite: true });
    expect(appReach(LISTING, "org/jaato")).toMatchObject({ canWrite: false, owner: "org", ownerInstalled: false });
    expect(appReach(LISTING, "alice/other")).toMatchObject({ canWrite: false, ownerInstalled: true });
  });
});

describe("mergeHits", () => {
  it("keeps listing entries first and drops a hit the listing already has", () => {
    const m = mergeHits(LISTING.repos, [
      { fullName: "alice/DOTS", private: false, defaultBranch: "main", appCanWrite: true },
      { fullName: "org/x", private: false, defaultBranch: "main", appCanWrite: false },
    ]);
    expect(m.map((r) => [r.fullName, r.appCanWrite])).toEqual([["alice/dots", true], ["org/x", false]]);
  });
});

describe("RepoPicker", () => {
  it("autocompletes an org repository through search, marks it and warns once picked", async () => {
    const calls = stubFetch();
    render(<Harness />);
    fireEvent.change(screen.getByPlaceholderText("owner/repo"), { target: { value: "Jaato-framework-and-examples/ja" } });
    const hit = await screen.findByRole("checkbox", { name: /Jaato-framework-and-examples\/jaato/ }, { timeout: 2000 });
    expect(calls.some((u) => u.includes("/search?q=Jaato-framework-and-examples%2Fja"))).toBe(true);
    expect(hit.textContent).toMatch(/App not installed/);
    fireEvent.click(hit);
    const note = await screen.findByTestId("reach-warning");
    expect(note.textContent).toMatch(/not installed on Jaato-framework-and-examples.*cannot push/);
    expect(screen.getByRole("link", { name: "Install the App" }).getAttribute("href")).toBe(LISTING.installUrl);
  });

  it("does not warn for a repository the installations cover", async () => {
    stubFetch();
    render(<Harness />);
    fireEvent.click(await screen.findByRole("checkbox", { name: /alice\/dots/ }));
    await waitFor(() => expect(screen.getAllByTestId("picked-repo")).toHaveLength(1));
    expect(screen.queryByTestId("reach-warning")).toBeNull();
  });
});
