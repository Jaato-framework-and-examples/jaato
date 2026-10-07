import { strict as assert } from "node:assert";
import { describe, test } from "node:test";
import { GitHubApiError, GitHubGrantRevoked, HttpGitHubApi, nextLink } from "../src/github-api.js";

/** A fetch double serving canned pages by URL, recording what it was asked and with which headers. */
function fakeFetch(pages: Record<string, { status?: number; body: unknown; link?: string }>) {
  const calls: Array<{ url: string; auth: string | null }> = [];
  const impl = (async (input: string | URL, init?: RequestInit) => {
    const url = String(input);
    const headers = new Headers(init?.headers);
    calls.push({ url, auth: headers.get("authorization") });
    const p = pages[url];
    if (!p) return new Response("{}", { status: 404 });
    return new Response(JSON.stringify(p.body), { status: p.status ?? 200, headers: p.link ? { link: p.link } : {} });
  }) as typeof fetch;
  return { impl, calls };
}

const API = "https://api.github.com";
const repo = (n: number) => ({ full_name: `o/r${n}`, private: n % 2 === 0, default_branch: "main", pushed_at: "2026-09-01T00:00:00Z" });

describe("HttpGitHubApi listing calls", () => {
  test("listInstallationRepos follows Link rel=next and maps fields", async () => {
    const p1 = `${API}/user/installations/7/repositories?per_page=100`;
    const p2 = `${API}/user/installations/7/repositories?per_page=100&page=2`;
    const { impl, calls } = fakeFetch({
      [p1]: { body: { total_count: 3, repositories: [repo(1), repo(2)] }, link: `<${p2}>; rel="next", <${p2}>; rel="last"` },
      [p2]: { body: { total_count: 3, repositories: [{ full_name: "o/r3", private: false, default_branch: "dev" }] } },
    });
    const api = new HttpGitHubApi({ clientId: "c", clientSecret: "s", fetchImpl: impl });
    const repos = await api.listInstallationRepos("tok", 7, 1000);
    assert.deepEqual(repos.map((r) => r.fullName), ["o/r1", "o/r2", "o/r3"]);
    assert.equal(repos[1]!.private, true);
    assert.equal(repos[2]!.defaultBranch, "dev");
    assert.equal(repos[2]!.pushedAt, undefined);
    assert.equal(calls.length, 2);
    assert.ok(calls.every((c) => c.auth === "Bearer tok"));
  });

  test("pagination stops at the cap", async () => {
    const p1 = `${API}/user/installations/7/repositories?per_page=100`;
    const p2 = `${p1}&page=2`;
    const { impl, calls } = fakeFetch({
      [p1]: { body: { repositories: [repo(1), repo(2), repo(3)] }, link: `<${p2}>; rel="next"` },
      [p2]: { body: { repositories: [repo(4)] } },
    });
    const api = new HttpGitHubApi({ clientId: "c", clientSecret: "s", fetchImpl: impl });
    assert.equal((await api.listInstallationRepos("tok", 7, 2)).length, 2);
    assert.equal(calls.length, 1, "no second page once the cap is reached");
  });

  test("a next link off the API origin is not followed (the token stays home)", async () => {
    const p1 = `${API}/repos/o/r/branches?per_page=100`;
    const { impl, calls } = fakeFetch({
      [p1]: { body: [{ name: "main" }], link: `<https://evil.example/steal?page=2>; rel="next"` },
    });
    const api = new HttpGitHubApi({ clientId: "c", clientSecret: "s", fetchImpl: impl });
    assert.deepEqual(await api.listBranches("tok", "o", "r", 500), ["main"]);
    assert.equal(calls.length, 1);
  });

  test("GHE api base url is honoured, 401 is a revoked grant, 404 an API error", async () => {
    const ghe = "https://ghe.example/api/v3";
    const { impl } = fakeFetch({
      [`${ghe}/repos/o/r/branches?per_page=100`]: { status: 401, body: {} },
    });
    const api = new HttpGitHubApi({ clientId: "c", clientSecret: "s", apiBaseUrl: ghe, fetchImpl: impl });
    await assert.rejects(api.listBranches("tok", "o", "r", 500), GitHubGrantRevoked);
    await assert.rejects(api.listInstallationRepos("tok", 9, 10), GitHubApiError);
  });

  test("nextLink parses GitHub's Link header", () => {
    assert.equal(nextLink(null), null);
    assert.equal(nextLink('<https://a/x?page=3>; rel="last"'), null);
    assert.equal(nextLink('<https://a/x?page=2>; rel="next", <https://a/x?page=3>; rel="last"'), "https://a/x?page=2");
  });
});
