/**
 * Test doubles for the GitHub connect feature: a fake ``GitHubApi`` that
 * mints deterministic tokens and counts refreshes (so the #683 lock is
 * observable), and a fake ``SessionReloader`` that records the users it was
 * asked to reload.  Neither touches the network or a registered App.
 */
import type { GitHubApi, GitHubIdentity, GitHubTokenSet } from "../src/github-api.js";
import { GitHubGrantRevoked } from "../src/github-api.js";
import type { SessionReloader } from "../src/github.js";

export class FakeGitHubApi implements GitHubApi {
  readonly gitHost = "github.com";
  exchanges = 0;
  refreshes = 0;
  revokes: string[] = [];
  /** Set to make the next refresh fail as a dead grant (a user who revoked at GitHub). */
  nextRefreshRevoked = false;
  /** Set to make the next refresh throw a transient error. */
  nextRefreshError: string | null = null;
  /** The identity fetchIdentity returns; defaults to alice. */
  identity: GitHubIdentity = { login: "alice", id: 4242, name: "Alice Example", noreplyEmail: "4242+alice@users.noreply.github.com", installations: [{ id: 7, account: "acme" }] };
  private _seq = 0;

  authorizeUrl(state: string, redirectUri: string): string {
    return `https://github.com/login/oauth/authorize?client_id=Iv1.test&state=${state}&redirect_uri=${encodeURIComponent(redirectUri)}`;
  }

  async exchangeCode(_code: string, _redirectUri: string): Promise<GitHubTokenSet> {
    this.exchanges += 1;
    this._seq += 1;
    return { accessToken: `access-${this._seq}`, accessExpiresInSeconds: 28800, refreshToken: `refresh-${this._seq}`, refreshExpiresInSeconds: 15897600 };
  }

  async refresh(refreshToken: string): Promise<GitHubTokenSet> {
    this.refreshes += 1;
    if (this.nextRefreshError) { const m = this.nextRefreshError; this.nextRefreshError = null; throw new Error(m); }
    if (this.nextRefreshRevoked) { this.nextRefreshRevoked = false; throw new GitHubGrantRevoked("bad_refresh_token"); }
    this._seq += 1;
    // Rotation: a new refresh token, voiding the presented one.
    return { accessToken: `access-${this._seq}`, accessExpiresInSeconds: 28800, refreshToken: `refresh-${this._seq}-from-${refreshToken}`, refreshExpiresInSeconds: 15897600 };
  }

  async revokeGrant(accessToken: string): Promise<void> {
    this.revokes.push(accessToken);
  }

  async fetchIdentity(_accessToken: string): Promise<GitHubIdentity> {
    return this.identity;
  }
}

export class FakeReloader implements SessionReloader {
  calls: string[] = [];
  answer: { status: string; reloaded: number } = { status: "ok", reloaded: 0 };
  async reloadUser(user: string): Promise<{ status: string; reloaded: number }> {
    this.calls.push(user);
    return this.answer;
  }
}
