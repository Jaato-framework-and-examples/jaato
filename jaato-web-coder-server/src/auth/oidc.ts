/**
 * OpenID Connect against Keycloak (or any conforming issuer), via
 * ``openid-client`` v6.
 *
 * Two things a same-host Keycloak needs and a generic setup would not
 * (``docs/design/web-server-bff.md`` §11):
 *
 * - **Front channel vs back channel.**  Tokens carry the PUBLIC issuer and
 *   are validated against it, but this process may reach Keycloak over
 *   loopback.  ``backchannelUrl`` swaps the origin on every request the
 *   library makes (discovery, token, JWKS, userinfo) while the ``iss``
 *   check keeps the public value.  Plain ``http`` is accepted for that
 *   back channel only when it is loopback (``config.ts`` enforces it).
 * - **Role-gated sign-in.**  ``requiredRole`` is looked up in the ID token's
 *   ``realm_access.roles`` and ``resource_access.<client>.roles``; a user
 *   without it is refused at the callback, before any ticket exists.
 *
 * PKCE (S256) is always sent, ``state`` and ``nonce`` are always checked.
 */
import * as client from "openid-client";
import { createRemoteJWKSet, jwtVerify, customFetch as joseCustomFetch } from "jose";
import type { OidcConfig } from "../config.js";
import { type AuthenticatedUser, type IdentityProvider, type LoginStart, type LogoutToken, SignInRefusedError } from "./identity.js";

const BACKCHANNEL_LOGOUT_EVENT = "http://schemas.openid.net/event/backchannel-logout";

/** A ``fetch`` that rewrites the issuer's public origin to the back-channel origin. */
export function rewritingFetch(publicIssuer: string, backchannelUrl: string): client.CustomFetch {
  const pub = new URL(publicIssuer);
  const back = new URL(backchannelUrl);
  return (url, options) => {
    const u = new URL(url);
    if (u.origin === pub.origin) {
      u.protocol = back.protocol;
      u.host = back.host;
    }
    // Keycloak serves the public hostname's metadata when asked with that Host,
    // so the request names the public host even though the socket is loopback.
    const headers = { ...options.headers, host: pub.host };
    return fetch(u, { ...options, headers } as RequestInit);
  };
}

export class OidcProvider implements IdentityProvider {
  private constructor(
    private readonly _cfg: OidcConfig,
    private readonly _config: client.Configuration,
    private readonly _jwks: ReturnType<typeof createRemoteJWKSet>,
  ) {}

  static async discover(cfg: OidcConfig): Promise<OidcProvider> {
    const execute: Array<(c: client.Configuration) => void> = [];
    const insecure = cfg.issuer.startsWith("http://") || (cfg.backchannelUrl?.startsWith("http://") ?? false);
    if (insecure) execute.push(client.allowInsecureRequests);
    const options: client.DiscoveryRequestOptions = { execute, timeout: 15 };
    let fetchImpl: client.CustomFetch | undefined;
    if (cfg.backchannelUrl) {
      fetchImpl = rewritingFetch(cfg.issuer, cfg.backchannelUrl);
      options[client.customFetch] = fetchImpl;
    }
    const config = await client.discovery(new URL(cfg.issuer), cfg.clientId, cfg.clientSecret, client.ClientSecretPost(cfg.clientSecret), options);
    const meta = config.serverMetadata();
    if (!meta.jwks_uri) throw new Error(`issuer ${cfg.issuer} publishes no jwks_uri; back-channel logout cannot be verified`);
    const jwks = createRemoteJWKSet(new URL(meta.jwks_uri), fetchImpl ? { [joseCustomFetch]: fetchImpl as unknown as typeof fetch } : {});
    return new OidcProvider(cfg, config, jwks);
  }

  async startLogin(redirectUri: string): Promise<LoginStart> {
    const codeVerifier = client.randomPKCECodeVerifier();
    const codeChallenge = await client.calculatePKCECodeChallenge(codeVerifier);
    const state = client.randomState();
    const nonce = client.randomNonce();
    const url = client.buildAuthorizationUrl(this._config, {
      redirect_uri: redirectUri,
      scope: this._cfg.scopes.join(" "),
      code_challenge: codeChallenge,
      code_challenge_method: "S256",
      state,
      nonce,
    });
    return { url: url.href, state, pending: { codeVerifier, nonce, state } };
  }

  async completeLogin(callbackUrl: URL, pending: Record<string, string>, _redirectUri: string): Promise<AuthenticatedUser> {
    const tokens = await client.authorizationCodeGrant(this._config, callbackUrl, {
      pkceCodeVerifier: pending.codeVerifier,
      expectedNonce: pending.nonce,
      expectedState: pending.state,
      idTokenExpected: true,
    });
    const claims = tokens.claims();
    if (!claims) throw new SignInRefusedError("the issuer returned no ID token");
    const user = claims[this._cfg.subjectClaim];
    if (typeof user !== "string" || !user) {
      throw new SignInRefusedError(`the ID token carries no usable '${this._cfg.subjectClaim}' claim`);
    }
    if (this._cfg.requiredRole && !hasRole(claims as Record<string, unknown>, this._cfg.clientId, this._cfg.requiredRole)) {
      throw new SignInRefusedError(`account lacks the required role '${this._cfg.requiredRole}'`);
    }
    return {
      sub: claims.sub,
      user,
      sid: typeof claims.sid === "string" ? claims.sid : undefined,
      idToken: tokens.id_token,
    };
  }

  endSessionUrl(idToken: string | undefined, postLogoutRedirectUri: string): string | null {
    if (!this._config.serverMetadata().end_session_endpoint) return null;
    const params: Record<string, string> = { post_logout_redirect_uri: postLogoutRedirectUri, client_id: this._cfg.clientId };
    if (idToken) params.id_token_hint = idToken;
    return client.buildEndSessionUrl(this._config, params).href;
  }

  async verifyLogoutToken(token: string): Promise<LogoutToken> {
    const { payload } = await jwtVerify(token, this._jwks, { issuer: this._cfg.issuer, audience: this._cfg.clientId });
    const events = payload.events as Record<string, unknown> | undefined;
    if (!events || !(BACKCHANNEL_LOGOUT_EVENT in events)) throw new Error("not a back-channel logout token (no events claim)");
    if ("nonce" in payload) throw new Error("a logout token must not carry a nonce");
    const sub = typeof payload.sub === "string" ? payload.sub : undefined;
    const sid = typeof payload.sid === "string" ? payload.sid : undefined;
    if (!sub && !sid) throw new Error("logout token names neither sub nor sid");
    return { sub, sid };
  }
}

/** Keycloak puts realm roles in ``realm_access.roles`` and client roles in ``resource_access.<client>.roles``. */
export function hasRole(claims: Record<string, unknown>, clientId: string, role: string): boolean {
  const realm = (claims.realm_access as { roles?: unknown } | undefined)?.roles;
  if (Array.isArray(realm) && realm.includes(role)) return true;
  const res = (claims.resource_access as Record<string, { roles?: unknown }> | undefined)?.[clientId]?.roles;
  return Array.isArray(res) && res.includes(role);
}
