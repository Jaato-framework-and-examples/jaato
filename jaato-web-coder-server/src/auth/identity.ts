/**
 * The seam between the HTTP routes and whoever authenticates the person.
 *
 * ``OidcProvider`` (``oidc.ts``) is the production implementation against
 * Keycloak or any OpenID Connect issuer; the tests drive the same routes
 * with an in-process fake.  Nothing in ``routes.ts`` knows which it has.
 */
export interface LoginStart {
  /** Where to send the browser. */
  url: string;
  /** Opaque per-login state the callback needs back (PKCE verifier, nonce, state). */
  state: string;
  pending: Record<string, string>;
}

export interface AuthenticatedUser {
  /** The stable OIDC subject. */
  sub: string;
  /** The claim configured as ``subject_claim`` — what ``ticket.bind`` is told. */
  user: string;
  /** OIDC session id, when the issuer sends one (back-channel logout matches on it). */
  sid?: string;
  idToken?: string;
}

export class SignInRefusedError extends Error {
  override name = "SignInRefusedError";
}

export interface LogoutToken {
  sub?: string;
  sid?: string;
}

export interface IdentityProvider {
  /** Begin a login; ``redirectUri`` is this server's callback URL. */
  startLogin(redirectUri: string): Promise<LoginStart>;
  /** Finish a login from the callback URL and the pending state saved at start. */
  completeLogin(callbackUrl: URL, pending: Record<string, string>, redirectUri: string): Promise<AuthenticatedUser>;
  /** RP-initiated logout URL, or ``null`` when the issuer offers none. */
  endSessionUrl(idToken: string | undefined, postLogoutRedirectUri: string): string | null;
  /** Validate an OIDC back-channel logout token; throws when it is not one this client should honour. */
  verifyLogoutToken(token: string): Promise<LogoutToken>;
}
