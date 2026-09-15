/**
 * What this bundle is, for the person staring at a connection failure.
 *
 * ``__JAATO_BUILD__`` is stamped by ``vite.config.ts`` at build time:
 * the UI version, the ``@jaato/sdk`` revision compiled in, the protocol
 * floor that SDK enforces (``MIN_PROTOCOL_VERSION``) and the commit.  The
 * same object is written to ``dist/build-info.json`` for the launcher's
 * ``--version``.  Under a test runner that does not inject it, every field
 * reads ``unknown`` rather than throwing.
 */
export interface BuildInfo {
  ui: string;
  sdk: string;
  protocolMin: string;
  commit: string;
  builtAt: string;
}

const UNKNOWN: BuildInfo = { ui: "unknown", sdk: "unknown", protocolMin: "unknown", commit: "unknown", builtAt: "" };

export const BUILD: BuildInfo = typeof __JAATO_BUILD__ !== "undefined" ? __JAATO_BUILD__ : UNKNOWN;

/** One line: ``jaato-web-coder-ui 0.1.0 · @jaato/sdk 0.6.0 · protocol ≥ 1.0 · abc1234``. */
export function buildLine(b: BuildInfo = BUILD): string {
  return `jaato-web-coder-ui ${b.ui} · @jaato/sdk ${b.sdk} · protocol ≥ ${b.protocolMin} · ${b.commit}`;
}
