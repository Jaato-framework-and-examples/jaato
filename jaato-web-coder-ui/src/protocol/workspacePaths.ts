/**
 * Workspace-relative path arithmetic for the Files panel's markdown viewer.
 *
 * A markdown document in the workspace links to its neighbours the way a
 * repository README does: ``docs/setup.md``, ``../README.md``,
 * ``./img/diagram.png``, ``/CONTRIBUTING.md`` (root-relative, as GitHub
 * reads it).  None of those is a URL the browser can load -- the page is
 * served from the web coder's origin, not from the workspace -- so the
 * viewer resolves them HERE into a workspace-relative path and fetches it
 * through the daemon (``workspace.file.fetch``), where containment and the
 * credential rule are enforced.
 *
 * This module decides only what a reference MEANS; it grants nothing.  A
 * reference that climbs above the workspace root resolves to ``null`` so
 * the viewer does not even ask, but the daemon's own ``unsafe_path``
 * refusal is what actually bounds a fetch.
 */

/**
 * True for a path the Files panel viewer RENDERS rather than shows as
 * text: a markdown document or an image.  What the tool row's ``view``
 * button is offered for (``viewablePathsForCall``).
 */
export function isRenderedPath(path: string): boolean {
  return isMarkdownPath(path) || imageMimeFor(path) !== null;
}

/** File extensions the viewer renders as markdown. */
const MARKDOWN_EXT = /\.(md|markdown|mdown|mkd|mdx)$/i;

/** True for a path the viewer should render as markdown rather than raw text. */
export function isMarkdownPath(path: string): boolean {
  return MARKDOWN_EXT.test(path);
}

const IMAGE_MIME: Record<string, string> = {
  png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", gif: "image/gif",
  webp: "image/webp", svg: "image/svg+xml", avif: "image/avif", bmp: "image/bmp", ico: "image/x-icon",
};

/**
 * The image mime for a workspace path, by extension, or ``null`` when the
 * extension is not an image format a browser renders.  The blob the viewer
 * builds carries this type, because a blob with no type is not reliably
 * decoded as an image -- SVG in particular needs its type to render.
 */
export function imageMimeFor(path: string): string | null {
  const ext = /\.([a-z0-9]+)$/i.exec(path)?.[1]?.toLowerCase();
  return ext ? (IMAGE_MIME[ext] ?? null) : null;
}

/** What a link or image reference in a workspace document points at. */
export type Reference =
  | { kind: "external"; href: string }
  | { kind: "workspace"; path: string; fragment: string }
  | { kind: "anchor"; fragment: string }
  | { kind: "blocked" };

const SCHEME = /^([a-z][a-z0-9+.-]*):/i;

/**
 * Classify ``href`` as written in the document at ``fromPath``.
 *
 * - ``http:`` / ``https:`` / ``mailto:`` → ``external`` (opened in a new tab).
 * - any other scheme (``javascript:``, ``data:``, ``file:``, ...) and a
 *   protocol-relative ``//host`` → ``blocked``: nothing in a workspace
 *   document has a reason to carry one, and each is a way out of the
 *   viewer's rules.
 * - ``#heading`` → ``anchor`` within the same document.
 * - anything else is a path: resolved against the document's directory
 *   (or the workspace root, for a leading ``/``), percent-decoded, and
 *   normalised.  One that climbs above the root is ``blocked``.
 */
export function classifyReference(href: string, fromPath: string): Reference {
  const raw = href.trim();
  if (!raw) return { kind: "blocked" };
  if (raw.startsWith("//")) return { kind: "blocked" };
  const scheme = SCHEME.exec(raw)?.[1]?.toLowerCase();
  if (scheme) {
    return scheme === "http" || scheme === "https" || scheme === "mailto" ? { kind: "external", href: raw } : { kind: "blocked" };
  }
  if (raw.startsWith("#")) return { kind: "anchor", fragment: raw.slice(1) };
  const hashAt = raw.indexOf("#");
  const fragment = hashAt >= 0 ? raw.slice(hashAt + 1) : "";
  const pathPart = (hashAt >= 0 ? raw.slice(0, hashAt) : raw).split("?")[0] ?? "";
  let decoded: string;
  try { decoded = decodeURIComponent(pathPart); } catch { return { kind: "blocked" }; }
  const path = resolveWorkspacePath(decoded, fromPath);
  return path === null ? { kind: "blocked" } : { kind: "workspace", path, fragment };
}

/**
 * Resolve ``target`` against the directory of ``fromPath`` (both
 * workspace-relative), returning a normalised workspace-relative path, or
 * ``null`` when the result is empty or climbs above the workspace root.
 */
export function resolveWorkspacePath(target: string, fromPath: string): string | null {
  const t = target.replace(/\\/g, "/");
  const base = t.startsWith("/") ? [] : fromPath.split("/").filter(Boolean).slice(0, -1);
  const out = [...base];
  for (const part of t.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") {
      if (!out.length) return null;
      out.pop();
      continue;
    }
    out.push(part);
  }
  return out.length ? out.join("/") : null;
}
