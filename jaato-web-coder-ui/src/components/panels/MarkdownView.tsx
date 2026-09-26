/**
 * Rendered view of a markdown file from the workspace, shown by the Files
 * panel's content viewer (``WorkspacePanel.tsx``) when the file it opened
 * is markdown.  Loaded lazily -- the parser, GFM and the highlighter
 * (``rehypeFenceHighlight``) are the heaviest code in the client, and most
 * sessions never view a ``.md``.
 *
 * **No ``innerHTML``, like the rest of the client.**  ``react-markdown``
 * turns the document into a syntax tree and renders ordinary React
 * elements.  Raw HTML in the document is DROPPED (no ``rehype-raw``), so a
 * file the agent wrote cannot inject markup into the page; the viewer's
 * raw mode still shows it as text.
 *
 * A ```` ```mermaid ```` fence is drawn as a diagram by ``MermaidDiagram``
 * (which loads mermaid itself, on the first diagram, and shows the result
 * as an image).
 *
 * **Every URL goes through ``classifyReference``** (``protocol/workspacePaths``):
 *
 * - ``http(s):`` / ``mailto:`` links open in a new tab, ``noopener``.
 * - A relative link (``docs/setup.md``, ``../README.md``, ``/CONTRIBUTING.md``)
 *   is a workspace path: clicking it calls ``onOpen``, which opens that file
 *   in the same viewer.  ``#heading`` scrolls within this document; headings
 *   carry ``md-``-prefixed ids (``rehype-slug``) so they cannot collide with
 *   the page's own element ids.
 * - A relative IMAGE is fetched through ``fetchFile`` -- the daemon's
 *   ``workspace.file.fetch``, which applies containment and the credential
 *   rule -- and shown from a blob URL that is revoked when the image
 *   unmounts.  A REMOTE image is not loaded by default: fetching it would
 *   tell a third party that this document was opened (the tracking-pixel
 *   shape), so it renders as a placeholder naming its host, with a
 *   ``load remote images`` button.  That button is PER DOCUMENT and lasts
 *   while the document is on screen (``key`` it by path to reset it); once
 *   pressed, every remote image in the document loads, with
 *   ``referrerPolicy="no-referrer"`` so the host is not told which page
 *   asked.
 * - Every other scheme (``javascript:``, ``data:``, ``file:``, ...) renders
 *   as plain text.
 */
import { useEffect, useRef, useState, type ReactNode } from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";
import type { Element } from "hast";
import rehypeFenceHighlight from "./rehypeFenceHighlight";
import { MermaidDiagram } from "./MermaidDiagram";
import { classifyReference, imageMimeFor } from "@/protocol/workspacePaths";

/** Prefix on every heading id this view generates. */
export const HEADING_ID_PREFIX = "md-";

export interface MarkdownViewProps {
  /** The document's text. */
  source: string;
  /** The document's workspace-relative path; relative references resolve against it. */
  path: string;
  /** Fetch a workspace file's bytes, or ``null`` when it cannot be fetched. */
  fetchFile: (path: string) => Promise<Uint8Array | null>;
  /** Open another workspace file in the viewer (a relative link was clicked). */
  onOpen: (path: string) => void;
}

/**
 * The source of a ```` ```mermaid ```` fence, when ``pre`` is one, else
 * ``null``.  Read off the hast node rather than the rendered children:
 * ``rehypeFenceHighlight`` leaves an unregistered language's text alone,
 * so the code element's children are the fence's text verbatim.
 */
function mermaidSource(pre: Element | undefined): string | null {
  const code = pre?.children.find((c): c is Element => c.type === "element" && c.tagName === "code");
  const cls: unknown = code?.properties?.className;
  if (!code || !Array.isArray(cls) || !cls.includes("language-mermaid")) return null;
  return code.children.map((c) => (c.type === "text" ? c.value : "")).join("");
}

function hostOf(href: string): string {
  try { return new URL(href).host; } catch { return href; }
}

function RemoteImage({ href, alt, allowed, onAllow }: { href: string; alt: string; allowed: boolean; onAllow: () => void }) {
  if (allowed) return <img src={href} alt={alt} title={href} className="md-img" referrerPolicy="no-referrer" loading="lazy" />;
  return (
    <span className="md-img-missing">
      [remote image: {alt || href} · {hostOf(href)}]{" "}
      <button type="button" className="link not-italic text-[11px]" onClick={onAllow} title="Fetch every remote image in this document from its host">
        load remote images
      </button>
    </span>
  );
}

function WorkspaceImage({ path, alt, fetchFile }: { path: string; alt: string; fetchFile: MarkdownViewProps["fetchFile"] }) {
  const [state, setState] = useState<{ url?: string; failed?: boolean }>({});
  const mime = imageMimeFor(path);
  useEffect(() => {
    if (!mime) return;
    let url: string | undefined;
    let cancelled = false;
    void fetchFile(path).then(
      (data) => {
        if (cancelled) return;
        if (!data) { setState({ failed: true }); return; }
        url = URL.createObjectURL(new Blob([data as BlobPart], { type: mime }));
        setState({ url });
      },
      () => { if (!cancelled) setState({ failed: true }); },
    );
    return () => { cancelled = true; if (url) URL.revokeObjectURL(url); };
  }, [path, mime, fetchFile]);
  if (!mime || state.failed) return <span className="md-img-missing" title={path}>[image: {alt || path}]</span>;
  if (!state.url) return <span className="md-img-missing" title={path}>[loading {alt || path}…]</span>;
  return <img src={state.url} alt={alt} title={path} className="md-img" />;
}

export default function MarkdownView({ source, path, fetchFile, onOpen }: MarkdownViewProps) {
  const root = useRef<HTMLDivElement>(null);
  const [remoteAllowed, setRemoteAllowed] = useState(false);

  const scrollToAnchor = (fragment: string) => {
    let id = fragment;
    try { id = decodeURIComponent(fragment); } catch { /* keep as written */ }
    const el = root.current?.querySelector(`#${CSS.escape(HEADING_ID_PREFIX + id.toLowerCase())}`);
    el?.scrollIntoView({ block: "start", behavior: "smooth" });
  };

  const components: Components = {
    a({ href, children }) {
      const ref = classifyReference(href ?? "", path);
      switch (ref.kind) {
        case "external":
          return <a href={ref.href} target="_blank" rel="noopener noreferrer" title={ref.href}>{children}</a>;
        case "anchor":
          return <a href={`#${HEADING_ID_PREFIX}${ref.fragment}`} onClick={(e) => { e.preventDefault(); scrollToAnchor(ref.fragment); }}>{children}</a>;
        case "workspace":
          return (
            <a href={`#${ref.path}`} title={ref.path} onClick={(e) => { e.preventDefault(); onOpen(ref.path); }}>
              {children}
            </a>
          );
        default:
          return <span className="md-link-blocked" title="Link not followed: only web, mail and workspace links are">{children}</span>;
      }
    },
    img({ src, alt }) {
      const ref = classifyReference(typeof src === "string" ? src : "", path);
      if (ref.kind === "workspace") return <WorkspaceImage path={ref.path} alt={alt ?? ""} fetchFile={fetchFile} />;
      if (ref.kind === "external" && !ref.href.toLowerCase().startsWith("mailto:")) {
        return <RemoteImage href={ref.href} alt={alt ?? ""} allowed={remoteAllowed} onAllow={() => setRemoteAllowed(true)} />;
      }
      return <span className="md-img-missing">[image: {alt || "blocked"}]</span>;
    },
    table({ children }): ReactNode {
      return <div className="overflow-x-auto"><table className="j-table">{children}</table></div>;
    },
    pre({ node, children }) {
      const mermaid = mermaidSource(node);
      if (mermaid !== null) return <MermaidDiagram source={mermaid} />;
      return <pre className="code-block plate plate-ground p-2.5 my-1.5 overflow-x-auto whitespace-pre">{children}</pre>;
    },
    code({ className, children }) {
      // Inside ``pre`` it is a fenced block (carries ``language-*`` / ``hljs``);
      // otherwise it is inline code.
      const block = /\b(language-|hljs)/.test(className ?? "");
      return <code className={block ? className : "inline"}>{children}</code>;
    },
  };

  return (
    <div ref={root} className="prose-j md-doc" data-testid="markdown-view">
      <Markdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeSlug, { prefix: HEADING_ID_PREFIX }], rehypeFenceHighlight]}
        components={components}
      >
        {source}
      </Markdown>
    </div>
  );
}
