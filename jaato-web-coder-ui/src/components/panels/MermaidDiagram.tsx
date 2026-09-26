/**
 * A ```` ```mermaid ```` fence in a workspace markdown document, drawn as a
 * diagram (``MarkdownView``'s ``pre`` renderer hands the fence here).
 *
 * **Loaded only when a document has one.**  ``mermaid`` is by far the
 * largest library the client can load, so it is imported on the first
 * diagram rather than with the markdown view, and every diagram after that
 * reuses the loaded module.
 *
 * **Shown as an image, never inserted as markup.**  ``mermaid.render``
 * produces an SVG *string*; putting it in the page would mean
 * ``innerHTML``, which this client never uses.  The SVG instead becomes a
 * blob URL on an ``<img>``: an image cannot run script, cannot reach the
 * page's DOM, and cannot load anything else, whatever the diagram source
 * contains.  Two settings follow from that and from the source being
 * untrusted text:
 *
 * - ``securityLevel: "strict"`` -- mermaid sanitises labels and disables
 *   click handlers and links (an image could not follow them anyway).
 * - ``htmlLabels: false`` -- labels are SVG ``<text>`` rather than HTML in
 *   a ``<foreignObject>``, which renders unreliably inside an image.
 *
 * The theme follows the client's (``isDarkTheme``): mermaid's ``dark`` on
 * a dark theme, ``neutral`` otherwise.
 *
 * A diagram mermaid cannot parse shows the error and the source as a
 * plain code block -- the document still says what its author wrote.
 */
import { useEffect, useState } from "react";
import { useJaato } from "@/store/store";
import { isDarkTheme } from "@/theme/themes";

type Mermaid = typeof import("mermaid").default;
let loading: Promise<Mermaid> | null = null;

/** The mermaid module, imported once on first use. */
function loadMermaid(): Promise<Mermaid> {
  loading ??= import("mermaid").then((m) => m.default);
  return loading;
}

let counter = 0;

type State = { status: "rendering" } | { status: "ok"; url: string; width: number | null; svg: string } | { status: "error"; error: string };

/**
 * The width mermaid drew the diagram at.  Its SVG says ``width="100%"``
 * with the real size as ``style="max-width: NNNpx"``; as an image that
 * would stretch a three-box graph across the whole panel, so the image is
 * given this width instead (and ``max-width: 100%`` still shrinks a wide one).
 */
export function naturalWidthOf(svg: string): number | null {
  const m = /max-width:\s*([\d.]+)px/.exec(svg);
  const n = m ? Number(m[1]) : NaN;
  return Number.isFinite(n) && n > 0 ? Math.ceil(n) : null;
}

/**
 * ``onOpen``, when given, makes the diagram a button that hands its SVG to
 * the viewer's image mode, where it can be zoomed -- a large diagram is
 * unreadable at the panel's width.
 */
/**
 * The SVG with an explicit size, for the image viewer.  Mermaid writes
 * ``width="100%"`` on the root, which leaves an image with no natural
 * size -- it is then laid out to whatever box holds it, and the viewer's
 * ``1:1`` and zoom readout have nothing to measure against.  The width is
 * the one mermaid drew at (``naturalWidthOf``) and the height follows the
 * ``viewBox``'s ratio.  Returned unchanged when either is missing.
 */
export function sizedSvg(svg: string): string {
  const width = naturalWidthOf(svg);
  const vb = /viewBox="\s*[-\d.]+[\s,]+[-\d.]+[\s,]+([\d.]+)[\s,]+([\d.]+)\s*"/.exec(svg);
  if (!width || !vb) return svg;
  const height = Math.ceil((width * Number(vb[2])) / Number(vb[1]));
  if (!Number.isFinite(height) || height <= 0) return svg;
  return svg.replace(/<svg\b([^>]*)>/, (_m, attrs: string) => {
    const rest = attrs.replace(/\s(width|height)="[^"]*"/g, "");
    return `<svg${rest} width="${width}" height="${height}">`;
  });
}

export function MermaidDiagram({ source, onOpen }: { source: string; onOpen?: (svg: string) => void }) {
  const dark = useJaato((s) => isDarkTheme(s.ui.theme));
  const [state, setState] = useState<State>({ status: "rendering" });

  useEffect(() => {
    let cancelled = false;
    let url: string | undefined;
    setState({ status: "rendering" });
    void (async () => {
      try {
        const mermaid = await loadMermaid();
        mermaid.initialize({ startOnLoad: false, securityLevel: "strict", htmlLabels: false, theme: dark ? "dark" : "neutral" });
        const { svg } = await mermaid.render(`jaato-mermaid-${++counter}`, source);
        if (cancelled) return;
        url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
        setState({ status: "ok", url, width: naturalWidthOf(svg), svg });
      } catch (err) {
        if (!cancelled) setState({ status: "error", error: err instanceof Error ? err.message : String(err) });
      }
    })();
    return () => { cancelled = true; if (url) URL.revokeObjectURL(url); };
  }, [source, dark]);

  if (state.status === "ok") {
    return (
      <figure className="md-mermaid my-2" data-testid="mermaid-diagram">
        {onOpen ? (
          <button type="button" className="md-img-open" onClick={() => onOpen(sizedSvg(state.svg))} aria-label="Open the diagram in the image viewer" title="Open in the image viewer to zoom">
            <img src={state.url} alt="Mermaid diagram" className="max-w-full h-auto" width={state.width ?? undefined} />
          </button>
        ) : (
          <img src={state.url} alt="Mermaid diagram" className="max-w-full h-auto" width={state.width ?? undefined} />
        )}
      </figure>
    );
  }
  return (
    <div className="my-1.5">
      {state.status === "rendering"
        ? <div className="text-[11px] text-text-muted italic">Drawing diagram…</div>
        : <div className="text-[11px] text-error" role="alert">Diagram could not be drawn: {state.error}</div>}
      <pre className="code-block plate plate-ground p-2.5 my-1 overflow-x-auto whitespace-pre"><code>{source}</code></pre>
    </div>
  );
}
