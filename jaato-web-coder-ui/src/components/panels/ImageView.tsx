/**
 * An image in the Files panel's content viewer (``WorkspacePanel.tsx``):
 * a workspace image file, or a diagram the markdown view drew and handed
 * over to be looked at up close.  Loaded lazily with the zoom library it
 * uses, so a session that never views an image never downloads either.
 *
 * **Shown from a blob URL on an ``<img>``, like every image in this
 * client.**  The bytes arrive through the daemon's ``workspace.file.fetch``
 * (containment and the credential rule are the daemon's), and an SVG shown
 * as an image cannot run script or load anything else, whatever it holds.
 * The URL is revoked when the bytes change or the view unmounts.
 *
 * **Fit is scale 1.**  The image is laid out to fit the viewer (``max-width``
 * / ``max-height`` 100%), and ``react-zoom-pan-pinch`` scales that layout:
 * wheel and pinch zoom, drag pans, double-click toggles.  ``1:1`` computes
 * the scale at which one image pixel is one CSS pixel from the natural and
 * the laid-out width, so it is exact however the panel is sized; the zoom
 * readout is relative to the natural size for the same reason — a readout
 * of the library's own scale would say "100%" about an image shrunk to fit.
 *
 * Transparent pixels sit on a checkerboard, so a transparent PNG is not
 * mistaken for one with a white background.
 */
import { useEffect, useRef, useState } from "react";
import { TransformComponent, TransformWrapper, useControls } from "react-zoom-pan-pinch";

export interface ImageViewProps {
  /** The image's bytes. */
  data: Uint8Array;
  /** Its mime type (``imageMimeFor``); the blob carries it, which SVG needs. */
  mime: string;
  /** What the image is, for its ``alt`` text. */
  label: string;
  /**
   * A Tailwind height class for the viewing area (the viewer's expand
   * state), given with ``!``: the zoom library injects an UNLAYERED
   * stylesheet sizing its wrapper to its content, and an unlayered rule
   * outranks every Tailwind utility (the ``.plate`` lesson in theme.css).
   */
  heightClass: string;
}

/** ``1920×1080 · PNG · 240 KB``, the parts that are known. */
export function describeImage(natural: { w: number; h: number } | null, mime: string, bytes: number): string {
  const kind = (mime.split("/")[1] ?? mime).replace(/\+xml$/, "").toUpperCase();
  const size = bytes < 1024 ? `${bytes} B` : bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return [natural ? `${natural.w}×${natural.h}` : null, kind, size].filter(Boolean).join(" · ");
}

const BTN = "link text-[11px]";

function Controls({ img, natural, zoom }: { img: HTMLImageElement | null; natural: { w: number; h: number } | null; zoom: number | null }) {
  const { zoomIn, zoomOut, resetTransform, centerView } = useControls();
  const actual = () => {
    // The laid-out (scale 1) width: getBoundingClientRect includes the
    // current transform, offsetWidth does not.
    if (!img || !natural || !img.offsetWidth) return;
    centerView(natural.w / img.offsetWidth, 150);
  };
  return (
    <div className="flex items-center gap-2.5 px-2 py-1 border-b hairline">
      <button type="button" className={BTN} onClick={() => resetTransform(150)} title="Fit the image to the viewer">fit</button>
      <button type="button" className={BTN} onClick={actual} disabled={!natural} title="One image pixel per screen pixel">1:1</button>
      <button type="button" className={BTN} onClick={() => zoomOut(0.4, 150)} aria-label="Zoom out">−</button>
      <button type="button" className={BTN} onClick={() => zoomIn(0.4, 150)} aria-label="Zoom in">+</button>
      {zoom !== null && <span className="font-mono text-[11px] text-text-muted" aria-live="polite">{Math.round(zoom * 100)}%</span>}
    </div>
  );
}

export default function ImageView({ data, mime, label, heightClass }: ImageViewProps) {
  // Created and revoked by the SAME effect, never memoised: a URL made in a
  // memo and revoked in an effect cleanup is dead after React's development
  // double-mount, which runs the cleanup once while the memo is kept.
  const [url, setUrl] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [scale, setScale] = useState(1);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const u = URL.createObjectURL(new Blob([data as BlobPart], { type: mime }));
    setUrl(u);
    setNatural(null);
    setFailed(false);
    setScale(1);
    return () => URL.revokeObjectURL(u);
  }, [data, mime]);

  const laidOut = imgRef.current?.offsetWidth ?? 0;
  const zoom = natural && laidOut ? (scale * laidOut) / natural.w : null;

  if (!url) return null;
  if (failed) {
    return <div className="px-2 py-2 text-[12px] text-text-muted italic">The browser could not decode this image -- use download.</div>;
  }
  return (
    <div data-testid="image-view">
      <TransformWrapper
        minScale={0.2}
        maxScale={32}
        centerOnInit
        doubleClick={{ mode: "toggle", step: 1 }}
        wheel={{ step: 0.15 }}
        onTransformed={(_ref, s) => setScale(s.scale)}
      >
        <Controls img={imgRef.current} natural={natural} zoom={zoom} />
        <TransformComponent wrapperClass={`img-checker !w-full ${heightClass}`} contentClass="!w-full !h-full flex items-center justify-center">
          <img
            ref={imgRef}
            src={url}
            alt={label}
            className="max-w-full max-h-full object-contain select-none"
            draggable={false}
            onLoad={(e) => { const i = e.currentTarget; setNatural(i.naturalWidth ? { w: i.naturalWidth, h: i.naturalHeight } : null); }}
            onError={() => setFailed(true)}
          />
        </TransformComponent>
      </TransformWrapper>
      <div className="px-2 py-1 text-[11px] text-text-muted font-mono border-t hairline" data-testid="image-info">
        {describeImage(natural, mime, data.byteLength)}
      </div>
    </div>
  );
}
