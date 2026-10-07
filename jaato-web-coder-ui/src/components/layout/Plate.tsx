/**
 * The redesign's unit of surface: a square, hairline-bordered plate with
 * registration marks at its corners, the way a component is drawn on a
 * blueprint.  ``edge`` recolours the border for a plate that means
 * something -- ``steel`` for the primary / focused one, ``warning`` for a
 * permission request, ``error`` for a failed tool -- and ``ground`` draws
 * it on the page ground rather than the surface (a code block inside a
 * white plate).  ``corners`` picks which marks to draw: all four by
 * default, two for a plate nested in another.
 *
 * Everything else about a plate is the caller's: ``className`` is
 * appended, the children are laid out by the caller, and ``as`` picks
 * the element (``section`` for a labelled region, ``form`` for a form).
 */
import type { ElementType, HTMLAttributes, ReactNode } from "react";

export type PlateEdge = "hairline" | "steel" | "warning" | "error";

export interface PlateProps extends HTMLAttributes<HTMLElement> {
  as?: ElementType;
  edge?: PlateEdge;
  ground?: boolean;
  corners?: "all" | "two" | "none";
  children?: ReactNode;
}

const EDGE_CLASS: Record<PlateEdge, string> = { hairline: "", steel: "plate-steel", warning: "plate-warning", error: "plate-error" };

export function Plate({ as, edge = "hairline", ground = false, corners = "all", className = "", children, ...rest }: PlateProps) {
  const Tag = (as ?? "div") as ElementType;
  return (
    <Tag className={`plate ${EDGE_CLASS[edge]} ${ground ? "plate-ground" : ""} ${className}`} {...rest}>
      {corners !== "none" && <i className="corner tl" aria-hidden="true" />}
      {corners === "all" && <i className="corner tr" aria-hidden="true" />}
      {corners === "all" && <i className="corner bl" aria-hidden="true" />}
      {corners !== "none" && <i className="corner br" aria-hidden="true" />}
      {children}
    </Tag>
  );
}

/** A plate header row: a kicker on the left, an optional monospace value on the right, hairline below. */
export function PlateHeader({ label, value, children, className = "" }: { label: ReactNode; value?: ReactNode; children?: ReactNode; className?: string }) {
  return (
    <div className={`flex items-baseline justify-between gap-3 px-3.5 py-2 border-b hairline ${className}`}>
      <span className="kicker">{label}</span>
      {children}
      {value != null && <span className="font-mono text-[11px] text-text-muted">{value}</span>}
    </div>
  );
}
