/**
 * A rehype plugin that syntax-highlights fenced code in the markdown view
 * (``MarkdownView.tsx``) with ``lowlight`` -- highlight.js producing a
 * syntax tree instead of an HTML string, so the result is still rendered
 * as React elements and nothing touches ``innerHTML``.
 *
 * It exists instead of ``rehype-highlight`` for one reason: that plugin
 * imports lowlight's whole ``common`` language pack at module level, so the
 * lazily-loaded markdown chunk carried ~37 grammars whichever ones it was
 * told to register.  Here the grammars are the ones ``highlighter.ts``
 * lists and no others.
 *
 * The grammars live in ``highlighter.ts``, shared with the Files viewer's
 * ``CodeView``.  Only a ``pre > code`` carrying ``language-<name>`` for a registered name
 * (or one of its aliases -- ``sh``, ``py``, ``ts``, ``yml``, ``html``,
 * ``toml``, ...) is highlighted, and nothing is auto-detected: a fence
 * with no language, or one not listed, stays plain monospace.  Guessing
 * would colour prose as code and costs a pass over every grammar.  Output
 * spans carry ``hljs-*`` classes, which ``theme.css`` maps onto the
 * client's ``tok-*`` palette.
 */
import type { Element, ElementContent, Root, RootContent } from "hast";
import { lowlight } from "./highlighter";

function textOf(node: ElementContent): string {
  if (node.type === "text") return node.value;
  if (node.type === "element") return node.children.map(textOf).join("");
  return "";
}

function languageOf(code: Element): string | null {
  const cls: unknown = code.properties?.className;
  const list = Array.isArray(cls) ? cls : typeof cls === "string" ? cls.split(/\s+/) : [];
  for (const c of list) {
    const m = /^lang(?:uage)?-(.+)$/.exec(String(c));
    if (m) return m[1]!;
  }
  return null;
}

function highlightCode(code: Element): void {
  const lang = languageOf(code);
  if (!lang || !lowlight.registered(lang)) return;
  const tree = lowlight.highlight(lang, code.children.map(textOf).join(""));
  code.children = tree.children as ElementContent[];
  const cls = code.properties.className;
  code.properties.className = [...(Array.isArray(cls) ? cls : []), "hljs"];
}

function walk(node: Root | RootContent): void {
  if (node.type !== "root" && node.type !== "element") return;
  if (node.type === "element" && node.tagName === "pre") {
    for (const child of node.children) if (child.type === "element" && child.tagName === "code") highlightCode(child);
    return;
  }
  for (const child of node.children) walk(child);
}

/** The plugin: ``rehypePlugins={[rehypeFenceHighlight]}``. */
export default function rehypeFenceHighlight() {
  return (tree: Root) => walk(tree);
}
