/**
 * A deliberately small, injection-free markdown reader for the prose
 * around ``<j-*>`` blocks.  Code fences and tables are already lifted
 * out by the server's formatters, so what remains is paragraphs,
 * headings, lists, block quotes, rules and inline emphasis / code /
 * links.  Output is an AST the React layer renders with ordinary
 * elements — no ``innerHTML``, so model output can never inject
 * markup into the page.
 */

export type Inline =
  | { kind: "text"; text: string }
  | { kind: "code"; text: string }
  | { kind: "strong"; children: Inline[] }
  | { kind: "em"; children: Inline[] }
  | { kind: "link"; href: string; children: Inline[] };

export type Block =
  | { kind: "paragraph"; children: Inline[] }
  | { kind: "heading"; level: number; children: Inline[] }
  | { kind: "list"; ordered: boolean; items: Inline[][] }
  | { kind: "quote"; children: Inline[] }
  | { kind: "rule" }
  | { kind: "pre"; text: string };

// Built fresh per call: ``parseInline`` recurses for emphasis children, and a
// shared ``/g`` regex would have its ``lastIndex`` clobbered by the inner call.
// Underscore emphasis needs a word boundary on both sides so ``snake_case_names``
// stay literal.
const INLINE_SRC =
  "(`+)([\\s\\S]*?[^`])\\1(?!`)|\\*\\*([^*]+)\\*\\*|(?<!\\w)__([^_]+)__(?!\\w)|\\*([^*\\s][^*]*?)\\*|(?<!\\w)_([^_\\s][^_]*?)_(?!\\w)|\\[([^\\]]+)\\]\\((https?:\\/\\/[^\\s)]+)\\)";

export function parseInline(text: string): Inline[] {
  const out: Inline[] = [];
  const re = new RegExp(INLINE_SRC, "g");
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push({ kind: "text", text: text.slice(last, m.index) });
    if (m[2] !== undefined) out.push({ kind: "code", text: m[2] });
    else if (m[3] !== undefined || m[4] !== undefined) out.push({ kind: "strong", children: parseInline(m[3] ?? m[4] ?? "") });
    else if (m[5] !== undefined || m[6] !== undefined) out.push({ kind: "em", children: parseInline(m[5] ?? m[6] ?? "") });
    else if (m[7] !== undefined && m[8] !== undefined) out.push({ kind: "link", href: m[8], children: parseInline(m[7]) });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

export function parseMarkdown(text: string): Block[] {
  const blocks: Block[] = [];
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  let i = 0;
  const para: string[] = [];
  const flushPara = () => {
    if (!para.length) return;
    blocks.push({ kind: "paragraph", children: parseInline(para.join("\n")) });
    para.length = 0;
  };
  while (i < lines.length) {
    const line = lines[i]!;
    if (/^\s*$/.test(line)) { flushPara(); i++; continue; }
    if (/^```/.test(line)) {
      flushPara();
      const buf: string[] = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i]!)) buf.push(lines[i]!), i++;
      i++;
      blocks.push({ kind: "pre", text: buf.join("\n") });
      continue;
    }
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) { flushPara(); blocks.push({ kind: "heading", level: h[1]!.length, children: parseInline(h[2]!) }); i++; continue; }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flushPara(); blocks.push({ kind: "rule" }); i++; continue; }
    if (/^\s*>\s?/.test(line)) {
      flushPara();
      const buf: string[] = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i]!)) buf.push(lines[i]!.replace(/^\s*>\s?/, "")), i++;
      blocks.push({ kind: "quote", children: parseInline(buf.join("\n")) });
      continue;
    }
    const li = /^\s*([-*+]|\d+[.)])\s+(.*)$/.exec(line);
    if (li) {
      flushPara();
      const ordered = /\d/.test(li[1]!);
      const items: Inline[][] = [];
      while (i < lines.length) {
        const cur = /^\s*([-*+]|\d+[.)])\s+(.*)$/.exec(lines[i]!);
        if (!cur) break;
        let item = cur[2]!;
        i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]!) && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i]!)) item += "\n" + lines[i]!.trim(), i++;
        items.push(parseInline(item));
      }
      blocks.push({ kind: "list", ordered, items });
      continue;
    }
    para.push(line);
    i++;
  }
  flushPara();
  return blocks;
}
