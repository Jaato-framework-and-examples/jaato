/**
 * Pygments short-token-name → semantic colour role.
 *
 * ``<j-tok t="…">`` carries the short name from
 * ``pygments.token.STANDARD_TYPES`` (``k`` keyword, ``nf`` function
 * name, ``s2`` double-quoted string…).  The TUI feeds these back
 * through Rich's Pygments theme; here they become a CSS class whose
 * colour is a theme variable, so code colouring follows the UI theme
 * the same way it does in the terminal.
 */
export type TokenRole =
  | "keyword" | "name" | "function" | "class" | "builtin" | "string" | "number"
  | "operator" | "punctuation" | "comment" | "decorator" | "tag" | "attribute"
  | "error" | "inserted" | "deleted" | "heading" | "prompt" | "text";

const EXACT: Record<string, TokenRole> = {
  nf: "function", fm: "function",
  nc: "class", nn: "class", ne: "class", nt: "tag",
  nb: "builtin", bp: "builtin", no: "builtin", kc: "keyword", kt: "class",
  nd: "decorator", na: "attribute", nv: "name", vc: "name", vg: "name", vi: "name", vm: "name",
  gd: "deleted", gi: "inserted", gh: "heading", gu: "heading", gp: "prompt", gr: "error", gt: "error",
  err: "error", w: "text", "": "text",
};

export function tokenRole(t: string): TokenRole {
  const exact = EXACT[t];
  if (exact) return exact;
  switch (t[0]) {
    case "k": return "keyword";
    case "n": return "name";
    case "s": return "string";
    case "l": return "string";
    case "m": return "number";
    case "i": return "number"; // il — Integer.Long
    case "o": return "operator";
    case "p": return "punctuation";
    case "c": return "comment";
    case "g": return "text";
    default: return "text";
  }
}

export function tokenClass(t: string): string {
  return `tok tok-${tokenRole(t)}`;
}
