/**
 * Which highlight.js grammar a workspace file is written in, by its name,
 * for the Files viewer's ``CodeView``.  Kept apart from the highlighter
 * (``components/panels/highlighter.ts``) so the panel can ask "is this
 * code?" without loading a single grammar; every name returned here is one
 * that module registers.
 */

const BY_EXT: Record<string, string> = {
  json: "json", jsonc: "json", json5: "json", ipynb: "json",
  py: "python", pyi: "python",
  ts: "typescript", tsx: "typescript", mts: "typescript", cts: "typescript",
  js: "javascript", jsx: "javascript", mjs: "javascript", cjs: "javascript",
  sh: "bash", bash: "bash", zsh: "bash",
  css: "css", diff: "diff", patch: "diff",
  go: "go", java: "java", rs: "rust", sql: "sql",
  ini: "ini", toml: "ini", cfg: "ini", conf: "ini", properties: "ini",
  xml: "xml", html: "xml", htm: "xml", xsd: "xml", plist: "xml",
  yaml: "yaml", yml: "yaml",
};

const BY_NAME: Record<string, string> = {
  dockerfile: "dockerfile", containerfile: "dockerfile",
  makefile: "bash", ".bashrc": "bash", ".profile": "bash", ".zshrc": "bash",
  ".gitconfig": "ini", ".editorconfig": "ini", ".env.example": "bash",
};

/**
 * The registered grammar for ``path``, by file name then extension, or
 * ``null``.  Markdown is deliberately absent: the viewer renders it.
 */
export function languageForPath(path: string): string | null {
  const name = (path.split("/").pop() ?? "").toLowerCase();
  if (BY_NAME[name]) return BY_NAME[name]!;
  if (name.startsWith("dockerfile.")) return "dockerfile";
  const ext = /\.([a-z0-9]+)$/.exec(name)?.[1];
  return ext ? (BY_EXT[ext] ?? null) : null;
}
