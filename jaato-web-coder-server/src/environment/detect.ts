/**
 * Marker detection: what a workspace's files suggest it needs
 * (docs/design/web-coder-environment-bootstrap.md §9).
 *
 * Detection only PROPOSES.  Nothing here installs, writes or runs anything;
 * the page shows a proposal and the user accepts or declines it.
 *
 * It reads the workspace root and its immediate subdirectories (#1332's
 * clone flow puts each repository at ``<ws>/<name>``), never deeper, and
 * never a dot-directory or ``node_modules``.  Every file read is bounded, and
 * a file that cannot be read is skipped: a proposal is a convenience.
 */
import { closeSync, openSync, readdirSync, readSync, statSync } from "node:fs";
import { join } from "node:path";
import type { ToolId } from "./catalog.js";

/** One suggestion: a toolchain, the version a file pins (if any), and the file that said so. */
export interface Detection {
  tool: ToolId;
  /** The version the repository pins, verbatim from the file; ``null`` when it names none. */
  pin: string | null;
  /** Workspace-relative path of the file that suggested it. */
  source: string;
}

/** How many subdirectories one scan looks into. */
export const MAX_SCANNED_DIRS = 64;
/** The most bytes read from one marker file. */
export const MAX_MARKER_BYTES = 64 * 1024;

const SKIP_DIRS = new Set(["node_modules", "worktrees", "__pycache__", "venv", "target", "dist", "build"]);

/** Read at most {@link MAX_MARKER_BYTES} of a regular file, or ``null``. */
export function readBounded(path: string): string | null {
  try {
    if (!statSync(path).isFile()) return null;
    const fd = openSync(path, "r");
    try {
      const buf = Buffer.alloc(MAX_MARKER_BYTES);
      const n = readSync(fd, buf, 0, MAX_MARKER_BYTES, 0);
      return buf.subarray(0, n).toString("utf8");
    } finally { closeSync(fd); }
  } catch { return null; }
}

function isFile(path: string): boolean {
  try { return statSync(path).isFile(); } catch { return false; }
}

/** The directories a scan looks at: ``""`` (the root) then each immediate, non-hidden subdirectory, sorted. */
export function scanDirs(workspace: string): string[] {
  const out = [""];
  let entries: string[] = [];
  try {
    entries = readdirSync(workspace, { withFileTypes: true })
      .filter((e) => e.isDirectory() && !e.name.startsWith(".") && !SKIP_DIRS.has(e.name))
      .map((e) => e.name)
      .sort();
  } catch { /* an unreadable root has only itself */ }
  return out.concat(entries.slice(0, MAX_SCANNED_DIRS));
}

function rel(dir: string, name: string): string {
  return dir ? `${dir}/${name}` : name;
}

/** ``.tool-versions`` lines: ``<tool> <version> …``. */
function parseToolVersions(body: string): Map<string, string> {
  const out = new Map<string, string>();
  for (const line of body.split("\n")) {
    const t = line.replace(/#.*/, "").trim();
    if (!t) continue;
    const [name, version] = t.split(/\s+/);
    if (name && version) out.set(name, version);
  }
  return out;
}

const TOOL_VERSION_NAMES: Array<[ToolId, string[]]> = [
  ["node", ["node", "nodejs"]],
  ["go", ["go", "golang"]],
  ["bun", ["bun"]],
  ["java", ["java"]],
  ["maven", ["maven"]],
  ["gradle", ["gradle"]],
];

/** The Java release a ``pom.xml`` compiles for, in the order Maven itself prefers them. */
function pomJavaRelease(pom: string): string | null {
  for (const tag of ["maven.compiler.release", "release", "java.version", "maven.compiler.source", "maven.compiler.target"]) {
    const m = new RegExp(`<${tag.replace(/\./g, "\\.")}>\\s*([0-9][0-9.]*)\\s*</`).exec(pom);
    if (m) return m[1]!;
  }
  return null;
}

/** The Java version a Gradle build names: a toolchain first, then the compatibility level. */
function gradleJavaVersion(build: string): string | null {
  const patterns = [
    /JavaLanguageVersion\.of\(\s*["']?(\d+)["']?\s*\)/,
    /jvmToolchain\(\s*(\d+)\s*\)/,
    /sourceCompatibility\s*=\s*JavaVersion\.VERSION_(\d+(?:_\d+)?)/,
    /sourceCompatibility\s*=\s*["']?(\d+(?:\.\d+)?)["']?/,
  ];
  for (const re of patterns) {
    const m = re.exec(build);
    if (m) return m[1]!.replace(/_/g, ".");
  }
  return null;
}

/** What one directory's files suggest. */
export function detectDir(workspace: string, dir: string): Detection[] {
  const base = join(workspace, dir);
  const found: Detection[] = [];
  const seen = new Set<ToolId>();
  const add = (tool: ToolId, pin: string | null, source: string) => {
    if (seen.has(tool)) return;
    seen.add(tool);
    found.push({ tool, pin: pin && pin.trim() ? pin.trim() : null, source });
  };

  // The most specific pin first: a version manager's own file beats a manifest.
  const tv = readBounded(join(base, ".tool-versions"));
  if (tv !== null) {
    const pins = parseToolVersions(tv);
    for (const [tool, names] of TOOL_VERSION_NAMES) {
      const name = names.find((n) => pins.has(n));
      if (name) add(tool, pins.get(name)!, rel(dir, ".tool-versions"));
    }
  }
  for (const f of [".nvmrc", ".node-version"]) {
    const body = readBounded(join(base, f));
    if (body !== null) add("node", body.split("\n")[0] ?? null, rel(dir, f));
  }
  const pkg = readBounded(join(base, "package.json"));
  if (pkg !== null) {
    let engine: string | null = null;
    try {
      const parsed = JSON.parse(pkg) as { engines?: { node?: unknown } };
      if (typeof parsed?.engines?.node === "string") engine = parsed.engines.node;
    } catch { /* a malformed package.json still suggests node */ }
    const bunLock = ["bun.lockb", "bun.lock", "bunfig.toml"].find((f) => isFile(join(base, f)));
    if (bunLock) add("bun", null, rel(dir, bunLock));
    add("node", engine, rel(dir, "package.json"));
  }
  const goVersion = readBounded(join(base, ".go-version"));
  if (goVersion !== null) add("go", goVersion.split("\n")[0] ?? null, rel(dir, ".go-version"));
  const gomod = readBounded(join(base, "go.mod"));
  if (gomod !== null) {
    const m = /^go\s+(\S+)/m.exec(gomod);
    add("go", m ? m[1]! : null, rel(dir, "go.mod"));
  }
  // Java: a version manager's file, then the build; Maven / Gradle only when the repository has no wrapper.
  const javaVersion = readBounded(join(base, ".java-version"));
  if (javaVersion !== null) add("java", javaVersion.split("\n")[0] ?? null, rel(dir, ".java-version"));
  const sdkman = readBounded(join(base, ".sdkmanrc"));
  if (sdkman !== null) {
    const m = /^\s*java\s*=\s*(\S+)/m.exec(sdkman);
    if (m) add("java", m[1]!, rel(dir, ".sdkmanrc"));
  }
  const pom = readBounded(join(base, "pom.xml"));
  if (pom !== null) {
    add("java", pomJavaRelease(pom), rel(dir, "pom.xml"));
    if (!isFile(join(base, "mvnw"))) add("maven", null, rel(dir, "pom.xml"));
  }
  const gradleFile = ["build.gradle.kts", "build.gradle", "settings.gradle.kts", "settings.gradle"].find((f) => isFile(join(base, f)));
  if (gradleFile) {
    add("java", gradleJavaVersion(readBounded(join(base, gradleFile)) ?? ""), rel(dir, gradleFile));
    if (!isFile(join(base, "gradlew"))) add("gradle", null, rel(dir, gradleFile));
  }
  const py = ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"].find((f) => isFile(join(base, f)));
  if (py) add("python", null, rel(dir, py));
  return found;
}

/**
 * Every suggestion in the workspace, one per toolchain: the first directory
 * (root first, then subdirectories in name order) that suggests a toolchain
 * decides its pin and source.
 */
export function detectWorkspace(workspace: string): Detection[] {
  const byTool = new Map<ToolId, Detection>();
  for (const dir of scanDirs(workspace)) {
    for (const d of detectDir(workspace, dir)) {
      const prev = byTool.get(d.tool);
      // A later directory's PIN fills in a first hit that named none.
      if (!prev) byTool.set(d.tool, d);
      else if (prev.pin === null && d.pin !== null) byTool.set(d.tool, d);
    }
  }
  return [...byTool.values()];
}
