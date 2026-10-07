import { strict as assert } from "node:assert";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, test } from "node:test";
import {
  managedContent,
  managedMarker,
  parseMarker,
  removeManagedFile,
  writeManagedFile,
  type ManagedFile,
} from "../src/managed-files.js";

/** A plain non-atomic write is enough for a test; the real one is github.ts's temp+rename. */
const plainWrite = (path: string, body: string, mode: number) => {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, body, { mode });
};

const ws = () => mkdtempSync(join(tmpdir(), "jwcs-managed-"));
const FILE = (version: number): ManagedFile => ({
  relativePath: ".jaato/instructions/40-x.md",
  markerId: "x-guidance",
  version,
  body: `# X v${version}\n\nbody line\n`,
});

describe("managed files — marker helpers", () => {
  test("marker + content shape, and parse round-trips", () => {
    assert.equal(
      managedMarker("x-guidance", 3),
      "<!-- jaato-managed: x-guidance v3 — delete this line to keep your own edits -->",
    );
    const content = managedContent(FILE(2));
    assert.ok(content.startsWith("<!-- jaato-managed: x-guidance v2 "), "marker is the first line");
    assert.ok(content.endsWith("\n"), "trailing newline ensured");
    const first = content.split("\n", 1)[0]!;
    assert.deepEqual(parseMarker(first), { markerId: "x-guidance", version: 2 });
  });

  test("parseMarker rejects a line that is not our marker", () => {
    assert.equal(parseMarker("# just a heading"), null);
    assert.equal(parseMarker("<!-- some other comment -->"), null);
  });
});

describe("writeManagedFile — the §5 rules", () => {
  test("absent → written", () => {
    const dir = ws();
    const out = writeManagedFile(dir, FILE(1), plainWrite);
    assert.deepEqual(out, { path: ".jaato/instructions/40-x.md", action: "written", reason: "absent" });
    const written = readFileSync(join(dir, ".jaato/instructions/40-x.md"), "utf8");
    assert.equal(written, managedContent(FILE(1)));
  });

  test("our marker, SAME version → no-op, bytes untouched", () => {
    const dir = ws();
    const dest = join(dir, ".jaato/instructions/40-x.md");
    // A user edited the BODY but kept the marker at the current version.
    const edited = `${managedMarker("x-guidance", 1)}\n# my own edits\n`;
    plainWrite(dest, edited, 0o644);
    const out = writeManagedFile(dir, FILE(1), plainWrite);
    assert.equal(out.action, "unchanged");
    assert.equal(readFileSync(dest, "utf8"), edited, "an edited body at the same version is left as-is");
  });

  test("our marker, DIFFERENT version → overwrite", () => {
    const dir = ws();
    const dest = join(dir, ".jaato/instructions/40-x.md");
    plainWrite(dest, `${managedMarker("x-guidance", 0)}\n# old shipped content\n`, 0o644);
    const out = writeManagedFile(dir, FILE(1), plainWrite);
    assert.deepEqual(out, { path: ".jaato/instructions/40-x.md", action: "written", reason: "version-changed", previousVersion: 0 });
    assert.equal(readFileSync(dest, "utf8"), managedContent(FILE(1)));
  });

  test("present WITHOUT our marker → skipped, never clobbered", () => {
    const dir = ws();
    const dest = join(dir, ".jaato/instructions/40-x.md");
    const theirs = "# entirely my own file\n";
    plainWrite(dest, theirs, 0o644);
    const out = writeManagedFile(dir, FILE(1), plainWrite);
    assert.equal(out.action, "skipped-user-file");
    assert.equal(readFileSync(dest, "utf8"), theirs, "the user's file is left byte-for-byte");
  });

  test("a different owner's marker is also left alone", () => {
    const dir = ws();
    const dest = join(dir, ".jaato/instructions/40-x.md");
    const theirs = `${managedMarker("someone-else", 1)}\nnot ours\n`;
    plainWrite(dest, theirs, 0o644);
    const out = writeManagedFile(dir, FILE(1), plainWrite);
    assert.equal(out.action, "skipped-user-file");
    assert.equal(readFileSync(dest, "utf8"), theirs);
  });

  test("an I/O failure is an error outcome, not a throw", () => {
    const dir = ws();
    const throwing = () => { throw new Error("disk full"); };
    const out = writeManagedFile(dir, FILE(1), throwing);
    assert.equal(out.action, "error");
    assert.match((out as { message: string }).message, /disk full/);
  });
});

describe("removeManagedFile — bind-to-none", () => {
  test("our marker → removed", () => {
    const dir = ws();
    const dest = join(dir, ".jaato/instructions/40-x.md");
    plainWrite(dest, managedContent(FILE(1)), 0o644);
    const out = removeManagedFile(dir, FILE(1));
    assert.equal(out.action, "removed");
    assert.ok(!existsSync(dest));
  });

  test("a user-replaced file (marker deleted) is left in place", () => {
    const dir = ws();
    const dest = join(dir, ".jaato/instructions/40-x.md");
    const theirs = "# my own file, marker deleted\n";
    plainWrite(dest, theirs, 0o644);
    const out = removeManagedFile(dir, FILE(1));
    assert.equal(out.action, "skipped-user-file");
    assert.equal(readFileSync(dest, "utf8"), theirs);
  });

  test("absent → nothing to do", () => {
    const dir = ws();
    const out = removeManagedFile(dir, FILE(1));
    assert.equal(out.action, "absent");
  });
});
