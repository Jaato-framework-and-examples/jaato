import { describe, expect, it } from "vitest";
import { STAGE_PER_FILE_LIMIT, STAGE_TOTAL_LIMIT, attachmentFooter, checkSizes, formatSize, normalizeFolder, stagedName } from "./attachments";

describe("stagedName", () => {
  it("stages a plain file at its own name under the workspace root", () => {
    expect(stagedName("notes.md", "", "")).toBe("notes.md");
  });
  it("keeps the path a dropped directory gave the file", () => {
    expect(stagedName("b.md", "docs/sub/b.md", "")).toBe("docs/sub/b.md");
  });
  it("prefixes the chosen folder, however it was spelled", () => {
    expect(stagedName("a.pdf", "", " inbox/ ")).toBe("inbox/a.pdf");
    expect(stagedName("a.pdf", "", "./inbox//deep/")).toBe("inbox/deep/a.pdf");
    expect(stagedName("a.pdf", "", "inbox\\win")).toBe("inbox/win/a.pdf");
  });
  it("refuses what the daemon refuses: climbing, absolute, empty", () => {
    expect(stagedName("a", "", "..")).toBeNull();
    expect(stagedName("a", "../a", "")).toBeNull();
    expect(stagedName("a", "", "x/../../y")).toBeNull();
    expect(stagedName("a", "/etc/passwd", "")).toBeNull();
    expect(stagedName("a", "C:/x", "")).toBeNull();
    expect(stagedName("", "", "")).toBeNull();
  });
});

describe("normalizeFolder", () => {
  it("is empty for the root however it is spelled", () => {
    expect(normalizeFolder("")).toBe("");
    expect(normalizeFolder(" / ")).toBe("");
    expect(normalizeFolder("./")).toBe("");
  });
});

describe("checkSizes", () => {
  it("refuses one file over the per-file cap and leaves the others", () => {
    const v = checkSizes([10, STAGE_PER_FILE_LIMIT + 1, 20]);
    expect(v.map((x) => x.reason === null)).toEqual([true, false, true]);
    expect(v[1]!.reason).toContain("per-file cap");
  });
  it("refuses the whole batch over the total cap, as the daemon does", () => {
    const half = Math.ceil(STAGE_TOTAL_LIMIT / 2) + 1;
    const v = checkSizes([half, half]);
    expect(v.every((x) => x.reason?.includes("exceeds cap"))).toBe(true);
  });
});

describe("checkSizes against a daemon's limits", () => {
  it("judges a file against the advertised per-file limit", () => {
    const v = checkSizes([500, 1500], { stagePerFileLimit: 1000, stageTotalLimit: 10_000, advertised: true });
    expect(v[0]!.reason).toBeNull();
    expect(v[1]!.reason).toContain("per-file cap 1000");
  });
  // The reported PDF: 1.4 MB, allowed by the 10 MB default, but a daemon
  // that advertises nothing closes the connection on anything over 1 MiB.
  it("refuses a file over an older daemon's 1 MiB message limit, naming the remedy", () => {
    const legacy = { stagePerFileLimit: 1024 * 1024, stageTotalLimit: STAGE_TOTAL_LIMIT, advertised: false };
    const [pdf] = checkSizes([1_400_000], legacy);
    expect(pdf!.reason).toMatch(/message limit; upgrade jaato-server/);
    expect(checkSizes([1_400_000])[0]!.reason).toBeNull();
  });
});

describe("attachmentFooter", () => {
  it("is empty with nothing staged, so the prompt goes as typed", () => {
    expect(attachmentFooter([])).toBe("");
  });
  it("names every staged path on one trailing line, quoting one with spaces", () => {
    expect(attachmentFooter(["a.pdf", "docs/my notes.md"])).toBe('\n\nAttached files, staged in the workspace: a.pdf, "docs/my notes.md"');
  });
});

describe("formatSize", () => {
  it("picks the unit", () => {
    expect(formatSize(312)).toBe("312 B");
    expect(formatSize(48 * 1024)).toBe("48 kB");
    expect(formatSize(1.25 * 1024 * 1024)).toBe("1.3 MB");
  });
});
