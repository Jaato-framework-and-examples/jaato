import { strict as assert } from "node:assert";
import { mkdtempSync, readFileSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import { FileNoteStore, MAX_NOTE_CHARS, MAX_NOTES_PER_OWNER, NoteError, validateNote, validateSessionId } from "../src/notes.js";

const KEY = "k".repeat(40);
const fresh = () => join(mkdtempSync(join(tmpdir(), "jwcs-note-")), "sub", "notes.json");

describe("note store", () => {
  test("put → list returns the text; the file holds none of it", () => {
    const path = fresh();
    const store = new FileNoteStore(path, KEY, () => new Date("2026-09-19T10:00:00Z"));
    const n = store.put("sub-alice", "20260919_084517", "waiting on the grace period answer, then re-run e2e");
    assert.deepEqual(n, { sessionId: "20260919_084517", text: "waiting on the grace period answer, then re-run e2e", updatedAt: "2026-09-19T10:00:00.000Z" });
    assert.deepEqual(store.list("sub-alice"), [n]);
    const onDisk = readFileSync(path, "utf8");
    assert.ok(!onDisk.includes("grace period"), "note text must not be in the file");
    assert.equal(statSync(path).mode & 0o777, 0o600);
  });

  test("owners are isolated: another sub sees nothing and cannot delete", () => {
    const store = new FileNoteStore(fresh(), KEY);
    store.put("sub-alice", "s1", "mine");
    assert.deepEqual(store.list("sub-bob"), []);
    assert.equal(store.remove("sub-bob", "s1"), false);
    assert.equal(store.list("sub-alice")[0]!.text, "mine");
  });

  test("two owners may note the SAME session, and neither sees the other's", () => {
    // A note on a session you did not create is YOUR note about their
    // session -- the documented consequence of keying on the OIDC sub.
    const store = new FileNoteStore(fresh(), KEY);
    store.put("sub-alice", "shared", "ask Bob about the schema");
    store.put("sub-bob", "shared", "Alice is waiting on me");
    assert.equal(store.list("sub-alice")[0]!.text, "ask Bob about the schema");
    assert.equal(store.list("sub-bob")[0]!.text, "Alice is waiting on me");
  });

  test("put on an existing session replaces rather than appending", () => {
    const store = new FileNoteStore(fresh(), KEY);
    store.put("s", "s1", "first");
    store.put("s", "s1", "second");
    assert.deepEqual(store.list("s").map((n) => n.text), ["second"]);
  });

  test("an emptied note is a forgotten note, not an empty row", () => {
    // The editor debounce-saves whatever is in the box, so clearing one has
    // to reach the store as a removal or it keeps counting against the cap.
    const store = new FileNoteStore(fresh(), KEY);
    store.put("s", "s1", "something");
    assert.equal(store.put("s", "s1", "   \n  "), null);
    assert.deepEqual(store.list("s"), []);
  });

  test("survives a reload: a second store over the same file decrypts what the first wrote", () => {
    const path = fresh();
    new FileNoteStore(path, KEY).put("s", "s1", "carried over");
    assert.equal(new FileNoteStore(path, KEY).list("s")[0]!.text, "carried over");
  });

  test("re-attributing a note in the file makes it undecryptable, not somebody else's", async () => {
    // The AAD binds owner and session id, which is what that buys.
    const path = fresh();
    new FileNoteStore(path, KEY).put("sub-alice", "s1", "secret plan");
    const raw = readFileSync(path, "utf8").replace(/"owner": "sub-alice"/, '"owner": "sub-mallory"');
    const { writeFileSync } = await import("node:fs");
    writeFileSync(path, raw, { mode: 0o600 });
    const store = new FileNoteStore(path, KEY);
    assert.throws(() => store.list("sub-mallory"));
  });

  test("newlines and tabs survive (checkboxes are the point); other control characters do not", () => {
    assert.equal(validateNote("- [ ] one\n- [x] two"), "- [ ] one\n- [x] two");
    assert.equal(validateNote("a\r\nb"), "a\nb", "CRLF is normalised");
    assert.throws(() => validateNote("bad\u0007bell"), NoteError);
  });

  test("caps: note length and notes per owner", () => {
    const store = new FileNoteStore(fresh(), KEY);
    assert.throws(() => store.put("s", "s1", "x".repeat(MAX_NOTE_CHARS + 1)), NoteError);
    for (let i = 0; i < MAX_NOTES_PER_OWNER; i++) store.put("s", `s${i}`, "n");
    assert.throws(() => store.put("s", "one-too-many", "n"), (e: unknown) => e instanceof NoteError && e.status === 409);
    // At the cap, EDITING an existing note still works -- the cap bounds the
    // file, and refusing an edit would strand somebody at 200 unreadable rows.
    assert.equal(store.put("s", "s0", "edited")!.text, "edited");
  });

  test("the session id is bounded, not modelled: shape only, and traversal is refused", () => {
    assert.equal(validateSessionId("20260919_084517"), "20260919_084517");
    assert.equal(validateSessionId("cascade:stage-2"), "cascade:stage-2");
    for (const bad of ["", "../etc/passwd", "a/b", "x".repeat(129), "-leading"]) {
      assert.throws(() => validateSessionId(bad), NoteError, `should refuse ${JSON.stringify(bad)}`);
    }
  });
});
