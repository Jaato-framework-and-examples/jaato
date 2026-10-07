/**
 * ``parseWorkspaceCreatedEvent`` -- the shape ``createWorkspace`` reads a
 * ``workspace.created`` reply through, mirrored from the store's own
 * ``WORKSPACE_CREATED`` extraction (``store.ts``) so the two cannot
 * disagree about what the daemon sent.  This is what ``createWorkspace``
 * hands to a caller that needs to identify the new workspace precisely --
 * the GitHub auto-bind is the caller (``app/github.ts``'s
 * ``autoBindDefaultGitHubAccount``), and it needs the ABSOLUTE ``path``,
 * not the bare name, so a wrong extraction here would silently record a
 * binding ``secret.resolve`` can never find.
 */
import { describe, expect, it } from "vitest";
import { parseWorkspaceCreatedEvent } from "./connection";

describe("parseWorkspaceCreatedEvent", () => {
  it("reads name and path off the daemon's `workspace` row", () => {
    const ev = { type: "workspace.created", workspace: { name: "proj", path: "/srv/ws/proj", configured: false } };
    expect(parseWorkspaceCreatedEvent(ev)).toEqual({ name: "proj", configured: false, path: "/srv/ws/proj" });
  });

  it("falls back to top-level name/path for an older daemon with no `workspace` row", () => {
    const ev = { type: "workspace.created", name: "proj", path: "/srv/ws/proj" };
    expect(parseWorkspaceCreatedEvent(ev)).toEqual({ name: "proj", configured: false, path: "/srv/ws/proj" });
  });

  it("a reply naming nothing is not a row -- null, never an entry with an empty name", () => {
    expect(parseWorkspaceCreatedEvent({ type: "workspace.created" })).toBeNull();
    expect(parseWorkspaceCreatedEvent({ type: "workspace.created", name: "", path: "" })).toBeNull();
  });

  it("a daemon that sends no path answers null for it, never an empty string", () => {
    const ev = { type: "workspace.created", workspace: { name: "proj", configured: true } };
    expect(parseWorkspaceCreatedEvent(ev)).toEqual({ name: "proj", configured: true, path: null });
  });
});
