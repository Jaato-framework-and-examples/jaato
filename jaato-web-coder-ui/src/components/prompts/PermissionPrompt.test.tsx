/**
 * The regrouped permission card (jaato/#1304 §2, the final rollout item
 * from the closed issue #1304). Three claims are asserted here rather
 * than described:
 *
 *  - **Allow / Deny are unchanged.** Same markup, same accessible name
 *    ("yes y" / "no n") as the pre-regroup card -- an existing caller
 *    that types the key or clicks the button by that name still works.
 *  - **Nothing the daemon sent is ever dropped.** t/i/a/all/never move
 *    into the "Allow for…" dropdown, c/yc become the note field, once/e
 *    hide behind "More…", and an option this vocabulary has no opinion
 *    about still gets its own button.
 *  - **The dropdown's outside-click exemption**, mirroring
 *    ``PermissionsPlate.test.tsx``: the trigger sits outside the
 *    popover, so a real browser click (mousedown then click) must not
 *    close what it just opened.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { PermissionPrompt } from "./PermissionPrompt";
import type { PendingPermission } from "@/store/types";

/** See ``PermissionsPlate.test.tsx``'s own docstring for why this matters. */
function realClick(el: Element): void {
  fireEvent.mouseDown(el);
  fireEvent.click(el);
}

afterEach(cleanup);

const FULL_OPTIONS = [
  { key: "y", label: "yes" },
  { key: "n", label: "no" },
  { key: "a", label: "always" },
  { key: "t", label: "turn" },
  { key: "i", label: "idle" },
  { key: "once", label: "once" },
  { key: "never", label: "never" },
  { key: "all", label: "all" },
  { key: "c", label: "deny-comment" },
  { key: "yc", label: "allow-comment" },
];

function permission(overrides: Partial<PendingPermission> = {}): PendingPermission {
  return {
    requestId: "r1",
    agentId: "main",
    toolName: "writeNewFile",
    toolArgs: { path: "a.py" },
    options: FULL_OPTIONS,
    promptLines: [],
    focus: 0,
    inputMode: false,
    ...overrides,
  };
}

describe("PermissionPrompt: primary buttons are unchanged", () => {
  it("renders Allow (yes y) and Deny (no n) with the pre-regroup accessible name", () => {
    render(<PermissionPrompt p={permission()} onRespond={() => undefined} />);
    expect(screen.getByRole("button", { name: /^yes y$/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^no n$/ })).toBeInTheDocument();
  });

  it("clicking Allow sends the plain 'y' key when no note is open", () => {
    const onRespond = vi.fn();
    render(<PermissionPrompt p={permission()} onRespond={onRespond} />);
    fireEvent.click(screen.getByRole("button", { name: /^yes y$/ }));
    expect(onRespond).toHaveBeenCalledWith("y");
  });

  it("a minimal y/n-only option set degrades to just the two primary buttons", () => {
    render(<PermissionPrompt p={permission({ options: [{ key: "y", label: "yes" }, { key: "n", label: "no" }] })} onRespond={() => undefined} />);
    expect(screen.getByRole("button", { name: /^yes y$/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^no n$/ })).toBeInTheDocument();
    expect(screen.queryByText(/Allow for/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Add a note/)).not.toBeInTheDocument();
    expect(screen.queryByText(/More…/)).not.toBeInTheDocument();
  });
});

describe("PermissionPrompt: the 'Allow for…' dropdown", () => {
  it("groups turn/idle/always/all and the destructive never, none as top-level buttons", () => {
    render(<PermissionPrompt p={permission()} onRespond={() => undefined} />);
    for (const label of [/^turn t$/, /^idle i$/, /^always a$/, /^all all$/, /^never never$/]) {
      expect(screen.queryByRole("button", { name: label })).not.toBeInTheDocument();
    }
    expect(screen.getByRole("button", { name: /Allow for/ })).toBeInTheDocument();
  });

  it("opens on click and stays open (the outside-click listener exempts its own trigger)", () => {
    render(<PermissionPrompt p={permission()} onRespond={() => undefined} />);
    const trigger = screen.getByRole("button", { name: /Allow for/ });
    realClick(trigger);
    expect(screen.getByRole("menu", { name: /Allow for a scope/ })).toBeInTheDocument();
  });

  it("picking a duration answers with that key and closes", () => {
    const onRespond = vi.fn();
    render(<PermissionPrompt p={permission()} onRespond={onRespond} />);
    realClick(screen.getByRole("button", { name: /Allow for/ }));
    fireEvent.click(screen.getByRole("menuitem", { name: /^turn t$/ }));
    expect(onRespond).toHaveBeenCalledWith("t");
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  it("the destructive 'never' lives in the SAME dropdown, styled apart", () => {
    const onRespond = vi.fn();
    render(<PermissionPrompt p={permission()} onRespond={onRespond} />);
    realClick(screen.getByRole("button", { name: /Allow for/ }));
    const never = screen.getByRole("menuitem", { name: /^never never$/ });
    expect(never.className).toContain("btn-danger");
    fireEvent.click(never);
    expect(onRespond).toHaveBeenCalledWith("never");
  });

  it("Escape closes the dropdown without answering", () => {
    const onRespond = vi.fn();
    render(<PermissionPrompt p={permission()} onRespond={onRespond} />);
    realClick(screen.getByRole("button", { name: /Allow for/ }));
    expect(screen.getByRole("menu")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    expect(onRespond).not.toHaveBeenCalled();
  });
});

describe("PermissionPrompt: hidden options (once, edit)", () => {
  it("once is not a top-level button until 'More…' is opened", () => {
    render(<PermissionPrompt p={permission()} onRespond={() => undefined} />);
    expect(screen.queryByRole("button", { name: /^once once$/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("More…"));
    expect(screen.getByRole("button", { name: /^once once$/ })).toBeInTheDocument();
  });
});

describe("PermissionPrompt: the note field (c / yc)", () => {
  it("Allow with a note sends yc:<text>, not the plain y", () => {
    const onRespond = vi.fn();
    render(<PermissionPrompt p={permission()} onRespond={onRespond} />);
    fireEvent.click(screen.getByText("+ Add a note"));
    fireEvent.change(screen.getByPlaceholderText(/the model reads this back/), { target: { value: "looks fine, but watch the perms" } });
    fireEvent.click(screen.getByRole("button", { name: /^yes y$/ }));
    expect(onRespond).toHaveBeenCalledWith("yc:looks fine, but watch the perms");
  });

  it("Deny with a note sends c:<text>", () => {
    const onRespond = vi.fn();
    render(<PermissionPrompt p={permission()} onRespond={onRespond} />);
    fireEvent.click(screen.getByText("+ Add a note"));
    fireEvent.change(screen.getByPlaceholderText(/the model reads this back/), { target: { value: "no, wrong file" } });
    fireEvent.click(screen.getByRole("button", { name: /^no n$/ }));
    expect(onRespond).toHaveBeenCalledWith("c:no, wrong file");
  });

  it("an empty note falls back to the plain key -- a blank note is not sent as feedback", () => {
    const onRespond = vi.fn();
    render(<PermissionPrompt p={permission()} onRespond={onRespond} />);
    fireEvent.click(screen.getByText("+ Add a note"));
    fireEvent.click(screen.getByRole("button", { name: /^yes y$/ }));
    expect(onRespond).toHaveBeenCalledWith("y");
  });
});

describe("PermissionPrompt: nothing the daemon sent is ever dropped", () => {
  it("an option key this vocabulary has no opinion about still gets its own button", () => {
    const onRespond = vi.fn();
    render(<PermissionPrompt p={permission({
      options: [{ key: "y", label: "yes" }, { key: "n", label: "no" }, { key: "x", label: "future option" }],
    })} onRespond={onRespond} />);
    const btn = screen.getByRole("button", { name: /^future option x$/ });
    fireEvent.click(btn);
    expect(onRespond).toHaveBeenCalledWith("x");
  });
});

describe("PermissionPrompt: risk tag and plain question", () => {
  it("shows the daemon's tool_class as a risk tag, and the write question", () => {
    render(<PermissionPrompt p={permission({ toolClass: "write" })} onRespond={() => undefined} />);
    expect(screen.getByText("Changes a file")).toBeInTheDocument();
    expect(screen.getByText("Let the agent change this file?")).toBeInTheDocument();
  });

  it("falls back to the client table when the daemon reported no tool_class", () => {
    render(<PermissionPrompt p={permission({ toolName: "createPlan", toolClass: null })} onRespond={() => undefined} />);
    expect(screen.getByText("Routine")).toBeInTheDocument();
    expect(screen.getByText("Let the agent do this bookkeeping step?")).toBeInTheDocument();
  });

  it("shows the command as the effect for an exec-class ask", () => {
    render(<PermissionPrompt p={permission({
      toolName: "cli_based_tool", toolClass: "exec", toolArgs: { command: "rm -rf /tmp/x" },
    })} onRespond={() => undefined} />);
    expect(screen.getByText("rm -rf /tmp/x")).toBeInTheDocument();
  });
});
