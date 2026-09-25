/**
 * The `permissions` command as a plate, opened from the status bar's
 * permission segment.
 *
 * That segment already reported the effective default and named the
 * command in its tooltip, so the one thing missing was a way to ACT on
 * what it says — the reading and the doing were in different places, and
 * only one of them was on screen.
 *
 * **What is here and what is not** is the whole design, and the split is
 * not arbitrary: this plate holds the verbs whose arguments are a CLOSED
 * set, so a click can express them exactly. `allow`, `deny` and `check`
 * take a tool NAME, which is an open set the composer already completes
 * from the daemon's own inventory; building a second, staler picker for
 * them here would be a worse answer than the one that exists. The plate
 * says so rather than leaving the gap unexplained.
 *
 * Every action goes through `submitInput`, the same path a typed command
 * takes — so a click and a keystroke are one mechanism, the transcript
 * records what was run, and the daemon's answer arrives where the user is
 * already reading. A button that spoke to the daemon directly would leave
 * a policy change with no record of who asked for it.
 */
import { useEffect, useRef, type RefObject } from "react";
import { submitInput } from "@/app/actions";
import { Plate, PlateHeader } from "@/components/layout/Plate";
import { useJaato } from "@/store/store";

/** `permissions default <policy>` — the closed set the daemon completes. */
const POLICIES: { value: string; label: string; help: string }[] = [
  { value: "ask", label: "Ask", help: "Prompt for each tool" },
  { value: "allow", label: "Allow", help: "Auto-approve all tools" },
  { value: "deny", label: "Deny", help: "Auto-deny all tools" },
];

export function PermissionsPlate({ onClose, anchor }: {
  onClose: () => void;
  /**
   * The control that opened this plate.
   *
   * Not decoration: the trigger is OUTSIDE the plate, so without this the
   * outside-click listener fires on its `mousedown`, closes, and the
   * trigger's own `click` — which arrives after — reopens.  The segment
   * would then be a button that cannot be clicked shut.  A test that
   * clicks with `fireEvent.click` alone cannot see this at all, because
   * that dispatches no `mousedown`.
   */
  anchor?: RefObject<HTMLElement | null>;
}) {
  const status = useJaato((s) => s.permissionStatus);
  const ref = useRef<HTMLDivElement>(null);

  // Escape closes, and a click outside does too -- this is a popover over
  // the page, not a question blocking the turn, so dismissing it must
  // never be a decision.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); onClose(); }
    };
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (ref.current?.contains(t)) return;
      if (anchor?.current?.contains(t)) return;   // the toggle closes itself
      onClose();
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("mousedown", onDown);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("mousedown", onDown);
    };
  }, [onClose, anchor]);

  const run = (command: string) => {
    onClose();
    void submitInput(command, false).catch(() => undefined);
  };

  const suspended = status?.suspensionScope != null;
  const current = status?.effectiveDefault ?? "";

  return (
    <div ref={ref} className="absolute bottom-[26px] left-0 z-30 w-[22rem] max-w-[calc(100vw-2rem)]">
      <Plate edge="steel" className="bg-surface shadow-lg" role="dialog" aria-label="Session permissions">
        <PlateHeader label="Permissions" value={suspended ? `suspended (${status?.suspensionScope})` : current} />

        <div className="px-3.5 py-2.5 border-b hairline">
          <p className="m-0 mb-1.5 text-[11px] text-text-muted">Default for tools with no rule of their own</p>
          <div className="flex gap-1.5">
            {POLICIES.map((p) => (
              <button
                key={p.value}
                type="button"
                title={p.help}
                aria-pressed={!suspended && current === p.value}
                onClick={() => run(`permissions default ${p.value}`)}
                className={`btn flex-1 ${!suspended && current === p.value ? "btn-primary" : ""}`}
              >
                {p.label}
              </button>
            ))}
          </div>
          {suspended && (
            <p className="m-0 mt-1.5 text-[11px] text-warning">
              Prompting is suspended, so the default is not being consulted.
            </p>
          )}
          {/* jaato/#1304 phase 3: `auto_allow_housekeeping` on
              PermissionStatusEvent, via the existing session.get_permission_status
              verb -- shown only when the daemon reports it TRUE. `false` and
              `null` (older daemon, or the enforcer could not be reached) say
              nothing extra here; the default line above is already accurate
              for both. */}
          {status?.autoAllowHousekeeping === true && (
            <p className="m-0 mt-1.5 text-[11px] text-text-muted">
              Read-only, low-risk tools are auto-approved regardless of the default above.
            </p>
          )}
        </div>

        <div className="px-3.5 py-2.5 border-b hairline">
          <p className="m-0 mb-1.5 text-[11px] text-text-muted">Stop being asked</p>
          <div className="flex gap-1.5">
            {suspended ? (
              <button type="button" className="btn flex-1" onClick={() => run("permissions resume")}>
                Resume prompting
              </button>
            ) : (
              <>
                <button type="button" className="btn flex-1" title="Suspend prompting for this turn only"
                        onClick={() => run("permissions suspend --turn")}>
                  This turn
                </button>
                <button type="button" className="btn flex-1" title="Suspend prompting until the session goes idle"
                        onClick={() => run("permissions suspend")}>
                  Until idle
                </button>
              </>
            )}
          </div>
        </div>

        <div className="px-3.5 py-2.5 border-b hairline flex gap-1.5">
          <button type="button" className="btn flex-1" title="Print the effective policy, whitelist and blacklist"
                  onClick={() => run("permissions show")}>
            Show policy
          </button>
          <button type="button" className="btn btn-danger flex-1" title="Drop every change made to permissions this session"
                  onClick={() => run("permissions clear")}>
            Reset session
          </button>
        </div>

        {/* The open-set verbs. Naming them here is what stops the plate
            reading as the whole command; the composer completes tool
            names from the daemon's own inventory, which nothing local can. */}
        <p className="px-3.5 py-2 m-0 text-[11px] text-text-muted">
          One tool at a time: type <span className="font-mono text-text">permissions allow</span>,{" "}
          <span className="font-mono text-text">deny</span> or <span className="font-mono text-text">check</span>{" "}
          in the composer — it completes tool names.
        </p>
      </Plate>
    </div>
  );
}
