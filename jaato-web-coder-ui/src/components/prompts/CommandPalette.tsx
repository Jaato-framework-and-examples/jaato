/**
 * The Ctrl/⌘+K command palette (#1304 §6), and the leader §7 names.
 *
 * One gesture serves both: opening it is striking the leader, and what
 * happens next depends on the very first keystroke.  While the search box
 * is still EMPTY, a bare P / B / F / T / [ / ] / 1-9 / O is a quick action
 * -- ``app/leaderKeys.ts`` resolves it, this component applies it and
 * closes -- so "Ctrl+K then P" opens Plan in one motion with no visible
 * list in between.  Anything else -- a letter the leader does not answer
 * for, or any key once the box already holds text -- is an ordinary
 * search character, filtering ``command.list`` (the same list the
 * composer's own proposals complete from; there is no second source of
 * command names).  The gate is "query still empty", not "first
 * keystroke": backspacing to empty and pressing ``o`` again re-arms the
 * quick action rather than search text landing where a shortcut fired
 * once already.
 *
 * Selecting an item -- Enter, or a click -- runs it exactly as if it had
 * been typed into the composer and submitted (``submitInput``), which is
 * also where ``help`` now lands instead of the ~500-line dump the
 * transcript used to get.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { submitInput } from "@/app/actions";
import { resolveLeaderKey, LEADER_KEY_HINT, type LeaderEffect } from "@/app/leaderKeys";
import { filterGroups, flattenGroups, groupCommands } from "@/protocol/palette";
import type { CommandSpec } from "@/protocol/commands";
import { Plate } from "@/components/layout/Plate";
import { runningToolCallIds, useJaato } from "@/store/store";

export function CommandPalette() {
  const commands = useJaato((s) => s.commands);
  const agentOrder = useJaato((s) => s.agentOrder);
  const selectedAgentId = useJaato((s) => s.selectedAgentId);
  const blocks = useJaato((s) => s.blocks);
  const popupCallId = useJaato((s) => s.ui.popupCallId);
  const setPaletteOpen = useJaato((s) => s.setPaletteOpen);
  const toggleUi = useJaato((s) => s.toggleUi);
  const setToolsExpanded = useJaato((s) => s.setToolsExpanded);
  const showTools = useJaato((s) => s.ui.showTools);
  const selectAgent = useJaato((s) => s.selectAgent);
  const setPopup = useJaato((s) => s.setPopup);

  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  const groups = useMemo(() => filterGroups(groupCommands(commands), query), [commands, query]);
  const flat = useMemo(() => flattenGroups(groups), [groups]);
  useEffect(() => { setActive(0); }, [query, commands]);

  const close = () => setPaletteOpen(false);

  const run = (c: CommandSpec) => {
    close();
    void submitInput(c.name, false);
  };

  const apply = (effect: LeaderEffect) => {
    switch (effect.kind) {
      case "toggleUi": toggleUi(effect.key); break;
      case "toggleTools": setToolsExpanded(!showTools); break;
      case "selectAgent": selectAgent(effect.id); break;
      case "cyclePopup": setPopup(effect.callId); break;
    }
    close();
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") { e.preventDefault(); close(); return; }
    if (e.key === "ArrowDown") { e.preventDefault(); setActive((i) => (flat.length ? (i + 1) % flat.length : 0)); return; }
    if (e.key === "ArrowUp") { e.preventDefault(); setActive((i) => (flat.length ? (i - 1 + flat.length) % flat.length : 0)); return; }
    if (e.key === "Enter") { e.preventDefault(); const c = flat[active]; if (c) run(c); return; }
    // The leader's own payload: only while nothing has been typed yet, and
    // only a bare key (no modifier -- Ctrl/⌘+<letter> is somebody else's
    // shortcut, not this one, and must reach the browser or another
    // handler untouched).
    if (!query && !e.ctrlKey && !e.metaKey && !e.altKey && e.key.length === 1) {
      const effect = resolveLeaderKey(e.key, {
        agentOrder, selectedAgentId, popupCallId,
        runningToolCallIds: runningToolCallIds({ blocks }, selectedAgentId),
      });
      if (effect) { e.preventDefault(); apply(effect); }
    }
  };

  return (
    <div
      className="palette-backdrop fixed inset-0 z-50 flex items-start justify-center pt-[12vh] px-4 bg-[color-mix(in_srgb,var(--c-text)_35%,transparent)]"
      onMouseDown={(e) => { if (e.target === e.currentTarget) close(); }}
      role="presentation"
    >
      <Plate
        edge="steel"
        className="palette-panel w-[min(640px,100%)] max-h-[70vh] flex flex-col bg-surface shadow-lg"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2.5 px-3.5 py-2.5 border-b hairline">
          <span className="font-mono text-steel" aria-hidden="true">›</span>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Search commands, or press a letter (P B F T [ ] 1-9 O)"
            aria-label="Search commands"
            className="flex-1 bg-transparent outline-none font-mono text-[13.5px]"
          />
          <button type="button" onClick={close} aria-label="Close" className="text-text-muted hover:text-text min-h-[44px] min-w-[44px]">✕</button>
        </div>
        <div role="listbox" aria-label="Commands" className="overflow-y-auto flex-1">
          {flat.length === 0 && <p className="px-3.5 py-3 m-0 text-[13px] text-text-muted">No commands match.</p>}
          {groups.map((g) => (
            <div key={g.area}>
              <div className="kicker kicker-muted px-3.5 pt-2 pb-1 text-[10px]">{g.area}</div>
              {g.items.map((c) => {
                const i = flat.indexOf(c);
                return (
                  <button
                    key={c.name}
                    type="button"
                    role="option"
                    aria-selected={i === active}
                    onMouseEnter={() => setActive(i)}
                    onClick={() => run(c)}
                    className={`w-full flex items-baseline gap-3 px-3.5 min-h-[44px] py-1.5 text-left ${i === active ? "bg-surface text-steel shadow-[inset_2px_0_0_var(--c-steel)]" : "hover:bg-tint/70"}`}
                  >
                    <span className="font-mono text-[13px]">{c.name}</span>
                    {c.description && <span className="text-text-muted text-xs truncate">{c.description}</span>}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
        <div className="px-3.5 py-2 text-[11px] text-text-muted border-t hairline flex flex-wrap gap-3">
          <span><kbd>↑↓</kbd> move</span><span><kbd>Enter</kbd> run</span><span><kbd>Esc</kbd> close</span>
          <span className="flex-1" />
          <span>leader: {LEADER_KEY_HINT}</span>
        </div>
      </Plate>
    </div>
  );
}
