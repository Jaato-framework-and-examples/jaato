/**
 * The exit choice as a plate (``app/exitChoice.ts``): the TUI's
 * "Exit options:" / "Task in progress. What would you like to do?" and
 * its lettered answers, drawn like the permission plate so the two read
 * as the same kind of question.  The focused option is the one solid
 * button, Return sits apart at the right edge, every button shows its
 * key, Tab cycles focus, and the composer forwards typed keys.
 */
import { useEffect } from "react";
import type { ExitChoice } from "@/store/types";
import { useJaato } from "@/store/store";
import { Plate } from "@/components/layout/Plate";

export function ExitPrompt({ x, onAnswer }: { x: ExitChoice; onAnswer: (key: string) => void }) {
  const focus = useJaato((s) => s.focusExitChoice);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const inField = t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT");
      if (e.key === "Tab" && !inField) {
        e.preventDefault();
        const n = x.options.length || 1;
        focus((x.focus + (e.shiftKey ? -1 : 1) + n) % n);
      } else if (e.key === "Escape") {
        e.preventDefault();
        onAnswer("r");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [x, focus, onAnswer]);

  const button = (o: ExitChoice["options"][number], i: number) => (
    <button
      key={o.key}
      type="button"
      onClick={() => onAnswer(o.key)}
      title={o.description}
      className={`btn ${i === x.focus ? "btn-primary" : o.key === "e" ? "btn-danger" : ""}`}
      aria-current={i === x.focus ? "true" : undefined}
    >
      {o.label} <span className="key">{o.key}</span>
    </button>
  );
  const back = x.options.findIndex((o) => o.key === "r");
  return (
    <Plate edge="warning" className="my-3" role="group" aria-label="Exit options">
      <div className="px-3.5 py-2.5 flex items-center gap-3 border-b hairline">
        <span className="text-warning">⚠</span>
        <span className="kicker text-warning">{x.running ? "Task in progress" : "Exit options"}</span>
        <span className="text-[13px] text-text-muted">{x.running ? "What would you like to do?" : "What should become of the session?"}</span>
      </div>
      <ul className="px-3.5 py-2 m-0 list-none text-[13px] text-text-muted flex flex-col gap-0.5">
        {x.options.map((o) => (
          <li key={o.key}><span className="font-mono text-text">[{o.key}]</span> {o.label} <span>— {o.description}</span></li>
        ))}
      </ul>
      <div className="px-3.5 py-2.5 flex items-center gap-2 border-t hairline">
        {x.options.map((o, i) => (o.key === "r" ? null : button(o, i)))}
        <span className="flex-1" />
        {back >= 0 && button(x.options[back]!, back)}
      </div>
    </Plate>
  );
}
