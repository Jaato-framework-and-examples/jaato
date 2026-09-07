/**
 * The prompt box, with jaato's command model.
 *
 * jaato commands are bare words (``model gpt-4o``, ``permissions status``),
 * not ``/slash`` verbs.  So the composer must answer "did the user mean
 * the command or the word?" — and it answers it the way the TUI does:
 *
 *  • While the first (and second) word is typed, a popup proposes the
 *    matching commands.  ``Tab`` / ``↑↓`` pick one; ``Tab`` accepts it.
 *  • ``Enter`` submits the line as typed.  If its first word names a
 *    command, the line runs as that command.
 *  • ``Esc`` dismisses the popup and marks the line **verbatim**: from
 *    then on, this line is a message even though it starts with a
 *    command word (``model this is broken`` reaches the model as text).
 *    Verbatim lasts until the first word changes or the box is cleared;
 *    pressing ``Tab`` again re-opens the proposal and cancels it.
 *  • A hint under the box always says what ``Enter`` will do, so the
 *    routing is never a surprise.
 *
 * Sigils are passed through untouched: ``@path``, ``@@path``, ``%prompt``
 * and ``/command`` are expanded server-side; their presence disables
 * command proposals, as in the TUI.
 *
 * Pending permission / clarification prompts take over ``Enter``: the
 * typed text becomes the answer (``y``, ``a``, an option key, a free-
 * text reply), exactly like typing into the TUI while a prompt is up.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { commandCompletions, wouldRouteAsCommand, type CommandSpec, type Completion } from "@/protocol/commands";

export interface ComposerProps {
  commands: CommandSpec[];
  disabled?: boolean;
  /** Set when a prompt (permission / clarification / reference) is waiting for input. */
  captureMode?: { kind: "permission" | "clarification" | "reference"; placeholder: string; suggestions?: string[] } | null;
  history: string[];
  onSubmit: (text: string, verbatim: boolean) => void;
  onEscapeEmpty?: () => void;
}

function firstWord(text: string): string {
  return (text.trimStart().split(/\s+/)[0] ?? "").toLowerCase();
}

export function Composer({ commands, disabled, captureMode, history, onSubmit, onEscapeEmpty }: ComposerProps) {
  const [text, setText] = useState("");
  const [verbatim, setVerbatim] = useState(false);
  const [dismissed, setDismissed] = useState(false); // popup closed for this first word
  const [selected, setSelected] = useState(0);
  const [histIdx, setHistIdx] = useState<number | null>(null);
  const [caret, setCaret] = useState(0);
  const ref = useRef<HTMLTextAreaElement>(null);
  const verbatimWord = useRef<string>("");

  const before = text.slice(0, caret);
  const completions: Completion[] = useMemo(() => {
    if (captureMode || verbatim || dismissed) return [];
    if (!text.trim() && !before.length) return [];
    return commandCompletions(before, commands).slice(0, 12);
  }, [before, text, commands, captureMode, verbatim, dismissed]);
  const popupOpen = completions.length > 0;

  useEffect(() => { setSelected(0); }, [completions.length, before]);

  // Verbatim / dismissed are per first-word: changing it re-arms proposals.
  useEffect(() => {
    const w = firstWord(text);
    if (!text.trim()) { setVerbatim(false); setDismissed(false); verbatimWord.current = ""; return; }
    if ((verbatim || dismissed) && w !== verbatimWord.current) { setVerbatim(false); setDismissed(false); }
  }, [text, verbatim, dismissed]);

  const routed = !captureMode && !verbatim ? wouldRouteAsCommand(text, commands) : null;
  const routedIfNotVerbatim = !captureMode && verbatim ? wouldRouteAsCommand(text, commands) : null;

  const resize = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = Math.min(el.scrollHeight, 240) + "px";
  }, []);
  useEffect(resize, [text, resize]);

  const accept = (c: Completion) => {
    const rest = text.slice(caret).replace(/^\S*/, "");
    const next = c.insert + " " + rest.replace(/^\s+/, "");
    setText(next);
    const pos = c.insert.length + 1;
    requestAnimationFrame(() => { ref.current?.setSelectionRange(pos, pos); setCaret(pos); });
  };

  const submit = () => {
    const t = text.replace(/\s+$/, "");
    if (!t && !captureMode) return;
    onSubmit(t, verbatim);
    setText("");
    setVerbatim(false);
    setDismissed(false);
    setHistIdx(null);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Tab" && !captureMode) {
      e.preventDefault();
      if (popupOpen) {
        if (e.shiftKey) setSelected((s) => (s - 1 + completions.length) % completions.length);
        else if (completions.length === 1 || selected >= 0) accept(completions[selected] ?? completions[0]!);
        return;
      }
      // Re-arm proposals (cancels verbatim / dismissed).
      setVerbatim(false);
      setDismissed(false);
      verbatimWord.current = "";
      return;
    }
    if (popupOpen && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      e.preventDefault();
      setSelected((s) => (e.key === "ArrowDown" ? (s + 1) % completions.length : (s - 1 + completions.length) % completions.length));
      return;
    }
    if (e.key === "Escape") {
      if (popupOpen) {
        e.preventDefault();
        // The caveat: dismissing the proposal means "this word, verbatim".
        setVerbatim(true);
        setDismissed(true);
        verbatimWord.current = firstWord(text);
        return;
      }
      if (!text) { onEscapeEmpty?.(); return; }
      if (routed) {
        // No popup (e.g. pasted line) but Enter would run a command: Esc still means verbatim.
        e.preventDefault();
        setVerbatim(true);
        setDismissed(true);
        verbatimWord.current = firstWord(text);
        return;
      }
      return;
    }
    if (e.key === "Enter" && !e.shiftKey && !e.altKey && !e.ctrlKey && !e.metaKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
      return;
    }
    if (e.key === "ArrowUp" && !popupOpen && history.length && (text === "" || histIdx !== null) && caretOnFirstLine()) {
      e.preventDefault();
      const idx = histIdx === null ? history.length - 1 : Math.max(0, histIdx - 1);
      setHistIdx(idx);
      setText(history[idx] ?? "");
      return;
    }
    if (e.key === "ArrowDown" && !popupOpen && histIdx !== null) {
      e.preventDefault();
      const idx = histIdx + 1;
      if (idx >= history.length) { setHistIdx(null); setText(""); }
      else { setHistIdx(idx); setText(history[idx] ?? ""); }
    }
  };

  const caretOnFirstLine = () => {
    const el = ref.current;
    if (!el) return true;
    return !el.value.slice(0, el.selectionStart).includes("\n");
  };

  const placeholder = captureMode
    ? captureMode.placeholder
    : "Message the agent — or type a command (help, tools, model …)";

  return (
    <div className="relative">
      {popupOpen && (
        <div role="listbox" aria-label="Command proposals" className="absolute bottom-full mb-1 left-0 w-[min(34rem,100%)] rounded-md border hairline bg-bg shadow-lg text-[13px] overflow-hidden z-30">
          {completions.map((c, i) => (
            <div
              key={c.insert}
              role="option"
              aria-selected={i === selected}
              onMouseDown={(e) => { e.preventDefault(); accept(c); }}
              className={`flex items-baseline gap-3 px-2 py-1 cursor-pointer ${i === selected ? "bg-surface text-primary" : "hover:bg-surface/60"}`}
            >
              <span className="font-mono">{c.label}</span>
              {c.description && <span className="text-text-muted text-xs truncate">{c.description}</span>}
            </div>
          ))}
          <div className="px-2 py-0.5 text-[11px] text-text-muted border-t hairline flex gap-3">
            <span><kbd>Tab</kbd> complete</span><span><kbd>Enter</kbd> run</span><span><kbd>Esc</kbd> not a command — send as text</span>
          </div>
        </div>
      )}
      <div className={`flex items-end gap-2 rounded-lg border px-3 py-2 surface-1 ${captureMode ? "border-warning/60" : "hairline"} focus-within:border-primary/60`}>
        <span className="font-mono text-text-muted select-none pb-[3px]">{captureMode ? "?" : "›"}</span>
        <textarea
          ref={ref}
          value={text}
          rows={1}
          disabled={disabled}
          placeholder={placeholder}
          aria-label="Prompt"
          spellCheck={!routed}
          onChange={(e) => { setText(e.target.value); setCaret(e.target.selectionStart ?? e.target.value.length); }}
          onSelect={(e) => setCaret((e.target as HTMLTextAreaElement).selectionStart ?? 0)}
          onKeyDown={onKeyDown}
          className="flex-1 resize-none bg-transparent outline-none font-mono text-[13.5px] leading-5 placeholder:text-text-muted/70 max-h-60"
        />
        <button type="button" onClick={submit} disabled={disabled || (!text.trim() && !captureMode)} className="text-xs px-2 py-1 rounded bg-surface text-text-muted hover:text-text disabled:opacity-40" aria-label="Send">
          ⏎
        </button>
      </div>
      <div className="h-5 px-1 text-[11px] text-text-muted flex items-center gap-3">
        {captureMode && captureMode.suggestions?.length ? (
          <span>Answer with {captureMode.suggestions.map((s) => <kbd key={s} className="mx-0.5">{s}</kbd>)} or type a reply</span>
        ) : routed ? (
          <span><kbd>Enter</kbd> runs command <span className="font-mono text-accent">{routed}</span> · <kbd>Esc</kbd> to send as text instead</span>
        ) : routedIfNotVerbatim ? (
          <span>Sending as text (not the <span className="font-mono">{routedIfNotVerbatim}</span> command) · <kbd>Tab</kbd> to make it a command</span>
        ) : (
          <span className="opacity-70"><kbd>Enter</kbd> send · <kbd>Shift</kbd>+<kbd>Enter</kbd> newline · <kbd>@file</kbd> <kbd>%prompt</kbd> references</span>
        )}
      </div>
    </div>
  );
}
