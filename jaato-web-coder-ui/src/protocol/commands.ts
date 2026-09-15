/**
 * Command routing for the composer — a faithful port of the TUI's
 * ``client_commands.parse_user_input`` plus the completion rules of
 * ``file_completer.CommandCompleter``.
 *
 * jaato does NOT use a ``/`` prefix for its built-in commands.  A user
 * types ``model gpt-4o`` or ``permissions status`` as bare words; the
 * client recognises the first word and routes the line as a command
 * instead of a prompt.  The completion popup is the disambiguation
 * surface: while the first word is being typed it proposes matching
 * commands, and DISMISSING it (Escape) is how a user says "no, I mean
 * this word verbatim".  That intent is carried here as the ``verbatim``
 * flag, which forces {@link parseUserInput} to route the line as a
 * message even when the first word names a command.
 *
 * Three sigils keep their TUI meaning and are never commands:
 *   ``@path``  file reference   ``@@path`` sandbox reference
 *   ``%name``  prompt-library reference   ``/name`` workspace slash command
 * (``/name`` is the one prefixed form, and it is user-authored under
 * ``.jaato/commands/`` — the server expands it, so it travels as message
 * text.)
 */

export type CommandAction =
  | "exit"
  | "stop"
  | "clear"
  | "help"
  | "context"
  | "history"
  | "server"
  | "message";

export interface ParsedCommand {
  action: CommandAction;
  /** Wire command name for ``action === "server"`` (e.g. ``tools.list``). */
  command?: string;
  /** argv-style arguments for ``action === "server"``. */
  args?: string[];
  /** Message text for ``action === "message"``. */
  text?: string;
}

/** ``{name, description}`` as carried by ``CommandListEvent.commands``. */
export interface CommandSpec {
  name: string;
  description?: string;
}

/**
 * Client-only commands, handled locally and never forwarded.  Mirrors
 * ``client_commands.CLIENT_COMMANDS``; ``keybindings``/``export`` are
 * TUI-specific and intentionally absent.
 */
export const CLIENT_COMMANDS: CommandSpec[] = [
  { name: "help", description: "Show available commands" },
  { name: "history", description: "Show conversation history" },
  { name: "exit", description: "Detach from the session" },
  { name: "quit", description: "Detach from the session" },
  { name: "stop", description: "Stop current model generation" },
  { name: "clear", description: "Clear output display" },
  { name: "context", description: "Show context window usage" },
  { name: "reset", description: "Clear conversation history" },
  { name: "theme", description: "Switch the UI theme (theme <name>)" },
];

const CLIENT_COMMAND_NAMES = new Set(CLIENT_COMMANDS.map((c) => c.name));

/**
 * First words that always route to the server, even before the daemon
 * has sent its command list.  Mirrors ``SERVER_COMMAND_PREFIXES``.
 */
export const SERVER_COMMAND_PREFIXES = new Set<string>([
  "tools", "session", "permissions", "model", "mcp", "save", "resume",
  "memory", "lsp", "todo", "waypoint", "background", "prompt-library",
  "clarification", "multimodal", "notebook", "references", "sandbox",
  "workspace", "reliability",
]);

/** Commands completed before the server list arrives (TUI ``DEFAULT_COMMANDS`` subset). */
export const DEFAULT_COMMANDS: CommandSpec[] = [
  { name: "tools", description: "Manage tools available to the model" },
  { name: "tools list", description: "List all tools with enabled/disabled status" },
  { name: "tools enable", description: "Enable a tool (tools enable <name>|all)" },
  { name: "tools disable", description: "Disable a tool (tools disable <name>|all)" },
  { name: "model", description: "Switch to a different model" },
  { name: "session", description: "Manage sessions" },
  { name: "session list", description: "List sessions" },
  { name: "session new", description: "Create a new session" },
  { name: "session attach", description: "Attach to an existing session" },
  { name: "permissions", description: "Manage tool permissions" },
  { name: "permissions status", description: "Show permission status" },
  { name: "permissions whitelist", description: "Whitelist tools or patterns" },
  { name: "permissions blacklist", description: "Blacklist tools or patterns" },
  { name: "save", description: "Save current session" },
  { name: "resume", description: "Resume a saved session" },
  { name: "sessions", description: "List saved sessions" },
  { name: "plan", description: "Show current plan status" },
];

export interface ParseOptions {
  /** Commands advertised by the daemon (``CommandListEvent``). */
  serverCommands?: CommandSpec[];
  /**
   * The user dismissed the command completion for this line: send the
   * text as a message regardless of what its first word looks like.
   */
  verbatim?: boolean;
}

/**
 * Decide what a submitted line means.  Lexical, like the TUI: the
 * decision is made from the first word (or the whole line for the
 * client-only commands) — unless ``verbatim`` says otherwise.
 */
export function parseUserInput(raw: string, opts: ParseOptions = {}): ParsedCommand {
  const text = raw.trim();
  if (!text) return { action: "message", text: "" };
  if (opts.verbatim) return { action: "message", text };

  const lower = text.toLowerCase();
  const parts = text.split(/\s+/);
  const cmd = (parts[0] ?? "").toLowerCase();
  const args = parts.slice(1);

  if (lower === "exit" || lower === "quit" || lower === "q") return { action: "exit" };
  if (lower === "stop") return { action: "stop" };
  if (lower === "clear") return { action: "clear" };
  if (lower === "help") return { action: "help" };
  if (lower === "context") return { action: "context" };
  if (lower === "history") return { action: "history" };
  if (cmd === "reset") return { action: "server", command: "reset", args };

  if (cmd === "tools" || cmd === "session") {
    const sub = args[0] ?? "list";
    return { action: "server", command: `${cmd}.${sub}`, args: args.slice(1) };
  }
  if (SERVER_COMMAND_PREFIXES.has(cmd)) return { action: "server", command: cmd, args };
  // ``theme`` is a client-side command that takes an argument (the TUI's
  // ``theme <name>``); it rides the "server" action and is intercepted locally.
  if (cmd === "theme") return { action: "server", command: "theme", args };

  // The daemon's list may be handed to us merged with CLIENT_COMMANDS (that is
  // what the completer wants).  The TUI only ever matches the *daemon's* names
  // here, and a client-only word is a command solely as a whole line — so
  // ``help me write a test`` must stay a prompt.
  for (const spec of opts.serverCommands ?? []) {
    const base = spec.name.trim().toLowerCase().split(/\s+/)[0];
    if (!base || CLIENT_COMMAND_NAMES.has(base)) continue;
    if (lower === base || lower.startsWith(base + " ")) {
      return { action: "server", command: base, args };
    }
  }
  return { action: "message", text };
}

/**
 * Would this line be routed as a command if submitted as-is?  Used by
 * the composer to show the "Enter runs <cmd> · Esc sends as text" hint.
 */
export function wouldRouteAsCommand(text: string, serverCommands?: CommandSpec[]): string | null {
  const parsed = parseUserInput(text, { serverCommands });
  if (parsed.action === "message") return null;
  return parsed.action === "server" ? (parsed.command ?? "") : parsed.action;
}

export interface Completion {
  /** Text to place in the composer when accepted. */
  insert: string;
  /** What the popup shows — the full command path. */
  label: string;
  description?: string;
}

/**
 * Command completions for the text before the caret.  Follows the TUI's
 * progressive rule: typing the first word completes base commands;
 * ``base ␣`` lists that command's subcommands; ``base su`` filters
 * them.  Lines containing ``@`` or starting with ``/``, ``%`` yield
 * nothing (those sigils have their own completers).  Only the first
 * two words participate — after ``tools enable `` the argument is the
 * user's.
 */
export function commandCompletions(
  textBeforeCaret: string,
  commands: CommandSpec[],
): Completion[] {
  const text = textBeforeCaret.replace(/^\s+/, "");
  if (text.includes("@") || text.startsWith("/") || text.startsWith("%")) return [];
  if (text.includes("\n")) return [];

  const trailingSpace = /\s$/.test(text);
  const words = text.trim() === "" ? [] : text.trim().split(/\s+/);
  const seen = new Set<string>();
  const out: Completion[] = [];
  const push = (insert: string, label: string, description?: string) => {
    const key = insert.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    out.push({ insert, label, description });
  };

  if (words.length === 0) {
    for (const c of commands) {
      const base = c.name.split(/\s+/)[0]!;
      if (base === c.name) push(base, c.name, c.description);
      else push(base, base, undefined);
    }
    return out;
  }

  const first = words[0]!.toLowerCase();
  if (words.length === 1 && !trailingSpace) {
    for (const c of commands) {
      const base = c.name.split(/\s+/)[0]!;
      if (!base.toLowerCase().startsWith(first)) continue;
      if (base === c.name) push(base, c.name, c.description);
      else push(base, base, undefined);
    }
    return out;
  }

  const partial = words.length === 1 ? "" : (words[1] ?? "").toLowerCase();
  if (words.length > 2 || (words.length === 2 && trailingSpace)) return [];
  for (const c of commands) {
    const parts = c.name.split(/\s+/);
    if (parts.length < 2 || parts[0]!.toLowerCase() !== first) continue;
    const sub = parts[1]!;
    if (!sub.toLowerCase().startsWith(partial)) continue;
    push(`${parts[0]} ${sub}`, `${parts[0]} ${sub}`, c.description);
  }
  return out;
}

/** Merge the static defaults with the daemon's list; the daemon wins on name collisions. */
export function mergeCommandSpecs(server: CommandSpec[]): CommandSpec[] {
  const byName = new Map<string, CommandSpec>();
  for (const c of [...CLIENT_COMMANDS, ...DEFAULT_COMMANDS]) byName.set(c.name, c);
  for (const c of server) if (c.name) byName.set(c.name, c);
  return [...byName.values()];
}
