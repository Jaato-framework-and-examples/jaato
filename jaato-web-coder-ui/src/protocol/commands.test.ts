import { describe, expect, it } from "vitest";
import {
  CLIENT_COMMANDS,
  DEFAULT_COMMANDS,
  commandCompletions,
  mergeCommandSpecs,
  parseUserInput,
  wouldRouteAsCommand,
} from "./commands";

const all = mergeCommandSpecs([{ name: "waypoint", description: "x" }, { name: "lsp status" }]);

describe("parseUserInput — TUI routing parity", () => {
  it("routes exact client commands", () => {
    expect(parseUserInput("help").action).toBe("help");
    expect(parseUserInput("QUIT").action).toBe("exit");
    expect(parseUserInput("stop").action).toBe("stop");
  });
  it("help with more words is a prompt, like the TUI — even with the merged completion list", () => {
    expect(parseUserInput("help me write a test")).toEqual({ action: "message", text: "help me write a test" });
    expect(parseUserInput("help me write a test", { serverCommands: all }).action).toBe("message");
    expect(parseUserInput("clear the table please", { serverCommands: all }).action).toBe("message");
  });
  it("theme <name> is a client command with an argument", () => {
    expect(parseUserInput("theme light", { serverCommands: all })).toEqual({ action: "server", command: "theme", args: ["light"] });
  });
  it("splits tools/session into dotted wire commands", () => {
    expect(parseUserInput("tools enable cli")).toEqual({ action: "server", command: "tools.enable", args: ["cli"] });
    expect(parseUserInput("session")).toEqual({ action: "server", command: "session.list", args: [] });
  });
  it("routes known prefixes even before the server list arrives", () => {
    expect(parseUserInput("model gpt-4o")).toEqual({ action: "server", command: "model", args: ["gpt-4o"] });
  });
  it("matches the daemon's command list by base word", () => {
    expect(parseUserInput("waypoint list", { serverCommands: [{ name: "waypoint list" }] }).command).toBe("waypoint");
    expect(parseUserInput("waypointer", { serverCommands: [{ name: "waypoint" }] }).action).toBe("message");
  });
  it("verbatim wins over every command match", () => {
    expect(parseUserInput("model this is broken", { verbatim: true })).toEqual({
      action: "message",
      text: "model this is broken",
    });
    expect(parseUserInput("help", { verbatim: true }).action).toBe("message");
  });
  it("never treats sigil lines as commands", () => {
    expect(parseUserInput("/review src").action).toBe("message");
    expect(parseUserInput("%summarise").action).toBe("message");
    expect(parseUserInput("@README.md explain").action).toBe("message");
  });
});

describe("wouldRouteAsCommand", () => {
  it("names the command a line would run", () => {
    expect(wouldRouteAsCommand("model x")).toBe("model");
    expect(wouldRouteAsCommand("tools disable all")).toBe("tools.disable");
    expect(wouldRouteAsCommand("clear")).toBe("clear");
    expect(wouldRouteAsCommand("please clear the table")).toBeNull();
  });
});

describe("commandCompletions — progressive, first two words only", () => {
  it("completes base commands while typing the first word", () => {
    const c = commandCompletions("to", all);
    expect(c.map((x) => x.insert)).toEqual(["tools"]);
  });
  it("lists subcommands after a trailing space", () => {
    const c = commandCompletions("tools ", all).map((x) => x.insert);
    expect(c).toEqual(["tools list", "tools enable", "tools disable"]);
  });
  it("filters subcommands by partial", () => {
    expect(commandCompletions("tools en", all).map((x) => x.insert)).toEqual(["tools enable"]);
  });
  it("stops after the second word", () => {
    expect(commandCompletions("tools enable ", all)).toEqual([]);
    expect(commandCompletions("tools enable cl", all)).toEqual([]);
  });
  it("offers nothing on sigil lines or multi-line input", () => {
    expect(commandCompletions("@src/x.ts to", all)).toEqual([]);
    expect(commandCompletions("/re", all)).toEqual([]);
    expect(commandCompletions("hello\nto", all)).toEqual([]);
  });
  it("dedupes base names that appear with several subcommands", () => {
    const inserts = commandCompletions("", all).map((x) => x.insert);
    expect(new Set(inserts).size).toBe(inserts.length);
    expect(inserts).toContain("help");
    expect(inserts).toContain("tools");
  });
  it("prefers the daemon's description over the static default", () => {
    const merged = mergeCommandSpecs([{ name: "model", description: "from daemon" }]);
    expect(merged.find((c) => c.name === "model")?.description).toBe("from daemon");
    expect(merged.length).toBe(CLIENT_COMMANDS.length + DEFAULT_COMMANDS.length);
  });
});
