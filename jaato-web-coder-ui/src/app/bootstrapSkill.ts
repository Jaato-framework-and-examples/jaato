/**
 * Bootstrap the ``jaato-sdk`` skill into every workspace this client serves
 * (#1263).
 *
 * The skill is what an agent — Claude Code opening the directory, or the jaato
 * web-session agent through ``prompt_library`` — reads to learn how to build
 * on the SDK.  The web coder does NOT carry the skill's text: it asks the
 * DAEMON to install it, because the copy is stamped with the version of
 * whichever ``jaato-server`` runs the session and the workspace directory is
 * on that host.  So the application can never ship a copy that has drifted
 * from the framework — the daemon's own ``jaato-scaffold integration
 * claude-code --refresh`` keeps it current, re-applying an ``absent`` /
 * ``stale`` / ``outdated`` copy and leaving an edited one alone.
 *
 * WHEN it runs, per the issue:
 *   • on workspace selection — which, right after ``workspace.create``, is
 *     workspace creation;
 *   • on session start — which covers a daemon that has since been upgraded
 *     (a re-refresh of an unchanged copy costs the daemon one ``compare()``).
 *
 * Both reduce to "a workspace context appeared": the daemon resolves the
 * caller's own workspace from the attached session or the selection, so a
 * single verb serves both moments.  It fires once per distinct context and
 * again after a reconnect (a new client whose daemon may be newer), deduped
 * by the context key so a select-then-session pair is not two installs.
 *
 * The outcome — including a SKIPPED refresh of an edited copy — is reported
 * in the Files-panel notice by the store's ``scaffold.integration.result``
 * handler, never silently.  A daemon below protocol 1.21 refuses the verb
 * (the SDK throws), and that is left quiet: an older daemon simply does not
 * bootstrap the skill, which is a deployment fact rather than a per-workspace
 * failure.
 */
import { useJaato } from "@/store/store";
import { getClient, isConnected } from "@/sdk/connection";

/** The integration whose payload is the ``jaato-sdk`` skill. */
export const SKILL_INTEGRATION = "claude-code";

/** The last context we asked the daemon to bootstrap, so a select-then-start
 *  pair is one install.  Cleared when the context goes away (disconnect,
 *  session end), so the next appearance — possibly against an upgraded
 *  daemon — asks again. */
let lastKey: string | null = null;

/** The workspace context this client is in right now, or ``null``. */
function contextKey(st: {
  sessionId?: string | null;
  workspace: { selected?: string | null };
}): string | null {
  if (st.sessionId) return `session:${st.sessionId}`;
  if (st.workspace.selected) return `workspace:${st.workspace.selected}`;
  return null;
}

/** Ask the daemon to install/refresh the skill into the caller's workspace. */
export async function bootstrapSkill(): Promise<void> {
  if (!isConnected()) return;
  try {
    await getClient().runScaffoldIntegration(SKILL_INTEGRATION);
  } catch {
    // A daemon below protocol 1.21 refuses the verb; the skill is simply not
    // bootstrapped.  Nothing to report — the result event's absence here is
    // an old daemon, not a failed install.
  }
}

/**
 * Install the store subscription that fires {@link bootstrapSkill} whenever a
 * workspace or session context appears.  Idempotent — importing this module
 * once is enough, and a second call is a no-op.
 */
let installed = false;
export function installSkillBootstrap(): void {
  if (installed) return;
  installed = true;
  useJaato.subscribe((st) => {
    const key = isConnected() ? contextKey(st) : null;
    if (key === lastKey) return;
    lastKey = key;
    if (key) void bootstrapSkill();
  });
}

// Self-install on import, exactly as ``app/staging.ts`` registers its own
// store subscription at module load.
installSkillBootstrap();
