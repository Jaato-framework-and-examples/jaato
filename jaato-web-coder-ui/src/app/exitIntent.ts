/**
 * "I left on purpose", carried across the screen change that follows.
 *
 * ``exit`` detaches and shows the connect screen.  Served with a
 * ``config.json`` that says ``autoConnect``, that screen would connect
 * again the moment it mounted -- the launcher config is a property of the
 * page load, and the screen cannot tell a fresh load from a return.  So
 * the exit path leaves this one-shot mark, and the connect screen consumes
 * it: everything else the launcher config decides (the daemon address,
 * the backend, who is signed in) still applies, only the automatic connect
 * is skipped once.  A reload starts clean, because the mark is consumed
 * the first time it is read.
 *
 * ``sessionStorage`` so the mark is per tab and never outlives it; every
 * access is guarded, since storage can be absent or refused.
 */
const KEY = "jaato.exited";

export function markExited(storage: Storage | null = safeStorage()): void {
  try { storage?.setItem(KEY, "1"); } catch { /* ignore */ }
}

/** True once per mark: reading it clears it. */
export function consumeExited(storage: Storage | null = safeStorage()): boolean {
  try {
    if (!storage || storage.getItem(KEY) !== "1") return false;
    storage.removeItem(KEY);
    return true;
  } catch {
    return false;
  }
}

function safeStorage(): Storage | null {
  try { return typeof sessionStorage === "undefined" ? null : sessionStorage; } catch { return null; }
}
