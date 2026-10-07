/**
 * Re-render on an interval while ``active``, returning the current clock
 * value each time -- the one place a component reaches for "wall time" for
 * a live elapsed/stall readout.
 *
 * Extracted from ``PhaseLine`` (which used a 1s-only local copy) so
 * ``AgentTabs`` and the stall banner can share it: a second hand is
 * presentation, and putting the clock in the store would commit React once
 * per interval for every subscriber of every slice.
 */
import { useEffect, useState } from "react";

export function useTick(active: boolean, intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const t = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(t);
  }, [active, intervalMs]);
  return now;
}
