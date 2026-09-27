/**
 * Scroll-back through a session's history, one page at a time (protocol
 * 1.28).
 *
 * The client declares ``history_replay: "paged"`` in its presentation
 * (``sdk/connection.ts``), so an attach no longer replays the whole
 * conversation as a stream of output events: the daemon sends the MOST
 * RECENT page as a ``HistoryPageEvent`` and this module fetches older ones
 * when the person scrolls to the top of the transcript.  Every page's
 * ``model`` text has been through the daemon's output formatter, and no
 * page splits a fenced block, a table or a notebook cell.
 *
 * The answer is folded into the transcript by the store's ``HISTORY_PAGE``
 * reducer (the SDK delivers it to every subscriber, the store included);
 * this module owns only the request and its ``loading`` bookkeeping.
 *
 * Against a daemon below 1.28 nothing here runs: ``pagedHistorySupported``
 * is false, the daemon ignores ``history_replay`` and replays the old way,
 * and ``attachSession`` keeps asking for ``history.request``.
 */
import { MIN_HISTORY_PAGE_PROTOCOL, isProtocolCompatible } from "@jaato/sdk";
import { MAIN_AGENT, useJaato } from "@/store/store";
import { getClient, isConnected } from "@/sdk/connection";

/** Rendered lines per page the client asks for; the daemon caps it. */
export const HISTORY_PAGE_LINES = 150;

/** Does the connected daemon serve paged history? */
export function pagedHistorySupported(protocolVersion?: string | null): boolean {
  const v = protocolVersion ?? useJaato.getState().connection.protocolVersion;
  return !!v && isProtocolCompatible(v, MIN_HISTORY_PAGE_PROTOCOL);
}

/**
 * Fetch the page just older than what is on screen for ``agentId``.
 *
 * No-op when there is nothing older, a page is already in flight, or the
 * daemon predates 1.28 -- so a scroll handler may call it on every
 * scroll-to-top without guarding.  A failure is recorded on the agent's
 * paging state (the pane shows it) rather than thrown.
 */
export async function loadOlderHistory(agentId: string = MAIN_AGENT): Promise<void> {
  const st = useJaato.getState();
  const paging = st.historyPaging[agentId];
  if (!paging || !paging.hasMore || paging.loading || !paging.before) return;
  if (!isConnected() || !pagedHistorySupported()) return;
  st.setHistoryPaging(agentId, { loading: true, error: null });
  try {
    await getClient().requestHistoryPage({ agentId, before: paging.before, maxLines: HISTORY_PAGE_LINES });
  } catch (err) {
    useJaato.getState().setHistoryPaging(agentId, {
      loading: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
}
