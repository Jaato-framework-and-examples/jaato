/**
 * Parser for the server's ``<nb-row>`` notebook markup (#1193).
 *
 * A notebook cell reaches a client as one row per part of the cell:
 *
 *   <nb-row type="input"  label="In [1]:">\n<j-code language="ipython">…</j-code>\n</nb-row>
 *   <nb-row type="stdout" label="Out [1]:">\n42\n</nb-row>
 *   <nb-row type="error"  label="Err [1]:">\nTraceback (most recent call last): …\n</nb-row>
 *
 * Emitted two ways and identical by the time a client sees them: the
 * notebook plugin writes ``<nb-row>`` directly for a cell that ran, and
 * ``notebook_output_formatter`` rewrites the ``<notebook-cell>`` markers of
 * an early exit (``label="Err:"``, no count).  ``type`` is one of ``input``,
 * ``stdout``, ``stderr``, ``result``, ``display``, ``error``; ``label`` may
 * be empty (the formatter gives stdout none).
 *
 * The web client had no parser for this family at all, so a cell reached
 * the transcript as text with its wrapper tags showing.  The TUI has had
 * one since the notebook shipped (``output_buffer._render_notebook_rows``).
 *
 * **Bounded to its own tags, on purpose.**  ``jmarkup.ts`` knows nothing
 * of ``<nb-row>`` and this module knows nothing of ``<j-*>`` -- the same
 * boundary the TUI pins with ``test_nb_row_is_not_j_markup``.  A row's
 * ``content`` is handed back RAW so the renderer can run ``parseJMarkup``
 * on it: an input cell carries a ``<j-code>`` block, and an output that
 * printed a markdown table carries a ``<j-table>``.
 *
 * Attribute values and program output inside a row are NOT escaped by the
 * server (only the interiors of ``<j-*>`` blocks are), so nothing here
 * unescapes anything.
 *
 * **An unterminated row stays text**, as an unterminated ``<j-code>`` does
 * in ``jmarkup.ts``.  The TUI renders a half-streamed row progressively,
 * but no producer the web client sees can split one -- tool output is
 * formatted per chunk and the notebook writes whole rows -- while a model
 * QUOTING the tag in prose is entirely possible, and rendering progressively
 * would swallow the rest of that message into a cell that does not exist.
 */

export interface NotebookRow {
  /** ``input`` | ``stdout`` | ``stderr`` | ``result`` | ``display`` | ``error`` -- kept verbatim. */
  type: string;
  /** ``In [1]:``, ``Out [1]:``, ``Err [1]:``, ``Err:``, or ``""``. */
  label: string;
  /** The row's body, raw: may hold ``<j-code>`` / ``<j-table>`` blocks and plain program output. */
  content: string;
}

export type NotebookSegment =
  | { kind: "text"; text: string }
  | { kind: "row"; row: NotebookRow };

// One leading and one trailing newline belong to the markup, not to the
// cell (the emitters write ``>\n{content}\n</nb-row>``); indentation inside
// the cell is content and is kept.  The newline after the closing tag is the
// row separator.  Instantiated per call: a shared ``/g`` regex carries
// ``lastIndex`` between calls.
const ROW_SRC = '<nb-row\\s+type="([^"]*)"\\s+label="([^"]*)">\\n?([\\s\\S]*?)\\n?<\\/nb-row>\\n?';

/** Does this text contain notebook markup this module would split? */
export function containsNbMarkup(text: string): boolean {
  return new RegExp(ROW_SRC).test(text);
}

/** Split a buffer into text / notebook-row segments. */
export function parseNbMarkup(text: string): NotebookSegment[] {
  const out: NotebookSegment[] = [];
  const ROW_RE = new RegExp(ROW_SRC, "g");
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = ROW_RE.exec(text)) !== null) {
    if (m.index > last) out.push({ kind: "text", text: text.slice(last, m.index) });
    out.push({ kind: "row", row: { type: m[1] ?? "", label: m[2] ?? "", content: m[3] ?? "" } });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

/** Rows whose body is a failure rather than a result: rendered in the error tone. */
export function isFailureRow(row: NotebookRow): boolean {
  return row.type === "error" || row.type === "stderr";
}
