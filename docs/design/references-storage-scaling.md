# Scaling the References Store (the wikiLLM's base)

**Status: proposed. Nothing here is built, and nothing here has been measured
yet.** Phase 0 is the measurement; every later phase starts only when a number
it produces crosses an agreed limit.

The references catalog that the wikiLLM rests on is a set of JSON files under
`<workspace>/.jaato/references/` (the root plus one directory per bundle
carrying `bundle.json`), with the user tier at `~/.jaato/references/`. A
bundle may also carry a derived vector index: `embedding_config.json` plus a
`references.embeddings.npy` sidecar matrix, rebuilt by `reconcile`. The
question this document answers is whether that will scale, and what to do
about it if it does not.

## 1. Verdict

**Files stay the source of truth. What gets added is a derived, rebuildable
index beside them, and only where a measurement shows the files are the
bottleneck.** Replacing the files with a database would undo properties the
wikiLLM design depends on (§2), and the costs that will grow first are
caused by how the code reads the files rather than by storing them as files
(§3).

## 2. What a file per reference buys, and a database would cost

| Property | Depends on files because |
|---|---|
| **Confinement** | What a runner may read or write is expressed as path grants: the catalog is write-denied in every runner body, `references-claims/` is writable (template v43), and promotion writes the catalog daemon-side (`server/reference_curation.py`) until #1422 makes `tool_hat` real. A database needs a socket and a new trust boundary for every runner. |
| **Review and provenance** | Curation, `revises` with digest-checked stale detection (#1437), `origin` stamps that are observed, never claimed, `git pull` / PR review of the wiki and `bundle merge` / `unpack` all assume one inspectable file per reference. |
| **Hand editing** | A person can open, fix and commit a reference with nothing but an editor. The #1145 refresh picks it up. |
| **The derived pattern already exists** | The `.npy` sidecar is derived from the files and rebuilt by `reconcile`. Deleting it loses nothing that cannot be recomputed. |

A primary database is therefore the most expensive change available and
should be the last one taken (§4, phase 4).

## 3. Where it will hurt, roughly in the order it will be hit

1. **Context, not storage.** `listReferences` already overflowed a session
   at 228k tokens (#1513). At thousands of references the listing must be
   paged and filtered (tag, bundle, query), or the model cannot use the wiki
   at any storage speed.
2. **Change detection is O(files) on every read path.** Since #1437,
   `catalog_watch` stats every `*.json` file individually on each
   `listReferences`, `selectReferences`, `proposeReference` and at each
   prompt-enrichment turn boundary (`_refresh_catalog_if_changed`). Any
   change reloads the whole catalog, in every runner.
3. **Recomputed graph work.** `LinkIndex` is rebuilt from the full catalog
   on each call that needs links (`plugin.py`, three sites). Transitive
   expansion reads and tokenises reference bodies at selection time
   (`_find_referenced_ids`), and a directory reference `rglob`s its subtree
   and concatenates it.
4. **Vectors.** Search is brute-force cosine over a full per-bundle matrix,
   and each runner loads its own copy of the matrix and of the embedding
   model (#1562, #1565). Fine up to tens of thousands of entries; beyond
   that, approximate nearest-neighbour search and one shared copy are
   needed. `reconcile` rewrites the whole sidecar on every change.
5. **Writes.** Daemon-side, one file per promotion through
   `write_contained`, one sidecar lock per bundle. Not a bottleneck at any
   plausible curation rate; listed so the absence is recorded.

## 4. Plan

### Phase 0: measure

A benchmark script (`scripts/bench_references_scale.py`, next to
`scripts/bench_cascade_fanout.py`) that generates catalogs of **1k, 10k and
100k** references with realistic shape: bodies of a few KB, 3 mentions and 1
to 3 declared links per reference, a few directory references, an indexed
bundle with a 384-dimension sidecar. It reports, per size:

| Measurement | Through |
|---|---|
| session bootstrap, references plugin `initialize()` | a real `bootstrap_session` on a warm slot |
| the refresh check with nothing changed, and after one file changed | `_refresh_catalog_if_changed` |
| `listReferences` latency and **result size in tokens** | the tool executor |
| `selectReferences` with transitive expansion, capped and uncapped | the tool executor |
| promotion into an indexed bundle, including reconcile | `reference.promote` against a live daemon |
| runner RSS with the index loaded | `/proc/<pid>/status` |

Run on a real host (enforcing AppArmor, real disk), not only in CI, because
the stat-per-file cost depends on the filesystem and the page cache. The
results go into this document as a table, and they set the limits that gate
phases 2 to 4.

### Phase 1: one store interface

Route every read the references plugin makes through a `ReferenceStore`:

```python
class ReferenceStore(Protocol):
    def generation(self) -> str: ...                      # changes when anything does
    def get(self, ref_id: str) -> Optional[ReferenceSource]: ...
    def list(self, *, tags=None, bundle=None, query=None,
             offset: int = 0, limit: int = 50) -> Page: ...
    def links_from(self, ref_id: str) -> List[ReferenceLink]: ...
    def links_to(self, ref_id: str) -> List[Dict[str, str]]: ...
    def nearest(self, vector, *, k: int, bundles=None) -> List[Tuple[str, float]]: ...
```

The first implementation is today's file code, so behaviour does not change.
The point is a single seam: every later phase replaces an implementation,
not the plugin's call sites. Writes stay where they are (daemon-side
curation, `reconcile`); the store is read-only to the runner.

### Phase 2: fix the algorithms, storage unchanged

- **A catalog generation counter.** Every framework writer (promotion,
  revision, link edit, merge, unpack, reconcile) bumps
  `.jaato/references/.generation` in the same contained write. The refresh
  check reads that one file instead of stating every JSON. The per-file
  stamps stay as the fallback that catches hand edits, checked at a coarser
  interval (once per turn rather than once per tool call).
- **Indexes cached per generation:** the `LinkIndex`, the tag index and the
  token index `_find_referenced_ids` needs, built once per generation
  instead of per call.
- **`listReferences` paged and filtered**, with the total and the filters
  applied in the answer so a cut listing is never read as the whole catalog.

Expected to carry the catalog to around 10k references; phase 0 says
whether it does.

### Phase 3: a derived SQLite index

`.jaato/.cache/references.db`, behind the phase 1 interface:

- **Contents:** reference metadata, tags, declared links (both directions),
  an FTS5 table over body text, and, once brute force stops being enough, a
  vector table (sqlite-vec, or an hnswlib file beside it).
- **Built by the daemon, read by runners.** If the model could write the
  index it could poison retrieval: the same reason the AppArmor fragment
  cache tier was write-denied (#1385). The runner profile gets read access
  only, and the index is keyed to the catalog generation it was built from.
- **Disposable.** Deleting it is always safe; a stale or missing index is
  rebuilt from the files, and the plugin answers from the files meanwhile.
- **One copy per workspace**, not per runner, which also addresses the
  per-runner matrix memory in §3 item 4.

### Phase 4: a server-backed store, only when something specific forces it

For example Postgres with pgvector, behind the same interface, with the file
layout kept as the import/export and review format. The triggers:

- one wiki shared by several hosts or daemons;
- many concurrent curators, so file-level conflicts become routine;
- catalogs past what a per-workspace index can hold.

Until one of those is real, this phase stays unbuilt.

## 5. What to avoid

Moving straight to a database as the primary store. It pays the confinement,
sealing, provenance and review costs of §2 now, before any measurement shows
the files are the bottleneck, and it does nothing for §3 item 1, the limit
most likely to be hit first.
