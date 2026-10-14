# Scaling the References Store (the wikiLLM's base)

**Status: proposed. Nothing here is built. Phase 0 is partly done: the
in-process and daemon-level measurements are in §4 ("Phase 0 results");
confinement is not measured.** Every later phase starts
only when a number crosses an agreed limit.

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

#### Phase 0 results (in-process, measured)

`scripts/bench_references_scale.py`, run 2026-10-14 on a developer host
(Linux 6.8, 8 cores, ext4, Python 3.12, AppArmor enabled but the plugin
driven in-process, so unconfined). It drives the real references plugin
(`create_plugin` + `initialize`, the tool executors, the refresh check), one
fresh process per size. Catalog shape: bundles of 1,000, local markdown
bodies of 3 KB, 3 id mentions and 1-3 declared links per reference, no
embedding index.

| Measurement | 1k | 10k | 100k |
|---|---|---|---|
| `initialize()` | 0.21 s | 2.2 s | 67 s |
| refresh check, nothing changed (median of 20) | 3.9 ms | 47 ms | 478 ms |
| refresh after one file edited in place (= full reload) | 0.21 s | 2.2 s | 23.6 s |
| `listReferences`, no arguments | 0.18 s | 2.1 s | 19.9 s |
| `listReferences` result, est. tokens (bytes / 4) | **147k** | 1.48M | 14.9M |
| `LinkIndex` build over the catalog | 1.5 ms | 25 ms | 615 ms |
| `selectReferences` one id, unbounded | 0.92 s (918 refs) | 23 s (8,328) | **842 s** (27,663, depth 10 binding) |
| same, `max_transitive_references: 25` | 23 ms | 0.26 s | 2.8 s |
| `enrich_prompt`, prompt naming 3 ids | 11 ms | 61 ms | 598 ms |
| process RSS after one session's `initialize()` | 53 MB | 71 MB | 263 MB |
| peak RSS (two more sessions, unbounded select) | 63 MB | 159 MB | 1,044 MB |

Where the time goes, profiled with `cProfile` at 10k:

1. **The listing outgrows the context first.** An unfiltered
   `listReferences` is ~147k tokens at 1,000 references, so it fills a
   typical context window at roughly 1,000-1,500 references. No storage
   change affects this; only paging and filtering do (§3 item 1 confirmed).
2. **Unbounded transitive selection is quadratic.** 10x the catalog cost 25x
   the time from 1k to 10k (the 100k run is cut short by the depth bound).
   97% of it is `_find_referenced_ids` -> `_partition_ids_by_boundary`,
   which iterates every catalog id for every node the walk visits: 6,690
   calls and ~67M dictionary lookups at 10k. Its own docstring notes the
   partition "is recomputed for every node of every traversal"; the
   per-id memo removes the regex test, not the iteration. With the cap at 25
   it stays cheap, and the cap is off by default.
3. **`listReferences` time is path resolution.** ~90% is
   `_resolve_path_for_access`, two `Path.resolve()` plus stats per reference
   on every call.
4. **The refresh check is O(files) per read path.** 478 ms at 100k on every
   `listReferences`, `selectReferences`, `proposeReference` and turn
   boundary, and any change reloads the whole catalog (23.6 s at 100k).
5. **`initialize()` grows faster than linearly** (2.2 s -> 67 s for 10x);
   not yet profiled.
6. **Memory is not the first limit.** Peak 1 GB at 100k comes from three
   sessions plus one unbounded selection of 27k references.

#### Phase 0 results (through a daemon, measured)

`scripts/bench_references_daemon.py` gives every bundle a vector index
(jaato-premium's `all-MiniLM-L6-v2`), starts a private daemon
(`ConformanceDaemon`: its own socket, pid file and HOME; pool size 1), and
drives it with `IPCClient`. Same catalog shape as above.

| | 1k | 10k |
|---|---|---|
| index every bundle (in-process, one-off, first run) | 18.6 s | 90 s (10 bundles) |
| `listReferences`, tags only, one `invoke_plugin_tool` | 0.49 s, 590 KB | 3.6 s, 5.9 MB |
| `selectReferences`, cap 25, tags only | 0.42 s | 2.6 s |
| `selectReferences`, cap 25, hybrid (ranked by similarity) | 8.3 s | 10.6 s |
| `session.new`, profile enabling `references` (hybrid) | 10.1 s | 15.0 s |
| `reference.promote` into an indexed bundle, reconcile `updated` | 2.3 s | 6.0 s |
| runner RSS, no model / model loaded | 128 / 766-809 MB | 128 / 800-843 MB |

Findings:

7. **The embedding model costs about 650 MB per runner that loads it**, and
   each runner loads its own copy. The hybrid calls and `session.new` are
   dominated by that load (~8 s), not by the catalog. This is a memory limit
   that arrives before any catalog size: five concurrent hybrid sessions are
   ~4 GB.
8. **`listReferences` through the daemon at 10k is a 5.9 MB answer.** The
   frame fits the 10 MB IPC cap, so 20k references would not, and the model
   could not read it anyway (~1.5M tokens). Paging (phase 2) is required
   before 10k, independently of storage.
9. **Promotion and reconcile scale with the bundle, not the catalog**
   (2.3 s -> 6.0 s for 1k -> 10k with 1k-reference bundles). Acceptable.

**A defect outside the references plugin, found by this harness.** On a
host where one plugin is skipped for a missing optional dependency (here
`interactive_shell` without `pexpect`) and many distributions are installed
(jaato-premium brings torch: 28k installed files), every plugin discovery
took 47 s: `registry._install_advice` -> `dependencies._metadata_index` calls
`_provided_import_names` once per required package (55 times), and each call
runs a full `importlib.metadata.packages_distributions()`, which stats every
installed file (cProfile: 1.1M `stat` calls, 96 s for two discoveries). The
`lru_cache` is per package name, so it never saves the scan. Consequences,
all from the captured daemon log: the daemon's event loop is stalled ~45 s
at startup, the template misses its 30 s READY deadline, a cold-spawned
runner misses the 35 s `session.bootstrap` deadline, so **the first
`invoke_plugin_tool` after a daemon start fails** (`DaemonLoopTimeout`). A
second defect follows: the late `READY` is then read as the first
`FORK_SLOT` reply, every later reply is off by one, and `recv(64)` with no
line framing concatenates several replies (`malformed FORKED reply
'FORKED:47930\nFORKED:47931...'`), orphaning the slots it forked. With
`pexpect` installed, discovery takes 0.6 s and every call above succeeds;
the numbers in the table were taken that way. Neither defect is fixed by
this plan.

Not measured yet: anything under AppArmor confinement (loading a profile
needs root on the benchmark host).

What this changes in the plan: items 1, 2 and 3 are algorithmic and fixed in
phase 2 without touching storage. Phase 3 (a derived index) is not yet
justified by these numbers; item 5 and the reload cost in item 4 are where it
would first be.

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
