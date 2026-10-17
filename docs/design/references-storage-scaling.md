# Scaling the References Store (the wikiLLM's base)

**Status: phase 1 built (`references/store.py`); phases 2-4 proposed.
Phase 0 is partly done: the in-process and daemon-level measurements are in
§4 ("Phase 0 results"); confinement is not measured. Phase 1's prototype
measurements and the measurements of the built store are in §4 ("Phase 1
measurements", "Phase 1 as built").** Every later phase starts
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
5. **`initialize()` is linear; the 2.2 s -> 67 s jump is the page cache.**
   Profiled with `cProfile` at 10k and 100k (2026-10-15): every call count
   is exactly 10x (100,100 file reads, 901k `lstat`, 100k `Path.resolve()`),
   and the cost per read is the same (0.33 ms at 10k, 0.32 ms at 100k, both
   runs cold). The benchmark's 10k run read warm files, its 100k run (~600 MB
   of JSON and bodies on a host with ~2 GB free) cold ones. Of the 77 s
   profiled at 100k, 42% is reading one small JSON file per reference
   (`config_loader.discover_references`) and 28% is `Path.resolve()` of each
   body path at load, about 9 `lstat` per reference.
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
justified by these numbers; item 5 (one file read per reference, cold) and
the reload cost in item 4 are where it would first be.

#### Fix options per finding (to be decided, nothing implemented)

For each finding, the alternatives considered and the two best, with the
code path the cost was measured in. Nothing here is built; one option per
finding is chosen before any of it is.

**A. Unbounded transitive selection is quadratic (item 2).**
`_resolve_transitive_references` passes the whole `catalog_ids` set to
`_mentions_of` for every node it visits, and `_find_referenced_ids` starts
with `_partition_ids_by_boundary(catalog_ids)`, which iterates every id to
split plain ids from ids containing a boundary character. Visiting V nodes of
an N-id catalog costs V x N before any content is read.

Considered: (1) partition once per traversal and pass the result down;
(2) look the content's tokens up in the catalog instead of building `plain`
(`{t for t in tokens if t in catalog_ids}` minus the exotic ids), with the
exotic ids computed once per catalog load; (3) a mention graph built at load
(every body read at `initialize()`); (4) cap `max_transitive_references` by
default; (5) an FTS index (phase 3); (6) parallel content reads.

| | Option 2: iterate the tokens, exotic ids once per load | Option 1: partition once per traversal |
|---|---|---|
| Cost per node | O(content), independent of N | O(content) + one O(N) pass per traversal |
| Change | `_find_referenced_ids` and the reload path that computes the exotic set | one argument threaded through `_mentions_of` |
| Pros | removes N from the hot path entirely; the exotic set is almost always empty, so the regex fallback runs on nothing | smallest diff; no new catalog-scoped state to invalidate |
| Cons | a new piece of state that must be recomputed on every reload (stale if a writer is missed) | a 100k catalog still pays 100k dictionary operations per selection |

Rejected: (4) treats the symptom and changes what an unconfigured workspace
gets; (3) moves the cost to every session start; (5) and (6) are larger than
the defect.

**B. `listReferences` spends ~90% in path resolution (item 3).** The loader
already resolves each body path once (`config_loader.py:128-148`,
`Path.resolve()`) and stores it project-relative. `_execute_list` then calls
`_resolve_path_for_access` for every LOCAL source on every call, which joins,
resolves and `exists()` up to four candidates, and `is_dir()` once more.

Considered: (1) store the absolute resolved path and `is_dir` on the source
at load, and have the listing read them; (2) memoise `_resolve_path_for_access`
per source, cleared on reload; (3) drop `resolved_path` / `is_directory` from
the listing (the path is returned on selection); (4) resolve only the page
being returned (needs C); (5) resolve in a thread pool.

| | Option 1: resolve at load, read at list | Option 2: memo per source, cleared on reload |
|---|---|---|
| Pros | one resolution per reference per load, where the loader already does one; every other caller (authorization, content read) can use the same fields | no extra work at `initialize()`; only references ever listed or read pay |
| Cons | the stored answer is as old as the last load: a body deleted afterwards still lists as present (the refresh watches catalog JSON, not bodies) | a second cache with its own invalidation rule beside the catalog's |

Both keep the listing's contract. (3) changes it, (4) depends on C, (5) adds
threads for a cost that should not be paid at all.

**C. The listing is unpaged (items 1 and 8).** 147k tokens at 1k
references; a 5.9 MB frame at 10k; 20k would exceed the 10 MB IPC cap.

Considered: (1) `offset` / `limit` with a default limit, returning `total`
and `next_offset`; (2) without a filter, return an overview (counts per tag
and per bundle) above a size threshold, entries only once filtered;
(3) compact entries (drop description and links); (4) a `query` argument
(substring over id, name, description, tags) plus paging; (5) cursor paging
by id; (6) cut to a token budget with a note.

| | Option 1 + 4: paging with a query filter | Option 2: an overview until filtered |
|---|---|---|
| Pros | one shape at every catalog size; a cut listing always says `total` and how to get the next page; `query` lets the model find a reference without walking pages | a model asking "what is there?" gets an answer it can read at any size and is led to filter |
| Cons | a model can still walk every page; the default limit is a number someone has to choose | two answer shapes; a small catalog switches shape when it crosses the threshold |

(3) shrinks the answer about 5x, not 100x; (5) adds nothing over offsets for
a catalog that is reread per call; (6) hides how much was dropped.

**D. The refresh check stats every catalog file (item 4, the check).**
`_catalog_watch_paths` watches every directory and, since #1437, every
`*.json` in them; `_refresh_catalog_if_changed` stats them all on every
`listReferences`, `selectReferences`, `proposeReference` and prompt
enrichment (478 ms at 100k).

Considered: (1) a generation file every framework writer bumps, with the
per-file stamps checked once per turn for hand edits; (2) inotify; (3) check
at most once per turn or per N seconds; (4) watch directories only (drops
#1437's in-place edits); (5) move the check to the daemon and push a
generation on the envelope.

| | Option 1: generation file + per-turn file sweep | Option 3: check at most once per turn |
|---|---|---|
| Pros | the per-call check is one `stat`; framework writes are seen at once | no writer changes; one place to edit |
| Cons | every writer must bump it (promotion, revision, link edit, merge, unpack, reconcile), and a missed one means a stale catalog until the sweep; hand edits wait for the sweep | still O(files), once per turn (478 ms per turn at 100k); a promotion made mid-turn is seen next turn |

(2) is Linux-only, costs watches per runner and is not readable under every
confinement; (4) reverts #1437; (5) needs a new daemon->runner notification.

**E. Any change reloads the whole catalog (item 4, the reload).** One edited
file at 100k is a 23.6 s `_reload_from_disk`, run inside the read that
noticed it.

Considered: (1) incremental reload: diff the stamp maps and re-parse only
files whose stamp moved, and only the directories whose listing changed;
(2) reload in the background and swap the catalog atomically, answering from
the old one meanwhile; (3) cache the parsed catalog per directory, keyed by
the directory stamp; (4) phase 3's index; (5) accept it.

| | Option 1: incremental reload | Option 2: background reload and swap |
|---|---|---|
| Pros | cost proportional to what changed; same thread, same answer as today | the read never waits; no change to how files are parsed |
| Cons | `_reload_from_disk` also re-authorizes selections and re-runs the transitive walk; those have to become per-change too, or still run whole | the read that noticed the change answers from the old catalog, which is the #1145 staleness the refresh exists to remove; a background thread per runner |

(3) is option 1 at a coarser grain; (4) is the phase 3 decision, not this
one.

**F. `initialize()` reads one file per reference (item 5).** Linear, as
above; the constant is a cold read of a ~400-byte JSON file plus a
`Path.resolve()` of the body path.

Considered: (1) store the body path normalised at load and resolve it only
when the body is read (removes ~9 `lstat` per reference); (2) a derived
per-bundle manifest (one JSON-lines file of every entry, rebuilt by the
framework writers, ignored when older than the directory); (3) read files in
a thread pool; (4) load the catalog once per workspace in the daemon and ship
it to runners; (5) phase 3's SQLite index; (6) leave it.

| | Option 1: no eager `resolve()` at load | Option 2: per-bundle manifest |
|---|---|---|
| Pros | ~28% of `initialize()` at 100k, one function, no new files | one read per bundle instead of one per reference: removes the dominant cost |
| Cons | leaves the 42% of reading the files; a symlinked body is resolved later, so the path a containment check sees must be resolved there | a derived file that can disagree with the directory, so it needs a validity rule (stamps) and a fallback to the files; every writer must rebuild it; it is phase 3 in miniature |

(3) helps cold I/O but adds threads to `initialize()`; (4) changes the
runner's trust model (the runner would read a catalog it did not load);
(5) and (6) belong to the phase 3 decision.

**G. The embedding model costs ~650 MB per runner (item 7).** Each runner
that uses `hybrid` loads its own copy.

Considered: (1) embed in the daemon, one copy, runners asking over the runner
RPC (the reverse of today's `RunnerEmbeddingProvider`); (2) a separate
embedding process per host shared by daemons; (3) load the model in the pool
template so slots share its pages copy-on-write; (4) a smaller or quantised
model in jaato-premium; (5) lexical matching (BM25 / FTS) instead of vectors;
(6) a cap on concurrent hybrid sessions.

| | Option 1: the daemon embeds | Option 3: model in the template, shared copy-on-write |
|---|---|---|
| Pros | one copy per daemon, whatever the number of sessions; the confined runner loads no torch at all | no new RPC; slots fork with the weights already mapped |
| Cons | the daemon process grows by the model and its threads; every embedding is a round trip; the daemon does work a runner did | not measured: whether the weight pages stay shared (Python reference counts write to object headers, torch may copy); torch in the template breaks the one-thread rule the template forks under; and #1565 retires a slot that loaded the model, which would have to change |

I don't have evidence for option 3's sharing yet; it would have to be
measured (PSS per slot) before it could be chosen. (2) is option 1 with a
new process; (4) is jaato-premium's decision; (5) changes what hybrid lookup
means; (6) limits users instead of the cost.

### Phase 1: one store interface

**Why there is a seam to add.** The plugin's catalog is one list,
`self._sources`, read at 94 sites and mutated at about 12. It mixes two
things: the catalog loaded from disk, and the session's working set (an
inline `sources:` narrowing, plus preselected and transitive references
appended from the master catalog). By-id lookups are linear scans
(`next(s for s in self._sources if s.id == x)`, about 20 sites).
`_resolve_source_for_context` rewrites paths on the shared objects. The
daemon side does not use the list at all: `reference_catalog.py`,
`reference_curation.py` and `claims.py` each walk and parse the files
themselves.

**The design (agreed, not built).** Three parts, and only the first stores
anything:

| Part | What it is | Who holds it |
|---|---|---|
| `ReferenceStore` | the source of references: today the files, later possibly a database or a server. `load()`, `generation()` | one per catalog per process |
| `CatalogSnapshot` | the immutable result of one load at one generation: entries by id, files by path, per-bundle index, `LinkIndex`, all built once | the store holds the current one per catalog, by weak reference; each plugin keeps the one it is working from |
| `WorkingSet` | one session's ids into the current snapshot | the plugin, per session |

Rules that follow from it:

- **Indexes cannot go stale.** A snapshot never changes, so an index built
  on it stays right for its lifetime. A change on disk produces a new
  snapshot; a tool call keeps the snapshot it started with.
- **The next generation is derived, not mutated.** On a change, the store
  re-parses only the files whose stamp moved and the directories whose
  listing changed, and builds G+1 sharing every unchanged entry with G.
  Writes from other processes (daemon curation, `git pull`, hand edits) are
  found by the existing `catalog_watch` stamps, not by events.
- **Sharing within a process, never across processes.** Every holder in one
  process (the daemon's per-session mirrors, its three readers) gets the
  same snapshot object. The cache key is everything a load depends on:
  workspace, `config_root`, and the user-tier home, which differs for a
  runner dropped to another uid (#1168).
- **No eviction policy.** A snapshot is freed when its last holder lets go:
  in the daemon when the last mirror for that catalog unloads, in a runner
  at `reset_for_next_session` / `shutdown`. An unconfined pool slot can
  move between workspaces (`accepts_unaffined` compares only the profile
  name), so a runner must not keep a snapshot across sessions by default.
  Keeping one strongly across sessions of the same confined boundary is a
  later, measured step.
- **Paths are resolved when read**, per session, against the session's
  workspace (`WorkingSet`), never written into the shared snapshot.
- **Writes stay where they are** (daemon-side curation, `reconcile`); the
  store is read-only to the runner.

The query interface sketched in earlier drafts (`get`, paged `list`,
`links_from`, `nearest`) is phase 4's shape: it is what the store becomes
when a backend cannot hand over a whole snapshot.

#### Phase 1 measurements (prototype, measured)

`scripts/bench_references_snapshot.py`, run 2026-10-17 on the phase 0 host
and catalogs. The design is prototyped in the script; the plugin is
unchanged. Each derived snapshot is compared with a full plugin reload of the
same disk, and was equal (same ids, same parsed entries) in every case below.

Step 1, the seam with a full reload:

| Measurement | 1k | 10k | 100k |
|---|---|---|---|
| full reload (`_reload_from_disk`) | 0.21 s | 2.3 s | 22.9 s |
| of which the parse (`_reload_catalog` + bundles) | 0.21 s | 2.4 s | 22.7 s |
| snapshot indexes (by id, by bundle, `LinkIndex`): build | 1.6 ms | 65 ms | 0.77 s |
| snapshot indexes: memory beyond the entries | 0.6 MB | 6.2 MB | 64 MB |
| one by-id lookup, linear scan as today | 16 µs | 192 µs | **7.2 ms** |
| one by-id lookup, dict | 0.14 µs | 0.15 µs | 0.16 µs |

Step 2, generation G+1 derived from G (median of 5) against a full reload of
the same change:

| Change | 1k | 10k | 100k derive | 100k full reload |
|---|---|---|---|---|
| detect (stat every watched path, unchanged) | 4.6 ms | 48 ms | 0.50 s | same |
| 1 file edited in place | 1.9 ms | 30 ms | **0.41 s** | 23.1 s |
| 10 files edited | 3.7 ms | 32 ms | 0.40 s | 23.4 s |
| 100 files edited | 21 ms | 60 ms | 0.42 s | 23.5 s |
| 1 file added | 7.4 ms | 39 ms | 0.43 s | 23.2 s |
| 1 file deleted | 7.2 ms | 36 ms | 0.42 s | 24.2 s |
| memory G+1 allocates beyond G | 0.7 MB | 6.7 MB | 73 MB | |

Who holds the snapshot, through a private daemon (`daemon` part), daemon
process RSS after each `session.new`, 3 sessions with a profile that does
not list `references` then 3 that do:

| Catalog | idle | +1 | +2 | +3 | +4 | +5 | +6 | `session.new` |
|---|---|---|---|---|---|---|---|---|
| 10 references | 91 MB | 135 | 137 | 138 | 139 | 140 | 142 | 2.4-3.8 s |
| 10,000 references | 91 MB | 160 | 186 | 211 | 236 | 262 | 297 | 9.4-25.5 s |

Read off the tables:

1. **Deriving makes a change cost what the indexes cost, not what the
   catalog costs**: 0.4 s at 100k whatever the number of edited files, about 56x
   less than the full reload. What remains is fixed: copying the three
   dicts and rebuilding the whole `LinkIndex`. Incremental `LinkIndex`
   maintenance would remove most of it, if a measurement ever asks.
2. **The full reload is all parse.** `_reload_from_disk`'s own work
   (selections, authorizations, matchers) is noise beside it, so deriving
   is the whole fix for finding E.
3. **Linear by-id lookups cost 7 ms each at 100k**, across about 20 sites,
   several inside loops. The snapshot's dict removes that.
4. **The daemon pays a catalog per session, even for sessions that do not
   use references.** Each session costs the daemon about 25 MB at 10k
   references against 1-2 MB at 10, with or without `references` in the
   profile: the daemon's mirror registry initializes every discovered
   plugin (#1566), and the references plugin loads its catalog in
   `initialize()` even when `session_enables_plugin` is false (only the
   embedder is deferred, #1563). `session.new` goes from ~2.5 s to 9-25 s
   for the same reason, on the daemon side, the runner side, or both; the
   split is not measured. A store shared within the daemon process makes
   that one copy; deferring the load when the session does not enable the
   plugin removes it for those sessions entirely.
5. **Not measured:** the `detect` cost is unchanged by this design (finding
   D, 0.5 s at 100k on every read path); confinement; and memory in a
   runner when a snapshot is shared rather than copied (a runner holds one
   session, so nothing is shared there).

#### Phase 1 as built (measured)

Built as designed, with two consumers fixed on the way that defeated the
deferral (they asked every plugin for its system instructions, and the
references plugin builds them from its catalog): the runtime's
configure-time instruction cache, which nothing read, and the instruction
budget, which walked every exposed plugin instead of the session's list.

Same host, 10,000 references, private daemon, three sessions per profile:

| | Before phase 1 | Built |
|---|---|---|
| daemon growth per session, profile without `references` | ~25 MB | ~5 MB |
| daemon growth per session, profile with `references` | 25-35 MB | 7-14 MB |
| `session.new`, profile without `references` | 9-25 s | 3.2-5.0 s |
| `session.new`, profile with `references` | 9-25 s | 22.6 s first, then 8-9 s |
| catalog loads for a profile without `references` | daemon and runner | none |

The store's derivation (`ReferenceStore._derive`, median of 5) against a
full reload of the same change, equal in every case:

| Change | 1k derive | 1k full | 10k derive | 10k full |
|---|---|---|---|---|
| 1 file edited | 12.8 ms | 0.24 s | 146 ms | 2.30 s |
| 10 files edited | 15.6 ms | 0.21 s | 150 ms | 2.29 s |
| 100 files edited | 37.2 ms | 0.22 s | 170 ms | 2.30 s |
| 1 file added | 63.0 ms | 0.23 s | 168 ms | 2.29 s |
| 1 file deleted | 66.0 ms | 0.23 s | 178 ms | 2.27 s |

The prototype measured 30-48 ms at 10k because it was handed the stamps;
the store stamps every watched path before and after (47 ms each at 10k),
so a write made while it derives is not lost. Phase 2's generation counter
is what removes that. 100k was not re-measured. Not built: the daemon's own
readers (`reference_catalog.py`, `reference_curation.py`, `claims.py`) still
read the files themselves. What makes up the ~5 MB per session the daemon
still grows is not measured.

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
