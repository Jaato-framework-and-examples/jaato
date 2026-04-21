# Tool Specification: `compute-embedding`

## Overview

A new jaato tool that computes vector embeddings for text or file content. Serves two roles:

1. **Agent-facing tool** — called by the indexing agent during `gen-references` execution to embed reference documents and store vectors in the reference schema
2. **Internal API** — called by the references plugin at runtime to embed streaming content pieces for semantic matching

Same implementation, single embedding model configuration, guaranteeing vector space consistency between index-time and query-time.

---

## Tool Definition (Gemini Function Calling)

```json
{
  "name": "compute_embedding",
  "description": "Compute a vector embedding for a text string or file contents. Returns a float array representing the semantic meaning of the input. Use this when building or updating reference indexes that require semantic search capability.",
  "parameters": {
    "type": "object",
    "properties": {
      "input": {
        "type": "string",
        "description": "The text to embed. Mutually exclusive with 'file'."
      },
      "file": {
        "type": "string",
        "description": "Path to a file whose contents should be embedded. Mutually exclusive with 'input'. For large files, content is truncated to the model's max input token limit."
      }
    }
  }
}
```

### Tool Response

```json
{
  "embedding": [0.0123, -0.0456, ...],
  "model": "all-MiniLM-L6-v2",
  "dimensions": 384,
  "input_tokens": 142
}
```

The response includes model metadata so the caller (or the reference schema) can record which model produced the vector — essential for compatibility validation.

---

## Internal API (Plugin-facing)

The references plugin does NOT call the tool through the function-calling interface. It imports the embedding function directly:

```python
from jaato.tools.compute_embedding import embed_text

# Single text
result = await embed_text("some content from the LLM stream")
# result.embedding -> list[float]
# result.model -> str
# result.dimensions -> int

# Batch (for efficiency when screening multiple pieces in one turn)
results = await embed_batch(["piece1", "piece2", "piece3"])
```

Both the tool handler and this internal API call the same underlying `_compute_embedding()` function, ensuring identical results.

---

## Configuration

Embedding model configuration lives in `jaato.yaml` under a shared section, not per-plugin:

```yaml
embedding:
  provider: local                       # or "vertexai"
  model: all-MiniLM-L6-v2              # sentence-transformers model
  # model: text-embedding-004           # Vertex AI alternative
  dimensions: 384                       # model-dependent
  batch_size: 64                        # max texts per batch (local is fast, can batch more)
  max_input_tokens: 512                 # MiniLM max sequence length
  eager_load: true                      # load model at startup, not on first call
```

### Provider options

| Provider   | Model                   | Dimensions | Latency (single) | Cold start | Notes                              |
|------------|-------------------------|------------|-------------------|------------|------------------------------------|
| `local`    | `all-MiniLM-L6-v2`     | 384        | ~5ms CPU          | ~1–2s      | Default. Zero network dependency   |
| `local`    | `nomic-embed-text-v1.5` | 768       | ~15ms CPU         | ~3–4s      | Higher quality, Matryoshka support |
| `vertexai` | `text-embedding-004`    | 768        | ~20–40ms + net    | none       | Native to Gemini stack             |

The `local` provider uses `sentence-transformers` and loads the model into memory once at startup when `eager_load` is true (recommended). This avoids a 1–2 second latency spike on the first screening pass.

### Local provider implementation

**Dependency:** `pip install sentence-transformers` (pulls `torch`, `transformers`, `huggingface-hub`). The model (~80MB) is downloaded and cached under `~/.cache/huggingface/` on first use.

**Interface:**

```python
from sentence_transformers import SentenceTransformer

# Load once at startup
model = SentenceTransformer("all-MiniLM-L6-v2")

# Single text → numpy array, shape (384,)
vec = model.encode("some text", normalize_embeddings=True)

# Batch → numpy array, shape (N, 384)
vecs = model.encode(["text one", "text two"], normalize_embeddings=True, batch_size=64)
```

Normalizing at encode time means dot product equals cosine similarity, which makes the matching step a single matrix multiply with no extra normalization.

**Async integration:**

`model.encode()` is a blocking CPU operation. To avoid stalling jaato's async event loop during streaming, the provider runs it in a thread executor:

```python
class LocalEmbeddingProvider:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimensions = self.model.get_sentence_embedding_dimension()

    async def embed_text(self, text: str) -> EmbeddingResult:
        vec = await asyncio.to_thread(
            self.model.encode, text, normalize_embeddings=True
        )
        return EmbeddingResult(
            embedding=vec.tolist(),
            model=self.model_name,
            dimensions=self.dimensions,
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        vecs = await asyncio.to_thread(
            self.model.encode, texts, normalize_embeddings=True, batch_size=64
        )
        return [
            EmbeddingResult(
                embedding=vec.tolist(),
                model=self.model_name,
                dimensions=self.dimensions,
            )
            for vec in vecs
        ]
```

Both the `compute_embedding` tool handler and the references plugin's internal API delegate to the same provider instance, ensuring identical vectors.

---

## Reference Schema Changes

### Current schema (tag-based)

```json
{
  "references": [
    {
      "id": "arch-overview",
      "file": "docs/architecture.md",
      "tags": ["architecture", "overview", "design"],
      "description": "High-level architecture document"
    }
  ]
}
```

### Updated schema (with embedding metadata)

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dimensions": 384,
  "embedding_sidecar": "references.embeddings.npy",
  "rows": ["arch-overview"],
  "references": [
    {
      "id": "arch-overview",
      "file": "docs/architecture.md",
      "tags": ["architecture", "overview", "design"],
      "description": "High-level architecture document",
      "embedding": {
        "source_hash": "sha256:a1b2c3d4..."
      }
    }
  ]
}
```

### Sidecar file: `references.embeddings.npy`

The vectors live in a separate NumPy binary file — a 2D `float32` array of shape `(N, D)` where N is the number of references with embeddings and D is the dimension count (384 for MiniLM). Row position is defined by the bundle's `rows` list (index i in `rows` corresponds to row i in the matrix), so the references themselves don't need to agree on numbering.

**Why a sidecar:**
- The JSON index stays clean and human-readable — no 768-element float arrays cluttering the schema
- The `.npy` file loads directly into a NumPy array with zero parsing overhead (`np.load()`)
- The sidecar can be regenerated independently of the index (e.g., when switching embedding models)
- Git-friendly: the JSON diffs cleanly, the binary file is a blob

**Lifecycle:**
- `gen-references` agent writes both `references.json` and `references.embeddings.npy` together
- The agent calls `compute_embedding` per document, collects all vectors, writes the matrix in the order declared by `rows`
- On re-index: if `source_hash` matches the current metadata, reuse the existing row; only re-embed changed refs, append new rows, drop orphans, then rewrite the sidecar

### Design notes

- **`tags` retained** — the hybrid lookup strategy (exact tag match → semantic fallback) means tags are still the fast path when the caller knows the exact tag
- **`source_hash`** — SHA-256 of the metadata that was embedded (name + description + tags + fetchHint). During index refresh, if the hash hasn't changed, skip re-embedding. Also enables staleness detection at runtime
- **`embedding_model` + `embedding_dimensions` at top level** — the references plugin validates at startup that the configured embedding model matches what's recorded in the index. Mismatch → warning + refuse semantic matching (fall back to tags only)
- **`rows`** — ordered list of reference ids, `rows[i]` is the id whose vector lives at matrix row `i`. Single source of truth for row↔id mapping, so inserting/removing a row only rewrites this list (not every reference JSON). References without embeddings simply omit the `embedding` property and don't appear in `rows`

See [References as Knowledge Bundles](design/references-knowledge-bundles.md) for how this schema participates in the per-bundle reconcile and merge flow.

---

## References Plugin Changes

### Configuration additions

```yaml
plugins:
  references:
    index_path: "references.json"
    lookup_strategy: "hybrid"           # "tags_only" | "semantic_only" | "hybrid"
    similarity_threshold: 0.75          # cosine similarity minimum for semantic match
    max_matches_per_piece: 3            # top-k semantic matches per screened piece
    skip_known_references: true         # don't embed content from already-resolved references
```

### Runtime flow (updated)

```
Piece arrives (prompt / tool output / model output)
    │
    ├─ Is this content from a known reference?
    │   └─ YES → skip (no embedding, no matching)
    │
    ├─ Tag scan (existing fast path)
    │   └─ exact tag matches → collect reference IDs
    │
    ├─ Semantic scan (new)
    │   ├─ embed_text(piece)
    │   ├─ cosine_similarity(piece_embedding, each ref.embedding.vector)
    │   ├─ filter by similarity_threshold
    │   ├─ top-k by max_matches_per_piece
    │   └─ collect reference IDs (excluding those already matched by tag)
    │
    └─ Inject matched reference IDs into context annotation
```

### Similarity computation

For the expected scale (tens to low hundreds of references), brute-force cosine similarity over a NumPy matrix is sufficient and avoids introducing FAISS or similar as a dependency:

```python
import numpy as np

def find_matches(query_vec: np.ndarray, index: np.ndarray, threshold: float, top_k: int) -> list[tuple[int, float]]:
    """
    query_vec: (D,) normalized
    index: (N, D) normalized reference embeddings
    Returns: list of (ref_index, score) above threshold, sorted desc, capped at top_k
    """
    scores = index @ query_vec                          # (N,) cosine similarities
    mask = scores >= threshold
    candidates = [(i, float(scores[i])) for i in np.where(mask)[0]]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]
```

Embeddings are normalized once at load time. The matrix multiply is sub-millisecond for N < 1000.

---

## gen-references Prompt Changes

The `gen-references.md` prompt that the indexing agent executes needs to instruct the agent to call the new tool. Key additions:

```markdown
For each reference document:
1. Read the file contents
2. Compute a SHA-256 hash of the file contents
3. If the reference already has an embedding and the source_hash matches, skip re-embedding
4. Otherwise, call the `compute_embedding` tool with the file path
5. Store the embedding metadata in the reference entry's `embedding` property:
   - `index`: the row position in the sidecar matrix
   - `source_hash`: SHA-256 of the file contents

After processing all references:
- Write the top-level `embedding_model`, `embedding_dimensions`, and
  `embedding_sidecar` fields to the index JSON
- Write all collected vectors as a NumPy float32 matrix to the sidecar file
```

---

## Error Handling

| Scenario                          | Behavior                                                    |
|-----------------------------------|-------------------------------------------------------------|
| Embedding API unavailable         | Log warning, fall back to tag-only matching for this turn   |
| Model mismatch (index vs config)  | Log warning at startup, disable semantic matching           |
| Input exceeds max tokens          | Truncate to `max_input_tokens`, log info                    |
| Empty input                       | Return zero vector, log debug                               |
| File not found (tool call)        | Return tool error to agent                                  |
| Batch partially fails             | Return successful embeddings, log errors for failures       |

---

## Observability

If OpenTelemetry is enabled:

- **Span per embedding call**: `compute_embedding` with attributes `input_tokens`, `model`, `provider`
- **Span per screening pass**: `references.semantic_scan` with attributes `piece_type` (prompt/tool_output/model_output), `num_matches`, `top_score`, `latency_ms`
- **Metric**: `jaato.references.embedding_calls_total` (counter), `jaato.references.semantic_matches_total` (counter), `jaato.references.embedding_latency_ms` (histogram)
