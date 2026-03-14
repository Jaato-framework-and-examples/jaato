---
description: Generate vector embeddings for reference documents in the workspace
params:
  references_dir:
    required: false
    default: .jaato/references
    description: Path to the references directory
  force:
    required: false
    default: "false"
    description: Re-embed all documents even if hashes match
tags: ['indexing', 'references', 'embeddings']
---

Index all reference documents in `{{references_dir}}` and generate a semantic embedding sidecar.

## Steps

1. **Discover** — List all `.json` files in `{{references_dir}}`. Parse each as a reference entry. Skip `embedding_config.json` (it's metadata, not a reference).

2. **Read existing state** — If `{{references_dir}}/embedding_config.json` exists, load it to get the current embedding model, dimensions, and sidecar path. Load the existing sidecar `.npy` matrix if present (for incremental re-use).

3. **Process each reference**:
   - Resolve the target: `type=local` → read file at `path`; `type=inline` → use `content` field; `type=url` → skip (not supported in offline mode).
   - Compute SHA-256 of the resolved content.
   - If `{{force}}` is `false` and the reference already has `embedding.source_hash` matching the computed hash, skip it — reuse the existing vector from the sidecar matrix.
   - Otherwise call `compute_embedding` with `file` (for local) or `input` (for inline).
   - Assign the next available `embedding.index` and record `source_hash`.

4. **Write outputs**:
   - Collect all vectors (reused + newly computed) into a single NumPy float32 matrix, ordered by `embedding.index`.
   - Write the matrix to `{{references_dir}}/references.embeddings.npy` using `numpy.save()`.
   - Write/update `{{references_dir}}/embedding_config.json` with model name, dimensions, and sidecar filename.
   - Update each reference `.json` file with its `embedding` metadata (`index` + `source_hash`).

5. **Report** — Print a summary table:
   - Total references found
   - Newly embedded (count)
   - Skipped / unchanged (count)
   - Failed / unresolvable (count + list)
