"""Semantic matching engine for the references plugin.

Loads the embedding sidecar matrix and provides cosine similarity search
over reference embeddings. Used by the enrichment pipeline to find
semantically relevant references when tag matching doesn't cover the
user's vocabulary.

The matching is brute-force cosine similarity via matrix multiplication.
For the expected scale (tens to low hundreds of references), this is
sub-millisecond and avoids introducing FAISS or similar as a dependency.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedding_types import SemanticMatch

logger = logging.getLogger(__name__)

# Max content length (in characters) below which we embed the whole piece.
# Above this, we use the first/last chunking strategy.
_SHORT_CONTENT_THRESHOLD = 1500  # ~400 tokens


class SemanticMatcher:
    """Cosine similarity matcher over a precomputed embedding matrix.

    Loads the sidecar ``.npy`` file at initialization and provides
    ``find_matches()`` to search for semantically similar references.

    Attributes:
        available: Whether the matcher is ready (matrix loaded, provider ready).
        embedding_model: The model name recorded in the index (for validation).
        embedding_dimensions: The dimensionality recorded in the index.
    """

    def __init__(self):
        self._matrix = None  # np.ndarray of shape (N, D), float32, normalized
        self._index_to_source_id: Dict[int, str] = {}  # row index → source ID
        self.embedding_model: Optional[str] = None
        self.embedding_dimensions: Optional[int] = None
        self._provider = None  # LocalEmbeddingProvider reference

    @property
    def available(self) -> bool:
        """Whether semantic matching can be performed."""
        return (
            self._matrix is not None
            and self._provider is not None
            and self._provider.available
        )

    def load_index(
        self,
        sidecar_path: str,
        embedding_model: str,
        embedding_dimensions: int,
        index_to_source_id: Dict[int, str],
    ) -> bool:
        """Load the embedding sidecar matrix from disk.

        Args:
            sidecar_path: Absolute path to the ``.npy`` sidecar file.
            embedding_model: Model name from the references index (for validation).
            embedding_dimensions: Expected dimensionality from the references index.
            index_to_source_id: Mapping from matrix row index to source ID.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not os.path.isfile(sidecar_path):
            logger.info(
                "Embedding sidecar not found at '%s' — semantic matching "
                "disabled (run gen-references to create it)",
                sidecar_path,
            )
            return False

        try:
            matrix = np.load(sidecar_path)
        except Exception as e:
            logger.warning(
                "Failed to load embedding sidecar '%s': %s", sidecar_path, e
            )
            return False

        # Validate shape
        if matrix.ndim != 2:
            logger.warning(
                "Embedding sidecar has unexpected shape %s (expected 2D)",
                matrix.shape,
            )
            return False

        if matrix.shape[1] != embedding_dimensions:
            logger.warning(
                "Embedding sidecar dimensions %d do not match index "
                "dimensions %d — semantic matching disabled",
                matrix.shape[1], embedding_dimensions,
            )
            return False

        # Ensure float32 and normalized
        matrix = matrix.astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        matrix = matrix / norms

        self._matrix = matrix
        self._index_to_source_id = dict(index_to_source_id)
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

        logger.info(
            "Loaded embedding sidecar: %d vectors, %d dimensions, model='%s'",
            matrix.shape[0], matrix.shape[1], embedding_model,
        )
        return True

    def set_provider(self, provider) -> None:
        """Set the embedding provider for query-time embedding.

        Args:
            provider: A ``LocalEmbeddingProvider`` instance.
        """
        self._provider = provider

    def validate_model(self, provider_model: str) -> bool:
        """Check that the provider's model matches the index's model.

        Args:
            provider_model: The model name configured on the provider.

        Returns:
            True if models match, False otherwise (logs a warning).
        """
        if self.embedding_model and provider_model != self.embedding_model:
            logger.warning(
                "Embedding model mismatch: provider uses '%s' but index "
                "was built with '%s' — semantic matching disabled. "
                "Re-run gen-references to rebuild the index.",
                provider_model, self.embedding_model,
            )
            return False
        return True

    def find_matches(
        self,
        query_vec,
        threshold: float = 0.75,
        top_k: int = 3,
        exclude_ids: Optional[set] = None,
    ) -> List[SemanticMatch]:
        """Find references whose embeddings are similar to the query vector.

        Args:
            query_vec: numpy array of shape (D,), normalized.
            threshold: Minimum cosine similarity to include.
            top_k: Maximum number of matches to return.
            exclude_ids: Source IDs to exclude from results (e.g., already
                matched by tags or already selected).

        Returns:
            List of SemanticMatch sorted by descending score.
        """
        if self._matrix is None:
            return []

        exclude_ids = exclude_ids or set()

        # Cosine similarity via dot product (both sides normalized)
        scores = self._matrix @ query_vec  # shape (N,)

        # Filter by threshold
        mask = scores >= threshold
        candidate_indices = np.where(mask)[0]

        # Build candidates, excluding specified IDs
        candidates: List[SemanticMatch] = []
        for idx in candidate_indices:
            idx_int = int(idx)
            source_id = self._index_to_source_id.get(idx_int)
            if source_id is None or source_id in exclude_ids:
                continue
            candidates.append(SemanticMatch(
                source_id=source_id,
                score=float(scores[idx]),
                embedding_index=idx_int,
            ))

        # Sort by score descending, cap at top_k
        candidates.sort(key=lambda m: m.score, reverse=True)
        return candidates[:top_k]

    def score_sources(
        self,
        query_vec,
        source_ids: set,
    ) -> Dict[str, float]:
        """Compute cosine similarity for specific source IDs.

        Used by the tag veto logic in hybrid mode: after tag matching
        finds candidate sources, this method scores each one so that
        low-similarity tag matches can be dropped as false positives.

        Args:
            query_vec: numpy array of shape (D,), normalized.
            source_ids: Source IDs to score.

        Returns:
            Dict mapping source_id → cosine similarity score.
            Sources not found in the index are omitted.
        """
        if self._matrix is None:
            return {}

        # Build reverse mapping: source_id → row index
        source_to_idx: Dict[str, int] = {
            sid: idx for idx, sid in self._index_to_source_id.items()
            if sid in source_ids
        }

        scores = self._matrix @ query_vec  # shape (N,)

        return {
            sid: float(scores[idx])
            for sid, idx in source_to_idx.items()
        }

    def embed_and_match(
        self,
        content: str,
        threshold: float = 0.75,
        top_k: int = 3,
        exclude_ids: Optional[set] = None,
    ) -> List[SemanticMatch]:
        """Embed content and find matching references in one call.

        For short content (< ~400 tokens), embeds the whole string.
        For long content, embeds the first ~400 tokens and last ~400 tokens
        separately, then takes the best match per reference across both.

        Args:
            content: The text to embed and match.
            threshold: Minimum cosine similarity.
            top_k: Maximum number of matches.
            exclude_ids: Source IDs to exclude.

        Returns:
            List of SemanticMatch sorted by descending score.
        """
        if not self.available or not content:
            return []

        if len(content) <= _SHORT_CONTENT_THRESHOLD:
            # Short content: embed whole thing
            query_vec = self._provider.embed_text_as_array(content)
            if query_vec is None:
                return []
            return self.find_matches(query_vec, threshold, top_k, exclude_ids)

        # Long content: embed first and last chunks, merge best scores
        first_chunk = content[:_SHORT_CONTENT_THRESHOLD]
        last_chunk = content[-_SHORT_CONTENT_THRESHOLD:]

        vec_first = self._provider.embed_text_as_array(first_chunk)
        vec_last = self._provider.embed_text_as_array(last_chunk)

        if vec_first is None and vec_last is None:
            return []

        # Collect matches from both chunks
        all_matches: Dict[str, SemanticMatch] = {}

        for vec in (vec_first, vec_last):
            if vec is None:
                continue
            for match in self.find_matches(vec, threshold, top_k * 2, exclude_ids):
                existing = all_matches.get(match.source_id)
                if existing is None or match.score > existing.score:
                    all_matches[match.source_id] = match

        # Sort by best score, cap at top_k
        results = sorted(all_matches.values(), key=lambda m: m.score, reverse=True)
        return results[:top_k]
