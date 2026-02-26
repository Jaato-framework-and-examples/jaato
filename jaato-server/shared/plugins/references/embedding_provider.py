"""Embedding provider for the references plugin.

Wraps sentence-transformers for local embedding computation. The provider
loads a model once and reuses it for all embedding calls — both the
``compute_embedding`` tool (called by the gen-references agent) and the
internal API (called by the enrichment pipeline for semantic matching).
"""

import logging
import numpy as np
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from .embedding_types import EmbeddingResult

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider:
    """Local embedding provider using sentence-transformers.

    Loads a SentenceTransformer model into memory and provides synchronous
    embedding methods. The caller (tool executor or enrichment pipeline) is
    responsible for running these in a thread executor if needed to avoid
    blocking an async event loop.

    The model is loaded eagerly at construction time when ``eager_load`` is
    True (default), or lazily on first ``embed_text`` / ``embed_batch`` call
    when False. Eager loading is recommended to avoid a 1-2 second latency
    spike on the first embedding call during a screening pass.

    Attributes:
        model_name: The sentence-transformers model identifier.
        dimensions: Embedding vector dimensionality (set after model load).
        available: Whether the provider is ready to compute embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_input_tokens: int = 512,
        batch_size: int = 64,
        eager_load: bool = True,
    ):
        """Initialize the local embedding provider.

        Args:
            model_name: sentence-transformers model to load.
            max_input_tokens: Maximum input sequence length (for truncation).
            batch_size: Maximum texts per batch encode call.
            eager_load: If True, load the model immediately. If False, defer
                to first embedding call.
        """
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.batch_size = batch_size
        self.dimensions: int = 0
        self._model = None

        if eager_load:
            self._load_model()

    @property
    def available(self) -> bool:
        """Whether the provider can compute embeddings."""
        return self._model is not None

    def _load_model(self) -> bool:
        """Load the sentence-transformers model.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._model is not None:
            return True

        try:
            # Suppress noisy transformer logs during model load
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

            self._model = SentenceTransformer(self.model_name)
            self.dimensions = self._model.get_sentence_embedding_dimension()
            logger.info(
                "Loaded embedding model '%s' (dimensions=%d)",
                self.model_name, self.dimensions
            )
            return True
        except Exception as e:
            logger.warning(
                "Failed to load embedding model '%s': %s",
                self.model_name, e
            )
            return False

    def embed_text(self, text: str) -> Optional[EmbeddingResult]:
        """Compute embedding for a single text string.

        This is a blocking CPU operation. When called from an async context,
        wrap in ``asyncio.to_thread()``.

        Args:
            text: The text to embed. Truncated to ``max_input_tokens`` if
                longer.

        Returns:
            EmbeddingResult with the embedding vector and metadata, or None
            if the provider is unavailable.
        """
        if not self._model and not self._load_model():
            return None

        if not text:
            return EmbeddingResult(
                embedding=[0.0] * self.dimensions,
                model=self.model_name,
                dimensions=self.dimensions,
                input_tokens=0,
            )

        vec = self._model.encode(text, normalize_embeddings=True)
        return EmbeddingResult(
            embedding=vec.tolist(),
            model=self.model_name,
            dimensions=self.dimensions,
            input_tokens=len(text.split()),  # rough approximation
        )

    def embed_batch(self, texts: List[str]) -> List[Optional[EmbeddingResult]]:
        """Compute embeddings for multiple texts in a single batch.

        More efficient than calling ``embed_text`` in a loop because the
        model processes all inputs in one forward pass.

        Args:
            texts: List of texts to embed.

        Returns:
            List of EmbeddingResult (one per input text), or empty list if
            the provider is unavailable.
        """
        if not self._model and not self._load_model():
            return []

        if not texts:
            return []

        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
        )
        return [
            EmbeddingResult(
                embedding=vec.tolist(),
                model=self.model_name,
                dimensions=self.dimensions,
                input_tokens=len(text.split()),
            )
            for vec, text in zip(vecs, texts)
        ]

    def embed_text_as_array(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding and return as a numpy array (not a list).

        Used internally by the semantic matching pipeline where the vector
        is immediately used for matrix multiplication, avoiding the
        list-to-array conversion overhead.

        Args:
            text: The text to embed.

        Returns:
            numpy ndarray of shape (D,), or None if unavailable.
        """
        if not self._model and not self._load_model():
            return None

        if not text:
            return np.zeros(self.dimensions, dtype="float32")

        return self._model.encode(text, normalize_embeddings=True)
