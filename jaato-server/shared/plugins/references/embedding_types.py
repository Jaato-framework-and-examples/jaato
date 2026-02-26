"""Embedding types for the references plugin.

Defines data structures for embedding results and matching.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class EmbeddingResult:
    """Result of computing a vector embedding.

    Attributes:
        embedding: The embedding vector as a list of floats.
        model: Name of the model that produced this embedding.
        dimensions: Dimensionality of the embedding vector.
        input_tokens: Approximate number of tokens in the input text.
    """
    embedding: List[float]
    model: str
    dimensions: int
    input_tokens: int = 0

    def to_dict(self) -> dict:
        """Serialize to a dict matching the tool response schema."""
        return {
            "embedding": self.embedding,
            "model": self.model,
            "dimensions": self.dimensions,
            "input_tokens": self.input_tokens,
        }


@dataclass
class SemanticMatch:
    """A reference matched by semantic similarity.

    Attributes:
        source_id: The reference source ID that matched.
        score: Cosine similarity score (0.0 to 1.0).
        embedding_index: Row index in the sidecar matrix.
    """
    source_id: str
    score: float
    embedding_index: int
