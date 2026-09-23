"""Embedding types for the references plugin.

Defines data structures for embedding results and semantic matching,
plus protocol definitions for injectable embedding providers.

Concrete implementations (``LocalEmbeddingProvider``, ``SemanticMatcher``)
are not included in jaato-server — they are provided by external packages
(e.g. jaato-premium) that register via the ``jaato.embedding`` entry point
group. See ``discover_embedding_subsystem()`` for the discovery mechanism.
"""

import importlib.metadata
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, runtime_checkable

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Protocols — define the contract for injectable embedding subsystem
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingProviderProtocol(Protocol):
    """Protocol for embedding providers that compute vector embeddings.

    Concrete implementations (e.g. ``LocalEmbeddingProvider`` using
    sentence-transformers) are registered via the ``jaato.embedding``
    entry point group and discovered at runtime by the references plugin.

    Lifecycle:
        1. Factory function called by ``discover_embedding_subsystem()``
           with a config dict containing ``embedding_model``,
           ``embedding_max_input_tokens``, and ``eager_load`` keys.
        2. Provider is assigned to the references plugin and optionally
           to a ``SemanticMatcherProtocol`` via ``set_provider()``.
        3. ``load_model()`` is called when the model needs to be loaded
           eagerly (e.g. for semantic matching at enrichment time).
        4. ``embed_text()`` / ``embed_batch()`` / ``embed_text_as_array()``
           are called during tool execution and enrichment.
    """

    @property
    def model_name(self) -> str:
        """The sentence-transformers model identifier."""
        ...

    @property
    def dimensions(self) -> int:
        """Embedding vector dimensionality (set after model load)."""
        ...

    @property
    def available(self) -> bool:
        """Whether the provider is ready to compute embeddings."""
        ...

    def load_model(self) -> bool:
        """Load the embedding model into memory.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        ...

    def embed_text(self, text: str) -> Optional[EmbeddingResult]:
        """Compute embedding for a single text string.

        Args:
            text: The text to embed.

        Returns:
            EmbeddingResult with the embedding vector and metadata,
            or None if the provider is unavailable.
        """
        ...

    def embed_batch(self, texts: List[str]) -> List[Optional[EmbeddingResult]]:
        """Compute embeddings for multiple texts in a single batch.

        Args:
            texts: List of texts to embed.

        Returns:
            List of EmbeddingResult (one per input text), or empty list
            if the provider is unavailable.
        """
        ...

    def embed_text_as_array(self, text: str) -> Optional[Any]:
        """Compute embedding and return as a numpy array.

        Used internally by the semantic matching pipeline where the
        vector is immediately used for matrix multiplication.

        Args:
            text: The text to embed.

        Returns:
            numpy ndarray of shape (D,), or None if unavailable.
            Typed as Any to avoid requiring numpy in core.
        """
        ...


@runtime_checkable
class SemanticMatcherProtocol(Protocol):
    """Protocol for semantic matchers that search precomputed embedding indices.

    Concrete implementations (e.g. ``SemanticMatcher`` using numpy matrix
    operations) are registered via the ``jaato.embedding`` entry point group
    alongside an ``EmbeddingProviderProtocol``.

    Lifecycle:
        1. Factory function called by ``discover_embedding_subsystem()``
           with no arguments — returns an uninitialized matcher.
        2. ``set_provider()`` is called to wire the embedding provider.
        3. ``load_index()`` is called to load the precomputed sidecar matrix.
        4. ``validate_model()`` checks provider/index model consistency.
        5. ``embed_and_match()`` / ``score_sources()`` / ``find_matches()``
           are called during enrichment.
    """

    @property
    def available(self) -> bool:
        """Whether semantic matching can be performed."""
        ...

    def validate_model(self, provider_model: str) -> bool:
        """Check that the provider's model matches the index's model.

        Args:
            provider_model: The model name configured on the provider.

        Returns:
            True if models match, False otherwise.
        """
        ...

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
            embedding_model: Model name from the references index.
            embedding_dimensions: Expected dimensionality.
            index_to_source_id: Mapping from matrix row to source ID.

        Returns:
            True if loaded successfully, False otherwise.
        """
        ...

    def set_provider(self, provider: EmbeddingProviderProtocol) -> None:
        """Set the embedding provider for query-time embedding.

        Args:
            provider: An embedding provider instance.
        """
        ...

    def score_sources(
        self,
        query_vec: Any,
        source_ids: Set[str],
    ) -> Dict[str, float]:
        """Compute cosine similarity for specific source IDs.

        Used by the tag veto logic in hybrid mode.

        Args:
            query_vec: numpy array of shape (D,), normalized.
            source_ids: Source IDs to score.

        Returns:
            Dict mapping source_id to cosine similarity score.
        """
        ...

    def find_matches(
        self,
        query_vec: Any,
        threshold: float = 0.75,
        top_k: int = 3,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[SemanticMatch]:
        """Find references whose embeddings are similar to the query.

        Args:
            query_vec: numpy array of shape (D,), normalized.
            threshold: Minimum cosine similarity to include.
            top_k: Maximum number of matches to return.
            exclude_ids: Source IDs to exclude from results.

        Returns:
            List of SemanticMatch sorted by descending score.
        """
        ...

    def embed_and_match(
        self,
        content: str,
        threshold: float = 0.75,
        top_k: int = 3,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[SemanticMatch]:
        """Embed content and find matching references in one call.

        Args:
            content: The text to embed and match.
            threshold: Minimum cosine similarity.
            top_k: Maximum number of matches.
            exclude_ids: Source IDs to exclude.

        Returns:
            List of SemanticMatch sorted by descending score.
        """
        ...


# ---------------------------------------------------------------------------
# Entry point discovery
# ---------------------------------------------------------------------------

_ENTRY_POINT_GROUP = "jaato.embedding"

# Cached entry point map — populated on first discovery call.
# Entry points don't change while the process is running.
_cached_ep_map: Optional[Dict[str, Any]] = None


def discover_embedding_subsystem(
    config: Dict[str, Any],
) -> Tuple[Optional[EmbeddingProviderProtocol], Optional[SemanticMatcherProtocol]]:
    """Discover embedding provider and semantic matcher via entry points.

    External packages (e.g. jaato-premium) register factory functions under
    the ``jaato.embedding`` entry point group::

        [project.entry-points."jaato.embedding"]
        embedding_provider = "my_package:create_embedding_provider"
        semantic_matcher = "my_package:create_semantic_matcher"

    Factory signatures:
        - ``embedding_provider``: ``(config: Dict[str, Any]) -> EmbeddingProviderProtocol``
        - ``semantic_matcher``: ``() -> SemanticMatcherProtocol``

    Args:
        config: Plugin configuration dict passed to the embedding provider
            factory. Expected keys: ``embedding_model``, ``embedding_max_input_tokens``,
            ``eager_load``.

    Returns:
        Tuple of (provider_or_none, matcher_or_none). Both are None when
        no entry point is registered or loading fails.
    """
    global _cached_ep_map

    provider: Optional[EmbeddingProviderProtocol] = None
    matcher: Optional[SemanticMatcherProtocol] = None

    if _cached_ep_map is None:
        try:
            if sys.version_info >= (3, 10):
                eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
            else:
                all_eps = importlib.metadata.entry_points()
                eps = all_eps.get(_ENTRY_POINT_GROUP, [])
        except Exception:
            _cached_ep_map = {}
            return (None, None)
        _cached_ep_map = {ep.name: ep for ep in eps}

    ep_map = _cached_ep_map

    # Discover embedding provider
    ep = ep_map.get("embedding_provider")
    if ep is not None:
        try:
            factory = ep.load()
            candidate = factory(config)
            if isinstance(candidate, EmbeddingProviderProtocol):
                provider = candidate
            else:
                logger.warning(
                    "Entry point '%s' returned %s which does not implement "
                    "EmbeddingProviderProtocol",
                    ep.value, type(candidate).__name__,
                )
        except Exception as exc:
            logger.warning(
                "Failed to load embedding_provider entry point '%s': %s",
                ep.value, exc,
            )

    # Discover semantic matcher
    ep = ep_map.get("semantic_matcher")
    if ep is not None:
        try:
            factory = ep.load()
            candidate = factory()
            if isinstance(candidate, SemanticMatcherProtocol):
                matcher = candidate
            else:
                logger.warning(
                    "Entry point '%s' returned %s which does not implement "
                    "SemanticMatcherProtocol",
                    ep.value, type(candidate).__name__,
                )
        except Exception as exc:
            logger.warning(
                "Failed to load semantic_matcher entry point '%s': %s",
                ep.value, exc,
            )

    return (provider, matcher)


def create_semantic_matcher() -> Optional[SemanticMatcherProtocol]:
    """Create a fresh semantic matcher instance via entry point discovery.

    Used by the references plugin to attach one matcher per bundle — each
    bundle has its own sidecar index, so matcher instances cannot be
    shared across bundles. This thin wrapper looks up the same
    ``semantic_matcher`` entry point as :func:`discover_embedding_subsystem`
    and invokes its factory with no arguments.

    Returns:
        A fresh :class:`SemanticMatcherProtocol` instance, or ``None`` when
        no entry point is registered or the factory fails.
    """
    global _cached_ep_map

    if _cached_ep_map is None:
        # Populate the cache using the same mechanism as discover_*.
        try:
            if sys.version_info >= (3, 10):
                eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
            else:
                all_eps = importlib.metadata.entry_points()
                eps = all_eps.get(_ENTRY_POINT_GROUP, [])
        except Exception:
            _cached_ep_map = {}
            return None
        _cached_ep_map = {ep.name: ep for ep in eps}

    ep = _cached_ep_map.get("semantic_matcher")
    if ep is None:
        return None
    try:
        factory = ep.load()
        candidate = factory()
    except Exception as exc:
        logger.warning(
            "Failed to instantiate semantic_matcher entry point '%s': %s",
            ep.value, exc,
        )
        return None
    if not isinstance(candidate, SemanticMatcherProtocol):
        logger.warning(
            "Entry point '%s' returned %s which does not implement "
            "SemanticMatcherProtocol",
            ep.value, type(candidate).__name__,
        )
        return None
    return candidate
