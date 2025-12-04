"""Benchmark metrics and aggregation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FactRetentionResult:
    """Result of testing a single fact's retention."""

    fact_id: str
    """Identifier of the fact tested."""

    category: str
    """Fact category (entity, number, date, decision)."""

    turn_index: int
    """Turn index where the fact originally appeared."""

    question: str
    """Verification question asked."""

    expected_answer: str
    """Expected correct answer."""

    actual_answer: str
    """Answer given by the model."""

    retained: bool
    """Whether the fact was correctly recalled."""

    confidence: float = 1.0
    """Confidence score (0-1) based on answer quality."""


@dataclass
class QualityMetrics:
    """Quality metrics from fact retention testing."""

    facts_tested: int
    """Total number of facts tested."""

    facts_retained: int
    """Number of facts correctly recalled."""

    retention_rate: float
    """Retention rate: facts_retained / facts_tested."""

    retention_by_category: Dict[str, float]
    """Retention rate by fact category."""

    retention_by_position: Dict[str, float]
    """Retention rate by position (early, middle, late)."""

    fact_results: List[FactRetentionResult]
    """Individual fact test results."""

    @classmethod
    def empty(cls) -> "QualityMetrics":
        """Create empty quality metrics (when quality testing is disabled)."""
        return cls(
            facts_tested=0,
            facts_retained=0,
            retention_rate=0.0,
            retention_by_category={},
            retention_by_position={},
            fact_results=[]
        )


@dataclass
class PluginRunMetrics:
    """Metrics from a single plugin run on a scenario."""

    plugin_name: str
    """Name of the GC plugin."""

    scenario_name: str
    """Name of the scenario."""

    # GC operation results
    success: bool
    """Whether GC completed successfully."""

    tokens_before: int
    """Token count before GC."""

    tokens_after: int
    """Token count after GC."""

    tokens_freed: int
    """Tokens freed by GC."""

    items_collected: int
    """Number of items (turns) collected."""

    trigger_reason: str
    """Why GC was triggered."""

    # Timing
    gc_duration_ms: float
    """GC operation duration in milliseconds."""

    # Quality (optional)
    quality_metrics: Optional[QualityMetrics] = None
    """Quality metrics if fact retention testing was enabled."""

    # Plugin-specific details
    details: Dict[str, Any] = field(default_factory=dict)
    """Plugin-specific details from GCResult."""

    error: Optional[str] = None
    """Error message if GC failed."""

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (lower is better compression)."""
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before

    @property
    def retention_rate(self) -> float:
        """Get retention rate from quality metrics, or 0 if not available."""
        if self.quality_metrics:
            return self.quality_metrics.retention_rate
        return 0.0


@dataclass
class ScenarioComparison:
    """Comparison of all plugins on a single scenario."""

    scenario_name: str
    """Name of the scenario."""

    plugin_results: Dict[str, PluginRunMetrics]
    """Results keyed by plugin name."""

    def best_by_tokens(self) -> str:
        """Get plugin that freed the most tokens."""
        return max(
            self.plugin_results.keys(),
            key=lambda p: self.plugin_results[p].tokens_freed
        )

    def best_by_retention(self) -> str:
        """Get plugin with highest fact retention rate."""
        return max(
            self.plugin_results.keys(),
            key=lambda p: self.plugin_results[p].retention_rate
        )

    def best_overall(
        self,
        token_weight: float = 0.3,
        quality_weight: float = 0.7
    ) -> str:
        """Get plugin with best weighted score."""
        def score(plugin_name: str) -> float:
            metrics = self.plugin_results[plugin_name]
            if not metrics.success:
                return 0.0

            # Normalize tokens freed (0-1 scale based on max in this scenario)
            max_freed = max(m.tokens_freed for m in self.plugin_results.values())
            token_score = metrics.tokens_freed / max_freed if max_freed > 0 else 0

            # Quality score is already 0-1
            quality_score = metrics.retention_rate

            return token_weight * token_score + quality_weight * quality_score

        return max(self.plugin_results.keys(), key=score)


@dataclass
class PluginSummary:
    """Aggregated metrics for a plugin across all scenarios."""

    plugin_name: str
    """Name of the plugin."""

    scenarios_run: int
    """Number of scenarios this plugin was tested on."""

    # Token efficiency
    avg_tokens_freed: float
    """Average tokens freed across scenarios."""

    avg_compression_ratio: float
    """Average compression ratio (tokens_after / tokens_before)."""

    total_tokens_freed: int
    """Total tokens freed across all scenarios."""

    # Quality
    avg_retention_rate: float
    """Average fact retention rate."""

    retention_by_scenario: Dict[str, float]
    """Retention rate per scenario."""

    # Reliability
    success_rate: float
    """Percentage of successful GC operations."""

    avg_gc_duration_ms: float
    """Average GC duration in milliseconds."""

    # Per-scenario breakdown
    scenario_results: Dict[str, PluginRunMetrics] = field(default_factory=dict)
    """Individual scenario results."""


@dataclass
class BenchmarkSummary:
    """Full benchmark summary across all scenarios and plugins."""

    # Configuration used
    model_name: str
    """Model used for testing."""

    gc_threshold_percent: float
    """GC threshold used."""

    preserve_recent_turns: int
    """Number of recent turns preserved."""

    # Results
    scenarios: Dict[str, ScenarioComparison]
    """Per-scenario comparisons."""

    plugin_summaries: Dict[str, PluginSummary]
    """Aggregated per-plugin summaries."""

    overall_ranking: List[Tuple[str, float]]
    """Plugins ranked by overall score (plugin_name, score)."""

    # Timing and metadata
    total_duration_s: float
    """Total benchmark duration in seconds."""

    total_llm_calls: int
    """Total LLM API calls made."""

    timestamp: str
    """When the benchmark was run."""

    quality_testing_enabled: bool
    """Whether fact retention testing was performed."""

    def get_winner(self) -> str:
        """Get the top-ranked plugin."""
        if not self.overall_ranking:
            return "unknown"
        return self.overall_ranking[0][0]


def calculate_plugin_summary(
    plugin_name: str,
    results: List[PluginRunMetrics]
) -> PluginSummary:
    """Calculate aggregated summary for a plugin."""
    if not results:
        return PluginSummary(
            plugin_name=plugin_name,
            scenarios_run=0,
            avg_tokens_freed=0.0,
            avg_compression_ratio=1.0,
            total_tokens_freed=0,
            avg_retention_rate=0.0,
            retention_by_scenario={},
            success_rate=0.0,
            avg_gc_duration_ms=0.0
        )

    successful = [r for r in results if r.success]
    success_rate = len(successful) / len(results) if results else 0.0

    total_freed = sum(r.tokens_freed for r in successful)
    avg_freed = total_freed / len(successful) if successful else 0.0

    compression_ratios = [r.compression_ratio for r in successful]
    avg_compression = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 1.0

    retention_rates = [r.retention_rate for r in results if r.quality_metrics]
    avg_retention = sum(retention_rates) / len(retention_rates) if retention_rates else 0.0

    retention_by_scenario = {
        r.scenario_name: r.retention_rate
        for r in results
        if r.quality_metrics
    }

    durations = [r.gc_duration_ms for r in results]
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    scenario_results = {r.scenario_name: r for r in results}

    return PluginSummary(
        plugin_name=plugin_name,
        scenarios_run=len(results),
        avg_tokens_freed=avg_freed,
        avg_compression_ratio=avg_compression,
        total_tokens_freed=total_freed,
        avg_retention_rate=avg_retention,
        retention_by_scenario=retention_by_scenario,
        success_rate=success_rate,
        avg_gc_duration_ms=avg_duration,
        scenario_results=scenario_results
    )


def calculate_overall_ranking(
    plugin_summaries: Dict[str, PluginSummary],
    token_weight: float = 0.3,
    quality_weight: float = 0.7
) -> List[Tuple[str, float]]:
    """Calculate overall plugin ranking."""
    scores: List[Tuple[str, float]] = []

    # Find max values for normalization
    max_freed = max(
        (ps.avg_tokens_freed for ps in plugin_summaries.values()),
        default=1.0
    )

    for name, summary in plugin_summaries.items():
        if summary.scenarios_run == 0:
            scores.append((name, 0.0))
            continue

        # Normalize token efficiency
        token_score = summary.avg_tokens_freed / max_freed if max_freed > 0 else 0

        # Quality score is already 0-1
        quality_score = summary.avg_retention_rate

        # Weight and combine
        total_score = (
            token_weight * token_score * summary.success_rate +
            quality_weight * quality_score
        )

        scores.append((name, total_score))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
