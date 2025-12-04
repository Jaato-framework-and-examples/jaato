"""Benchmark configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for GC benchmark runs."""

    # Plugins to benchmark
    plugins: List[str] = field(default_factory=lambda: [
        "gc_truncate", "gc_summarize", "gc_hybrid"
    ])

    # Scenarios to run
    scenarios: List[str] = field(default_factory=lambda: ["all"])

    # GC trigger settings (same for all plugins for fair comparison)
    gc_threshold_percent: float = 80.0
    preserve_recent_turns: int = 5

    # Plugin-specific configuration overrides
    plugin_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # LLM settings for summarization and quality testing
    project_id: str = ""
    location: str = ""
    model_name: str = "gemini-2.5-flash"

    # Quality testing
    enable_fact_retention: bool = True
    fact_question_count: int = 10

    # Output
    output_formats: List[str] = field(default_factory=lambda: ["console", "json"])
    output_path: str = "gc_benchmark_results.json"

    # Runtime options
    verbose: bool = False

    def validate(self) -> None:
        """Validate configuration."""
        if not self.project_id:
            raise ValueError("project_id is required")
        if not self.location:
            raise ValueError("location is required")
        if not self.plugins:
            raise ValueError("At least one plugin must be specified")
        if self.gc_threshold_percent <= 0 or self.gc_threshold_percent > 100:
            raise ValueError("gc_threshold_percent must be between 0 and 100")
        if self.preserve_recent_turns < 0:
            raise ValueError("preserve_recent_turns must be non-negative")
