"""Benchmark result reporters."""

import json
from dataclasses import asdict
from typing import Protocol

from metrics import BenchmarkSummary


class Reporter(Protocol):
    """Protocol for benchmark result reporters."""

    def report(self, summary: BenchmarkSummary) -> None:
        """Output the benchmark results."""
        ...


class ConsoleReporter:
    """Human-readable console output."""

    def report(self, summary: BenchmarkSummary) -> None:
        """Print benchmark results to console."""
        self._print_header(summary)
        self._print_scenario_table(summary)
        self._print_plugin_summary(summary)

        if summary.quality_testing_enabled:
            self._print_quality_breakdown(summary)

        self._print_winner(summary)

    def _print_header(self, summary: BenchmarkSummary) -> None:
        """Print benchmark header."""
        print()
        print("=" * 80)
        print("  GC Plugin Benchmark Results")
        print("=" * 80)
        print(f"  Model: {summary.model_name}")
        print(f"  Scenarios: {len(summary.scenarios)}")
        print(f"  Plugins: {', '.join(summary.plugin_summaries.keys())}")
        print(f"  Duration: {summary.total_duration_s:.1f}s")
        print(f"  LLM Calls: {summary.total_llm_calls}")
        print(f"  Quality Testing: {'Enabled' if summary.quality_testing_enabled else 'Disabled'}")
        print()

    def _print_scenario_table(self, summary: BenchmarkSummary) -> None:
        """Print per-scenario comparison table."""
        print("-" * 80)
        if summary.quality_testing_enabled:
            print(f"{'Scenario':<20} {'Plugin':<15} {'Tokens Freed':>12} "
                  f"{'Retention':>10} {'Time (ms)':>10}")
        else:
            print(f"{'Scenario':<20} {'Plugin':<15} {'Tokens Freed':>12} "
                  f"{'Compression':>12} {'Time (ms)':>10}")
        print("-" * 80)

        for scenario_name, comparison in sorted(summary.scenarios.items()):
            for plugin_name, metrics in sorted(comparison.plugin_results.items()):
                if summary.quality_testing_enabled:
                    retention = f"{metrics.retention_rate:.0%}" if metrics.quality_metrics else "N/A"
                    print(f"{scenario_name:<20} {plugin_name:<15} "
                          f"{metrics.tokens_freed:>12} {retention:>10} "
                          f"{metrics.gc_duration_ms:>10.1f}")
                else:
                    compression = f"{metrics.compression_ratio:.2f}"
                    print(f"{scenario_name:<20} {plugin_name:<15} "
                          f"{metrics.tokens_freed:>12} {compression:>12} "
                          f"{metrics.gc_duration_ms:>10.1f}")
        print()

    def _print_plugin_summary(self, summary: BenchmarkSummary) -> None:
        """Print aggregate plugin comparison."""
        print("-" * 80)
        print("  Plugin Summary")
        print("-" * 80)

        if summary.quality_testing_enabled:
            print(f"{'Plugin':<20} {'Avg Tokens Freed':>16} "
                  f"{'Avg Retention':>14} {'Success Rate':>13} {'Avg Time (ms)':>14}")
            print("-" * 80)

            for name, ps in sorted(summary.plugin_summaries.items()):
                print(f"{name:<20} {ps.avg_tokens_freed:>16.0f} "
                      f"{ps.avg_retention_rate:>13.0%} {ps.success_rate:>12.0%} "
                      f"{ps.avg_gc_duration_ms:>14.1f}")
        else:
            print(f"{'Plugin':<20} {'Avg Tokens Freed':>16} "
                  f"{'Avg Compression':>16} {'Success Rate':>13} {'Avg Time (ms)':>14}")
            print("-" * 80)

            for name, ps in sorted(summary.plugin_summaries.items()):
                print(f"{name:<20} {ps.avg_tokens_freed:>16.0f} "
                      f"{ps.avg_compression_ratio:>15.2f} {ps.success_rate:>12.0%} "
                      f"{ps.avg_gc_duration_ms:>14.1f}")
        print()

    def _print_quality_breakdown(self, summary: BenchmarkSummary) -> None:
        """Print quality metrics breakdown."""
        print("-" * 80)
        print("  Quality Breakdown (Fact Retention by Category)")
        print("-" * 80)

        # Collect all categories across all plugins
        all_categories = set()
        for ps in summary.plugin_summaries.values():
            for scenario_metrics in ps.scenario_results.values():
                if scenario_metrics.quality_metrics:
                    all_categories.update(
                        scenario_metrics.quality_metrics.retention_by_category.keys()
                    )

        if not all_categories:
            print("  No category data available")
            print()
            return

        # Header
        categories = sorted(all_categories)
        header = f"{'Plugin':<20}"
        for cat in categories:
            header += f" {cat:>10}"
        print(header)
        print("-" * 80)

        # Per-plugin category averages
        for name, ps in sorted(summary.plugin_summaries.items()):
            category_totals: dict = {c: [] for c in categories}

            for scenario_metrics in ps.scenario_results.values():
                if scenario_metrics.quality_metrics:
                    for cat, rate in scenario_metrics.quality_metrics.retention_by_category.items():
                        category_totals[cat].append(rate)

            row = f"{name:<20}"
            for cat in categories:
                values = category_totals.get(cat, [])
                if values:
                    avg = sum(values) / len(values)
                    row += f" {avg:>9.0%}"
                else:
                    row += f" {'N/A':>10}"
            print(row)

        print()

    def _print_winner(self, summary: BenchmarkSummary) -> None:
        """Print the winning plugin."""
        print("=" * 80)

        if summary.overall_ranking:
            winner_name, winner_score = summary.overall_ranking[0]
            print(f"  WINNER: {winner_name} (score: {winner_score:.2f})")

            if len(summary.overall_ranking) > 1:
                print()
                print("  Full Ranking:")
                for i, (name, score) in enumerate(summary.overall_ranking, 1):
                    print(f"    {i}. {name}: {score:.2f}")
        else:
            print("  No ranking available")

        print("=" * 80)
        print()


class JsonReporter:
    """Machine-readable JSON output."""

    def __init__(self, output_path: str):
        """Initialize JSON reporter.

        Args:
            output_path: Path to write JSON output.
        """
        self._output_path = output_path

    def report(self, summary: BenchmarkSummary) -> None:
        """Write benchmark results to JSON file."""
        output = self._build_output(summary)

        with open(self._output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Results written to {self._output_path}")

    def _build_output(self, summary: BenchmarkSummary) -> dict:
        """Build JSON-serializable output structure."""
        return {
            "meta": {
                "timestamp": summary.timestamp,
                "model": summary.model_name,
                "gc_threshold_percent": summary.gc_threshold_percent,
                "preserve_recent_turns": summary.preserve_recent_turns,
                "total_duration_s": summary.total_duration_s,
                "total_llm_calls": summary.total_llm_calls,
                "quality_testing_enabled": summary.quality_testing_enabled
            },
            "scenarios": self._serialize_scenarios(summary),
            "plugin_summaries": self._serialize_plugin_summaries(summary),
            "ranking": [
                {"plugin": name, "score": score}
                for name, score in summary.overall_ranking
            ]
        }

    def _serialize_scenarios(self, summary: BenchmarkSummary) -> dict:
        """Serialize scenario results."""
        scenarios = {}

        for scenario_name, comparison in summary.scenarios.items():
            scenarios[scenario_name] = {
                "best_by_tokens": comparison.best_by_tokens(),
                "best_by_retention": comparison.best_by_retention() if summary.quality_testing_enabled else None,
                "plugins": {}
            }

            for plugin_name, metrics in comparison.plugin_results.items():
                plugin_data = {
                    "success": metrics.success,
                    "tokens_before": metrics.tokens_before,
                    "tokens_after": metrics.tokens_after,
                    "tokens_freed": metrics.tokens_freed,
                    "compression_ratio": metrics.compression_ratio,
                    "items_collected": metrics.items_collected,
                    "gc_duration_ms": metrics.gc_duration_ms,
                    "error": metrics.error
                }

                if metrics.quality_metrics:
                    plugin_data["quality"] = {
                        "facts_tested": metrics.quality_metrics.facts_tested,
                        "facts_retained": metrics.quality_metrics.facts_retained,
                        "retention_rate": metrics.quality_metrics.retention_rate,
                        "retention_by_category": metrics.quality_metrics.retention_by_category,
                        "retention_by_position": metrics.quality_metrics.retention_by_position
                    }

                scenarios[scenario_name]["plugins"][plugin_name] = plugin_data

        return scenarios

    def _serialize_plugin_summaries(self, summary: BenchmarkSummary) -> dict:
        """Serialize plugin summaries."""
        return {
            name: {
                "scenarios_run": ps.scenarios_run,
                "avg_tokens_freed": ps.avg_tokens_freed,
                "avg_compression_ratio": ps.avg_compression_ratio,
                "total_tokens_freed": ps.total_tokens_freed,
                "avg_retention_rate": ps.avg_retention_rate,
                "retention_by_scenario": ps.retention_by_scenario,
                "success_rate": ps.success_rate,
                "avg_gc_duration_ms": ps.avg_gc_duration_ms
            }
            for name, ps in summary.plugin_summaries.items()
        }
