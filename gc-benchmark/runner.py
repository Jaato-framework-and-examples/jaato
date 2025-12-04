"""Main benchmark runner orchestrator."""

import sys
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.genai import types

from shared.jaato_client import JaatoClient
from shared.plugins.gc import GCConfig, GCPlugin, GCTriggerReason, load_gc_plugin

from config import BenchmarkConfig
from metrics import (
    BenchmarkSummary,
    PluginRunMetrics,
    QualityMetrics,
    ScenarioComparison,
    calculate_overall_ranking,
    calculate_plugin_summary,
)
from quality import QualityTester
from reporters import ConsoleReporter, JsonReporter, Reporter
from scenarios import BenchmarkScenario, ScenarioFactory


class BenchmarkRunner:
    """Main orchestrator for GC plugin benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        self._config = config
        self._client: Optional[JaatoClient] = None
        self._quality_tester: Optional[QualityTester] = None
        self._reporters: List[Reporter] = []
        self._llm_call_count = 0

    def initialize(self) -> None:
        """Initialize LLM client and reporters."""
        self._config.validate()

        # Connect to Vertex AI
        if self._config.verbose:
            print(f"Connecting to {self._config.model_name}...")

        self._client = JaatoClient()
        self._client.connect(
            self._config.project_id,
            self._config.location,
            self._config.model_name
        )

        # Create quality tester with a generate function
        if self._config.enable_fact_retention:
            self._quality_tester = QualityTester(self._generate_with_history)

        # Create reporters
        for fmt in self._config.output_formats:
            if fmt == "console":
                self._reporters.append(ConsoleReporter())
            elif fmt == "json":
                self._reporters.append(JsonReporter(self._config.output_path))

        if self._config.verbose:
            print("Initialization complete")

    def run(self) -> BenchmarkSummary:
        """Run the full benchmark suite.

        Returns:
            BenchmarkSummary with all results.
        """
        start_time = time.perf_counter()

        scenarios = self._load_scenarios()
        plugins = self._load_plugins()

        if self._config.verbose:
            print(f"Running {len(scenarios)} scenarios with {len(plugins)} plugins")

        results: Dict[str, ScenarioComparison] = {}

        for scenario in scenarios:
            if self._config.verbose:
                print(f"\nScenario: {scenario.name}")

            scenario_results: Dict[str, PluginRunMetrics] = {}

            for plugin_name, plugin in plugins.items():
                if self._config.verbose:
                    print(f"  Plugin: {plugin_name}")

                metrics = self._run_single(plugin, scenario)
                scenario_results[plugin_name] = metrics

            results[scenario.name] = ScenarioComparison(
                scenario_name=scenario.name,
                plugin_results=scenario_results
            )

        # Calculate summaries
        plugin_summaries = self._calculate_summaries(results, plugins.keys())
        overall_ranking = calculate_overall_ranking(plugin_summaries)

        total_duration = time.perf_counter() - start_time

        summary = BenchmarkSummary(
            model_name=self._config.model_name,
            gc_threshold_percent=self._config.gc_threshold_percent,
            preserve_recent_turns=self._config.preserve_recent_turns,
            scenarios=results,
            plugin_summaries=plugin_summaries,
            overall_ranking=overall_ranking,
            total_duration_s=total_duration,
            total_llm_calls=self._llm_call_count,
            timestamp=datetime.now().isoformat(),
            quality_testing_enabled=self._config.enable_fact_retention
        )

        # Report results
        for reporter in self._reporters:
            reporter.report(summary)

        return summary

    def _run_single(
        self,
        plugin: GCPlugin,
        scenario: BenchmarkScenario
    ) -> PluginRunMetrics:
        """Run a single plugin on a single scenario.

        Args:
            plugin: The GC plugin to test.
            scenario: The scenario to run.

        Returns:
            PluginRunMetrics with results.
        """
        # Create GC config (same for all plugins for fair comparison)
        gc_config = GCConfig(
            threshold_percent=self._config.gc_threshold_percent,
            preserve_recent_turns=self._config.preserve_recent_turns,
            auto_trigger=False  # Manual trigger for controlled testing
        )

        # Build context usage dict (simulated)
        context_usage = self._make_context_usage(scenario)

        # Time the GC operation
        start_time = time.perf_counter()
        try:
            new_history, result = plugin.collect(
                scenario.history,
                context_usage,
                gc_config,
                GCTriggerReason.MANUAL
            )
            gc_duration = (time.perf_counter() - start_time) * 1000
            error = result.error
        except Exception as e:
            gc_duration = (time.perf_counter() - start_time) * 1000
            return PluginRunMetrics(
                plugin_name=plugin.name,
                scenario_name=scenario.name,
                success=False,
                tokens_before=0,
                tokens_after=0,
                tokens_freed=0,
                items_collected=0,
                trigger_reason=GCTriggerReason.MANUAL.value,
                gc_duration_ms=gc_duration,
                error=str(e)
            )

        # Test quality if enabled and GC succeeded
        quality_metrics: Optional[QualityMetrics] = None
        if (self._config.enable_fact_retention and
            result.success and
            self._quality_tester and
            scenario.embedded_facts):

            if self._config.verbose:
                print(f"    Testing fact retention ({len(scenario.embedded_facts)} facts)...")

            quality_metrics = self._quality_tester.test_fact_retention(
                new_history,
                scenario.embedded_facts,
                verbose=self._config.verbose
            )

            if self._config.verbose:
                print(f"    Retention rate: {quality_metrics.retention_rate:.0%}")

        return PluginRunMetrics(
            plugin_name=plugin.name,
            scenario_name=scenario.name,
            success=result.success,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            tokens_freed=result.tokens_freed,
            items_collected=result.items_collected,
            trigger_reason=result.trigger_reason.value,
            gc_duration_ms=gc_duration,
            quality_metrics=quality_metrics,
            details=result.details,
            error=error
        )

    def _load_scenarios(self) -> List[BenchmarkScenario]:
        """Load benchmark scenarios based on config."""
        if "all" in self._config.scenarios:
            return ScenarioFactory.get_all_scenarios()

        scenarios = []
        for name in self._config.scenarios:
            scenarios.append(ScenarioFactory.get_scenario(name))
        return scenarios

    def _load_plugins(self) -> Dict[str, GCPlugin]:
        """Load and configure GC plugins."""
        plugins: Dict[str, GCPlugin] = {}

        for name in self._config.plugins:
            if self._config.verbose:
                print(f"Loading plugin: {name}")

            plugin = load_gc_plugin(name)

            # Get plugin-specific config
            plugin_config = self._config.plugin_configs.get(name, {})

            # For summarize/hybrid, inject real summarizer
            if name in ("gc_summarize", "gc_hybrid"):
                plugin_config["summarizer"] = self._create_summarizer()

            plugin.initialize(plugin_config)
            plugins[name] = plugin

        return plugins

    def _create_summarizer(self) -> Callable[[str], str]:
        """Create a summarizer function using the LLM."""
        def summarize(text: str) -> str:
            self._llm_call_count += 1

            prompt = (
                "Summarize the following conversation concisely. "
                "Focus on key information, decisions, and important context.\n\n"
                f"{text}\n\n"
                "Summary:"
            )

            # Use a fresh client for summarization to avoid state issues
            response = self._client.send_message(prompt)
            self._client.reset_session()
            return response

        return summarize

    def _generate_with_history(
        self,
        history: List[types.Content],
        prompt: str
    ) -> str:
        """Generate a response using history as context.

        Args:
            history: Conversation history to use as context.
            prompt: The prompt/question to ask.

        Returns:
            Model response string.
        """
        self._llm_call_count += 1

        # Reset session and inject history
        self._client.reset_session(history)

        # Send the prompt
        response = self._client.send_message(prompt)

        # Reset again to clean state
        self._client.reset_session()

        return response

    def _make_context_usage(self, scenario: BenchmarkScenario) -> Dict[str, Any]:
        """Create a context usage dict for a scenario.

        This simulates what JaatoClient.get_context_usage() would return.
        """
        # Simple estimation: assume we're at the threshold
        from shared.plugins.gc.utils import estimate_history_tokens

        estimated_tokens = estimate_history_tokens(scenario.history)

        # Assume a typical context limit
        context_limit = 1_000_000  # 1M tokens typical for Gemini

        return {
            "model": self._config.model_name,
            "context_limit": context_limit,
            "total_tokens": estimated_tokens,
            "prompt_tokens": estimated_tokens,
            "output_tokens": 0,
            "turns": scenario.target_turns,
            "percent_used": (estimated_tokens / context_limit) * 100,
            "tokens_remaining": context_limit - estimated_tokens
        }

    def _calculate_summaries(
        self,
        results: Dict[str, ScenarioComparison],
        plugin_names: List[str]
    ) -> Dict[str, Any]:
        """Calculate per-plugin summaries from results."""
        # Group results by plugin
        by_plugin: Dict[str, List[PluginRunMetrics]] = {
            name: [] for name in plugin_names
        }

        for comparison in results.values():
            for plugin_name, metrics in comparison.plugin_results.items():
                by_plugin[plugin_name].append(metrics)

        # Calculate summary for each plugin
        return {
            name: calculate_plugin_summary(name, metrics_list)
            for name, metrics_list in by_plugin.items()
        }

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            self._client.disconnect()
