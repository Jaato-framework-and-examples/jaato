#!/usr/bin/env python3
"""CLI entry point for GC plugin benchmark.

Usage:
    .venv/bin/python gc-benchmark/run_benchmark.py --env-file .env
    .venv/bin/python gc-benchmark/run_benchmark.py --no-quality --plugins gc_truncate
"""

import argparse
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from config import BenchmarkConfig
from runner import BenchmarkRunner


def main() -> int:
    """Run the GC plugin benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark GC plugins for context quality and efficiency"
    )

    parser.add_argument(
        "--plugins",
        default="all",
        help="Comma-separated list of plugins to benchmark, or 'all' (default: all)"
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Comma-separated list of scenarios, or 'all' (default: all)"
    )
    parser.add_argument(
        "--output",
        default="gc_benchmark_results.json",
        help="Path for JSON output (default: gc_benchmark_results.json)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="GC trigger threshold percentage (default: 80.0)"
    )
    parser.add_argument(
        "--preserve-turns",
        type=int,
        default=5,
        help="Number of recent turns to preserve (default: 5)"
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to environment file (default: .env)"
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip fact retention testing (faster, tokens only)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output during benchmark"
    )
    parser.add_argument(
        "--console-only",
        action="store_true",
        help="Only output to console, skip JSON file"
    )

    args = parser.parse_args()

    # Load environment
    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)
    else:
        print(f"Warning: Environment file '{args.env_file}' not found")

    # Check required environment variables
    project_id = os.environ.get("PROJECT_ID")
    location = os.environ.get("LOCATION")

    if not project_id:
        print("Error: PROJECT_ID environment variable is required")
        return 1
    if not location:
        print("Error: LOCATION environment variable is required")
        return 1

    # Parse plugins and scenarios
    if args.plugins == "all":
        plugins = ["gc_truncate", "gc_summarize", "gc_hybrid"]
    else:
        plugins = [p.strip() for p in args.plugins.split(",")]

    if args.scenarios == "all":
        scenarios = ["all"]
    else:
        scenarios = [s.strip() for s in args.scenarios.split(",")]

    # Determine output formats
    output_formats = ["console"]
    if not args.console_only:
        output_formats.append("json")

    # Build config
    config = BenchmarkConfig(
        plugins=plugins,
        scenarios=scenarios,
        gc_threshold_percent=args.threshold,
        preserve_recent_turns=args.preserve_turns,
        project_id=project_id,
        location=location,
        model_name=os.environ.get("MODEL_NAME", "gemini-2.5-flash"),
        enable_fact_retention=not args.no_quality,
        output_formats=output_formats,
        output_path=args.output,
        verbose=args.verbose
    )

    # Run benchmark
    runner = BenchmarkRunner(config)

    try:
        runner.initialize()
        summary = runner.run()

        # Return success if all GC operations succeeded
        all_success = all(
            metrics.success
            for comparison in summary.scenarios.values()
            for metrics in comparison.plugin_results.values()
        )
        return 0 if all_success else 1

    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        runner.shutdown()


if __name__ == "__main__":
    sys.exit(main())
