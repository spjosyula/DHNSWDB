"""Run all adaptive ef_search test scenarios.

This script executes all four test scenarios sequentially and generates
a comprehensive summary report comparing static vs adaptive ef_search.

Scenarios:
1. Large Document Corpus - 5000 docs, 1000 queries
2. High-Dimensional Embeddings - 768 dims, 800 queries
3. Dynamic Query Patterns - Shifting patterns, 1200 queries
4. Intent Diversity - 8 intents, 1500 queries

Usage:
    python adaptive_ef_tests/run_all_scenarios.py
"""

import sys
import os
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_scenario(scenario_name: str, scenario_module: str) -> dict:
    """Run a single scenario and return summary results.

    Args:
        scenario_name: Human-readable scenario name
        scenario_module: Module name to import and run

    Returns:
        Dictionary with scenario results
    """
    print("\n" + "=" * 100)
    print(f"RUNNING SCENARIO: {scenario_name}")
    print("=" * 100)

    start_time = time.time()

    try:
        # Import and run scenario
        module = __import__(scenario_module)
        module.main()

        elapsed = time.time() - start_time

        # Load results
        static_file = f"adaptive_ef_tests/results/{scenario_module.split('_', 1)[1]}_static.json"
        adaptive_file = f"adaptive_ef_tests/results/{scenario_module.split('_', 1)[1]}_adaptive.json"

        with open(static_file) as f:
            static_results = json.load(f)
        with open(adaptive_file) as f:
            adaptive_results = json.load(f)

        # Extract key metrics
        static_metrics = static_results["overall_metrics"]
        adaptive_metrics = adaptive_results["overall_metrics"]

        efficiency_improvement = (
            (adaptive_metrics["avg_efficiency"] - static_metrics["avg_efficiency"])
            / static_metrics["avg_efficiency"]
            * 100
        )

        return {
            "name": scenario_name,
            "status": "success",
            "elapsed_seconds": elapsed,
            "static_efficiency": static_metrics["avg_efficiency"],
            "adaptive_efficiency": adaptive_metrics["avg_efficiency"],
            "improvement_pct": efficiency_improvement,
            "num_queries": static_metrics["num_queries"],
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nERROR: Scenario failed with exception: {e}")
        return {
            "name": scenario_name,
            "status": "failed",
            "elapsed_seconds": elapsed,
            "error": str(e),
        }


def print_summary_report(results: list) -> None:
    """Print comprehensive summary report.

    Args:
        results: List of scenario result dictionaries
    """
    print("\n\n")
    print("=" * 100)
    print("COMPREHENSIVE SUMMARY REPORT: ADAPTIVE EF_SEARCH EVALUATION")
    print("=" * 100)

    # Overall statistics
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\nScenarios completed: {len(successful)}/{len(results)}")
    if failed:
        print(f"Scenarios failed: {len(failed)}")
        for r in failed:
            print(f"  - {r['name']}: {r['error']}")

    if not successful:
        print("\nNo successful scenarios to report.")
        return

    # Per-scenario summary
    print("\n" + "-" * 100)
    print("PER-SCENARIO RESULTS")
    print("-" * 100)

    print(
        f"\n{'Scenario':<40} | {'Queries':>8} | {'Static Eff':>12} | {'Adaptive Eff':>12} | {'Improvement':>12}"
    )
    print("-" * 100)

    for r in successful:
        print(
            f"{r['name']:<40} | {r['num_queries']:>8} | "
            f"{r['static_efficiency']:>12.2f} | {r['adaptive_efficiency']:>12.2f} | "
            f"{r['improvement_pct']:>11.1f}%"
        )

    # Aggregate analysis
    print("\n" + "-" * 100)
    print("AGGREGATE ANALYSIS")
    print("-" * 100)

    improvements = [r["improvement_pct"] for r in successful]
    avg_improvement = sum(improvements) / len(improvements)
    min_improvement = min(improvements)
    max_improvement = max(improvements)

    total_queries = sum(r["num_queries"] for r in successful)
    total_time = sum(r["elapsed_seconds"] for r in successful)

    print(f"\nTotal queries tested: {total_queries}")
    print(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\nEfficiency improvements:")
    print(f"  Average: {avg_improvement:.1f}%")
    print(f"  Minimum: {min_improvement:.1f}%")
    print(f"  Maximum: {max_improvement:.1f}%")

    # Verdict
    print("\n" + "=" * 100)
    print("FINAL VERDICT")
    print("=" * 100)

    if avg_improvement > 5.0:
        print(f"\nAdaptive ef_search shows {avg_improvement:.1f}% average efficiency improvement.")
        print("STRONG RECOMMENDATION: Adaptive learning provides significant benefit.")
        print("The Q-learning approach successfully optimizes ef_search per query intent.")
    elif avg_improvement > 2.0:
        print(f"\nAdaptive ef_search shows {avg_improvement:.1f}% average efficiency improvement.")
        print("MODERATE RECOMMENDATION: Adaptive learning provides measurable benefit.")
        print("Consider enabling for workloads with diverse query patterns.")
    elif avg_improvement > 0:
        print(f"\nAdaptive ef_search shows {avg_improvement:.1f}% average efficiency improvement.")
        print("MARGINAL BENEFIT: Improvement exists but may not justify complexity.")
        print("Evaluate based on specific use case requirements.")
    else:
        print(f"\nAdaptive ef_search shows {avg_improvement:.1f}% average efficiency change.")
        print("NO BENEFIT: Static ef_search performs as well or better.")
        print("Adaptive learning overhead may not be worthwhile.")

    print("\n" + "=" * 100)


def main():
    """Run all scenarios and generate summary report."""
    print("\n" + "=" * 100)
    print("ADAPTIVE EF_SEARCH TEST SUITE")
    print("=" * 100)
    print("\nThis will run all 4 test scenarios:")
    print("  1. Large Document Corpus (5000 docs, 1000 queries)")
    print("  2. High-Dimensional Embeddings (768 dims, 800 queries)")
    print("  3. Dynamic Query Patterns (shifting patterns, 1200 queries)")
    print("  4. Intent Diversity (8 intents, 1500 queries)")
    print("\nTotal: ~4500 queries across diverse scenarios")
    print("Estimated time: 15-25 minutes (depending on hardware)")

    # Check dependencies
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\nERROR: sentence-transformers not installed!")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    print("\nPress Enter to continue or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(0)

    # Create results directory
    Path("adaptive_ef_tests/results").mkdir(parents=True, exist_ok=True)

    # Run all scenarios
    scenarios = [
        ("Scenario 1: Large Document Corpus", "scenario_1_large_corpus"),
        ("Scenario 2: High-Dimensional Embeddings", "scenario_2_high_dimensional"),
        ("Scenario 3: Dynamic Query Patterns", "scenario_3_dynamic_patterns"),
        ("Scenario 4: Intent Diversity", "scenario_4_intent_diversity"),
    ]

    results = []
    for scenario_name, scenario_module in scenarios:
        result = run_scenario(scenario_name, f"adaptive_ef_tests.{scenario_module}")
        results.append(result)

    # Generate summary report
    print_summary_report(results)

    # Save summary to file
    summary_file = "adaptive_ef_tests/results/SUMMARY_REPORT.json"
    with open(summary_file, 'w') as f:
        json.dump({"scenarios": results}, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")
    print("\nAll scenarios complete!")


if __name__ == "__main__":
    main()
