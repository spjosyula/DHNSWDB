"""Run all epsilon decay test scenarios.

This script runs all four scenarios sequentially and generates a summary report.
"""

import sys
import os
import subprocess
import time
from pathlib import Path


def run_scenario(scenario_name: str, script_path: str) -> bool:
    """Run a single scenario script.

    Args:
        scenario_name: Human-readable scenario name
        script_path: Path to the scenario script

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 100)
    print(f"RUNNING {scenario_name.upper()}")
    print("=" * 100)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(script_path)),
            capture_output=False,
            text=True,
        )

        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        if result.returncode == 0:
            print(f"\n{scenario_name} completed successfully in {minutes}m {seconds}s")
            return True
        else:
            print(f"\n{scenario_name} FAILED with return code {result.returncode}")
            return False

    except Exception as e:
        print(f"\n{scenario_name} FAILED with exception: {e}")
        return False


def main():
    """Run all scenarios and generate summary."""
    print("\n" + "=" * 100)
    print("EPSILON DECAY TEST SUITE - ALL SCENARIOS")
    print("=" * 100)
    print("\nThis will run all 4 test scenarios:")
    print("  1. Large Action Space (20 ef candidates, 500 queries)")
    print("  2. More Intents (10 intents, 600 queries)")
    print("  3. Long Horizon (5000 queries)")
    print("  4. Non-Stationary Environment (1000 queries)")
    print("\nTotal queries across all scenarios: ~7100")
    print("Estimated runtime: 30-45 minutes")
    print("\n" + "=" * 100)

    input("\nPress Enter to start all scenarios, or Ctrl+C to cancel...")

    # Get script directory
    script_dir = Path(__file__).parent

    # Define scenarios
    scenarios = [
        ("Scenario 1: Large Action Space", script_dir / "scenario_1_large_action_space.py"),
        ("Scenario 2: More Intents", script_dir / "scenario_2_more_intents.py"),
        ("Scenario 3: Long Horizon", script_dir / "scenario_3_long_horizon.py"),
        ("Scenario 4: Non-Stationary", script_dir / "scenario_4_non_stationary.py"),
    ]

    # Track results
    results = {}
    overall_start = time.time()

    # Run each scenario
    for scenario_name, script_path in scenarios:
        success = run_scenario(scenario_name, str(script_path))
        results[scenario_name] = success

    # Calculate total time
    total_elapsed = time.time() - overall_start
    total_minutes = int(total_elapsed // 60)
    total_seconds = int(total_elapsed % 60)

    # Print summary
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print(f"\nTotal runtime: {total_minutes}m {total_seconds}s")
    print("\nScenario Results:")

    all_passed = True
    for scenario_name, success in results.items():
        status = "PASS" if success else "FAIL"
        symbol = "[+]" if success else "[X]"
        print(f"  {symbol} {scenario_name}: {status}")
        if not success:
            all_passed = False

    print("\nResults saved to:")
    print("  epsilon_decay_tests/results/scenario_1_*.json")
    print("  epsilon_decay_tests/results/scenario_2_*.json")
    print("  epsilon_decay_tests/results/scenario_3_*.json")
    print("  epsilon_decay_tests/results/scenario_4_*.json")

    print("\n" + "=" * 100)

    if all_passed:
        print("ALL SCENARIOS PASSED")
    else:
        print("SOME SCENARIOS FAILED - See output above for details")

    print("=" * 100)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
