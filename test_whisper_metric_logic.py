#!/usr/bin/env python3
"""
Test script to simulate the actual logic used by the submission checker
when determining which metric to use for whisper.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools', 'submission'))

from submission_checker.constants import (
    RESULT_FIELD_BENCHMARK_OVERWRITE, 
    RESULT_FIELD_NEW
)

def simulate_metric_selection(model, scenario, version):
    """
    Simulate the logic from performance_check.py and utils.py
    to determine which metric field to use.
    """
    
    # First, get the default metric for the scenario
    default_metric = RESULT_FIELD_NEW[version][scenario]
    
    # Then check if there's a benchmark-specific override
    if (
        version in RESULT_FIELD_BENCHMARK_OVERWRITE
        and model in RESULT_FIELD_BENCHMARK_OVERWRITE[version]
        and scenario in RESULT_FIELD_BENCHMARK_OVERWRITE[version][model]
    ):
        override_metric = RESULT_FIELD_BENCHMARK_OVERWRITE[version][model][scenario]
        return override_metric, True  # True = override was applied
    
    return default_metric, False  # False = using default


def test_whisper_metric_logic():
    """Test that whisper correctly uses tokens per second instead of samples per second."""
    
    print("=" * 80)
    print("Testing Whisper Metric Selection Logic")
    print("=" * 80)
    
    test_cases = [
        ("v5.0", "whisper", "Offline"),
        ("v5.1", "whisper", "Offline"),
        ("v6.0", "whisper", "Offline"),
    ]
    
    all_passed = True
    
    for version, model, scenario in test_cases:
        print(f"\n[{version}] {model} - {scenario}")
        print("-" * 80)
        
        metric, is_override = simulate_metric_selection(model, scenario, version)
        
        print(f"  Selected metric: {metric}")
        print(f"  Override applied: {is_override}")
        
        # Verify the metric is correct
        expected_metric = "result_tokens_per_second"
        if metric == expected_metric:
            print(f"  ✓ PASS: Using correct metric '{metric}'")
        else:
            print(f"  ✗ FAIL: Expected '{expected_metric}', got '{metric}'")
            all_passed = False
        
        # Verify that override was applied
        if is_override:
            print(f"  ✓ PASS: Override was correctly applied")
        else:
            print(f"  ✗ FAIL: Override was not applied (would use wrong metric)")
            all_passed = False
    
    # Test what would happen without the override
    print("\n" + "=" * 80)
    print("Comparison: What would happen WITHOUT the fix")
    print("=" * 80)
    
    for version, model, scenario in test_cases:
        default_metric = RESULT_FIELD_NEW[version][scenario]
        print(f"\n[{version}] {model} - {scenario}")
        print(f"  Would use default: {default_metric}")
        print(f"  ✗ This is WRONG for whisper (should be tokens, not samples)")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe submission checker will now correctly:")
        print("  1. Detect that whisper has a benchmark-specific override")
        print("  2. Use 'result_tokens_per_second' instead of 'result_samples_per_second'")
        print("  3. Log the correct metric that matches the dashboard")
        print("\nThis fixes issue #2449: 'Submission checker logging Whisper")
        print("performance with wrong metric'")
    else:
        print("✗ SOME TESTS FAILED!")
        print("=" * 80)
        return False
    
    return True


if __name__ == "__main__":
    try:
        success = test_whisper_metric_logic()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
