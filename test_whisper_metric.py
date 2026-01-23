#!/usr/bin/env python3
"""
Test script to validate that the whisper metric fix is working correctly.
This tests that whisper uses "result_tokens_per_second" instead of "result_samples_per_second"
for the Offline scenario across all versions.
"""

import sys
import os

# Add the submission_checker to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools', 'submission'))

from submission_checker.constants import RESULT_FIELD_BENCHMARK_OVERWRITE, RESULT_FIELD_NEW

def test_whisper_metrics():
    """Test that whisper has the correct metric override for all versions."""
    
    print("=" * 80)
    print("Testing Whisper Metric Configuration")
    print("=" * 80)
    
    # Test v5.0
    print("\n[v5.0] Testing whisper configuration...")
    if "v5.0" in RESULT_FIELD_BENCHMARK_OVERWRITE:
        if "whisper" in RESULT_FIELD_BENCHMARK_OVERWRITE["v5.0"]:
            whisper_config = RESULT_FIELD_BENCHMARK_OVERWRITE["v5.0"]["whisper"]
            print(f"  ✓ whisper found in v5.0")
            print(f"    Config: {whisper_config}")
            
            if "Offline" in whisper_config:
                metric = whisper_config["Offline"]
                if metric == "result_tokens_per_second":
                    print(f"  ✓ Offline metric is correct: {metric}")
                else:
                    print(f"  ✗ ERROR: Offline metric is wrong: {metric}")
                    print(f"           Expected: result_tokens_per_second")
                    return False
            else:
                print(f"  ✗ ERROR: 'Offline' not found in whisper config")
                return False
        else:
            print(f"  ✗ ERROR: whisper not found in v5.0")
            return False
    else:
        print(f"  ✗ ERROR: v5.0 not found in RESULT_FIELD_BENCHMARK_OVERWRITE")
        return False
    
    # Test v5.1
    print("\n[v5.1] Testing whisper configuration...")
    if "v5.1" in RESULT_FIELD_BENCHMARK_OVERWRITE:
        if "whisper" in RESULT_FIELD_BENCHMARK_OVERWRITE["v5.1"]:
            whisper_config = RESULT_FIELD_BENCHMARK_OVERWRITE["v5.1"]["whisper"]
            print(f"  ✓ whisper found in v5.1")
            print(f"    Config: {whisper_config}")
            
            if "Offline" in whisper_config:
                metric = whisper_config["Offline"]
                if metric == "result_tokens_per_second":
                    print(f"  ✓ Offline metric is correct: {metric}")
                else:
                    print(f"  ✗ ERROR: Offline metric is wrong: {metric}")
                    print(f"           Expected: result_tokens_per_second")
                    return False
            else:
                print(f"  ✗ ERROR: 'Offline' not found in whisper config")
                return False
        else:
            print(f"  ✗ ERROR: whisper not found in v5.1")
            return False
    else:
        print(f"  ✗ ERROR: v5.1 not found in RESULT_FIELD_BENCHMARK_OVERWRITE")
        return False
    
    # Test v6.0
    print("\n[v6.0] Testing whisper configuration...")
    if "v6.0" in RESULT_FIELD_BENCHMARK_OVERWRITE:
        if "whisper" in RESULT_FIELD_BENCHMARK_OVERWRITE["v6.0"]:
            whisper_config = RESULT_FIELD_BENCHMARK_OVERWRITE["v6.0"]["whisper"]
            print(f"  ✓ whisper found in v6.0")
            print(f"    Config: {whisper_config}")
            
            if "Offline" in whisper_config:
                metric = whisper_config["Offline"]
                if metric == "result_tokens_per_second":
                    print(f"  ✓ Offline metric is correct: {metric}")
                else:
                    print(f"  ✗ ERROR: Offline metric is wrong: {metric}")
                    print(f"           Expected: result_tokens_per_second")
                    return False
            else:
                print(f"  ✗ ERROR: 'Offline' not found in whisper config")
                return False
        else:
            print(f"  ✗ ERROR: whisper not found in v6.0")
            return False
    else:
        print(f"  ✗ ERROR: v6.0 not found in RESULT_FIELD_BENCHMARK_OVERWRITE")
        return False
    
    # Test default behavior (without override)
    print("\n[DEFAULT] Testing default Offline metric (without override)...")
    default_metric = RESULT_FIELD_NEW["v5.0"]["Offline"]
    print(f"  Default Offline metric: {default_metric}")
    print(f"  This would be used if whisper was NOT in RESULT_FIELD_BENCHMARK_OVERWRITE")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nSummary:")
    print("  - whisper is correctly configured in v5.0, v5.1, and v6.0")
    print("  - Offline scenario uses 'result_tokens_per_second' (correct)")
    print("  - Without override, it would use 'result_samples_per_second' (wrong)")
    print("\nThis fix ensures the submission checker logs the correct metric")
    print("for Whisper, matching what's displayed in the final dashboard.")
    
    return True

if __name__ == "__main__":
    try:
        success = test_whisper_metrics()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
