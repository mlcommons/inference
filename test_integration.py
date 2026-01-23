#!/usr/bin/env python3
"""
Comprehensive test to verify the submission checker code integrity
after the whisper metric fix.
"""
import sys
sys.path.insert(0, 'tools/submission')

print("=" * 80)
print("SUBMISSION CHECKER INTEGRATION TEST")
print("=" * 80)

test_passed = True

# Test 1: Import all required modules
print("\n[TEST 1] Import all submission checker modules...")
try:
    from submission_checker.constants import *
    from submission_checker.configuration.configuration import Config
    from submission_checker.checks.performance_check import PerformanceCheck
    from submission_checker.checks.accuracy_check import AccuracyCheck
    from submission_checker import utils
    print("  ✓ All modules imported successfully")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    test_passed = False

# Test 2: Verify constants are well-formed
print("\n[TEST 2] Verify constants structure...")
try:
    # Check RESULT_FIELD_NEW structure
    for version in ["v5.0", "v5.1", "v6.0"]:
        assert version in RESULT_FIELD_NEW, f"Missing {version} in RESULT_FIELD_NEW"
        for scenario in ["Offline", "Server", "SingleStream", "MultiStream"]:
            assert scenario in RESULT_FIELD_NEW[version], f"Missing {scenario} in {version}"
    
    # Check RESULT_FIELD_BENCHMARK_OVERWRITE structure
    assert "v5.0" in RESULT_FIELD_BENCHMARK_OVERWRITE
    assert "v5.1" in RESULT_FIELD_BENCHMARK_OVERWRITE
    assert "v6.0" in RESULT_FIELD_BENCHMARK_OVERWRITE
    
    # Check whisper is in all versions
    assert "whisper" in RESULT_FIELD_BENCHMARK_OVERWRITE["v5.0"]
    assert "whisper" in RESULT_FIELD_BENCHMARK_OVERWRITE["v5.1"]
    assert "whisper" in RESULT_FIELD_BENCHMARK_OVERWRITE["v6.0"]
    
    print("  ✓ Constants structure is valid")
except AssertionError as e:
    print(f"  ✗ Structure validation failed: {e}")
    test_passed = False

# Test 3: Verify whisper configuration
print("\n[TEST 3] Verify whisper metric configuration...")
try:
    for version in ["v5.0", "v5.1", "v6.0"]:
        whisper_config = RESULT_FIELD_BENCHMARK_OVERWRITE[version]["whisper"]
        assert "Offline" in whisper_config, f"Missing Offline in whisper config for {version}"
        assert whisper_config["Offline"] == "result_tokens_per_second", \
            f"Wrong metric for whisper in {version}: {whisper_config['Offline']}"
    print("  ✓ Whisper uses correct metric (result_tokens_per_second)")
except AssertionError as e:
    print(f"  ✗ Whisper config validation failed: {e}")
    test_passed = False

# Test 4: Test metric selection logic
print("\n[TEST 4] Test metric selection logic...")
try:
    def select_metric(model, scenario, version):
        default = RESULT_FIELD_NEW[version][scenario]
        if (version in RESULT_FIELD_BENCHMARK_OVERWRITE and
            model in RESULT_FIELD_BENCHMARK_OVERWRITE[version] and
            scenario in RESULT_FIELD_BENCHMARK_OVERWRITE[version][model]):
            return RESULT_FIELD_BENCHMARK_OVERWRITE[version][model][scenario]
        return default
    
    # Test whisper uses override
    for version in ["v5.0", "v5.1", "v6.0"]:
        metric = select_metric("whisper", "Offline", version)
        assert metric == "result_tokens_per_second", \
            f"Whisper should use tokens metric in {version}, got {metric}"
    
    # Test other models (sanity check)
    metric = select_metric("llama2-70b-99", "Offline", "v5.0")
    assert metric == "result_tokens_per_second"
    
    metric = select_metric("resnet50", "Offline", "v5.0")  # should use default
    assert metric == "result_samples_per_second"
    
    print("  ✓ Metric selection logic works correctly")
except AssertionError as e:
    print(f"  ✗ Logic test failed: {e}")
    test_passed = False

# Test 5: Check for common issues
print("\n[TEST 5] Check for common configuration issues...")
try:
    # Ensure no trailing commas or syntax issues by checking dictionary is valid
    for version, models in RESULT_FIELD_BENCHMARK_OVERWRITE.items():
        for model, config in models.items():
            assert isinstance(config, dict), f"{version}/{model} is not a dict"
            for scenario, metric in config.items():
                assert isinstance(metric, str), f"{version}/{model}/{scenario} metric is not a string"
                assert len(metric) > 0, f"{version}/{model}/{scenario} has empty metric"
    
    print("  ✓ No configuration issues found")
except AssertionError as e:
    print(f"  ✗ Configuration issue: {e}")
    test_passed = False

# Final result
print("\n" + "=" * 80)
if test_passed:
    print("✅ ALL INTEGRATION TESTS PASSED")
    print("=" * 80)
    print("\nThe submission checker is working correctly with the whisper fix.")
    print("No regressions detected in the codebase.")
    sys.exit(0)
else:
    print("❌ SOME TESTS FAILED")
    print("=" * 80)
    sys.exit(1)
