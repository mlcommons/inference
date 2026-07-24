#!/usr/bin/env python3
# Copyright 2025 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Accuracy Evaluation for RAG-DB Workload

This script evaluates the accuracy of the datasetup workload by:
1. Reading the loadgen accuracy logs (mlperf_log_accuracy.json)
2. Verifying that all files returned success (1) except possibly a few failures
3. Verifying that the last file returned a valid MD5 hash
4. Computing the actual MD5 of the saved database
5. Comparing the returned MD5 with the actual MD5
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from datetime import datetime


def compute_md5(file_path):
    """Compute MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def parse_accuracy_log(log_path):
    """
    Parse mlperf_log_accuracy.json and extract responses.

    Handles both formats:
    - JSON array format: [{"qsl_idx": ...}, {"qsl_idx": ...}, ...]
    - JSONL format: one JSON object per line

    Returns:
        dict: qsl_idx -> response_data mapping
    """
    results = {}

    with open(log_path, 'r') as f:
        content = f.read().strip()

        # Try parsing as JSON array first
        try:
            entries = json.loads(content)
            if isinstance(entries, list):
                # JSON array format
                for entry in entries:
                    try:
                        qsl_idx = entry['qsl_idx']

                        # Handle data field - can be hex string or list of
                        # bytes
                        data_field = entry['data']
                        if isinstance(data_field, str):
                            # Hex string - convert to bytes
                            data = bytes.fromhex(data_field)
                        elif isinstance(data_field, list):
                            # List of byte values
                            data = bytes(data_field)
                        else:
                            data = bytes(data_field)

                        results[qsl_idx] = {
                            'data': data,
                            'seq_id': entry.get('seq_id', -1)
                        }
                    except (KeyError, ValueError) as e:
                        print(f"Warning: Skipping malformed entry: {e}")
                        continue
                return results
        except json.JSONDecodeError:
            pass

    # Fall back to JSONL format (one JSON object per line)
    results = {}
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line in ['[', ']']:
                continue
            # Remove trailing comma if present
            if line.endswith(','):
                line = line[:-1]
            try:
                entry = json.loads(line)
                qsl_idx = entry['qsl_idx']

                # Handle data field - can be hex string or list of bytes
                data_field = entry['data']
                if isinstance(data_field, str):
                    # Hex string - convert to bytes
                    data = bytes.fromhex(data_field)
                elif isinstance(data_field, list):
                    # List of byte values
                    data = bytes(data_field)
                else:
                    data = bytes(data_field)

                results[qsl_idx] = {
                    'data': data,
                    'seq_id': entry.get('seq_id', -1)
                }
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: Skipping malformed log entry: {e}")
                continue

    return results


def validate_database(database_path, retriever_model):
    """
    Run comprehensive database validation checks.

    Args:
        database_path: Path to database file
        retriever_model: Path to retriever model

    Returns:
        dict: Validation results
    """
    print("Running database validation checks...")
    print()

    validation_results = {
        "checks": [],
        "passed": True,
    }

    try:
        from retrieve import VectorDB

        # Load database
        db = VectorDB(
            retriever_model=retriever_model,
            device="cpu",
            embedding_device="cpu",
            load_embeddings=False,
        )
        db.from_serialized(database_path)

        # Check 1: Vector count
        if hasattr(db, '_vector_store') and hasattr(db._vector_store, 'index'):
            vector_count = db._vector_store.index.ntotal
            check_passed = vector_count > 0
            validation_results["checks"].append({
                "name": "vector_count",
                "description": "Database contains vectors",
                "result": "PASS" if check_passed else "FAIL",
                "value": vector_count
            })
            validation_results["passed"] &= check_passed
            print(f"  ✓ Vector count: {vector_count}")
        else:
            validation_results["checks"].append({
                "name": "vector_count",
                "description": "Database contains vectors",
                "result": "FAIL",
                "error": "Cannot access vector store"
            })
            validation_results["passed"] = False
            print(f"  ✗ Vector count: Cannot access vector store")

        # Check 2: Docstore consistency
        if hasattr(db, '_vector_store') and hasattr(
                db._vector_store, 'index_to_docstore_id'):
            docstore_count = len(db._vector_store.index_to_docstore_id)
            check_passed = (docstore_count == vector_count)
            validation_results["checks"].append({
                "name": "docstore_consistency",
                "description": "Vector count matches docstore count",
                "result": "PASS" if check_passed else "FAIL",
                "vector_count": vector_count,
                "docstore_count": docstore_count
            })
            validation_results["passed"] &= check_passed
            if check_passed:
                print(f"  ✓ Docstore consistency: {docstore_count} documents")
            else:
                print(
                    f"  ✗ Docstore consistency: {vector_count} vectors but {docstore_count} documents")
        else:
            validation_results["checks"].append({
                "name": "docstore_consistency",
                "description": "Vector count matches docstore count",
                "result": "FAIL",
                "error": "Cannot access docstore"
            })
            validation_results["passed"] = False
            print(f"  ✗ Docstore consistency: Cannot access docstore")

        # Check 3: Index dimension
        if hasattr(db, '_vector_store') and hasattr(
                db._vector_store.index, 'd'):
            dimension = db._vector_store.index.d
            expected_dim = 768  # e5-base-v2
            check_passed = (dimension == expected_dim)
            validation_results["checks"].append({
                "name": "index_dimension",
                "description": f"Index dimension matches expected ({expected_dim})",
                "result": "PASS" if check_passed else "WARN",
                "value": dimension,
                "expected": expected_dim
            })
            # Don't fail on dimension mismatch, just warn
            print(
                f"  {'✓' if check_passed else '⚠'} Index dimension: {dimension} (expected {expected_dim})")
        else:
            validation_results["checks"].append({
                "name": "index_dimension",
                "description": "Index dimension matches expected",
                "result": "SKIP",
                "error": "Cannot access index dimension"
            })
            print(f"  - Index dimension: Cannot access")

        # Check 4: Sample retrieval test
        try:
            test_query = "test query"
            results = db.lookup(test_query, k=1)
            check_passed = len(results) > 0
            validation_results["checks"].append({
                "name": "sample_retrieval",
                "description": "Database can perform retrieval",
                "result": "PASS" if check_passed else "FAIL",
                "retrieved_count": len(results)
            })
            validation_results["passed"] &= check_passed
            if check_passed:
                print(
                    f"  ✓ Sample retrieval: Retrieved {len(results)} results")
            else:
                print(f"  ✗ Sample retrieval: No results returned")
        except Exception as e:
            validation_results["checks"].append({
                "name": "sample_retrieval",
                "description": "Database can perform retrieval",
                "result": "FAIL",
                "error": str(e)
            })
            validation_results["passed"] = False
            print(f"  ✗ Sample retrieval: {e}")

    except Exception as e:
        validation_results["checks"].append({
            "name": "database_load",
            "description": "Database can be loaded",
            "result": "FAIL",
            "error": str(e)
        })
        validation_results["passed"] = False
        print(f"  ✗ Database load failed: {e}")

    print()
    return validation_results


def evaluate_accuracy(log_dir, output_dir, database_path,
                      retriever_model=None):
    """
    Evaluate accuracy of datasetup workload.

    Args:
        log_dir: Directory containing loadgen logs
        output_dir: Directory containing SUT output files
        database_path: Path to the saved database file
        retriever_model: Path to retriever model (for validation)
        manifest_path: Path to reference DB manifest for cross-system
            verification. If None, the manifest check is skipped.
        cosine_threshold: Minimum sample-embedding cosine similarity for the
            manifest check.
        top_k_depth: Probe-query top-K rank match depth for the manifest check.

    Returns:
        dict: Accuracy results
    """
    print("=" * 80)
    print("RAG-DB Accuracy Evaluation")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Read accuracy log
    accuracy_log_path = os.path.join(log_dir, "mlperf_log_accuracy.json")

    if not os.path.exists(accuracy_log_path):
        print(f"ERROR: Accuracy log not found: {accuracy_log_path}")
        return {"error": "accuracy_log_not_found"}

    print(f"Reading accuracy log: {accuracy_log_path}")
    results = parse_accuracy_log(accuracy_log_path)

    print(f"Found {len(results)} responses in accuracy log")
    print()

    # Analyze responses
    success_count = 0
    failure_count = 0
    md5_response = None
    md5_qsl_idx = None

    for qsl_idx, result in sorted(results.items()):
        data = result['data']

        if len(data) == 1:
            # Single byte response: 0 or 1
            value = data[0]
            if value == 1:
                success_count += 1
            else:
                failure_count += 1
        elif len(data) == 32:
            # MD5 hash (32 hex characters = 32 bytes when encoded)
            try:
                md5_str = data.decode('utf-8')
                if len(md5_str) == 32 and all(
                        c in '0123456789abcdef' for c in md5_str):
                    md5_response = md5_str
                    md5_qsl_idx = qsl_idx
                    success_count += 1  # MD5 response counts as success
                else:
                    print(
                        f"Warning: Invalid MD5 format at qsl_idx {qsl_idx}: {md5_str}")
                    failure_count += 1
            except UnicodeDecodeError:
                print(f"Warning: Cannot decode response at qsl_idx {qsl_idx}")
                failure_count += 1
        else:
            print(
                f"Warning: Unexpected response length {len(data)} at qsl_idx {qsl_idx}")
            failure_count += 1

    print(f"Response Summary:")
    print(f"  Successful files: {success_count}")
    print(f"  Failed files: {failure_count}")
    print(f"  Total files: {success_count + failure_count}")
    print()

    # Verify MD5 was returned
    if md5_response is None:
        print("ERROR: No MD5 hash found in responses (last file did not return MD5)")
        return {
            "passed": False,
            "error": "no_md5_found",
            "success_count": success_count,
            "failure_count": failure_count,
        }

    print(f"MD5 Response (from qsl_idx {md5_qsl_idx}): {md5_response}")
    print()

    # Compute actual MD5 of database
    if not os.path.exists(database_path):
        print(f"ERROR: Database file not found: {database_path}")
        return {
            "passed": False,
            "error": "database_not_found",
            "md5_response": md5_response,
            "success_count": success_count,
            "failure_count": failure_count,
        }

    print(f"Computing MD5 of database: {database_path}")
    actual_md5 = compute_md5(database_path)
    print(f"Actual MD5: {actual_md5}")
    print()

    # Compare MD5s
    md5_match = (md5_response == actual_md5)

    print("=" * 80)
    print("Accuracy Results:")
    print("=" * 80)
    print(f"  Total files processed: {success_count + failure_count}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failure_count}")
    print(
        f"  Success rate: {100.0 * success_count / (success_count + failure_count):.2f}%")
    print()
    print(f"  MD5 returned: {md5_response}")
    print(f"  MD5 actual:   {actual_md5}")
    print(f"  MD5 match: {'✅ PASS' if md5_match else '❌ FAIL'}")
    print()

    # Overall pass/fail
    # Pass criteria:
    # 1. At least 99% of files succeeded
    # 2. MD5 matches
    min_success_rate = 0.99
    actual_success_rate = success_count / \
        (success_count + failure_count) if (success_count + failure_count) > 0 else 0

    passed = (actual_success_rate >= min_success_rate) and md5_match

    print(f"Overall: {'✅ PASSED' if passed else '❌ FAILED'}")
    print("=" * 80)
    print()

    # Save results
    accuracy_results = {
        "passed": passed,
        "total_files": success_count + failure_count,
        "successful_files": success_count,
        "failed_files": failure_count,
        "success_rate": actual_success_rate,
        "min_success_rate_required": min_success_rate,
        "md5_response": md5_response,
        "md5_actual": actual_md5,
        "md5_match": md5_match,
        "database_path": database_path,
    }

    output_path = os.path.join(output_dir, "accuracy_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(accuracy_results, f, indent=2)

    print(f"Accuracy results saved to: {output_path}")
    print()

    # Run database validation if retriever model provided
    validation_results = None
    if retriever_model and os.path.exists(database_path):
        print("=" * 80)
        print("Database Validation")
        print("=" * 80)
        print()

        validation_results = validate_database(database_path, retriever_model)

        accuracy_results["validation"] = validation_results

        # Overall pass requires both accuracy and validation to pass
        overall_passed = passed and validation_results["passed"]
        accuracy_results["passed"] = overall_passed

        print("=" * 80)
        print("Validation Summary:")
        print("=" * 80)
        for check in validation_results["checks"]:
            status_symbol = "✓" if check["result"] == "PASS" else (
                "⚠" if check["result"] == "WARN" else "✗")
            print(
                f"  {status_symbol} {check['description']}: {check['result']}")
        print()
        print(
            f"Validation: {'✅ PASSED' if validation_results['passed'] else '❌ FAILED'}")
        print(f"Overall: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        print("=" * 80)
        print()

    # Cross-system manifest verification (corpus fingerprint, sample-embedding
    # cosine, probe-query top-K ranks) against a reference manifest.
    manifest_results = None
    if manifest_path:
        print("="*80)
        print("DB Manifest Verification")
        print("="*80)
        print(f"Manifest: {manifest_path}")
        print()

        if not os.path.exists(manifest_path):
            manifest_results = {
                "passed": False,
                "error": "manifest_not_found",
                "manifest_path": manifest_path,
            }
            print(f"  ✗ Manifest not found: {manifest_path}")
        elif not os.path.exists(database_path):
            manifest_results = {
                "passed": False,
                "error": "database_not_found",
                "database_path": database_path,
            }
            print(f"  ✗ Database not found: {database_path}")
        else:
            try:
                from db_manifest import verify_manifest
                manifest_results = verify_manifest(
                    database_path,
                    manifest_path,
                    retriever_model=retriever_model,
                    cosine_threshold=cosine_threshold,
                    top_k_depth=top_k_depth,
                )
                if manifest_results["passed"]:
                    print("  ✓ Manifest verification PASSED")
                else:
                    print("  ✗ Manifest verification FAILED:")
                    for failure in manifest_results["failures"]:
                        print(f"    - {failure}")
            except Exception as e:
                manifest_results = {"passed": False, "error": str(e)}
                print(f"  ✗ Manifest verification error: {e}")

        accuracy_results["manifest"] = manifest_results
        # Overall pass now also requires the manifest check to pass.
        accuracy_results["passed"] = accuracy_results["passed"] and manifest_results["passed"]

        print()
        print(f"Manifest: {'✅ PASSED' if manifest_results['passed'] else '❌ FAILED'}")
        print("="*80)
        print()

    # Write accuracy.txt in MLPerf format
    accuracy_txt_path = os.path.join(log_dir, "accuracy.txt")
    with open(accuracy_txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RAG-DB Accuracy Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Database: {database_path}\n")
        f.write("\n")

        f.write("File Processing Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total files: {accuracy_results['total_files']}\n")
        f.write(f"Successful: {accuracy_results['successful_files']}\n")
        f.write(f"Failed: {accuracy_results['failed_files']}\n")
        f.write(f"Success rate: {accuracy_results['success_rate']*100:.2f}%\n")
        f.write(
            f"Required rate: {accuracy_results['min_success_rate_required']*100:.2f}%\n")
        f.write(
            f"Status: {'PASS' if actual_success_rate >= min_success_rate else 'FAIL'}\n")
        f.write("\n")

        f.write("MD5 Verification:\n")
        f.write("-" * 80 + "\n")
        f.write(f"MD5 returned by SUT: {accuracy_results['md5_response']}\n")
        f.write(f"MD5 actual (computed): {accuracy_results['md5_actual']}\n")
        f.write(
            f"MD5 match: {'PASS' if accuracy_results['md5_match'] else 'FAIL'}\n")
        f.write("\n")

        if validation_results:
            f.write("Database Validation:\n")
            f.write("-" * 80 + "\n")
            for check in validation_results["checks"]:
                f.write(f"  {check['name']}: {check['result']}\n")
                f.write(f"    - {check['description']}\n")
                if 'value' in check:
                    f.write(f"    - Value: {check['value']}\n")
                if 'error' in check:
                    f.write(f"    - Error: {check['error']}\n")
            f.write(
                f"Validation status: {'PASS' if validation_results['passed'] else 'FAIL'}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write(
            f"Overall Result: {'PASS' if accuracy_results['passed'] else 'FAIL'}\n")
        f.write("=" * 80 + "\n")

    print(f"Accuracy report saved to: {accuracy_txt_path}")
    print()

    return accuracy_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy for RAG-DB workload"
    )

    parser.add_argument(
        "--log_dir",
        required=True,
        help="Directory containing loadgen logs (mlperf_log_accuracy.json)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for output files"
    )
    parser.add_argument(
        "--database",
        required=True,
        help="Path to database file to verify"
    )
    parser.add_argument(
        "--retriever_model",
        default="intfloat_e5-base-v2/e5-base-v2",
        help="Path to retriever model (for validation)"
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to reference DB manifest (.json/.json.gz) for cross-system "
             "verification. If omitted, the manifest check is skipped."
    )
    parser.add_argument(
        "--cosine_threshold",
        type=float,
        default=0.9999,
        help="Minimum sample-embedding cosine similarity for the manifest check"
    )
    parser.add_argument(
        "--top_k_depth",
        type=int,
        default=3,
        help="Probe-query top-K rank match depth for the manifest check"
    )

    args = parser.parse_args()

    results = evaluate_accuracy(
        args.log_dir,
        args.output_dir,
        args.database,
        args.retriever_model)

    # Exit with appropriate code
    if results.get("passed", False):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
