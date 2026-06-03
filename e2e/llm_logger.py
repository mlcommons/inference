#!/usr/bin/env python3
"""
LLM call logger for tracking all LLM requests, responses, and token usage.
"""

import os
import uuid
import time
import json
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional


class LLMLogger:
    """Logger for tracking all LLM calls with full input/output and metrics."""

    def __init__(self, output_file: str = None, experiment_metadata: Dict[str, Any] = None):
        self.session_id = str(uuid.uuid4())
        self.queries = []
        self.output_file = output_file
        self.experiment_metadata = experiment_metadata or {}
        self._lock = threading.Lock()
        self._local = threading.local()

        # Initialize file with header if output_file provided
        if self.output_file:
            self._initialize_file()

    @property
    def current_query(self):
        return getattr(self._local, 'current_query', None)

    @current_query.setter
    def current_query(self, value):
        self._local.current_query = value

    def start_query(self, query_id: str, original_query: str):
        """Start logging a new query"""
        self.current_query = {
            "query_id": query_id,
            "original_query": original_query,
            "timestamp_start": datetime.utcnow().isoformat() + "Z",
            "llm_calls": []
        }

    def log_llm_call(self,
                     component: str,
                     hop_count: Optional[int],
                     payload: Dict,
                     response: Dict,
                     latency_ms: float,
                     context: Dict[str, Any] = None,
                     simulated_response: Optional[str] = None):
        """Log a single LLM call with full input/output

        Args:
            simulated_response: In perf test mode, the cached response that was returned
                              to the pipeline (instead of the real LLM response)
        """

        usage = response.get('usage', {})
        isl = usage.get('prompt_tokens', 0)
        osl = usage.get('completion_tokens', 0)

        call_record = {
            "call_id": str(uuid.uuid4()),
            "component": component,
            "hop_count": hop_count,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "input": {
                "messages": payload.get('messages', []),
                "model": payload.get('model'),
                "temperature": payload.get('temperature'),
                "max_tokens": payload.get('max_tokens'),
                "top_p": payload.get('top_p'),
                "top_k": payload.get('top_k'),
                "reasoning_effort": payload.get('reasoning_effort'),
                "other_params": {
                    k: v for k, v in payload.items()
                    if k not in ['messages', 'model', 'temperature', 'max_tokens', 'top_p', 'top_k', 'reasoning_effort']
                }
            },
            "output": {
                "actual_output": self._extract_response_text(response),
                "cached_output": simulated_response if simulated_response is not None else None,
                "finish_reason": response.get('choices', [{}])[0].get('finish_reason') if response.get('choices') else None,
                "note": "In perf test mode: actual_output is from real LLM (for measurement), cached_output is returned to pipeline (for determinism)" if simulated_response is not None else None
            },
            "metrics": {
                "isl": isl,
                "osl": osl,
                "total_tokens": isl + osl,
                "latency_ms": round(latency_ms, 2),
                "tokens_per_second": round(osl / (latency_ms / 1000), 2) if latency_ms > 0 else 0
            },
            "context": context or {}
        }

        if self.current_query:
            self.current_query["llm_calls"].append(call_record)

    def _extract_response_text(self, response: Dict) -> str:
        """Extract response text from API response"""
        if not response or 'choices' not in response:
            return ""

        message = response['choices'][0].get('message', {})
        content = (message.get('content') or '').strip()

        # Fallback for thinking models
        if not content:
            content = message.get('reasoning_content', '').strip()

        return content

    def _initialize_file(self):
        """Initialize JSON file with metadata header"""
        initial_data = {
            "experiment_metadata": {
                "session_id": self.session_id,
                **self.experiment_metadata
            },
            "queries": [],
            "experiment_summary": {}
        }
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)

    def _append_query_to_file(self, query_data: Dict):
        """Append a completed query to the JSON file"""
        if not self.output_file:
            return

        if not os.path.exists(self.output_file):
            self._initialize_file()

        # Read current file
        with open(self.output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Append query
        data['queries'].append(query_data)

        # Update experiment summary
        data['experiment_summary'] = self._calculate_experiment_summary(data['queries'])

        # Write back
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _calculate_experiment_summary(self, queries: List[Dict]) -> Dict:
        """Calculate aggregate statistics across all queries"""
        all_calls = []
        for q in queries:
            all_calls.extend(q["llm_calls"])

        if not all_calls:
            return {}

        experiment_summary = {
            "total_queries": len(queries),
            "total_llm_calls": len(all_calls),
            "total_input_tokens": sum(c["metrics"]["isl"] for c in all_calls),
            "total_output_tokens": sum(c["metrics"]["osl"] for c in all_calls),
            "total_tokens": sum(c["metrics"]["total_tokens"] for c in all_calls),
            "total_latency_ms": round(sum(c["metrics"]["latency_ms"] for c in all_calls), 2),
            "average_tokens_per_second": round(sum(c["metrics"]["tokens_per_second"] for c in all_calls) / len(all_calls), 2) if all_calls else 0,
            "average_hops_per_query": round(sum(q["summary"]["total_hops"] for q in queries) / len(queries), 2) if queries else 0,
            "components_used": list(set(c["component"] for c in all_calls))
        }

        # Add retrieval/answer metrics if available
        queries_with_retrieval = [q for q in queries if "retrieval_results" in q]
        if queries_with_retrieval:
            experiment_summary["retrieval_metrics"] = {
                "average_precision": round(sum(q["retrieval_results"].get("precision", 0) for q in queries_with_retrieval) / len(queries_with_retrieval), 4),
                "average_recall": round(sum(q["retrieval_results"].get("recall", 0) for q in queries_with_retrieval) / len(queries_with_retrieval), 4),
                "average_f1": round(sum(q["retrieval_results"].get("f1", 0) for q in queries_with_retrieval) / len(queries_with_retrieval), 4),
            }

        queries_with_answers = [q for q in queries if "answer_results" in q]
        if queries_with_answers:
            correct_count = sum(1 for q in queries_with_answers if q["answer_results"].get("judge_score", 0) >= 4)
            experiment_summary["answer_metrics"] = {
                "average_judge_score": round(sum(q["answer_results"].get("judge_score", 0) for q in queries_with_answers) / len(queries_with_answers), 2),
                "queries_correct": correct_count,
                "queries_incorrect": len(queries_with_answers) - correct_count,
                "accuracy": round(correct_count / len(queries_with_answers), 4) if queries_with_answers else 0
            }

        return experiment_summary

    def end_query(self, retrieval_results: Dict = None, answer_results: Dict = None, wall_time_s: float = None):
        """Finish logging current query, compute summary, and write to file"""
        if self.current_query:
            self.current_query["timestamp_end"] = datetime.utcnow().isoformat() + "Z"

            # Calculate summary
            llm_calls = self.current_query["llm_calls"]
            hop_counts = [c["hop_count"] for c in llm_calls if c["hop_count"] is not None]

            summary = {
                "total_llm_calls": len(llm_calls),
                "total_hops": max(hop_counts) if hop_counts else 0,
                "total_input_tokens": sum(c["metrics"]["isl"] for c in llm_calls),
                "total_output_tokens": sum(c["metrics"]["osl"] for c in llm_calls),
                "total_tokens": sum(c["metrics"]["total_tokens"] for c in llm_calls),
                "total_latency_ms": round(sum(c["metrics"]["latency_ms"] for c in llm_calls), 2),
                "average_tokens_per_second": round(sum(c["metrics"]["tokens_per_second"] for c in llm_calls) / len(llm_calls), 2) if llm_calls else 0,
                "components_used": list(set(c["component"] for c in llm_calls))
            }
            if wall_time_s is not None:
                summary["total_wall_time_ms"] = round(wall_time_s * 1000, 2)
            self.current_query["summary"] = summary

            if retrieval_results:
                self.current_query["retrieval_results"] = retrieval_results
            if answer_results:
                self.current_query["answer_results"] = answer_results

            with self._lock:
                self.queries.append(self.current_query)

                # Write to file immediately after completing query
                if self.output_file:
                    self._append_query_to_file(self.current_query)

            self.current_query = None

    def save(self, output_file: str = None, experiment_metadata: Dict[str, Any] = None):
        """Save all logs to JSON file (legacy method for backward compatibility).

        Note: If logger was initialized with output_file, logs are already written
        incrementally. This method can be used to write to a different file or
        when using the old non-incremental mode.
        """
        # Use provided file or fall back to instance file
        target_file = output_file or self.output_file

        if not target_file:
            print("Warning: No output file specified, logs not saved")
            return

        # Calculate experiment summary
        experiment_summary = self._calculate_experiment_summary(self.queries)

        # Use provided metadata or instance metadata
        metadata = experiment_metadata or self.experiment_metadata

        output = {
            "experiment_metadata": {
                "session_id": self.session_id,
                **metadata
            },
            "queries": self.queries,
            "experiment_summary": experiment_summary
        }

        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"LLM logs saved to: {target_file}")
        print(f"Total queries: {len(self.queries)}")
        if experiment_summary:
            print(f"Total LLM calls: {experiment_summary.get('total_llm_calls', 0)}")
            print(f"Total tokens: {experiment_summary.get('total_tokens', 0):,} (input: {experiment_summary.get('total_input_tokens', 0):,}, output: {experiment_summary.get('total_output_tokens', 0):,})")
            # Per-query latency distribution (wall time: query to answer)
            per_query_latencies = sorted(
                (q["summary"].get("total_wall_time_ms") or q["summary"].get("total_latency_ms", 0)) / 1000
                for q in self.queries
                if "summary" in q
            )
            n = len(per_query_latencies)
            if n > 0:
                mean_lat = sum(per_query_latencies) / n
                median_lat = per_query_latencies[n // 2] if n % 2 == 1 else (per_query_latencies[n // 2 - 1] + per_query_latencies[n // 2]) / 2
                p90_lat = per_query_latencies[int(n * 0.90)] if n >= 10 else per_query_latencies[-1]
                p99_lat = per_query_latencies[int(n * 0.99)] if n >= 100 else per_query_latencies[-1]
                total_latency_s = sum(per_query_latencies)
                print(f"Per-query latency (query-to-answer):  mean={mean_lat:.2f}s  median={median_lat:.2f}s  p90={p90_lat:.2f}s  p99={p99_lat:.2f}s")
                print(f"Throughput:    {n / total_latency_s:.4f} queries/sec  ({total_latency_s / n:.2f}s per query)")
        print(f"{'='*80}\n")
