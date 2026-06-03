"""
Performance Test Cache Module

This module provides caching functionality for performance testing mode.
It loads LLM responses from a previous run's log file and provides lookup
by (query_id, component, hop_count) to replay deterministic workloads.

Usage:
    cache = PerfTestCache('llm_logs_multi_shot_20260520_191452.json')
    response = cache.get_response('11', 'generate_search_queries', 1)
"""

import json
from typing import Dict, Optional, Tuple


class PerfTestCache:
    """Cache for LLM responses to enable deterministic performance testing."""

    def __init__(self, log_file_path: str):
        """
        Load log file and build cache index.

        Args:
            log_file_path: Path to LLM log JSON file from a previous run

        Raises:
            FileNotFoundError: If log file doesn't exist
            json.JSONDecodeError: If log file is not valid JSON
            KeyError: If log file doesn't have expected structure
        """
        self.log_file_path = log_file_path
        self.data = None
        self.cache: Dict[Tuple[str, str, int], str] = {}

        # Load and validate log file
        with open(log_file_path, 'r') as f:
            self.data = json.load(f)

        if 'queries' not in self.data:
            raise KeyError(f"Log file {log_file_path} missing 'queries' field")

        # Build cache index
        self._build_index()

    def _build_index(self):
        """
        Build lookup index: (query_id, component, hop_count) -> response.

        This allows O(1) lookup of cached LLM responses during performance testing.
        """
        self.cache = {}

        for query in self.data['queries']:
            query_id = query.get('query_id')
            if not query_id:
                continue

            for call in query.get('llm_calls', []):
                component = call.get('component')
                hop_count = call.get('hop_count')

                if component is None or hop_count is None:
                    continue

                # Extract response from output field
                output = call.get('output', {})
                response = output.get('response')

                if response:
                    key = (str(query_id), component, hop_count)
                    self.cache[key] = response

        print(f"  Built cache index with {len(self.cache)} LLM responses")

    def get_response(self, query_id: str, component: str, hop_count: int) -> Optional[str]:
        """
        Retrieve cached LLM response.

        Args:
            query_id: Query identifier (matches query_id in log file)
            component: Component name (e.g., 'generate_search_queries')
            hop_count: Iteration/hop number

        Returns:
            Cached LLM response string, or None if not found
        """
        key = (str(query_id), component, hop_count)
        return self.cache.get(key)

    def has_response(self, query_id: str, component: str, hop_count: int) -> bool:
        """
        Check if cached response exists.

        Args:
            query_id: Query identifier
            component: Component name
            hop_count: Iteration/hop number

        Returns:
            True if cached response exists, False otherwise
        """
        key = (str(query_id), component, hop_count)
        return key in self.cache

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get statistics about the cache.

        Returns:
            Dict with cache statistics (total_responses, unique_queries, unique_components)
        """
        unique_queries = set()
        unique_components = set()

        for (query_id, component, hop_count) in self.cache.keys():
            unique_queries.add(query_id)
            unique_components.add(component)

        return {
            'total_responses': len(self.cache),
            'unique_queries': len(unique_queries),
            'unique_components': len(unique_components)
        }


def load_perf_test_cache(log_file_path: str) -> PerfTestCache:
    """
    Convenience function to load a performance test cache.

    Args:
        log_file_path: Path to LLM log JSON file

    Returns:
        Initialized PerfTestCache instance

    Raises:
        FileNotFoundError: If log file doesn't exist
        json.JSONDecodeError: If log file is not valid JSON
        KeyError: If log file doesn't have expected structure
    """
    return PerfTestCache(log_file_path)
