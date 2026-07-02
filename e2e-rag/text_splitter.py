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


#!/usr/bin/env python3
"""
Text Splitter Module

Shared text splitting functionality for both PDF and HTML pipelines.
Provides intelligent text chunking with sentence-aware boundaries and overlap.
"""

import re
from typing import List, Optional


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove excessive newlines but preserve paragraph structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    return text


def find_sentence_boundary(text: str, start: int, end: int,
                           search_window: int = 100) -> int:
    """
    Find the best sentence boundary within the search window.

    Args:
        text: The text to search in
        start: Start position of the passage
        end: Desired end position
        search_window: Number of characters to look back for sentence boundary

    Returns:
        Best boundary position
    """
    if end >= len(text):
        return len(text)

    # Look for sentence endings within the search window
    search_start = max(start, end - search_window)
    sentence_endings = ['.', '!', '?', '\n']

    best_break = end
    for i in range(end - 1, search_start - 1, -1):
        if text[i] in sentence_endings:
            # Check if it's followed by whitespace and uppercase letter (proper
            # sentence end)
            if i + 1 < len(text) and text[i + 1].isspace():
                # Look for the next non-whitespace character
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                if j < len(text) and (text[j].isupper() or text[j].isdigit()):
                    best_break = i + 1
                    break

    return best_break


def split_into_passages(text: str, max_length: int = 512,
                        overlap: int = 50) -> List[str]:
    """
    Split text into passages suitable for retrieval systems like ColBERT.

    Args:
        text: Input text to split
        max_length: Maximum length of each passage in characters
        overlap: Number of characters to overlap between passages

    Returns:
        List of passage texts
    """
    # Clean up the text
    text = clean_text(text)

    if len(text) <= max_length:
        return [text] if text else []

    passages = []
    start = 0

    while start < len(text):
        end = start + max_length

        # If we're not at the end of the text, try to break at a sentence
        # boundary
        if end < len(text):
            end = find_sentence_boundary(text, start, end)

        passage = text[start:end].strip()
        if passage:
            passages.append(passage)

        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break

    return passages


def split_into_fixed_passages(
        text: str, fixed_length: int = 256, overlap: int = 32) -> List[str]:
    """
    Split text into fixed-length passages with exact character counts.
    Useful for consistent passage lengths across datasets.

    Args:
        text: Input text to split
        fixed_length: Exact length of each passage in characters
        overlap: Number of characters to overlap between passages

    Returns:
        List of passage texts with fixed lengths
    """
    # Clean up the text
    text = clean_text(text)

    if len(text) <= fixed_length:
        return [text] if text else []

    passages = []
    start = 0

    while start < len(text):
        end = min(start + fixed_length, len(text))
        passage = text[start:end].strip()

        if passage:
            passages.append(passage)

        # Move start position with overlap
        start = start + fixed_length - overlap
        if start >= len(text):
            break

    return passages


def create_passage_metadata(
        filename: str, passage_index: int, original_url: Optional[str] = None) -> dict:
    """
    Create standardized metadata for a passage.

    Args:
        filename: Source filename (PDF or HTML)
        passage_index: Index of this passage within the document
        original_url: Original URL if available

    Returns:
        Dictionary containing passage metadata
    """
    # Extract base filename without extension
    base_filename = filename
    if '.' in filename:
        base_filename = '.'.join(filename.split('.')[:-1])

    metadata = {
        'index': passage_index,
        'base_filename': base_filename
    }

    if original_url:
        metadata['original_url'] = original_url

    return metadata


def estimate_passage_count(
        text: str, max_length: int = 512, overlap: int = 50) -> int:
    """
    Estimate the number of passages that will be created from text.
    Useful for progress tracking without actually splitting.
    """
    if len(text) <= max_length:
        return 1 if text.strip() else 0

    # Rough estimate based on overlap
    effective_length = max_length - overlap
    return max(1, (len(text) - overlap) // effective_length)


def split_into_hierarchical_passages(
    text: str,
    parent_length: int = 2048,
    parent_overlap: int = 32,
    child_length: int = 512,
    child_overlap: int = 100
) -> List[dict]:
    """
    Split text into hierarchical parent-child passages for retrieval.

    This creates a two-level hierarchy where:
    - Parent chunks (large): Provide complete context for LLM
    - Child chunks (small): Enable precise retrieval matching

    Retrieval strategy: Search using child embeddings, return parent text to LLM.

    Args:
        text: Input text to split
        parent_length: Length of parent chunks (default: 2048 chars)
        parent_overlap: Overlap between parent chunks (default: 32 chars)
        child_length: Length of child chunks (default: 512 chars)
        child_overlap: Overlap between child chunks (default: 100 chars)

    Returns:
        List of dicts with keys:
        - 'child_passage': Small chunk text (for embedding/retrieval)
        - 'parent_passage': Large parent chunk text (for LLM context)
        - 'parent_id': ID of parent chunk
        - 'child_index': Index of this child within parent
    """
    # Clean up the text
    text = clean_text(text)

    if len(text) <= parent_length:
        # Single parent case - still create child chunks
        parent_text = text
        children = split_into_passages(
            parent_text, child_length, child_overlap)

        results = []
        for child_idx, child_text in enumerate(children):
            results.append({
                'child_passage': child_text,
                'parent_passage': parent_text,
                'parent_id': 0,
                'child_index': child_idx
            })
        return results

    # Split into parent chunks first
    parent_chunks = split_into_passages(text, parent_length, parent_overlap)

    # For each parent, split into children
    all_results = []
    for parent_id, parent_text in enumerate(parent_chunks):
        # Split parent into children
        children = split_into_passages(
            parent_text, child_length, child_overlap)

        # Create hierarchical entries
        for child_idx, child_text in enumerate(children):
            all_results.append({
                'child_passage': child_text,
                'parent_passage': parent_text,
                'parent_id': parent_id,
                'child_index': child_idx
            })

    return all_results
