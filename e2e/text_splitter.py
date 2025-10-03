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


def find_sentence_boundary(text: str, start: int, end: int, search_window: int = 100) -> int:
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
            # Check if it's followed by whitespace and uppercase letter (proper sentence end)
            if i + 1 < len(text) and text[i + 1].isspace():
                # Look for the next non-whitespace character
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                if j < len(text) and (text[j].isupper() or text[j].isdigit()):
                    best_break = i + 1
                    break
    
    return best_break


def split_into_passages(text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
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

        # If we're not at the end of the text, try to break at a sentence boundary
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


def split_into_fixed_passages(text: str, fixed_length: int = 256, overlap: int = 32) -> List[str]:
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


def create_passage_metadata(filename: str, passage_index: int, original_url: Optional[str] = None) -> dict:
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


def estimate_passage_count(text: str, max_length: int = 512, overlap: int = 50) -> int:
    """
    Estimate the number of passages that will be created from text.
    Useful for progress tracking without actually splitting.
    """
    if len(text) <= max_length:
        return 1 if text.strip() else 0
    
    # Rough estimate based on overlap
    effective_length = max_length - overlap
    return max(1, (len(text) - overlap) // effective_length)