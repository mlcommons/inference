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

from typing import List
from legacy_helpers import __levenshtein


def compute_wer_with_concatenation(prediction, reference):
    """
    Compute WER considering concatenated words as correct matches using kaldialign
    Args:
        reference: Reference text string
        prediction: Hypothesis text string
    Returns:
        WER score treating concatenated words as matches
    """

    ref_words = reference
    hyp_words = prediction

    # Create alignment matrix considering concatenations
    alignment = []
    i = 0
    j = 0

    while i < len(ref_words) and j < len(hyp_words):
        # Check exact match
        if ref_words[i] == hyp_words[j]:
            alignment.append((ref_words[i], hyp_words[j]))
            i += 1
            j += 1
            continue

        # Check concatenated matches
        ref_concat = ref_words[i]
        hyp_concat = hyp_words[j]

        # Try concatenating up to 3 words
        ref_match_len = 1
        hyp_match_len = 1
        match_found = False

        for k in range(1, 4):
            if i + k <= len(ref_words):
                ref_concat = ''.join(ref_words[i:i + k])
                if ref_concat == hyp_words[j]:
                    ref_match_len = k
                    hyp_match_len = 1
                    match_found = True
                    break

            if j + k <= len(hyp_words):
                hyp_concat = ''.join(hyp_words[j:j + k])
                if hyp_concat == ref_words[i]:
                    ref_match_len = 1
                    hyp_match_len = k
                    match_found = True
                    break

        if match_found:
            # Add concatenated match
            alignment.append((' '.join(ref_words[i:i + ref_match_len]),
                              ' '.join(hyp_words[j:j + hyp_match_len])))
            i += ref_match_len
            j += hyp_match_len

        else:
            # No match found, mark as substitution
            alignment.append((ref_words[i], hyp_words[j]))
            i += 1
            j += 1

    # Handle remaining words
    while i < len(ref_words):
        alignment.append((ref_words[i], None))
        i += 1
    while j < len(hyp_words):
        alignment.append((None, hyp_words[j]))
        j += 1

    # Calculate WER using kaldialign
    ref_aligned = [x[0].replace(" ", "")
                   for x in alignment if x[0] is not None]
    hyp_aligned = [x[1].replace(" ", "")
                   for x in alignment if x[1] is not None]
    distance = __levenshtein(ref_aligned, hyp_aligned)
    wer = distance / len(ref_words) if ref_words else 0

    return distance, len(ref_words) if ref_words else 0


def expand_concatenations(
        words_list: List, reference_dict: dict, reference_list: List):
    """
    Finds matching compound words in 'words_list' which exist as keys in 'reference_dict', if any.
    If found, the compound word will be separated using reference_dict if the substitution reduces
    the 'Levenshtein distance' between 'words_list' and 'reference_list'.
    Args:
        words_list: List of English word strings
        reference_dict: Dictionary mapping compound words to a list a separated word strings.
        reference_list: List of English word strings
    Returns:
        Modified 'word_string' with compound words replaced by individual strings, if any
    """
    score = __levenshtein(words_list, reference_list)

    # Searches each word in 'word_list' for separability using the reference list. Once all options are
    # considered, the modified 'word_list' is returned. Length of 'word_list'
    # can grow, but not contract.
    i = 0
    words_length = len(words_list)
    while i < words_length:
        if words_list[i] in reference_dict.keys():
            words_candidate = words_list[:i] + \
                reference_dict[words_list[i]] + words_list[i + 1:]

            # If levenshtein distance reduced, cache new word_list and resume
            # search
            candidate_levenshtein = __levenshtein(
                words_candidate, reference_list)
            if candidate_levenshtein < score:
                words_list = words_candidate
                words_length = len(words_list)
                score = candidate_levenshtein
        i += 1
    return words_list


def get_expanded_wordlist(words_list: List, reference_list: List):
    """
    Provided two lists of English words, the two will be compared, and any compound words found in
        'word_list' which are separated in 'reference_list' will be separated and the modified
        'word_list' will be returned.
    Args:
        word_list: List of English word strings
        reference_list: List of English word strings
    Returns:
        List of words modified from 'word_list' after expanding referenced compound words
    """

    # If levenshtein distance < 2, there cannot be any compound word
    # separation issues.
    if __levenshtein(words_list, reference_list) < 2:
        return words_list

    # Adding two-word compouding candidates to checklist
    checklist = {}
    for i in range(len(reference_list) - 1):
        compound = reference_list[i] + reference_list[i + 1]
        checklist[compound] = [reference_list[i], reference_list[i + 1]]

    # Adding three-word compounding candidates to checklist
    for i in range(len(reference_list) - 2):
        compound = reference_list[i] + \
            reference_list[i + 1] + reference_list[i + 2]
        checklist[compound] = [reference_list[i],
                               reference_list[i + 1], reference_list[i + 2]]

    # All compiled candidates will be checked, and after checking for minimal Levenshtein
    # distance, the modified list (or original if compounding not found) is
    # directly returned
    return expand_concatenations(words_list, checklist, reference_list)
