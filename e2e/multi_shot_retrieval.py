"""
Multi-shot Retrieval System

This module implements multi-shot retrieval with query decomposition:
1. Takes a complex query
2. Uses LLM to rewrite/decompose into multiple sub-queries (max k=3)
3. Retrieves documents for each sub-query
4. Optionally reranks combined results
5. Evaluates performance

Architecture:
    Prompt → Query Rewriter (LLM) → k Sub-queries → Retrieval → Reranking → Evaluation
"""

import argparse
import json
import time
import os
from typing import List, Dict, Any, Optional
import pandas as pd

# Set no_proxy to bypass proxy for localhost/127.0.0.1
original_no_proxy = os.environ.get('no_proxy', '')
os.environ['no_proxy'] = '127.0.0.1,localhost,' + original_no_proxy
os.environ['NO_PROXY'] = '127.0.0.1,localhost,' + original_no_proxy

from retrieve import VectorDB, BM25DB
from evaluation import evaluate_retrieval_query, run_evaluation
from utils import (set_deterministic_seeds, filter_dataset_by_difficulty, 
                   setup_llm_config, get_device_config)
from params import add_all_args
import requests

# Prompts
QUERY_REWRITER_PROMPT = """\
You evaluate documents and generate search queries for complex multi-hop questions.

QUESTION: {question}
DOCUMENTS: {context}
SEARCH HISTORY: {history}
FEEDBACK: {feedback_history}

TASK 1: EVALUATE AND SUMMARIZE NEW DOCUMENTS
**ONLY if there are NEW documents to evaluate:**
For each NEW document:
- Mark: 1 if relevant to any part of question, 0 if irrelevant
- MANDATORY: If marked as relevant (1), you MUST provide a detailed summary preserving ALL key facts, names, dates, numbers, and relationships mentioned in the document that could be useful for answering the question
- If marked as irrelevant (0), provide empty string ""

**If there are NO NEW documents to evaluate, skip this task and go to TASK 3**

SUMMARY REQUIREMENTS: 
- Include specific names, dates, locations, numbers, and relationships
- Preserve mother's names, maiden names, birth places, assassination details, etc.
- Keep family relationships and biographical details intact
- Don't just say "contains information about X" - specify WHAT information

TASK 2: CHECK IF SUFFICIENT AND CONNECT INFORMATION
Review KEPT documents and summaries. Look for connections between entities:
- If you have "Person A is the 15th X" and "Person A's mother is Y", you can connect them
- If you have "Event happened in Year Z" and "Person was at Position in Year Z", connect them
- If you have partial chain (A→B, B→C), look for missing links to complete (A→B→C→D)
If you have all facts with clear evidence and connections, provide final answer.

TASK 3: GENERATE QUERIES (if not sufficient)
Decompose the complex question into at most {k} simpler sub-queries that, when answered together, would help answer the original question.
First, analyze failed queries in SEARCH HISTORY. For each failed query (0 docs), identify WHY it failed:
- Too specific/combined terms: Break into single entities  
- Wrong terminology: Try official names, abbreviations, alternative spellings
- Missing context: Search broader category first

MULTI-HOP STRATEGY by reasoning type:
- Temporal reasoning: Search specific years, dates, "as of [date]", event timelines, chronological lists
- Multiple constraints: Find each constraint separately (e.g., "15th first lady" then "her mother" separately)  
- Tabular/Numerical reasoning: Search for data tables, census data, statistics, rankings, population figures
- Entity chains: Break chains (Person → Birth location → Population data, or Event → Year → Other events that year)

ATOMIC EXAMPLES (use these patterns for ANY complex question):
- Entity + Attribute: "Barack Obama birth place", "Tesla Model S price", "Nuclear Blast first Billboard 200 band"
- Family Relationships: "Harriet Lane mother", "James Garfield family", "Pietro Barbo papal name" 
- Temporal Queries: "FIFA World Cup winner 2018", "London Champions League winners", "Charlotte Bronte 1847 book"
- Numerical/Rankings: "Dewey decimal classification literature", "tallest buildings New York 2024", "Billboard 200 chart history"
- Specific Names: Always use exact names from documents rather than descriptions

QUERY GENERATION RULES (CRITICAL - READ CAREFULLY):
1. **USE SPECIFIC NAMES FROM DOCUMENTS**: If you found "Harriet Lane" in documents, search "Harriet Lane mother" NOT "15th first lady mother"
2. **USE EXACT KEYWORDS**: If documents mention "Nuclear Blast", "James Garfield", "Pietro Barbo" - USE THESE EXACT NAMES
3. **CONNECT INFORMATION**: If you have "Harriet Lane = 15th first lady" and need her mother, search "Harriet Lane mother Jane"
4. **ONE CONCEPT PER QUERY**: Don't combine multiple facts in one search
5. **AVOID DESCRIPTIONS**: Never use "second assassinated president" if you found "James Garfield"
6. **NEVER REPEAT SEARCHES**: Check SEARCH HISTORY carefully to avoid duplicates
7. **BUILD ON FINDINGS**: Use names/facts from summaries to create precise next searches

RESPONSE FORMAT:
**If there are NEW documents:**
If sufficient: {{"relevance": [1,0,1], "summaries": ["Entity A has attribute X and connection Y", "", "Entity B was discovered by Person C in year Z"], "answer": "Final answer based on connected facts"}}
If not: {{"relevance": [1,0,1], "summaries": ["Entity A (dates) served as Role under Person B, making them the Nth person", "", "Entity C was Event in Year, making them the Position after Entity D"], "queries": ["Entity A specific detail", "Entity C family information"], "feedback": "Missing: Specific details needed. Next: Search using exact names found."}}

**If there are NO NEW documents:**
{{"relevance": [], "summaries": [], "queries": ["main entity keyword", "secondary entity keyword"], "feedback": "Starting search for required information."}}

EXAMPLES OF CONNECTING INFORMATION:
- "Jane Eyre by Charlotte Brontë published 1847, Dewey classification 823 for English fiction"
- "Chelsea won UEFA Champions League 2012, France held FIFA World Cup at that time"  
- "Meshuggah first Nuclear Blast band on Billboard 200 at position 165, vocalist Jens Kidman"

CRITICAL REQUIREMENTS:
- If NO NEW documents: return empty arrays: "relevance": [], "summaries": []  
- If {len_new_docs} NEW documents: return exactly {len_new_docs} relevance scores and {len_new_docs} summaries
- For relevant docs (relevance=1): summary MUST contain specific facts, names, dates, numbers
- For irrelevant docs (relevance=0): summary MUST be empty string ""
- USE EXACT NAMES from documents in your next queries, not generic descriptions
- Connect information: if you found "X is Y" and need "Y's Z", search "X Z" using the specific name
- Each query must be genuinely different from SEARCH HISTORY and use specific keywords
- Count your arrays before responding - this is critical for system stability

FORMAT VALIDATION: Array lengths MUST match document count - double check before responding!
KEYWORD USAGE: Always use specific names/terms found in documents for next queries!
Respond only in JSON format"""


def summarize_document(doc_content: str, max_length: int = 100) -> str:
    """Summarize document content to key facts."""
    # Simple extractive summarization - take first sentence and key phrases
    sentences = doc_content.split('.')[:2]  # First 2 sentences
    summary = '. '.join(sentences).strip()
    
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    
    return summary if summary else doc_content[:max_length] + "..."


def query_rewriter(question: str, new_documents: List[str],
                   kept_documents: List[str],
                   max_queries: int = 3,
                   reasoning_effort: str = "medium",
                   query_history: Optional[List[str]] = None,
                   query_results: Optional[List[int]] = None,
                   previous_feedback: str = "",
                   feedback_history: Optional[List[str]] = None,
                   llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Evaluates documents AND generates new queries in one LLM call.
    
    Args:
        question: The user's original question
        new_documents: List of NEW document texts to evaluate
        kept_documents: List of KEPT document texts (already marked relevant)
        max_queries: Maximum number of new queries to generate
        reasoning_effort: LLM reasoning level
        query_history: List of previous search queries
        query_results: List of number of documents found for each query (parallel to query_history)
        previous_feedback: Feedback from previous iteration about what's missing
        
    Returns:
        Dict with:
        - 'relevance' (list of 0/1 for ONLY new_documents)
        - 'queries' (list of new search queries, empty if answer provided)
        - 'feedback' (what's missing, empty if answer provided)
        - 'answer' (final answer if sufficient, empty otherwise)
    """
    # Format KEPT documents with summaries
    kept_context = ""
    if kept_documents:
        for i, doc in enumerate(kept_documents, 1):
            # Use the doc as-is if it's already a summary, otherwise summarize
            if len(doc) <= 200:  # Likely already a summary
                kept_context += f"\n[KEPT {i}] {doc}\n"
            else:
                summary = summarize_document(doc, max_length=150)
                kept_context += f"\n[KEPT {i}] {summary}\n"
    else:
        kept_context = "None"
    
    # Format NEW documents with summaries
    new_context = ""
    if new_documents:
        for i, doc in enumerate(new_documents, 1):
            summary = summarize_document(doc, max_length=150)
            new_context += f"\n[NEW {i}] {summary}\n"
    else:
        new_context = "None"
    
    # Combine for context
    context = f"KEPT DOCUMENTS (already relevant):\n{kept_context}\n\nNEW DOCUMENTS (evaluate these):\n{new_context}"
    
    # Format query history with results - focus on failures for learning
    if query_history:
        failed_queries = []
        successful_queries = []
        if query_results and len(query_results) == len(query_history):
            for q, num_docs in zip(query_history, query_results):
                if num_docs == 0:
                    failed_queries.append(q)
                else:
                    successful_queries.append(f"{q} ({num_docs} docs)")
        
        history_parts = []
        if failed_queries:
            history_parts.append(f"FAILED: {', '.join(failed_queries[-3:])}")  # Last 3 failures
        if successful_queries:
            history_parts.append(f"SUCCESS: {', '.join(successful_queries[-2:])}")  # Last 2 successes
        
        history_text = "; ".join(history_parts) if history_parts else "No queries yet"
    else:
        history_text = "No queries yet"
    
    # Build feedback history - show progression of what was tried and learned
    if feedback_history and len(feedback_history) > 0:
        # Show last 2-3 unique feedback entries to show progression
        unique_feedback = []
        for fb in reversed(feedback_history):
            if fb and fb not in unique_feedback:
                unique_feedback.append(fb)
            if len(unique_feedback) >= 3:
                break
        
        if unique_feedback:
            feedback_text = "PREVIOUS ATTEMPTS: " + " → ".join(reversed(unique_feedback))
        else:
            feedback_text = f"Iteration {len(query_history) + 1 if query_history else 1}"
    else:
        feedback_text = f"Iteration {len(query_history) + 1 if query_history else 1} - Initial search"
    
    prompt = QUERY_REWRITER_PROMPT.format(
        question=question,
        context=context,
        history=history_text,
        feedback_history=feedback_text,
        k=max_queries,
        len_new_docs=len(new_documents)
    )
    
    system_message = f"You are an expert at multi-hop reasoning and strategic search. CRITICAL: Never repeat failed queries. Always try completely different approaches when queries return 0 docs. Focus on atomic facts and progressive strategies. Reasoning: {reasoning_effort}."
    
    # Use LLM config if provided, otherwise use defaults
    if llm_config:
        model_name = llm_config["model_name"]
        service_url = llm_config["service_url"]
        max_tokens = llm_config["max_tokens"]
    else:
        model_name = "/mnt/weka/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct/"
        service_url = "http://127.0.0.1:8123/v1/chat/completions"
        max_tokens = 10240

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(service_url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        message = result['choices'][0]['message']
        llm_output = message.get('content')
        
        # For debugging: check if reasoning_content exists
        reasoning_content = message.get('reasoning_content', '')
        if reasoning_content and not llm_output:
            print(f"    DEBUG: reasoning_content exists but content is empty")
            print(f"    DEBUG: reasoning_content snippet: {reasoning_content[:200]}")
        
        # Use only content field
        if llm_output is None or not llm_output.strip():
            print(f"    Warning: LLM returned empty content")
            return {
                "relevance": [0] * len(new_documents),
                "summaries": [""] * len(new_documents),
                "queries": [question] if not query_history else [],
                "feedback": "LLM returned empty response",
                "answer": ""
            }
        
        llm_output = llm_output.strip()
        
        print(f"    Combined output: {llm_output[:200]}...")
        
        # Parse JSON output - handle markdown code blocks
        if llm_output.startswith("```"):
            llm_output = llm_output.split("```")[1]
            if llm_output.startswith("json"):
                llm_output = llm_output[4:]
            llm_output = llm_output.strip()
        
        result_data = json.loads(llm_output)
        
        # Validate format - no longer require "sufficient" field
        required_fields = ["relevance"]
        for field in required_fields:
            if field not in result_data:
                print(f"Warning: Missing required field '{field}' in response")
                result_data[field] = [0] * len(new_documents)
        
        # Ensure we have either "answer" OR "queries"+"feedback"
        if "answer" not in result_data:
            result_data["answer"] = ""
        if "queries" not in result_data:
            result_data["queries"] = []
        if "feedback" not in result_data:
            result_data["feedback"] = ""
        if "summaries" not in result_data:
            result_data["summaries"] = [""] * len(new_documents)
        
        # Ensure relevance array matches NEW document count - fix mismatches by padding/truncating
        if len(result_data["relevance"]) != len(new_documents):
            print(f"Warning: Relevance array length mismatch. Expected {len(new_documents)}, got {len(result_data['relevance'])}. Auto-fixing.")
            relevance = result_data["relevance"][:len(new_documents)]  # Truncate if too long
            while len(relevance) < len(new_documents):  # Pad with 0s if too short
                relevance.append(0)
            result_data["relevance"] = relevance
            print(f"Fixed relevance array: {relevance}")
        
        # Ensure summaries array matches NEW document count - fix mismatches by padding/truncating  
        if len(result_data["summaries"]) != len(new_documents):
            print(f"Warning: Summaries array length mismatch. Expected {len(new_documents)}, got {len(result_data['summaries'])}. Auto-fixing.")
            summaries = result_data["summaries"][:len(new_documents)]  # Truncate if too long
            while len(summaries) < len(new_documents):  # Pad with empty strings if too short
                summaries.append("")
            result_data["summaries"] = summaries
            print(f"Fixed summaries array length: {len(summaries)}")
        
        # Validate that relevant documents have non-empty summaries
        for i, (rel, summary) in enumerate(zip(result_data["relevance"], result_data["summaries"])):
            if rel == 1 and not summary.strip():
                print(f"Warning: Document {i+1} marked relevant but has empty summary. This defeats the summarization purpose.")
                # Don't auto-fix here - let it be empty to debug the issue
        
        # Ensure queries is a list
        if not isinstance(result_data["queries"], list):
            result_data["queries"] = []
        
        return result_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling combined LLM: {e}")
        return {
            "relevance": [0] * len(new_documents),
            "summaries": [""] * len(new_documents),
            "queries": [question] if not query_history else [],
            "feedback": f"API error: {str(e)}",
            "answer": ""
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM output: {e}")
        print(f"LLM output: {llm_output[:200]}")
        return {
            "relevance": [0] * len(new_documents),
            "summaries": [""] * len(new_documents),
            "queries": [question] if not query_history else [],
            "feedback": f"JSON parse error: {str(e)}",
            "answer": ""
        }
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "relevance": [0] * len(new_documents),
            "summaries": [""] * len(new_documents),
            "queries": [question] if not query_history else [],
            "feedback": f"Unexpected error: {str(e)}",
            "answer": ""
        }


def query_rewriter_llm(original_query: str, max_queries: int = 3, reasoning_effort: str = "medium",
                       history: Optional[List[str]] = None, retrieved_docs: Optional[List[str]] = None,
                       llm_config: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Use LLM to decompose a complex query into multiple sub-queries, or generate new queries
    based on iterative feedback.
    
    Args:
        original_query: The original complex query
        max_queries: Maximum number of sub-queries to generate (default: 3)
        reasoning_effort: LLM reasoning level (low/medium/high)
        history: Optional list of previous search queries (for iterative mode)
        retrieved_docs: Optional list of retrieved document texts (for iterative mode)
        
    Returns:
        List of sub-queries
    """
    
    # Determine if this is initial decomposition or iterative refinement
    is_iterative = history is not None and retrieved_docs is not None
    
    if is_iterative:
        # Iterative mode: Generate new queries based on what's been retrieved
        history_text = "\n".join(f"- {q}" for q in history) if history else "None yet"
        
        results_text = ""
        for i, doc in enumerate(retrieved_docs, 1):
            results_text += f"\n[Document {i}]\n{doc[:300]}...\n"
        
        if not results_text:
            results_text = "None yet"
        
        prompt = QUERY_REWRITER_ITERATIVE_PROMPT.format(
            k=max_queries,
            user_question=original_query,
            history=history_text,
            results=results_text
        )
        system_prompt = f"You are a helpful assistant that generates search queries. Reasoning: {reasoning_effort}."
    else:
        # Initial decomposition mode
        system_prompt = f"""You are an expert at decomposing complex multi-hop questions into simpler sub-questions. Reasoning: {reasoning_effort}.

Your task: Given a complex question, break it down into 1-{max_queries} simpler sub-questions that, when answered together, would help answer the original question.

Guidelines:
1. Identify the key facts/entities needed to answer the question
2. Create sub-questions that retrieve each piece of information
3. Order sub-questions logically (dependencies first)
4. Keep sub-questions clear and specific
5. If the question is already simple, return just the original question

Output format: Return ONLY a JSON array of sub-questions, nothing else.
Example: ["What year did X happen?", "Who won Y in that year?"]
"""
        prompt = f"""Original question: {original_query}

Decompose this into at most {max_queries} sub-questions. Return only the JSON array."""

    # Use LLM config if provided, otherwise use defaults
    if llm_config:
        model_name = llm_config["model_name"]
        service_url = llm_config["service_url"]
        max_tokens = llm_config["max_tokens"]
    else:
        model_name = "/mnt/weka/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct/"
        service_url = "http://127.0.0.1:8123/v1/chat/completions"
        max_tokens = 10240

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(service_url, json=payload, timeout=60)
        
        if response.status_code != 200:
            print(f"Error response: {response.text[:500]}")
            print(f"Falling back to original query")
            return [original_query]
        
        result = response.json()
        
        # Use only content field
        message = result['choices'][0]['message']
        llm_output = message.get('content')
        
        # For debugging: check if reasoning_content exists
        reasoning_content = message.get('reasoning_content', '')
        if reasoning_content and not llm_output:
            print(f"DEBUG: reasoning_content exists but content is empty")
            print(f"DEBUG: reasoning_content snippet: {reasoning_content[:200]}")
        
        if llm_output is None or not llm_output.strip():
            print(f"Warning: LLM returned empty content for query rewriting")
            return [original_query]
        
        llm_output = llm_output.strip()
        
        if not is_iterative:
            print(f"LLM output: {llm_output[:200]}...")
        
        # Parse JSON output - handle markdown code blocks
        if llm_output.startswith("```json"):
            llm_output = llm_output.replace("```json", "").replace("```", "").strip()
        elif llm_output.startswith("```"):
            llm_output = llm_output.split("```")[1]
            if llm_output.startswith("json"):
                llm_output = llm_output[4:]
            llm_output = llm_output.strip()
        
        sub_queries = json.loads(llm_output)
        
        # Validate output
        if not isinstance(sub_queries, list):
            print(f"Warning: LLM output is not a list, using original query")
            return [original_query]
        
        # Limit to max_queries
        sub_queries = sub_queries[:max_queries]
        
        # Ensure at least the original query is included if list is empty
        if not sub_queries:
            return [original_query]
        
        if not is_iterative:
            print(f"\n{'='*80}")
            print(f"QUERY DECOMPOSITION")
            print(f"{'='*80}")
            print(f"Original: {original_query}")
            print(f"Sub-queries ({len(sub_queries)}):")
            for i, sq in enumerate(sub_queries, 1):
                print(f"  {i}. {sq}")
            print(f"{'='*80}\n")
        
        return sub_queries
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM service: {e}")
        print(f"Falling back to original query")
        return [original_query]
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM output as JSON: {e}")
        print(f"LLM output: {llm_output[:200]}")
        print(f"Falling back to original query")
        return [original_query]
    except Exception as e:
        print(f"Unexpected error in query rewriting: {e}")
        import traceback
        traceback.print_exc()
        print(f"Falling back to original query")
        return [original_query]


def multi_shot_retrieval(rag_db, original_query: str, expected_urls: List[str],
                         expected_answer: str = "",
                         max_sub_queries: int = 3,
                         top_k_retriever: int = 10,
                         top_k_reranking: int = 10,
                         max_iterations: int = 10,
                         no_rerank: bool = False,
                         retrieval_strategy: str = "fixed_k",
                         verbose: bool = True,
                         reasoning_effort: str = "medium",
                         llm_config: Optional[Dict[str, Any]] = None,
                         **strategy_params) -> Dict[str, Any]:
    """
    Multi-shot retrieval with iterative query refinement and document evaluation.
    
    Algorithm:
    1. Generate initial search queries based on the original question
    2. Retrieve documents for each query
    3. Evaluate documents and check if sufficient to answer
    4. If not sufficient: generate new queries based on what's missing, go to step 2
    5. Repeat until sufficient or max_iterations reached
    
    Args:
        rag_db: RAG database instance
        original_query: Original user question
        expected_urls: Expected ground truth URLs for evaluation
        max_sub_queries: Maximum number of sub-queries per iteration
        top_k_retriever: Number of documents to retrieve per sub-query
        top_k_reranking: Final number of documents to return
        max_iterations: Maximum number of retrieval iterations (default: 10)
        no_rerank: Skip reranking step
        retrieval_strategy: Strategy for retrieval
        verbose: Print detailed information
        reasoning_effort: LLM reasoning level
        **strategy_params: Additional parameters for retrieval strategy
        
    Returns:
        Dictionary containing evaluation metrics and iteration statistics
    """
    
    start_time = time.perf_counter()
    
    # Track iteration history
    query_history = []
    query_results = []  # Track how many docs each query found
    kept_docs = []  # List of (url, content, summary) tuples that were marked relevant
    new_docs = []   # List of (url, content) tuples just retrieved this iteration
    all_retrieved_urls = set()
    iteration_times = []
    previous_feedback = ""  # Feedback from previous iteration
    feedback_history = []  # Track all feedback to show progression
    
    sufficient = False
    iteration = 0
    final_answer = ""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"MULTI-SHOT RETRIEVAL")
        print(f"{'='*80}")
        print(f"Original Query: {original_query}")
        print(f"Max iterations: {max_iterations}")
        print(f"Max sub-queries per iteration: {max_sub_queries}")
        print(f"{'='*80}\n")
    
    while not sufficient and iteration < max_iterations:
        iteration += 1
        iteration_start = time.perf_counter()
        
        if verbose:
            print(f"\n{'─'*80}")
            print(f"ITERATION {iteration}/{max_iterations}")
            print(f"{'─'*80}")
        
        # Step 1: Use combined function to grade NEW docs AND generate new queries
        if verbose:
            print(f"\n  Evaluating documents and generating queries...")
        
        # Aggressive summarization: use summaries after iteration 2 to improve information connection
        total_content_length = sum(len(doc[1]) for doc in kept_docs)
        # Use summaries if we have more than 2 docs OR after iteration 2, to help LLM connect information better
        use_summaries = len(kept_docs) > 2 or iteration > 2
        
        if use_summaries:
            kept_contents = [doc[2] if len(doc) >= 3 and doc[2] else doc[1] for doc in kept_docs]
            if verbose:
                summary_count = sum(1 for doc in kept_docs if len(doc) >= 3 and doc[2])
                print(f"    Using {summary_count}/{len(kept_docs)} document summaries for better information connection")
        else:
            kept_contents = [doc[1] for doc in kept_docs]  # Use full content for first few iterations
            if verbose:
                print(f"    Using full document content for {len(kept_docs)} docs (early iteration for detail)")
        new_contents = [doc[1] for doc in new_docs]
        print("    new contents", new_contents)

        result = query_rewriter(
            original_query, 
            new_documents=new_contents,
            kept_documents=kept_contents,
            max_queries=max_sub_queries,
            reasoning_effort=reasoning_effort,
            query_history=query_history,
            query_results=query_results,
            previous_feedback=previous_feedback,
            feedback_history=feedback_history,
            llm_config=llm_config
        )
        
        # Check if we have an answer (sufficient)
        sufficient = bool(result.get("answer", "").strip())
        relevance = result["relevance"]  # Only for NEW documents
        summaries = result.get("summaries", [""] * len(new_contents))  # Summaries for NEW documents
        sub_queries = result["queries"]
        current_feedback = result["feedback"]
        final_answer = result.get("answer", "")
        reasoning_steps = result.get("reasoning", "")
        
        # Add to feedback history if it's new and meaningful
        if current_feedback and current_feedback.strip() and current_feedback != previous_feedback:
            feedback_history.append(current_feedback.strip())
        
        previous_feedback = current_feedback
        
        if verbose:
            print(f"    Sufficient: {'yes' if sufficient else 'no'}")
            print(f"    Kept docs: {len(kept_docs)}")
            if new_contents:
                print(f"    New docs evaluated: {len(new_contents)}")
                print(f"    Relevant new docs: {sum(relevance)}/{len(relevance)}")
                print(f"    Relevance array: {relevance}")
                # Show summary quality
                if summaries:
                    non_empty_summaries = [s for s in summaries if s.strip()]
                    print(f"    Generated summaries: {len(non_empty_summaries)}/{len(summaries)} non-empty")
                    for i, summary in enumerate(summaries):
                        if summary.strip() and relevance[i] == 1:
                            print(f"      Summary {i+1}: {summary[:100]}...")
            if reasoning_steps:
                print(f"    Reasoning: {reasoning_steps[:300]}...")
            if not sufficient:
                print(f"    Feedback: {previous_feedback}")
                print(f"    Generated {len(sub_queries)} new queries")
        
        # Filter NEW documents by relevance and add to kept_docs
        if new_contents:
            for i, (url, content) in enumerate(new_docs):
                if i < len(relevance) and relevance[i] == 1:
                    summary = summaries[i] if i < len(summaries) and summaries[i] else ""
                    kept_docs.append((url, content, summary))
            
            if verbose:
                print(f"    Added {sum(relevance)} new docs to kept set")
                print(f"    Total kept docs now: {len(kept_docs)}")
        
        # Clear new_docs for next iteration
        new_docs = []
        
        # If sufficient, we're done
        if sufficient:
            if verbose:
                print(f"\n  ✓ Sufficient information found!")
                if final_answer:
                    print(f"  Answer: {final_answer[:200]}...")
            iteration_times.append(time.perf_counter() - iteration_start)
            break
        
        # If no queries generated, break
        if not sub_queries:
            if verbose:
                print(f"\n  ⚠ No new queries generated, stopping")
            iteration_times.append(time.perf_counter() - iteration_start)
            break
        
        if verbose:
            print(f"\n  New queries:")
            for i, q in enumerate(sub_queries, 1):
                print(f"    {i}. {q}")
        
        # Step 2: Retrieve for each sub-query and track results
        num_sub_queries = len(sub_queries)
        #docs_per_subquery = max(1, top_k_retriever // num_sub_queries)
        docs_per_subquery = max(1, top_k_retriever)
        
        iteration_results = []
        per_query_counts = []  # Track new docs found by each query
        
        # Calculate target docs per subquery after reranking
        target_docs_per_subquery = max(3, top_k_retriever // num_sub_queries)
        
        for i, sub_query in enumerate(sub_queries, 1):
            if verbose:
                print(f"\n  Retrieving for query {i}: {sub_query[:60]}...")
            
            query_start_count = len(new_docs)  # Track docs before this query
            
            # Retrieve
            if retrieval_strategy == "fixed_k":
                results = rag_db.lookup(sub_query, k=docs_per_subquery)
            else:
                from retrieve.filter import filter
                original_max_results = strategy_params.get("max_results", 20)
                #adjusted_max_results = max(1, original_max_results // num_sub_queries)
                adjusted_max_results = max(1, original_max_results)
                strategy_params_copy = strategy_params.copy()
                strategy_params_copy["max_results"] = adjusted_max_results
                results = filter(rag_db, sub_query, method=retrieval_strategy, **strategy_params_copy)
            
            # Apply per-subquery reranking if enabled
            if not no_rerank and len(results) > target_docs_per_subquery:
                if verbose:
                    print(f"    Reranking {len(results)} docs for this subquery to top {target_docs_per_subquery}...")
                
                # Extract contents for reranking
                contents = [r.page_content for r in results]
                scored_passages = rag_db.rerank(sub_query, contents)
                
                # Reorder results by reranking scores and take top-k
                reranked_indices = [i for i, _ in sorted(enumerate(scored_passages), 
                                                         key=lambda x: x[1][1], reverse=True)]
                results = [results[idx] for idx in reranked_indices[:target_docs_per_subquery]]
                
                if verbose:
                    print(f"    After reranking: keeping top {len(results)} docs")
            elif len(results) > target_docs_per_subquery:
                # No reranking, just limit to target
                results = results[:target_docs_per_subquery]
            
            # Add to new_docs for evaluation (avoid duplicates)
            for result in results:
                if 'original_url' in result.metadata and result.metadata['original_url']:
                    url = result.metadata['original_url']
                    if url not in all_retrieved_urls:
                        all_retrieved_urls.add(url)
                        new_docs.append((url, result.page_content))
                        iteration_results.append(result)
            
            # Track how many NEW docs this query found
            docs_found_by_query = len(new_docs) - query_start_count
            per_query_counts.append(docs_found_by_query)
            
            if verbose:
                print(f"    Retrieved {len(results)} docs, {docs_found_by_query} new unique docs from this query")
        
        # Add queries and their results to history
        for sub_query, count in zip(sub_queries, per_query_counts):
            query_history.append(sub_query)
            query_results.append(count)
        
        if verbose:
            print(f"  Total kept docs: {len(kept_docs)}, new docs to evaluate: {len(new_docs)}")
        
        iteration_time = time.perf_counter() - iteration_start
        iteration_times.append(iteration_time)
        
        if iteration >= max_iterations:
            if verbose:
                print(f"\n  ⚠ Maximum iterations reached")
            break
    
    # Final processing
    total_time = time.perf_counter() - start_time
    
    # Extract URLs from kept_docs
    retrieved_urls = []
    for doc in kept_docs:
        if len(doc) >= 3:
            retrieved_urls.append(doc[0])  # url is first element
        elif len(doc) == 2:  # Handle old format for backward compatibility
            retrieved_urls.append(doc[0])  # url is first element
    
    # Limit to top_k_reranking (reranking already done per-subquery)
    retrieved_urls = retrieved_urls[:top_k_reranking]
    
    # Calculate metrics
    from evaluation import calculate_retrieval_metrics
    expected_set = set(url for url in expected_urls if url and url.strip())
    metrics = calculate_retrieval_metrics(list(expected_set), retrieved_urls)
    
    # Add iteration statistics
    metrics.update({
        'total_time': total_time,
        'num_iterations': iteration,
        'total_queries': len(query_history),
        'final_docs_count': len(retrieved_urls),
        'sufficient': sufficient,
        'avg_iteration_time': sum(iteration_times) / len(iteration_times) if iteration_times else 0
    })
    
    # Print final results
    if verbose:
        print(f"\n{'='*80}")
        print(f"MULTI-SHOT RETRIEVAL RESULTS")
        print(f"{'='*80}")
        print(f"Original Query: {original_query[:100]}...")
        print(f"Iterations: {iteration}")
        print(f"Total queries issued: {len(query_history)}")
        print(f"Sufficient: {'Yes' if sufficient else 'No'}")
        if final_answer:
            print(f"LLM Answer: {final_answer}")
        if expected_answer:
            print(f"Expected Answer: {expected_answer}")
        print(f"Expected ({len(expected_set)}): {sorted(list(expected_set)[:3])}{'...' if len(expected_set) > 3 else ''}")
        print(f"Retrieved ({len(retrieved_urls)} unique docs): {retrieved_urls[:3]}{'...' if len(retrieved_urls) > 3 else ''}")
        matches = len(expected_set.intersection(set(retrieved_urls)))
        print(f"Matches: {matches}")
        print(f"\nMetrics:")
        print(f"  P@N: {metrics.get('precision@N', 0.0):.3f}")
        print(f"  R@N: {metrics.get('recall@N', 0.0):.3f}")
        print(f"  F1@N: {metrics.get('f1@N', 0.0):.3f}")
        print(f"  MAP: {metrics.get('average_precision', 0.0):.3f}")
        print(f"\nTiming:")
        print(f"  Avg per iteration: {metrics['avg_iteration_time']*1000:.1f}ms")
        print(f"  Total: {total_time*1000:.1f}ms")
        print(f"{'='*80}\n")
    
    return metrics


def run_multi_shot_evaluation(rag_db, dataset_path: str,
                              max_sub_queries: int = 3,
                              top_k_retriever: int = 10,
                              top_k_reranking: int = 10,
                              max_queries: Optional[int] = None,
                              no_rerank: bool = False,
                              retrieval_strategy: str = "fixed_k",
                              reasoning_effort: str = "medium",
                              detailed_analysis: bool = False,
                              difficulty: int = 0,
                              max_iterations: int = 10,
                              llm_config: Optional[Dict[str, Any]] = None,
                              **strategy_params) -> Dict[str, float]:
    """
    Run multi-shot evaluation on a dataset.
    
    Args:
        rag_db: RAG database instance
        dataset_path: Path to dataset TSV file
        max_sub_queries: Maximum number of sub-queries per query
        top_k_retriever: Number of documents to retrieve per sub-query
        top_k_reranking: Number of documents after final reranking
        max_queries: Maximum number of queries to evaluate
        no_rerank: Skip reranking step
        retrieval_strategy: Strategy for retrieval
        reasoning_effort: LLM reasoning level
        detailed_analysis: Enable detailed complexity-based analysis
        difficulty: Minimum number of answer links required (0 = no filtering)
        max_iterations: Maximum iterations for iterative retrieval (default: 10)
        **strategy_params: Additional parameters for retrieval strategy
        
    Returns:
        Dictionary of averaged metrics
    """
    
    df = pd.read_csv(dataset_path, sep='\t')
    
    # Filter by difficulty if specified
    df = filter_dataset_by_difficulty(df, difficulty)
    
    if isinstance(max_queries, int) and max_queries > 0:
        df = df.head(max_queries)
    else:
        max_queries = len(df)
    
    print(f"\n{'='*80}")
    print(f"MULTI-SHOT EVALUATION")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_path}")
    print(f"Queries: {max_queries}")
    print(f"Max sub-queries: {max_sub_queries}")
    print(f"Retrieval strategy: {retrieval_strategy}")
    print(f"LLM reasoning effort: {reasoning_effort}")
    print(f"Detailed analysis: {detailed_analysis}")
    print(f"Max iterations: {max_iterations}")
    if difficulty > 0:
        print(f"Difficulty filter: >= {difficulty} answer links")
    print(f"{'='*80}\n")
    
    total_metrics = {}
    valid_queries = 0
    all_query_metrics = []  # For detailed analysis
    
    for idx, row in df.iterrows():
        print(f"\n[Query {idx+1}/{max_queries}]")
        
        # Extract expected URLs
        expected_urls = []
        for col in df.columns:
            if col.startswith('wikipedia_link_') and pd.notna(row[col]):
                expected_urls.append(row[col].strip())
        
        # Extract expected answer
        expected_answer = row.get('Answer', '').strip() if 'Answer' in row and pd.notna(row.get('Answer')) else ""
        
        if expected_urls:
            # Multi-shot retrieval with iterative refinement
            metrics = multi_shot_retrieval(
                rag_db, row['Prompt'], expected_urls,
                expected_answer=expected_answer,
                max_sub_queries=max_sub_queries,
                top_k_retriever=top_k_retriever,
                top_k_reranking=top_k_reranking,
                max_iterations=max_iterations,
                no_rerank=no_rerank,
                retrieval_strategy=retrieval_strategy,
                verbose=True,
                reasoning_effort=reasoning_effort,
                llm_config=llm_config,
                **strategy_params
            )
            
            # Accumulate metrics
            for metric_name, value in metrics.items():
                if metric_name not in total_metrics:
                    total_metrics[metric_name] = 0.0
                total_metrics[metric_name] += value
            
            valid_queries += 1
            
            # Collect metrics for detailed analysis
            if detailed_analysis:
                all_query_metrics.append(metrics.copy())
    
    if valid_queries > 0:
        # Calculate averages
        avg_metrics = {name: total / valid_queries for name, total in total_metrics.items()}
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"MULTI-SHOT EVALUATION SUMMARY ({valid_queries} queries)")
        print(f"{'='*80}")
        print(f"\nPRECISION METRICS:")
        print(f"  Precision@N:                {avg_metrics.get('precision@N', 0.0):.3f}")
        print(f"\nRECALL METRICS:")
        print(f"  Recall@N:                   {avg_metrics.get('recall@N', 0.0):.3f}")
        print(f"\nF1 METRICS:")
        print(f"  F1@N:                       {avg_metrics.get('f1@N', 0.0):.3f}")
        print(f"\nRANKING METRICS:")
        print(f"  Mean Average Precision:     {avg_metrics.get('average_precision', 0.0):.3f}")
        print(f"\nRETRIEVAL STATISTICS:")
        print(f"  Avg Sub-queries:            {avg_metrics.get('num_sub_queries', 0.0):.1f}")
        print(f"  Avg Passages Retrieved:     {avg_metrics.get('retrieved_passages_count', 0.0):.1f}")
        print(f"  Avg Unique Docs (N):        {avg_metrics.get('retrieved_docs_count', 0.0):.1f}")
        print(f"\nTIMING:")
        print(f"  Avg Decomposition Time:     {avg_metrics.get('decomposition_time', 0.0)*1000:.1f}ms")
        print(f"  Avg Retrieval Time:         {avg_metrics.get('retrieval_time', 0.0)*1000:.1f}ms")
        if avg_metrics.get('reranking_time', 0.0) > 0:
            print(f"  Avg Reranking Time:         {avg_metrics.get('reranking_time', 0.0)*1000:.1f}ms")
        print(f"  Avg Total Time:             {avg_metrics.get('total_time', 0.0)*1000:.1f}ms")
        print(f"{'='*80}\n")
        
        # Print detailed analysis if requested
        if detailed_analysis and all_query_metrics:
            from evaluation import _print_detailed_analysis
            _print_detailed_analysis(df, all_query_metrics, valid_queries)
        
        return avg_metrics
    else:
        print("No valid queries found!")
        return {}


if __name__ == "__main__":
    args = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                   description="Multi-shot retrieval with query decomposition")
    
    # Add all standard parameters
    add_all_args(args)
    
    # Add multi-shot specific parameters
    args.add_argument('--max-sub-queries', type=int, default=3,
                     help='Maximum number of sub-queries to generate (default: 3)')
    args.add_argument('--reasoning', type=str, default='medium',
                     choices=['low', 'medium', 'high'],
                     help='LLM reasoning level for query decomposition (default: medium)')
    args.add_argument('--max-iterations', type=int, default=10,
                     help='Maximum number of retrieval iterations (default: 10)')
    
    # Special handling for --eval argument
    for action in args._actions:
        if '--eval' in action.option_strings:
            action.type = lambda x: int(x) if x.isdigit() else True
            action.const = True
            break
    
    args = args.parse_args()
    
    # Set deterministic seeds
    set_deterministic_seeds(args.seed)
    
    # Setup LLM configuration with auto-detection
    llm_config = setup_llm_config(args)
    print(f"LLM Config: {llm_config}")
    
    # Setup device-specific environment
    device_config = get_device_config()
    print(f"Device Config: {device_config}")
    
    # Initialize database
    if args.retrieval_method == "bm25":
        db_class = BM25DB
    else:
        db_class = VectorDB
    
    if args.database is None:
        args.database = db_class.get_default_db_name()
    
    db_file_path = args.database if args.database.endswith('.db') else f"{args.database}.db"
    db_base_name = args.database.replace('.db', '') if args.database.endswith('.db') else args.database
    
    rag_db = db_class(
        retriever_model=args.retriever_model, 
        reranker_model=args.reranker_model, 
        device=args.device,
        k1=args.bm25_k1, b=args.bm25_b, method=args.bm25_method, 
        database=db_base_name,
        delta=args.bm25_delta, backend=args.bm25_backend, 
        stopwords=args.bm25_stopwords,
        show_progress=args.bm25_show_progress, stemmer=args.bm25_stemmer,
        vector_index_method=args.vector_index_method, 
        ivf_nprobe=args.ivf_nprobe,
        load_embeddings=args.load_embeddings, 
        num_embedding_devices=args.num_embedding_devices,
        benchmark=args.benchmark
    )
    
    # Load database
    if os.path.exists(db_file_path):
        print(f"Loading existing database from {db_file_path}")
        rag_db.from_serialized(db_file_path)
    else:
        raise ValueError(f"Database not found: {db_file_path}. Please create it first using single_shot_retrieval.py")
    
    # Build strategy parameters
    strategy_params = {"max_results": args.max_results}
    if args.retrieval_strategy == "top_p":
        strategy_params["p"] = args.top_p
    elif args.retrieval_strategy == "relative":
        strategy_params["ratio"] = args.relative_ratio
    
    # Run evaluation or single query
    if args.eval:
        max_queries = args.eval if isinstance(args.eval, int) and not isinstance(args.eval, bool) and args.eval > 0 else None
        
        metrics = run_multi_shot_evaluation(
            rag_db, args.dataset,
            max_sub_queries=args.max_sub_queries,
            top_k_retriever=args.top_k_retriever,
            top_k_reranking=args.top_k_reranking,
            max_queries=max_queries,
            no_rerank=args.no_rerank,
            retrieval_strategy=args.retrieval_strategy,
            reasoning_effort=args.reasoning,
            detailed_analysis=True,  # Enable detailed complexity analysis
            difficulty=args.difficulty,
            max_iterations=args.max_iterations,
            llm_config=llm_config,
            **strategy_params
        )
        
        # Save results
        results_data = {
            "multi_shot": True,
            "max_sub_queries": args.max_sub_queries,
            "reasoning_effort": args.reasoning,
            "metrics": metrics
        }
        
        with open("multi_shot_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to multi_shot_results.json")
        
    else:
        # Single query multi-shot retrieval
        if not args.query:
            args.query = "Who won the French Open Mens Singles tournament the year that New York City FC won their first MLS Cup title?"
        
        print(f"\nRunning multi-shot retrieval for single query...")
        multi_shot_retrieval(
            rag_db, args.query, expected_urls=[],
            max_sub_queries=args.max_sub_queries,
            top_k_retriever=args.top_k_retriever,
            top_k_reranking=args.top_k_reranking,
            no_rerank=args.no_rerank,
            retrieval_strategy=args.retrieval_strategy,
            verbose=True,
            reasoning_effort=args.reasoning,
            llm_config=llm_config,
            **strategy_params
        )
