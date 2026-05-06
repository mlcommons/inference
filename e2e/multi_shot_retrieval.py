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
import re
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


RELEVANCE_CHECK_PROMPT = """\
You are a document relevance classifier. Evaluate if documents are relevant to answering a question.

QUESTION: {question}

NEW DOCUMENTS TO EVALUATE:
{new_docs}

KEPT DOCUMENTS (context only):
{kept_docs}

TASK: For each NEW document, mark 1 if it contains ANY information relevant to answering the question, 0 if completely irrelevant.

Be GENEROUS in marking relevance:
- Mark 1 if document mentions any entity/person/place/event in the question
- Mark 1 if document has related information that could help answer
- Mark 1 if document has facts that could be connected with other docs
- Only mark 0 if truly unrelated content

OUTPUT FORMAT (JSON only):
{{"relevance": [1, 0, 1, 0, 1]}}

The array MUST have exactly {num_docs} elements (one per NEW document).
"""

QUERY_GENERATION_PROMPT = """\
You generate search queries for complex multi-hop questions.

QUESTION: {question}

KEPT DOCUMENTS (already marked relevant):
{kept_docs}

SEARCH HISTORY: {history}
FEEDBACK: {feedback_history}

TASK 1: CHECK IF SUFFICIENT
Review ALL kept documents. Try to connect information across documents to answer the question:
- Match entity names across documents
- Chain relationships (A → B → C)
- Cross-reference dates, positions, attributes
- Build complete answer chains

If you can construct a complete answer with specific facts from kept documents, provide it.

TASK 2: GENERATE QUERIES (if not sufficient)
Analyze what's MISSING, then generate up to {max_queries} strategic search queries.

CRITICAL ANALYSIS OF FAILURES:
Look at SEARCH HISTORY. If queries repeatedly failed (0 docs or 3+ attempts with same pattern):
- STOP that approach immediately
- ESCALATE to broader searches
- Get main Wikipedia articles first
- DO NOT repeat failed query patterns

ESCALATION STRATEGIES (when stuck):
1. **Get full articles**: Search just entity name "Harriet Lane" (not "Harriet Lane mother")
2. **Use list articles**: "List of presidents", "List of first ladies"
3. **Search related entities**: If can't find X's mother, search X's family/siblings
4. **Try alternate names**: Full name, nickname, maiden name

QUERY PATTERNS:
- **Biography**: Person's name alone → gets full Wikipedia with infobox family details
- **Lists/Rankings**: "List of X", "Timeline of Y"
- **Simple is better**: 2-4 words, entity names not descriptions

RESPONSE FORMAT (JSON only):
If sufficient: {{"sufficient": true, "answer": "Jane Ballou"}}
If not: {{"sufficient": false, "queries": ["Harriet Lane", "James Garfield mother"], "feedback": "Found: 15th first lady is Harriet Lane. Missing: Her mother's name, 2nd assassinated president's mother's maiden name."}}

RULES:
- Make queries SHORT (2-4 words)
- Use entity NAMES not descriptions
- Never repeat failed query patterns
- Escalate after 3 failed attempts on same info
- If you cannot find enough information to give a specific answer, "answer": "Unknown" rather than guessing
"""

# Legacy monolithic prompt (still used by old query_rewriter function)
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
- Extract and preserve specific details
- only facts from the document

TASK 2: CHECK IF SUFFICIENT AND CONNECT INFORMATION
Review ALL KEPT documents and summaries. Actively connect facts across documents:
- Identify entities by matching names across summaries 
- Chain relationships (A → B → C)
- Cross-reference dates/events 
- Build complete chains: Person → Family member → Attribute, or Event → Year → Cross-reference
If you can construct a complete answer chain with specific names/facts from kept documents, provide final answer.

TASK 3: GENERATE QUERIES (if not sufficient)
Analyze what's MISSING to complete the answer, then generate at most {k} strategic queries.

CRITICAL ANALYSIS OF FAILURES:
Look at SEARCH HISTORY. If a query returned 0 documents OR if same query failed 2+ times:
- **STOP IMMEDIATELY** - This approach is not working
- **ESCALATE TO BROADER SEARCH** - Get the main Wikipedia article first
- **DO NOT REUSE FAILED PATTERNS** - Varying descriptions of the same thing won't help

ESCALATION STRATEGIES (use when stuck 3+ iterations):

**When repeated queries fail:**
1. **Get full article first**: Search just the entity name to get their complete Wikipedia page
2. **Search related entities**: If can't find X's mother, search for X's siblings, X's biography, X's family tree
3. **Use list articles**: "List of presidents", "List of first ladies", "Timeline of X"
4. **Try alternate names**: Full formal name, nickname, maiden name, married name

STRATEGIC QUERY PATTERNS:

**For People/Biography:**
- Full article: Just the person's name "Harriet Lane"
- Family: "Person X family", "Person X parents"  
- Specific relative: "Person X" then extract family, don't search "Person X mother" repeatedly

**For Events/Dates:**
- Specific year: "X 2012 winner"
- Timeline: "X dates", "List of X events"
- Cross-reference: "List tallest buildings X City"

**For Numerical/Classification:**
- List/table: "Billboard 200 Nuclear Blast records", "Dewey Decimal X"

**Progressive refinement strategy:**
1. **Get main articles FIRST**: Search entity names to get full Wikipedia pages
2. **Extract connections**: From articles, identify related entities and search those
3. **Get details**: Once you have related entities, search for their specific attributes
4. **Cross-verify**: Use list articles to confirm order/position/relationships

QUERY GENERATION RULES (FOLLOW EXACTLY):
1. **GET FULL ARTICLES FIRST**: To find family/biographical info about Person X, search "Person X" alone to get their complete Wikipedia article with infobox and family details
2. **USE ENTITY NAMES, NOT DESCRIPTIONS**: Once you identify an entity name, always use the name in queries, never use descriptions
3. **ONE CONCEPT PER QUERY**: Each query should target one piece of information
4. **NEVER REPEAT FAILURES**: Check SEARCH HISTORY - if a query pattern failed (0 docs or 3+ attempts), use a completely different approach
5. **ESCALATE WHEN STUCK**: After a failed attempt on same information, escalate to broader queries:
   - Search the main entity name alone (get full Wikipedia article)
   - Search "List of X" for ordinal/positional questions
   - Search related entities instead

RESPONSE FORMAT:
**If there are NEW documents:**
If sufficient: {{"relevance": [1,0,1], "summaries": ["Person served as X from dates, family details include mother named Y with maiden name Z", "", "Person was Nth X, mother was Y Z with maiden name Ballou"], "answer": "Y Z"}}
If not: {{"relevance": [1,0,1], "summaries": ["Person A served from dates, mentioned as related to B", "", "Person C was Nth X, mother mentioned as Y"], "queries": ["Person A", "Person C family"], "feedback": "Found: Entity names and roles. Missing: Complete family details. Next: Get full biographical articles."}}

**If there are NO NEW documents:**
{{"relevance": [], "summaries": [], "queries": ["Nth position holder name", "specific event list"], "feedback": "Starting with direct entity/list searches."}}

CRITICAL REQUIREMENTS:
- If NO NEW documents: return empty arrays: "relevance": [], "summaries": []  
- If {len_new_docs} NEW documents: return exactly {len_new_docs} relevance scores and {len_new_docs} summaries
- For relevant docs (relevance=1): summary MUST extract specific facts (names, dates, relationships, family details from infobox/text)
- For irrelevant docs (relevance=0): summary MUST be empty string ""
- **ESCALATION TRIGGER**: If same query type failed 3+ times, MUST escalate to full article or list search
- Make queries SHORT and SIMPLE (2-4 words best) - Wikipedia article titles are short
- Search entity names alone to get complete biographical articles with family sections
- Wikipedia infoboxes contain: parents, born, died, spouse, children - search person's name to get this

FORMAT VALIDATION: Array lengths MUST match document count - double check before responding!
SEARCH STRATEGY: Simple entity names work better than complex descriptive queries!
Respond only in JSON format"""





# ==============================================================================
# FIX 1: SPLIT APPROACH - Two separate LLM calls for better performance
# ==============================================================================

def evaluate_document_relevance(question: str,
                                new_documents: List[tuple],
                                kept_documents: List[tuple],
                                llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    FIX 1 - Call 1: Simple binary relevance classification.
    """
    if not new_documents:
        return {"relevance": []}

    service_url = llm_config.get('service_url', 'http://127.0.0.1:8123/v1/chat/completions')
    model_name = llm_config.get('model_name', '/model/gpt-oss-20b-mxfp4')
    # Use 4096 so thinking models have enough budget for reasoning + final JSON output
    max_tokens = 4096

    # Format NEW documents
    new_docs_text = ""
    for i, (url, content) in enumerate(new_documents):
        snippet = content[:800] if len(content) > 800 else content
        new_docs_text += f"\n[NEW {i+1}] {snippet}\n"

    # Format KEPT documents
    kept_docs_text = ""
    if kept_documents:
        kept_docs_text = f"[{len(kept_documents)} documents already kept as relevant]\n"
        for i, doc in enumerate(kept_documents[:5]):
            content = doc[1] if len(doc) >= 2 else ""
            snippet = content[:300] if len(content) > 300 else content
            kept_docs_text += f"[KEPT {i+1}] {snippet}...\n"

    prompt = RELEVANCE_CHECK_PROMPT.format(
        question=question,
        new_docs=new_docs_text,
        kept_docs=kept_docs_text if kept_docs_text else "None",
        num_docs=len(new_documents)
    )

    seed = llm_config.get('seed') if llm_config else None
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1,
        "top_p": 1,
        "top_k": -1,
        "max_tokens": max_tokens
    }
    if seed is not None:
        payload["seed"] = seed

    try:
        response = requests.post(service_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        message = result['choices'][0]['message']
        llm_output = (message.get('content') or '').strip()

        # Fallback: thinking models put output in reasoning_content
        if not llm_output:
            reasoning_content = message.get('reasoning_content') or ''
            if reasoning_content:
                print(f"    [DEBUG] content empty, reasoning_content snippet: {reasoning_content[-300:]}...")
                json_match = re.search(r'\{.*\}', reasoning_content, re.DOTALL)
                if json_match:
                    llm_output = json_match.group(0)

        print(f"    [DEBUG] Relevance LLM raw output: {llm_output[:200]}...")

        if not llm_output:
            print(f"    Warning: Relevance check returned empty, marking all as relevant")
            return {"relevance": [1] * len(new_documents)}

        if llm_output.startswith("```"):
            llm_output = llm_output.split("```")[1]
            if llm_output.startswith("json"):
                llm_output = llm_output[4:]
            llm_output = llm_output.strip()

        relevance_result = json.loads(llm_output)
        relevance = relevance_result.get("relevance", [])

        if len(relevance) != len(new_documents):
            print(f"    Warning: Relevance mismatch. Expected {len(new_documents)}, got {len(relevance)}")
            return {"relevance": [1] * len(new_documents)}

        return {"relevance": relevance}

    except Exception as e:
        print(f"    Error in relevance check: {e}")
        return {"relevance": [1] * len(new_documents)}


def generate_search_queries(question: str,
                           kept_documents: List[tuple],
                           max_queries: int = 3,
                           query_history: Optional[List[str]] = None,
                           query_results: Optional[List[int]] = None,
                           feedback_history: Optional[List[str]] = None,
                           llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    FIX 1 - Call 2: Generate search queries OR final answer.
    """
    service_url = llm_config.get('service_url', 'http://127.0.0.1:8123/v1/chat/completions')
    # Use query_model_name if set (e.g. gpt-oss-120b), fall back to model_name (gpt-oss-20b)
    model_name = llm_config.get('query_model_name', llm_config.get('model_name', '/model/gpt-oss-20b-mxfp4'))
    max_tokens = llm_config.get('max_tokens', 10240)

    # Format KEPT documents
    kept_docs_text = ""
    if kept_documents:
        for i, doc in enumerate(kept_documents):
            content = doc[1] if len(doc) >= 2 else ""
            snippet = content[:1200] if len(content) > 1200 else content
            kept_docs_text += f"\n[DOC {i+1}] {snippet}\n"
    else:
        kept_docs_text = "None"

    # Format history
    history_text = ""
    if query_history and query_results:
        for q, count in zip(query_history, query_results):
            history_text += f"- Query: '{q}' → {count} docs\n"
    if not history_text:
        history_text = "No queries yet"

    # Format feedback
    feedback_text = "\n".join(feedback_history) if feedback_history else "Iteration 1 - Initial search"

    prompt = QUERY_GENERATION_PROMPT.format(
        question=question,
        kept_docs=kept_docs_text,
        history=history_text,
        feedback_history=feedback_text,
        max_queries=max_queries
    )

    seed = llm_config.get('seed') if llm_config else None
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1,
        "top_p": 1,
        "top_k": -1,
        "max_tokens": max_tokens
    }
    if seed is not None:
        payload["seed"] = seed

    try:
        response = requests.post(service_url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        message = result['choices'][0]['message']
        llm_output = (message.get('content') or '').strip()

        # Fallback: thinking models put output in reasoning_content
        if not llm_output:
            reasoning_content = message.get('reasoning_content') or ''
            if reasoning_content:
                json_match = re.search(r'\{.*\}', reasoning_content, re.DOTALL)
                if json_match:
                    llm_output = json_match.group(0)

        print(f"    [DEBUG] Query gen LLM raw output: {llm_output[:200]}...")

        if not llm_output:
            print(f"    Warning: Query generation returned empty")
            return {"sufficient": False, "queries": [question], "feedback": "LLM returned empty"}

        if llm_output.startswith("```"):
            llm_output = llm_output.split("```")[1]
            if llm_output.startswith("json"):
                llm_output = llm_output[4:]
            llm_output = llm_output.strip()

        # Try to extract JSON object even from mixed text/markdown responses
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if json_match:
            llm_output = json_match.group(0)

        query_result = json.loads(llm_output)
        sufficient = query_result.get("sufficient", False)

        if sufficient:
            return {"sufficient": True, "answer": query_result.get("answer", ""), "feedback": "Answer generated"}
        else:
            return {"sufficient": False, "queries": query_result.get("queries", [question]), 
                    "feedback": query_result.get("feedback", "Generating queries")}

    except Exception as e:
        print(f"    Error in query generation: {e}")
        # Last-ditch: if the raw output looks like a conclusive answer in free-text, surface it
        if llm_output:
            answer_match = re.search(
                r'(?:answer is|most recently described(?:\s+genus)?(?:\s+is)?)[:\s]+([^\n.]+)',
                llm_output, re.IGNORECASE
            )
            if answer_match:
                extracted = answer_match.group(1).strip()
                print(f"    [FALLBACK] Extracted answer from free-text: {extracted}")
                return {"sufficient": True, "answer": extracted, "feedback": "Extracted from free-text response"}
        return {"sufficient": False, "queries": [question], "feedback": f"Error: {str(e)}"}


def query_rewriter(question: str, new_documents: List[tuple],
                   kept_documents: List[tuple],
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
            kept_context += f"\n[KEPT {i}] {doc[2]}\n"
    else:
        kept_context = "None"
    
    # Format NEW documents 
    new_context = ""
    if new_documents:
        for i, doc in enumerate(new_documents, 1):
            new_context += f"\n[NEW {i}] {doc[1]}\n"
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
            history_parts.append(f"FAILED: {', '.join(failed_queries)}")  # Last 3 failures
        if successful_queries:
            history_parts.append(f"SUCCESS: {', '.join(successful_queries)}")  # Last 2 successes
        
        history_text = "; ".join(history_parts) if history_parts else "No queries yet"
    else:
        history_text = "No queries yet"
    
    # Build feedback history - show progression of what was tried and learned
    if feedback_history and len(feedback_history) > 0:
        unique_feedback = []
        for fb in reversed(feedback_history):
            if fb and fb not in unique_feedback:
                unique_feedback.append(fb)
        
        if unique_feedback:
            feedback_text = "PREVIOUS ATTEMPTS: " + " → ".join(reversed(unique_feedback))
        else:
            feedback_text = f"Iteration {len(query_history) + 1 if query_history else 1}"
    else:
        feedback_text = f"Iteration {len(query_history) + 1 if query_history else 1} - Initial search"
    
    print(f"Context: {context}")
    print(f"History: {history_text}")
    print(f"Feedback: {feedback_text}")

    prompt = QUERY_REWRITER_PROMPT.format(
        question=question,
        context=context,
        history=history_text,
        feedback_history=feedback_text,
        k=max_queries,
        len_new_docs=len(new_documents)
    )
    
    system_message = f"""You are an expert at multi-hop reasoning and strategic search. 
                        CRITICAL: Never repeat failed queries. 
                        Always try completely different approaches when queries return 0 docs. 
                        Focus on atomic facts and progressive strategies."""
    
    # Use LLM config if provided, otherwise use defaults
    if llm_config:
        model_name = llm_config["model_name"]
        service_url = llm_config["service_url"]
        max_tokens = llm_config["max_tokens"]
    else:
        model_name = "/mnt/weka/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct/"
        service_url = "http://127.0.0.1:8123/v1/chat/completions"
        max_tokens = 10240

    seed = llm_config.get('seed') if llm_config else None
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 1,
        "top_p": 1,
        "top_k": -1,
        "max_tokens": max_tokens
    }
    if seed is not None:
        payload["seed"] = seed

    try:
        response = requests.post(service_url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        
        message = result['choices'][0]['message']
        llm_output = message.get('content')
        
        # Fallback: use reasoning_content if content is empty (thinking models)
        reasoning_content = message.get('reasoning_content', '')
        if reasoning_content and not llm_output:
            print(f"    DEBUG: reasoning_content exists but content is empty, extracting JSON from reasoning_content")
            # Try to extract JSON from reasoning_content
            json_match = re.search(r'\{.*\}', reasoning_content, re.DOTALL)
            if json_match:
                llm_output = json_match.group(0)
                print(f"    DEBUG: Extracted JSON from reasoning_content ({len(llm_output)} chars)")
            else:
                print(f"    DEBUG: No JSON found in reasoning_content snippet: {reasoning_content[:200]}")

        if llm_output is None or not llm_output.strip():
            print(f"    Warning: LLM returned empty content, using original query as fallback")
            # Always fall back to original query - never return empty queries
            return {
                "relevance": [0] * len(new_documents),
                "summaries": [""] * len(new_documents),
                "queries": [question],
                "feedback": "LLM returned empty response",
                "answer": ""
            }
        
        llm_output = llm_output.strip()
        
        # Parse JSON output - handle markdown code blocks
        if llm_output.startswith("```"):
            llm_output = llm_output.split("```")[1]
            if llm_output.startswith("json"):
                llm_output = llm_output[4:]
            llm_output = llm_output.strip()
        
        result_data = json.loads(llm_output)
        
        # Validate format 
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

        # ============================================================
        # FIX 1: SPLIT APPROACH WITH ITERATION 1 SPECIAL HANDLING
        # ============================================================

        # Special handling for iteration 1: decompose original query first
        if iteration == 1 and not new_docs and not kept_docs:
            if verbose:
                print(f"  [ITERATION 1] Decomposing original query into sub-queries via generate_search_queries...")

            # Use generate_search_queries for initial decomposition (uses query_model_name / gpt-oss-120b)
            query_result = generate_search_queries(
                question=original_query,
                kept_documents=[],
                max_queries=max_sub_queries,
                query_history=None,
                query_results=None,
                feedback_history=None,
                llm_config=llm_config
            )

            sub_queries = query_result.get("queries", [original_query])
            if not sub_queries:
                sub_queries = [original_query]

            if verbose:
                print(f"    Generated {len(sub_queries)} initial sub-queries")
                for i, q in enumerate(sub_queries, 1):
                    print(f"      {i}. {q}")

            sufficient = False
            final_answer = ""
            current_feedback = query_result.get("feedback", "Initial query decomposition")
            relevance = []
            summaries = []
            reasoning_steps = ""

        else:
            # FIX 1: Split approach for iterations 2+

            # CALL 1: Evaluate document relevance (if we have new docs)
            relevance = []
            if new_docs:
                if verbose:
                    print(f"  [FIX1-CALL1] Evaluating {len(new_docs)} new documents...")

                relevance_result = evaluate_document_relevance(
                    question=original_query,
                    new_documents=new_docs,
                    kept_documents=kept_docs,
                    llm_config=llm_config
                )
                relevance = relevance_result.get("relevance", [1] * len(new_docs))

                # Add relevant docs to kept_docs IMMEDIATELY
                for i, (url, content) in enumerate(new_docs):
                    if i < len(relevance) and relevance[i] == 1:
                        kept_docs.append((url, content, content[:1000]))

                if verbose:
                    print(f"    Marked {sum(relevance)} of {len(new_docs)} docs as relevant")
                    print(f"    Relevance array: {relevance}")
                    print(f"    Total kept docs now: {len(kept_docs)}")

            # CALL 2: Generate queries or answer
            # Cap kept_docs sent to CALL2 to avoid context overflow (most recent docs are most useful)
            MAX_DOCS_FOR_QUERY_GEN = 12
            docs_for_query_gen = kept_docs[-MAX_DOCS_FOR_QUERY_GEN:] if len(kept_docs) > MAX_DOCS_FOR_QUERY_GEN else kept_docs
            if verbose:
                print(f"  [FIX1-CALL2] Generating queries from {len(kept_docs)} kept documents (capped to {len(docs_for_query_gen)})...")

            query_result = generate_search_queries(
                question=original_query,
                kept_documents=docs_for_query_gen,
                max_queries=max_sub_queries,
                query_history=query_history,
                query_results=query_results,
                feedback_history=feedback_history,
                llm_config=llm_config
            )

            sufficient = query_result.get("sufficient", False)
            # Never declare sufficient with no evidence - force more retrieval
            if sufficient and not kept_docs:
                sufficient = False
                if verbose:
                    print(f"    [GUARD] Overriding sufficient=True: no kept docs, forcing more retrieval")
            sub_queries = query_result.get("queries", [])
            current_feedback = query_result.get("feedback", "")
            final_answer = str(query_result.get("answer", ""))
            summaries = []
            reasoning_steps = ""

        
        # Add to feedback history if it's new and meaningful
        if current_feedback and current_feedback.strip() and current_feedback != previous_feedback:
            feedback_history.append(current_feedback.strip())
        
        previous_feedback = current_feedback
        
        if verbose:
            print(f"    Sufficient: {'yes' if sufficient else 'no'}")
            print(f"    Kept docs: {len(kept_docs)}")
            if new_docs:
                print(f"    New docs evaluated: {len(new_docs)}")
                print(f"    Relevant new docs: {sum(relevance)}/{len(relevance)}")
                print(f"    Relevance array: {relevance}")
                # Show summary quality
                if summaries:
                    non_empty_summaries = [s for s in summaries if s.strip()]
                    print(f"    Generated summaries: {len(non_empty_summaries)}/{len(summaries)} non-empty")
                    for i, summary in enumerate(summaries):
                        if summary.strip() and relevance[i] == 1:
                            print(f"      Summary {i+1}: {summary}...")
            if reasoning_steps:
                print(f"    Reasoning: {reasoning_steps[:300]}...")
            if not sufficient:
                print(f"    Feedback: {previous_feedback}")
                print(f"    Generated {len(sub_queries)} new queries")
        
        # Clear new_docs for next iteration (already added to kept_docs in CALL1 block above)
        new_docs = []
        
        # If sufficient, we're done
        if sufficient:
            if verbose:
                print(f"\n  ✓ Sufficient information found!")
                if final_answer:
                    print(f"  Answer: {final_answer[:200]}...")
            iteration_times.append(time.perf_counter() - iteration_start)
            break
        
        # If no queries generated, fall back to original query rather than stopping
        if not sub_queries:
            if verbose:
                print(f"\n  ⚠ No new queries generated, falling back to original query")
            sub_queries = [original_query]
        
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
                for j, result in enumerate(results, 1):
                    url = result.metadata.get('original_url', 'N/A')
                    passage = result.page_content[:300].replace('\n', ' ')
                    print(f"      [{j}] {url}\n          {passage}...")
        
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
        'avg_iteration_time': sum(iteration_times) / len(iteration_times) if iteration_times else 0,
        'llm_answer': final_answer,
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
    all_results = []  # For per-query results (used by evaluate.py)
    
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
                if isinstance(value, (int, float)):
                    total_metrics[metric_name] += value
            
            valid_queries += 1

            # Collect per-query results for evaluate.py scoring
            all_results.append({
                "prompt": row['Prompt'],
                "llm_answer": metrics.get('llm_answer', ''),
                "ground_truth": expected_answer,
            })

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
        
        avg_metrics['_per_query_results'] = all_results
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
        per_query_results = metrics.pop('_per_query_results', [])
        results_data = {
            "multi_shot": True,
            "max_sub_queries": args.max_sub_queries,
            "reasoning_effort": args.reasoning,
            "metrics": metrics,
            "results": per_query_results,
        }
        
        with open("result_multi_shot.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to result_multi_shot.json")
        
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
