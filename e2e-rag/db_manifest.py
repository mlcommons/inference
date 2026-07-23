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
"""Cross-system vector DB sanity check.

Workflow:
    # System A (after building DB):
    python3 db_manifest.py write \\
        --db vector_html_hnsw_len768_ov32_word.db \\
        --output manifest_intel_xpu.json

    # System B (after building DB independently):
    python3 db_manifest.py verify \\
        --db vector_html_hnsw_len768_ov32_word.db \\
        --manifest manifest_intel_xpu.json

The passage corpus is fingerprinted from the DB's docstore directly — no
external passages file needed.
"""

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from retrieve import VectorDB


def _open_manifest(path: str, mode: str):
    """Open a manifest file, transparently gzip-compressing if path ends in .gz."""
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


SAMPLE_SEED = 0xC0FFEE
NUM_SAMPLE_EMBEDDINGS = 50
# Larger probe set so the mean-retrieval-overlap gate is statistically
# meaningful (a 99.9% threshold on 10 queries is effectively all-or-nothing).
NUM_PROBE_QUERIES = 100
PROBE_TOP_K = 5
DEFAULT_COSINE_THRESHOLD = 0.9999
DEFAULT_TOP_K_DEPTH = 3
# Cross-system reproducibility gate. Exact byte/passage-count/sha256 matches are
# NOT required: different implementations chunk and index the corpus at
# different times, so passages arrive in different order and the sha256 / count
# legitimately differ. What must hold is that the DB *retrieves the same
# documents* for a fixed set of probe queries. This threshold is the minimum
# mean top-K document-URL overlap (recall) across the probe queries.
DEFAULT_RETRIEVAL_THRESHOLD = 0.999


def _sha256_docstore(db: "VectorDB") -> str:
    """SHA256 of all passages in index order; identifies the source corpus."""
    h = hashlib.sha256()
    n = len(db._vector_store.index_to_docstore_id)
    for i in range(n):
        doc_id = db._vector_store.index_to_docstore_id[i]
        doc = db._vector_store.docstore.search(doc_id)
        h.update(doc.page_content.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def _normalize_url(url: str) -> str:
    """Reduce a document identifier to a corpus-stable key.

    Different implementations store the source differently: a full URL
    (``https://en.wikipedia.org/wiki/James_Cameron#Filmography``) on one system,
    a sanitized filename (``en.wikipedia.org_wiki_James_Cameron#Filmography.html``)
    on another. Normalize both to the same key so retrieval overlap is compared
    on document identity, not on incidental path formatting.
    """
    if not url:
        return ""
    u = url.strip()
    # Drop scheme.
    u = re.sub(r"^[a-zA-Z]+://", "", u)
    # Drop a trailing .html the filename form appends.
    if u.endswith(".html"):
        u = u[:-len(".html")]
    # Unify path separators: the filename form replaces '/' with '_'.
    u = u.replace("/", "_")
    # Drop the in-page anchor: the same article may be chunked with or without
    # a section fragment, but it is the same source document.
    u = u.split("#", 1)[0]
    return u.lower()


def _cosine(a: List[float], b: List[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _load_db(db_path: str, retriever_model: str) -> VectorDB:
    db_path_obj = Path(db_path if db_path.endswith(".db") else f"{db_path}.db")
    if not db_path_obj.exists():
        raise FileNotFoundError(f"DB file not found: {db_path_obj}")

    db = VectorDB(
        retriever_model=retriever_model,
        device="cpu",
        embedding_device="cpu",
        load_embeddings=False,
    )
    db.from_serialized(db_path_obj.as_posix())
    return db


def _load_probe_queries(dataset_path: str, n: int) -> List[Dict]:
    df = pd.read_csv(dataset_path, sep="\t")
    rng = random.Random(SAMPLE_SEED)
    indices = sorted(rng.sample(range(len(df)), min(n, len(df))))
    return [{"index": i, "prompt": str(df.iloc[i]["Prompt"])} for i in indices]


def _gather_top_k(db: VectorDB, queries: List[Dict], k: int) -> List[Dict]:
    out = []
    for q in queries:
        results = db.lookup(q["prompt"], k=k)
        urls = []
        for doc in results:
            md = getattr(doc, "metadata", None) or {}
            url = md.get("original_url") or md.get(
                "source") or md.get("base_filename") or ""
            urls.append(url)
        out.append({"index": q["index"], "top_k_urls": urls})
    return out


def _gather_sample_embeddings(db: VectorDB, total: int, n: int) -> Dict:
    rng = random.Random(SAMPLE_SEED)
    indices = sorted(rng.sample(range(total), min(n, total)))

    docstore = db._vector_store.docstore
    embeddings = []
    for idx in indices:
        # docstore is keyed by string ids; FAISS internally maps int->id->doc.
        doc_id = db._vector_store.index_to_docstore_id.get(idx)
        if doc_id is None:
            raise RuntimeError(f"docstore has no entry for index {idx}")
        doc = docstore.search(doc_id)
        emb = db.embed_query(doc.page_content)
        embeddings.append(list(emb))
    return {"indices": indices, "embeddings": embeddings}


def cmd_write(args):
    db = _load_db(args.db, args.retriever_model)
    total_passages = len(db._vector_store.index_to_docstore_id)

    print(
        f"[manifest] DB has {total_passages} passages, dim={db._embedding_dimension}")

    corpus_sha = _sha256_docstore(db)
    sample_block = _gather_sample_embeddings(
        db, total_passages, NUM_SAMPLE_EMBEDDINGS)
    probe_queries = _load_probe_queries(args.dataset, NUM_PROBE_QUERIES)
    probe_block = _gather_top_k(db, probe_queries, PROBE_TOP_K)

    manifest = {
        "version": 1,
        "corpus_sha256": corpus_sha,
        "retriever_model": args.retriever_model,
        "vector_index_method": "hnsw",
        "total_passages": total_passages,
        "embedding_dim": db._embedding_dimension,
        "sample_seed": SAMPLE_SEED,
        "sample_embeddings": sample_block,
        "probe_queries": probe_queries,
        "probe_top_k": probe_block,
    }

    with _open_manifest(args.output, "wt") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] wrote {args.output}")


def verify_manifest(db_path: str, manifest_path: str,
                    retriever_model: str = None,
                    cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
                    top_k_depth: int = DEFAULT_TOP_K_DEPTH,
                    retrieval_threshold: float = DEFAULT_RETRIEVAL_THRESHOLD) -> Dict:
    """Verify a vector DB against a reference manifest.

    The gate is *retrieval reproducibility*, not byte-for-byte identity. Two
    correct implementations chunk and index the corpus at different times and in
    different order, so passage counts, the corpus sha256, and per-index sample
    embeddings all legitimately differ. What must hold for a valid submission is
    that the DB returns the *same documents* for a fixed set of probe queries.

    Accordingly:
      * PASS/FAIL is decided solely by mean probe-query top-K document overlap
        (``retrieval_accuracy``) meeting ``retrieval_threshold``.
      * passage count, corpus sha256, and sample-embedding cosine are recorded
        in ``metrics`` for diagnostics but never cause a failure.

    Args:
        db_path: Path to the local vector DB to check.
        manifest_path: Path to the reference manifest (.json or .json.gz).
        retriever_model: Retriever model to load the DB with. If None, falls
            back to the manifest's stored ``retriever_model``. The manifest
            value is often a system-specific absolute path, so callers on other
            systems should pass their own local model path here.
        cosine_threshold: Informational sample-embedding cosine threshold; a
            value below it is reported but does not fail the check.
        top_k_depth: Probe-query top-K depth used for the overlap computation.
        retrieval_threshold: Minimum mean top-K document-URL overlap across the
            probe queries required to pass.

    Returns:
        dict with keys ``passed`` (bool), ``failures`` (list[str]), and
        ``metrics`` (dict of observed values). Never raises on mismatch; the
        CLI wrapper is responsible for translating a failure into an exit code.
    """
    with _open_manifest(manifest_path, "rt") as f:
        manifest = json.load(f)

    # Prefer an explicit retriever model; the manifest's value may be an
    # absolute path that only exists on the system that wrote it.
    model = retriever_model or manifest["retriever_model"]
    db = _load_db(db_path, model)
    total_passages = len(db._vector_store.index_to_docstore_id)

    failures = []
    metrics = {
        "total_passages": total_passages,
        "manifest_total_passages": manifest["total_passages"],
        "embedding_dim": db._embedding_dimension,
        "retriever_model": model,
    }

    # --- Informational diagnostics (never fail the check) --------------------
    # Embedding dimension is the one structural invariant: a different dim means
    # a different retriever model, which would invalidate the comparison.
    if db._embedding_dimension != manifest["embedding_dim"]:
        failures.append(
            f"embedding_dim mismatch: local={db._embedding_dimension} "
            f"manifest={manifest['embedding_dim']} "
            f"(different retriever model — comparison is not meaningful)"
        )

    # Corpus fingerprint: recorded for provenance, but an implementation that
    # chunks/orders passages differently will differ here legitimately.
    local_corpus_sha = _sha256_docstore(db)
    metrics["corpus_sha256_match"] = (local_corpus_sha == manifest["corpus_sha256"])

    # Sample-embedding cosine compares the *same index position* across DBs.
    # When passage ordering differs, index i is a different passage, so this is
    # informational only — the retrieval overlap below is the real signal.
    cosines = []
    for idx, ref_emb in zip(manifest["sample_embeddings"]["indices"],
                            manifest["sample_embeddings"]["embeddings"]):
        doc_id = db._vector_store.index_to_docstore_id.get(idx)
        if doc_id is None:
            continue
        doc = db._vector_store.docstore.search(doc_id)
        local_emb = db.embed_query(doc.page_content)
        cosines.append((idx, _cosine(local_emb, ref_emb)))

    if cosines:
        worst_idx, worst_cos = min(cosines, key=lambda x: x[1])
        mean_cos = sum(c for _, c in cosines) / len(cosines)
        metrics["sample_cosine_mean"] = mean_cos
        metrics["sample_cosine_min"] = worst_cos
        print(f"[verify] sample embeddings (informational): mean cosine={mean_cos:.6f} "
              f"min={worst_cos:.6f} (idx={worst_idx}) threshold={cosine_threshold}")

    # --- Retrieval reproducibility (the PASS/FAIL gate) ----------------------
    # For each probe query, compare the set of retrieved document URLs against
    # the reference, normalized to a corpus-stable key and compared as a set so
    # ordering and path formatting do not matter. The score is the mean overlap
    # (recall of the reference documents) across all probe queries.
    probe_queries = manifest["probe_queries"]
    local_top = _gather_top_k(db, probe_queries, PROBE_TOP_K)
    ref_top = {r["index"]: r["top_k_urls"] for r in manifest["probe_top_k"]}

    overlaps = []
    low_overlap = []
    for entry in local_top:
        local_urls = {_normalize_url(u) for u in entry["top_k_urls"][:top_k_depth] if u}
        ref_urls = {_normalize_url(u) for u in ref_top.get(entry["index"], [])[:top_k_depth] if u}
        if not ref_urls:
            continue
        overlap = len(local_urls & ref_urls) / len(ref_urls)
        overlaps.append(overlap)
        if overlap < 1.0:
            low_overlap.append(
                f"  query idx {entry['index']}: overlap={overlap:.2f}\n"
                f"    local : {sorted(local_urls)}\n"
                f"    ref   : {sorted(ref_urls)}"
            )

    retrieval_accuracy = sum(overlaps) / len(overlaps) if overlaps else 0.0
    metrics["probe_queries_total"] = len(overlaps)
    metrics["probe_queries_full_match"] = sum(1 for o in overlaps if o >= 1.0)
    metrics["retrieval_accuracy"] = retrieval_accuracy
    metrics["retrieval_threshold"] = retrieval_threshold
    print(f"[verify] probe queries: {len(overlaps)} queries, "
          f"mean top-{top_k_depth} retrieval overlap={retrieval_accuracy:.4f} "
          f"(threshold={retrieval_threshold})")

    if retrieval_accuracy < retrieval_threshold:
        detail = "\n".join(low_overlap[:10])
        failures.append(
            f"retrieval accuracy below threshold: "
            f"{retrieval_accuracy:.4f} < {retrieval_threshold}\n"
            f"  {metrics['probe_queries_full_match']}/{len(overlaps)} probe "
            f"queries fully matched; sample of divergent queries:\n{detail}"
        )

    return {"passed": not failures, "failures": failures, "metrics": metrics}


def cmd_verify(args):
    result = verify_manifest(
        args.db,
        args.manifest,
        retriever_model=args.retriever_model,
        cosine_threshold=args.cosine_threshold,
        top_k_depth=args.top_k_depth,
        retrieval_threshold=args.retrieval_threshold,
    )
    if not result["passed"]:
        print("\n[verify] FAILED:")
        for f in result["failures"]:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[verify] OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser(
        "write",
        help="Generate a reference manifest from a DB.")
    pw.add_argument("--db", required=True)
    pw.add_argument(
        "--retriever_model",
        default="intfloat_e5-base-v2/e5-base-v2")
    pw.add_argument("--dataset", default="data/frames_dataset.tsv")
    pw.add_argument("--output", required=True)
    pw.set_defaults(func=cmd_write)

    pv = sub.add_parser(
        "verify",
        help="Verify a DB against a reference manifest.")
    pv.add_argument("--db", required=True)
    pv.add_argument("--manifest", required=True)
    pv.add_argument(
        "--retriever_model",
        default=None,
        help="Retriever model to load the DB with. Defaults to the manifest's "
             "stored value, which may be a system-specific absolute path; pass "
             "your local model path to verify on a different system.",
    )
    pv.add_argument(
        "--cosine-threshold", type=float, default=DEFAULT_COSINE_THRESHOLD,
        help="Informational sample-embedding cosine threshold; does not affect "
             "pass/fail.",
    )
    pv.add_argument("--top-k-depth", type=int, default=DEFAULT_TOP_K_DEPTH)
    pv.add_argument(
        "--retrieval-threshold", type=float, default=DEFAULT_RETRIEVAL_THRESHOLD,
        help="Minimum mean probe-query top-K document overlap required to pass "
             f"(default: {DEFAULT_RETRIEVAL_THRESHOLD}).",
    )
    pv.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
