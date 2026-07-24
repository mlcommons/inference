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
NUM_PROBE_QUERIES = 10
PROBE_TOP_K = 5
DEFAULT_COSINE_THRESHOLD = 0.9999
DEFAULT_TOP_K_DEPTH = 3


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
                    top_k_depth: int = DEFAULT_TOP_K_DEPTH) -> Dict:
    """Verify a vector DB against a reference manifest.

    Args:
        db_path: Path to the local vector DB to check.
        manifest_path: Path to the reference manifest (.json or .json.gz).
        retriever_model: Retriever model to load the DB with. If None, falls
            back to the manifest's stored ``retriever_model``. The manifest
            value is often a system-specific absolute path, so callers on other
            systems should pass their own local model path here.
        cosine_threshold: Minimum sample-embedding cosine similarity.
        top_k_depth: Probe-query top-K rank match depth.

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
        "embedding_dim": db._embedding_dimension,
        "retriever_model": model,
    }

    # Exact-match fields.
    if total_passages != manifest["total_passages"]:
        failures.append(
            f"total_passages mismatch: local={total_passages} manifest={manifest['total_passages']}"
        )
    if db._embedding_dimension != manifest["embedding_dim"]:
        failures.append(
            f"embedding_dim mismatch: local={db._embedding_dimension} "
            f"manifest={manifest['embedding_dim']}"
        )

    # Corpus fingerprint (sha256 of all passage texts in index order).
    local_corpus_sha = _sha256_docstore(db)
    metrics["corpus_sha256_match"] = (
        local_corpus_sha == manifest["corpus_sha256"])
    if local_corpus_sha != manifest["corpus_sha256"]:
        failures.append(
            f"corpus sha256 mismatch:\n"
            f"  local    = {local_corpus_sha}\n"
            f"  manifest = {manifest['corpus_sha256']}"
        )

    # Sample-embedding cosine similarity.
    cosines = []
    for idx, ref_emb in zip(manifest["sample_embeddings"]["indices"],
                            manifest["sample_embeddings"]["embeddings"]):
        doc_id = db._vector_store.index_to_docstore_id.get(idx)
        if doc_id is None:
            failures.append(f"sample idx {idx}: not present in local DB")
            continue
        doc = db._vector_store.docstore.search(doc_id)
        local_emb = db.embed_query(doc.page_content)
        cosines.append((idx, _cosine(local_emb, ref_emb)))

    if cosines:
        worst_idx, worst_cos = min(cosines, key=lambda x: x[1])
        mean_cos = sum(c for _, c in cosines) / len(cosines)
        metrics["sample_cosine_mean"] = mean_cos
        metrics["sample_cosine_min"] = worst_cos
        print(f"[verify] sample embeddings: mean cosine={mean_cos:.6f} "
              f"min={worst_cos:.6f} (idx={worst_idx}) threshold={cosine_threshold}")
        if worst_cos < cosine_threshold:
            failures.append(
                f"sample embedding cosine below threshold: "
                f"min={worst_cos:.6f} (idx={worst_idx}) < threshold={cosine_threshold}\n"
                f"  mean={mean_cos:.6f}"
            )

    # Probe-query top-K rank check.
    probe_queries = manifest["probe_queries"]
    local_top = _gather_top_k(db, probe_queries, PROBE_TOP_K)
    ref_top = {r["index"]: r["top_k_urls"] for r in manifest["probe_top_k"]}

    rank_failures = []
    for entry in local_top:
        local_urls = entry["top_k_urls"][:top_k_depth]
        ref_urls = ref_top.get(entry["index"], [])[:top_k_depth]
        if local_urls != ref_urls:
            rank_failures.append(
                f"  query idx {entry['index']}: top-{top_k_depth} differs\n"
                f"    local : {local_urls}\n"
                f"    ref   : {ref_urls}"
            )

    metrics["probe_queries_total"] = len(probe_queries)
    metrics["probe_queries_matched"] = len(probe_queries) - len(rank_failures)
    print(f"[verify] probe queries: {len(probe_queries)} queries, "
          f"top-{top_k_depth} {len(probe_queries) - len(rank_failures)}/"
          f"{len(probe_queries)} match")
    if rank_failures:
        failures.append(
            "probe-query top-K rank mismatch:\n" +
            "\n".join(rank_failures))

    return {"passed": not failures, "failures": failures, "metrics": metrics}


def cmd_verify(args):
    result = verify_manifest(
        args.db,
        args.manifest,
        retriever_model=args.retriever_model,
        cosine_threshold=args.cosine_threshold,
        top_k_depth=args.top_k_depth,
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
        "--cosine-threshold",
        type=float,
        default=DEFAULT_COSINE_THRESHOLD)
    pv.add_argument("--top-k-depth", type=int, default=DEFAULT_TOP_K_DEPTH)
    pv.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
