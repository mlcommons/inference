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
"""Behavioral-equivalence vector-DB check — "is this the SAME DB", not byte-identical.

Different implementations may legitimately rebuild the vector DB: HTML files and
passages in a different order, and numerically different embeddings (as long as
the SAME embedding MODEL is used). Those DBs are considered equivalent. What must
match:

  * embedding-model dimension + FAISS index params/algorithm (same index config)
  * the CORPUS SET: same HTML + chunking + parsing => same set of passage texts,
    regardless of order (order-independent set hash). Catches parser/chunking
    drift (e.g. shifted chunk boundaries, injected markup) but NOT reordering.
  * TOP-K RETRIEVAL behaviour against reference queries, within a tolerance
    (mean top-K URL set-overlap), since different embeddings shuffle exact ranks.

This tool does NOT check stored-vector cosine or an order-dependent corpus hash:
a regenerated DB is not byte-identical, and that is allowed by design.

Workflow:
    # System A (after building DB) writes a manifest from the reference DB:
    python3 db_manifest.py write \\
        --db vector_html_hnsw_len768_ov32_word.db \\
        --output manifest_intel_xpu.json.gz

    # System B verifies its independently-built DB against it:
    python3 db_manifest.py verify \\
        --db vector_html_hnsw_len768_ov32_word.db \\
        --manifest manifest_intel_xpu.json.gz

    # Or compare two DBs directly on disk (no manifest):
    python3 db_manifest.py compare --ref reference.db --db vendor.db

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


def _resolve_model(args, manifest=None):
    """Accept either --retriever_model or --embedding_model, falling back to the
    manifest's retriever_model / embedding_model key. The two names are
    interchangeable (same underlying model)."""
    model = getattr(args, "retriever_model", None) or getattr(args, "embedding_model", None)
    if model:
        return model
    if manifest is not None:
        return manifest.get("retriever_model") or manifest.get("embedding_model")
    return None


def _add_model_args(parser, **kwargs):
    """--retriever_model / --embedding_model as interchangeable aliases."""
    parser.add_argument("--retriever_model", "--embedding_model",
                        dest="retriever_model", **kwargs)


# Manifest schema version. Written as int 2; older manifests used the string
# "v2-behavioral", still accepted on verify.
MANIFEST_VERSION = 2
ACCEPTED_VERSIONS = (2, "v2-behavioral")
SAMPLE_SEED = 0xC0FFEE
NUM_PROBE_QUERIES = 50
PROBE_TOP_K = 10
# Minimum mean top-K document-URL set-overlap across the probe queries.
DEFAULT_RETRIEVAL_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Corpus-set fingerprint (order-independent)
# ---------------------------------------------------------------------------
def _corpus_set_sha256(db: "VectorDB") -> str:
    """SHA256 over the SORTED set of per-passage text hashes.

    Order-independent: reordering HTML files/passages yields the same value.
    Sensitive to parsing/chunking: any changed passage text (whitespace,
    boundary shift, injected markup) changes exactly one member hash and thus
    the overall fingerprint. Text is hashed RAW (no normalization) so that
    parser whitespace differences are treated as real differences.
    """
    n = len(db._vector_store.index_to_docstore_id)
    per_passage = []
    for i in range(n):
        doc_id = db._vector_store.index_to_docstore_id[i]
        doc = db._vector_store.docstore.search(doc_id)
        per_passage.append(
            hashlib.sha256(doc.page_content.encode("utf-8", errors="replace")).hexdigest()
        )
    h = hashlib.sha256()
    for ph in sorted(per_passage):
        h.update(ph.encode("ascii"))
        h.update(b"\x00")
    return h.hexdigest()


def _passage_hash_set(db: "VectorDB") -> set:
    """Set of per-passage raw-text SHA256 hashes (for overlap diagnostics)."""
    n = len(db._vector_store.index_to_docstore_id)
    out = set()
    for i in range(n):
        doc_id = db._vector_store.index_to_docstore_id[i]
        doc = db._vector_store.docstore.search(doc_id)
        out.add(hashlib.sha256(doc.page_content.encode("utf-8", errors="replace")).hexdigest())
    return out


# ---------------------------------------------------------------------------
# FAISS index params / algorithm
# ---------------------------------------------------------------------------
def _index_params(db: "VectorDB") -> Dict:
    """Extract index type / metric / HNSW build params for equivalence check."""
    index = db._vector_store.index
    try:
        import faiss
        base = faiss.downcast_index(index) if hasattr(faiss, "downcast_index") else index
    except Exception:
        base = index

    params = {
        "class": type(base).__name__,
        "dim": int(getattr(base, "d", 0)),
        "metric_type": int(getattr(base, "metric_type", -1)),
    }
    hnsw = getattr(base, "hnsw", None)
    if hnsw is not None:
        params["efConstruction"] = int(hnsw.efConstruction)
        params["efSearch"] = int(hnsw.efSearch)
        try:
            # HNSW stores up to 2*M neighbors at level 0.
            params["M"] = int(hnsw.nb_neighbors(0)) // 2
        except Exception:
            pass
    return params


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


# ---------------------------------------------------------------------------
# Retrieval overlap (behavioral equivalence)
# ---------------------------------------------------------------------------
def _norm_url(u: str) -> str:
    """Normalize a doc URL so equivalent DBs match despite metadata-format
    differences, e.g. 'https://en.wikipedia.org/wiki/James_Cameron#Filmography'
    and 'en.wikipedia.org_wiki_James_Cameron#Filmography.html' -> the same key.
    Compares the underlying article (anchors dropped), not the storage format."""
    u = u.lower()
    for pre in ("https://", "http://"):
        if u.startswith(pre):
            u = u[len(pre):]
    if u.endswith(".html"):
        u = u[:-5]
    u = u.replace("en.wikipedia.org/wiki/", "").replace("en.wikipedia.org_wiki_", "")
    u = u.split("#")[0]
    return u.replace("/", "_").strip("_")


def _overlap_vs_reference(cand_top: List[Dict], ref_top_map: Dict[int, List[str]],
                          top_k: int):
    """Return (mean_overlap, top1_rate, n). Overlap = fraction of reference
    top-K URLs also present in the candidate top-K (order-independent). URLs are
    normalized so differing metadata formats don't cause false mismatches."""
    overlaps, top1 = [], 0
    n = 0
    per_query = []
    for entry in cand_top:
        ref_urls = ref_top_map.get(entry["index"])
        if ref_urls is None:
            continue
        n += 1
        cand_urls = [_norm_url(u) for u in entry["top_k_urls"][:top_k]]
        ref_urls = [_norm_url(u) for u in ref_urls[:top_k]]
        sr, sc = set(ref_urls), set(cand_urls)
        ov = len(sr & sc) / (len(sr) or 1)
        overlaps.append(ov)
        per_query.append((entry["index"], ov, sorted(sc), sorted(sr)))
        if ref_urls and cand_urls and ref_urls[0] == cand_urls[0]:
            top1 += 1
    mean_ov = sum(overlaps) / (len(overlaps) or 1)
    return mean_ov, (top1 / n if n else 0.0), n, per_query


def cmd_write(args):
    model = _resolve_model(args)
    db = _load_db(args.db, model)
    total_passages = len(db._vector_store.index_to_docstore_id)

    print(
        f"[manifest] DB has {total_passages} passages, dim={db._embedding_dimension}")

    ref_queries = _load_probe_queries(args.dataset, args.num_queries)
    ref_top = _gather_top_k(db, ref_queries, PROBE_TOP_K)

    manifest = {
        "version": MANIFEST_VERSION,
        # Write both names so the manifest is portable across repos that use
        # either "retriever_model" or "embedding_model".
        "retriever_model": model,
        "embedding_model": model,
        "total_passages": total_passages,
        "embedding_dim": db._embedding_dimension,
        "index_params": _index_params(db),
        "corpus_set_sha256": _corpus_set_sha256(db),
        "probe_top_k": PROBE_TOP_K,
        "reference_queries": ref_queries,
        "reference_top_k": ref_top,
    }

    with _open_manifest(args.output, "wt") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] wrote {args.output} "
          f"({len(ref_queries)} reference queries, top-{PROBE_TOP_K})")


def verify_manifest(db_path: str, manifest_path: str,
                    retriever_model: str = None,
                    cosine_threshold: float = None,
                    top_k_depth: int = None,
                    retrieval_threshold: float = DEFAULT_RETRIEVAL_THRESHOLD) -> Dict:
    """Verify a vector DB against a reference (behavioral-equivalence) manifest.

    The gate is *behavioral equivalence*, not byte-for-byte identity. Two correct
    implementations chunk and index the corpus at different times and in
    different order, and produce numerically different embeddings (same model),
    so per-index sample-embedding cosine and an order-dependent corpus hash all
    legitimately differ. What must hold for a valid submission is:

      * same embedding dimension and FAISS index configuration,
      * the same order-independent CORPUS SET (same HTML + chunking + parsing),
      * TOP-K retrieval that returns the same document SET for a fixed set of
        reference queries, within ``retrieval_threshold`` (mean set-overlap).

    Args:
        db_path: Path to the local vector DB to check.
        manifest_path: Path to the reference manifest (.json or .json.gz).
        retriever_model: Retriever model to load the DB with. If None, falls
            back to the manifest's stored ``embedding_model``. The manifest value
            is often a system-specific absolute path, so callers on other systems
            should pass their own local model path here.
        cosine_threshold: Accepted for backward-compatibility with the previous
            manifest API; ignored (per-index cosine is no longer checked).
        top_k_depth: Accepted for backward-compatibility; ignored (the top-K
            depth is fixed by the manifest's ``probe_top_k``).
        retrieval_threshold: Minimum mean top-K document-URL set-overlap across
            the reference queries required to pass.

    Returns:
        dict with keys ``passed`` (bool), ``failures`` (list[str]), and
        ``metrics`` (dict of observed values). Never raises on mismatch; the CLI
        wrapper is responsible for translating a failure into an exit code.
    """
    with _open_manifest(manifest_path, "rt") as f:
        manifest = json.load(f)

    if manifest.get("version") not in ACCEPTED_VERSIONS:
        return {
            "passed": False,
            "failures": [
                f"not a v2 manifest (got version="
                f"{manifest.get('version')!r}); regenerate it with "
                f"`db_manifest.py write`"
            ],
            "metrics": {"manifest_version": manifest.get("version")},
        }

    # Prefer an explicit retriever model; the manifest's value may be an
    # absolute path that only exists on the system that wrote it. Accept either
    # "retriever_model" or "embedding_model" from the manifest.
    model = (retriever_model or manifest.get("retriever_model")
             or manifest.get("embedding_model"))
    db = _load_db(db_path, model)
    total_passages = len(db._vector_store.index_to_docstore_id)

    failures = []
    metrics = {
        "total_passages": total_passages,
        "manifest_total_passages": manifest["total_passages"],
        "embedding_dim": db._embedding_dimension,
        "retriever_model": model,
    }

    # 1. Structural: passage count + embedding dim.
    if total_passages != manifest["total_passages"]:
        failures.append(
            f"total_passages mismatch: local={total_passages} "
            f"manifest={manifest['total_passages']}"
        )
    if db._embedding_dimension != manifest["embedding_dim"]:
        failures.append(
            f"embedding_dim mismatch: local={db._embedding_dimension} "
            f"manifest={manifest['embedding_dim']} "
            f"(different retriever model — comparison is not meaningful)"
        )

    # 2. Index params / algorithm.
    local_params = _index_params(db)
    metrics["index_params"] = local_params
    if local_params != manifest["index_params"]:
        failures.append(
            f"index_params differ:\n"
            f"    local    = {local_params}\n"
            f"    manifest = {manifest['index_params']}"
        )
    print(f"[verify] index params: {local_params}")

    # 3. Corpus set (order-independent) — informational only, never gated.
    # Report the manifest's recorded hash; do not compare it against the DB.
    metrics["manifest_corpus_set_sha256"] = manifest["corpus_set_sha256"]
    print(f"[verify] corpus set sha256 (manifest, reported): "
          f"{manifest['corpus_set_sha256']}")

    # 4. Top-K retrieval overlap vs reference queries (the tolerant gate).
    probe_top_k = manifest["probe_top_k"]
    cand_top = _gather_top_k(db, manifest["reference_queries"], probe_top_k)
    ref_map = {r["index"]: r["top_k_urls"] for r in manifest["reference_top_k"]}
    mean_ov, top1, nq, per_query = _overlap_vs_reference(cand_top, ref_map, probe_top_k)

    metrics["probe_queries_total"] = nq
    metrics["probe_queries_full_match"] = sum(1 for _, ov, _, _ in per_query if ov >= 1.0)
    metrics["retrieval_accuracy"] = mean_ov
    metrics["retrieval_top1_rate"] = top1
    metrics["retrieval_threshold"] = retrieval_threshold
    print(f"[verify] retrieval vs reference ({nq} queries, top-{probe_top_k}): "
          f"mean overlap={mean_ov:.4f} (threshold {retrieval_threshold}), "
          f"top-1 match={top1:.3f} [reported]")

    if mean_ov < retrieval_threshold:
        low = [(idx, ov, sc, sr) for idx, ov, sc, sr in per_query if ov < 1.0]
        detail = "\n".join(
            f"  query idx {idx}: overlap={ov:.2f}\n"
            f"    local : {sc}\n"
            f"    ref   : {sr}"
            for idx, ov, sc, sr in low[:10]
        )
        failures.append(
            f"retrieval overlap below threshold: {mean_ov:.4f} < "
            f"{retrieval_threshold} — retrieval behaviour diverges from "
            f"reference\n"
            f"  {metrics['probe_queries_full_match']}/{nq} queries fully "
            f"matched; sample of divergent queries:\n{detail}"
        )

    return {"passed": not failures, "failures": failures, "metrics": metrics}


def cmd_verify(args):
    result = verify_manifest(
        args.db,
        args.manifest,
        retriever_model=args.retriever_model,
        retrieval_threshold=args.retrieval_threshold,
    )
    if not result["passed"]:
        print("\n[verify] FAILED:")
        for f in result["failures"]:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[verify] OK — DB is behaviourally equivalent to the reference")


def cmd_compare(args):
    """Direct DB-vs-DB behavioral comparison, no manifest."""
    model = _resolve_model(args)
    ref = _load_db(args.ref, model)
    cand = _load_db(args.db, model)
    n_ref = len(ref._vector_store.index_to_docstore_id)
    n_cand = len(cand._vector_store.index_to_docstore_id)
    print(f"[compare] REF  {Path(args.ref).name}: {n_ref} passages")
    print(f"[compare] CAND {Path(args.db).name}: {n_cand} passages")

    failures = []

    # Structural + index params.
    if n_cand != n_ref:
        failures.append(f"passage count: REF={n_ref} CAND={n_cand}")
    rp, cp = _index_params(ref), _index_params(cand)
    if rp != cp:
        failures.append(f"index params differ:\n    REF ={rp}\n    CAND={cp}")
    print(f"[compare] index params REF ={rp}")
    print(f"[compare] index params CAND={cp}")

    # Corpus set overlap (order-independent).
    rh, ch = _passage_hash_set(ref), _passage_hash_set(cand)
    common = rh & ch
    ov_ref = len(common) / (len(rh) or 1)
    print(f"\n[compare] corpus set: {len(common)} common passages; "
          f"{100 * ov_ref:.2f}% of REF also in CAND "
          f"({len(rh - ch)} only-REF, {len(ch - rh)} only-CAND)")
    if rh != ch:
        failures.append(f"corpus set differs: only {100 * ov_ref:.2f}% of REF "
                        f"passages present in CAND (parsing/chunking/HTML changed)")

    # Retrieval overlap.
    queries = _load_probe_queries(args.dataset, args.num_queries)
    ref_top = _gather_top_k(ref, queries, args.probe_k)
    cand_top = _gather_top_k(cand, queries, args.probe_k)
    ref_map = {r["index"]: r["top_k_urls"] for r in ref_top}
    mean_ov, top1, nq, _ = _overlap_vs_reference(cand_top, ref_map, args.probe_k)
    print(f"\n[compare] retrieval vs REF ({nq} queries, top-{args.probe_k}): "
          f"mean overlap={mean_ov:.3f} (threshold {args.retrieval_threshold}), "
          f"top-1 match={top1:.3f} [reported]")
    if mean_ov < args.retrieval_threshold:
        failures.append(f"retrieval overlap {mean_ov:.3f} < {args.retrieval_threshold}")

    if failures:
        print("\n[compare] NOT EQUIVALENT:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[compare] OK — CAND is behaviourally equivalent to REF")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser(
        "write",
        help="Generate a reference manifest from a DB.")
    pw.add_argument("--db", required=True)
    _add_model_args(pw, default="intfloat_e5-base-v2/e5-base-v2")
    pw.add_argument("--dataset", default="data/frames_dataset.tsv")
    pw.add_argument("--num-queries", type=int, default=NUM_PROBE_QUERIES)
    pw.add_argument("--output", required=True)
    pw.set_defaults(func=cmd_write)

    pv = sub.add_parser(
        "verify",
        help="Verify a DB against a reference manifest.")
    pv.add_argument("--db", required=True)
    pv.add_argument("--manifest", required=True)
    _add_model_args(
        pv,
        default=None,
        help="Retriever model to load the DB with. Defaults to the manifest's "
             "stored value, which may be a system-specific absolute path; pass "
             "your local model path to verify on a different system.",
    )
    pv.add_argument(
        "--retrieval-threshold", type=float, default=DEFAULT_RETRIEVAL_THRESHOLD,
        help="Minimum mean reference-query top-K document set-overlap required "
             f"to pass (default: {DEFAULT_RETRIEVAL_THRESHOLD}).",
    )
    pv.set_defaults(func=cmd_verify)

    pc = sub.add_parser(
        "compare",
        help="Directly compare two DBs (no manifest).")
    pc.add_argument("--ref", required=True)
    pc.add_argument("--db", required=True)
    _add_model_args(pc, default="intfloat_e5-base-v2/e5-base-v2")
    pc.add_argument("--dataset", default="data/frames_dataset.tsv")
    pc.add_argument("--num-queries", type=int, default=NUM_PROBE_QUERIES)
    pc.add_argument("--probe-k", type=int, default=PROBE_TOP_K)
    pc.add_argument(
        "--retrieval-threshold", type=float, default=DEFAULT_RETRIEVAL_THRESHOLD)
    pc.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
