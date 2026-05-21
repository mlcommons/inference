"""
Shared reranker queue for thread-safe reranking across multiple worker threads.

Uses ColBERT late-interaction (MaxSim) scoring.
When multiple query threads need reranking, they submit requests to this queue.
A dedicated worker thread processes them sequentially to avoid concurrent model access.
"""

import threading
import queue
from typing import List, Tuple


class RerankerQueue:
    """Thread-safe queue for reranking requests with a dedicated worker."""

    def __init__(self, reranker_model, reranker_tokenizer, device="cpu"):
        self._model = reranker_model
        self._tokenizer = reranker_tokenizer
        self._device = device
        self._request_queue = queue.Queue()
        self._worker_thread = None
        self._running = False
        self.total_requests = 0
        self.total_documents = 0
        self.total_latency_ms = 0.0

    def start(self):
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        self._running = False
        self._request_queue.put(None)
        if self._worker_thread:
            self._worker_thread.join(timeout=10)

    def submit(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        """Submit a rerank request and block until result is ready."""
        result_event = threading.Event()
        result_container = {}
        self._request_queue.put((query, passages, result_event, result_container))
        result_event.wait()
        if "error" in result_container:
            raise result_container["error"]
        return result_container["result"]

    def _worker_loop(self):
        import torch
        while self._running:
            try:
                item = self._request_queue.get(timeout=1.0)
                if item is None:
                    break
                query, passages, event, container = item
                try:
                    import time as _time
                    self.total_requests += 1
                    self.total_documents += len(passages)
                    t0 = _time.perf_counter()
                    container["result"] = self._do_rerank(query, passages)
                    elapsed_ms = (_time.perf_counter() - t0) * 1000
                    self.total_latency_ms += elapsed_ms
                except Exception as e:
                    container["error"] = e
                event.set()
            except queue.Empty:
                continue

    def _do_rerank(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        """ColBERT late-interaction reranking with MaxSim scoring."""
        import torch

        with torch.no_grad():
            q_inputs = self._tokenizer(
                query, return_tensors='pt', truncation=True,
                max_length=32, padding='max_length'
            )
            q_inputs = {k: v.to(self._device) for k, v in q_inputs.items()}
            q_emb = self._model(**q_inputs).last_hidden_state.squeeze(0)
            q_mask = q_inputs['attention_mask'].squeeze(0).bool()
            q_emb = q_emb[q_mask]

            d_inputs = self._tokenizer(
                passages, return_tensors='pt', truncation=True,
                max_length=512, padding=True
            )
            d_inputs = {k: v.to(self._device) for k, v in d_inputs.items()}
            d_emb = self._model(**d_inputs).last_hidden_state
            d_mask = d_inputs['attention_mask']

            sim = torch.einsum('qd,bld->bql', q_emb, d_emb)
            sim = sim.masked_fill(~d_mask.unsqueeze(1).bool(), float('-inf'))
            scores = sim.max(dim=-1).values.sum(dim=-1)

        scored_passages = list(zip(passages, scores.float().tolist()))
        scored_passages.sort(key=lambda x: x[1], reverse=True)
        return scored_passages
