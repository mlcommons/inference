"""
Shared reranker queue for thread-safe reranking across multiple worker threads.

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
                    container["result"] = self._do_rerank(query, passages)
                except Exception as e:
                    container["error"] = e
                event.set()
            except queue.Empty:
                continue

    def _do_rerank(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        import torch
        pairs = [[query, passage] for passage in passages]

        with torch.no_grad():
            inputs = self._tokenizer(pairs, padding=True, return_tensors='pt',
                                     truncation=True, max_length=512)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            scores = self._model(**inputs).logits.view(-1).float()

        scored_passages = list(zip(passages, scores.cpu().tolist()))
        scored_passages.sort(key=lambda x: x[1], reverse=True)
        return scored_passages
