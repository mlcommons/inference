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


"""Out-of-process reranker worker (ColBERT late-interaction MaxSim).

The reranker model lives in its own multiprocessing.Process so it can have:
- Its own OMP_NUM_THREADS (separate OMP runtime).
- Its own CPU affinity + memory binding (per-NUMA-node placement).
- Isolation from the main process's GIL / thread pool.

Public API kept: RerankerQueue.submit(query, passages) -> [(passage, score), ...]
"""

import ctypes
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
from typing import List, Tuple


# Linux prctl(PR_SET_PDEATHSIG, signo): kernel sends `signo` to this process when
# its parent dies, even if parent dies via SIGKILL.
_PR_SET_PDEATHSIG = 1


def _set_parent_death_signal(signo: int = signal.SIGTERM) -> None:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signo, 0, 0, 0)
    except Exception:
        pass   # non-Linux or no libc; daemon=True still kills on normal exit


def _reranker_worker_main(
    model_name: str,
    device: str,
    numa_node,
    omp_threads,
    request_q: mp.Queue,
    response_q: mp.Queue,
    ready_event,
):
    """Child-process entry point.

    Order matters: pin NUMA → set OMP_NUM_THREADS → import torch → load model.
    Setting OMP_NUM_THREADS after torch is imported has no effect.
    """
    _set_parent_death_signal()

    if numa_node is not None:
        from utils import _physical_cores_for_node, pin_worker_to_node
        cores = _physical_cores_for_node(numa_node)
        if omp_threads:
            cores = cores[:omp_threads]
        pin_worker_to_node(numa_node, cores)
    elif omp_threads:
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    if device == "cpu":
        from utils import apply_cpu_threading_env
        apply_cpu_threading_env()

    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"[reranker child] loading {model_name} on {device}")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"[reranker child] ready (pid={os.getpid()})")

    ready_event.set()

    while True:
        item = request_q.get()
        if item is None:
            break
        request_id, query, passages = item
        try:
            result = _do_rerank(model, tokenizer, device, query, passages)
            response_q.put((request_id, result, None))
        except Exception as e:
            response_q.put((request_id, None, repr(e)))


def _do_rerank(model, tokenizer, device: str, query: str, passages: List[str]) -> List[Tuple[str, float]]:
    """ColBERT late-interaction reranking with MaxSim scoring."""
    import torch

    with torch.no_grad():
        q_inputs = tokenizer(
            query, return_tensors='pt', truncation=True,
            max_length=32, padding='max_length'
        )
        q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
        q_emb = model(**q_inputs).last_hidden_state.squeeze(0)
        q_mask = q_inputs['attention_mask'].squeeze(0).bool()
        q_emb = q_emb[q_mask]

        d_inputs = tokenizer(
            passages, return_tensors='pt', truncation=True,
            max_length=512, padding=True
        )
        d_inputs = {k: v.to(device) for k, v in d_inputs.items()}
        d_emb = model(**d_inputs).last_hidden_state
        d_mask = d_inputs['attention_mask']

        sim = torch.einsum('qd,bld->bql', q_emb, d_emb)
        sim = sim.masked_fill(~d_mask.unsqueeze(1).bool(), float('-inf'))
        scores = sim.max(dim=-1).values.sum(dim=-1)

    scored_passages = list(zip(passages, scores.float().tolist()))
    scored_passages.sort(key=lambda x: x[1], reverse=True)
    return scored_passages


class RerankerQueue:
    """Submit reranking requests to an out-of-process ColBERT model.

    NUMA placement (optional):
        numa_node:  pin child to this NUMA node (CPU + memory). None = inherit.
        omp_threads: cap OMP threads in the child. None = use all node cores.
    """

    def __init__(self, reranker_model_name: str, device: str = "cpu",
                 numa_node=None, omp_threads=None):
        self._model_name = reranker_model_name
        self._device = device
        self._numa_node = numa_node
        self._omp_threads = omp_threads

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        ctx = mp.get_context('spawn')
        self._request_q = ctx.Queue()
        self._response_q = ctx.Queue()
        self._ready_event = ctx.Event()
        self._process = None

        # Shared response queue + dispatcher: one entry per pending request.
        self._pending: dict = {}
        self._pending_lock = threading.Lock()
        self._next_id = 0
        self._dispatcher_thread = None
        self._dispatcher_running = False

        # Stats (best-effort; reranker child does the actual work).
        self.total_requests = 0
        self.total_documents = 0
        self.total_latency_ms = 0.0

    def start(self):
        ctx = mp.get_context('spawn')
        self._process = ctx.Process(
            target=_reranker_worker_main,
            args=(self._model_name, self._device, self._numa_node, self._omp_threads,
                  self._request_q, self._response_q, self._ready_event),
            daemon=True,
        )
        self._process.start()

        if not self._ready_event.wait(timeout=300):
            raise RuntimeError("reranker child failed to become ready within 300s")

        self._dispatcher_running = True
        self._dispatcher_thread = threading.Thread(target=self._dispatcher_loop, daemon=True)
        self._dispatcher_thread.start()

    def stop(self):
        if self._process is None:
            return
        try:
            self._request_q.put(None)
        except Exception:
            pass
        self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._dispatcher_running = False

    def _dispatcher_loop(self):
        """Pull responses off the shared queue, route to the right awaiter."""
        while self._dispatcher_running:
            try:
                item = self._response_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            request_id, result, err = item
            with self._pending_lock:
                slot = self._pending.pop(request_id, None)
            if slot is None:
                continue
            event, container = slot
            if err is not None:
                container["error"] = RuntimeError(f"reranker child error: {err}")
            else:
                container["result"] = result
            event.set()

    def submit(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        if self._process is None or not self._process.is_alive():
            raise RuntimeError("reranker process is not running")

        with self._pending_lock:
            request_id = self._next_id
            self._next_id += 1
            event = threading.Event()
            container = {}
            self._pending[request_id] = (event, container)

        t0 = time.perf_counter()
        self._request_q.put((request_id, query, passages))
        event.wait()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if "error" in container:
            raise container["error"]
        self.total_requests += 1
        self.total_documents += len(passages)
        self.total_latency_ms += elapsed_ms
        return container["result"]
