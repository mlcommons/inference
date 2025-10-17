import faiss
import torch
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .ragdb import RagDB

# Worker function for parallel embedding generation (must be at module level for multiprocessing)
def _parallel_embed_worker(device_id, chunk_indices, chunks, result_queue, model_name, encode_kwargs, base_device):
    """Worker function to generate embeddings on a specific device.
    
    This worker processes multiple chunks on a single device to avoid
    loading the model multiple times.
    
    Args:
        device_id: Device index (0, 1, 2, etc.)
        chunk_indices: List of chunk indices this worker should process
        chunks: List of passage chunks to embed
        result_queue: Multiprocessing queue for results
        model_name: Name of the embedding model
        encode_kwargs: Encoding arguments
        base_device: Base device type from --device option ('xpu', 'cuda', 'cpu')
    """
    try:
        import torch
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Format device string: CPU doesn't use indices, others do
        if base_device == 'cpu':
            device = 'cpu'
        else:
            device = f'{base_device}:{device_id}'
        
        model_kwargs = {'device': device}
        
        # Load model once for this device
        embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        print(f"✓ Device {device}: Loaded model, processing {len(chunks)} chunk(s)")
        
        # Process all chunks assigned to this device
        for chunk_idx, chunk in zip(chunk_indices, chunks):
            embeddings = embedder.embed_documents(chunk)
            result_queue.put((chunk_idx, embeddings))
        
    except Exception as e:
        print(f"❌ Error on device {device_id} ({base_device}): {e}")
        import traceback
        traceback.print_exc()
        # Put None for all failed chunks
        for chunk_idx in chunk_indices:
            result_queue.put((chunk_idx, None))
            result_queue.put((chunk_idx, None))

class VectorDB(RagDB):
    @classmethod
    def get_default_db_name(cls) -> str:
        """Get the default database filename for VectorDB."""
        return "vector.db"
    
    def __init__(self,
            retriever_model: str = None,
            reranker_model: str = None,
            device: str = "auto",
            vector_index_method: str = "hnsw",
            ivf_nprobe: int = 10,
            load_embeddings: bool = True,
            num_embedding_devices: int = 1,
            benchmark: bool = False,
            **kwargs
        ):
        super().__init__(reranker_model, device, benchmark)
        self._retriever_model_name = retriever_model
        self._reranker_model_name = reranker_model
        self._vector_index_method = vector_index_method
        self._ivf_nprobe = ivf_nprobe
        self._load_embeddings = load_embeddings
        self._num_embedding_devices = num_embedding_devices

        # Initialize embedding model with device configuration
        model_kwargs = {'device': self._device}
        encode_kwargs = {'normalize_embeddings': True}
        
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=self._retriever_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        self._embedding_dimension = len(self._embedding_model.embed_query("hello world"))
        
        # Check the dtype of the embedding without using numpy
        test_embedding_raw = self._embedding_model.embed_query("test")
        
        # Calculate dtype and itemsize from Python native list
        if isinstance(test_embedding_raw, list) and len(test_embedding_raw) > 0:
            test_element = test_embedding_raw[0]
            embedding_dtype = type(test_element)
            embedding_itemsize = test_element.__sizeof__()  # Size in bytes of one element
            self._embedding_bytes_per_element = embedding_itemsize
        else:
            raise ValueError("Embedding query did not return a valid list of floats.")

        if self._benchmark:
            print(f"   Embedding element type: {embedding_dtype}")
            print(f"   Bytes per element: {embedding_itemsize}")

        # The index defines the algorithm used for the similarity search
        # Support multiple vector index types (currently FAISS-based)
        self._index = self._create_vector_index(self._vector_index_method, self._embedding_dimension)

        # The docstore is used to store the documents and their metadata
        self._docstore = InMemoryDocstore()

        self._vector_store = FAISS(
            embedding_function=self._embedding_model,
            index=self._index,
            docstore=self._docstore,
            index_to_docstore_id={}, # This will be populated as documents are added
        )
        
        # Keep track of ingested documents for consistency with BM25DB
        self._doc_list = []
    
    def _create_vector_index(self, method: str, dimension: int):
        """Create a vector index based on the specified method.
        
        Currently uses FAISS backend, but abstracted to allow future support
        for other vector databases (e.g., Milvus, Qdrant, Weaviate).
        
        Args:
            method: Index method - 'flat', 'hnsw', or 'ivf'
            dimension: Embedding dimension
            
        Returns:
            Vector index object (FAISS index)
            
        Index Method Details:
        
        1. FLAT (IndexFlatL2):
           - Exact brute-force search using L2 distance
           - Pros: Perfect accuracy, simple
           - Cons: O(N) search time, slow for large datasets
           - Best for: Small datasets (<10K), when accuracy is critical
        
        2. HNSW (Hierarchical Navigable Small World):
           - Graph-based approximate nearest neighbor search
           - Pros: Very fast search O(log N), excellent recall, no training needed
           - Cons: Higher memory usage (stores graph), slower indexing
           - Best for: Most use cases, default choice
        
        3. IVF (Inverted File):
           - Clustering-based approximate search
           - Parameters:
             * nlist: number of clusters (auto-adjusted to ~2*sqrt(N))
             * nprobe: clusters to search per query (default: 10)
               - nprobe=1: fastest but lowest accuracy (~80-90%)
               - nprobe=10: good balance (~95-98% accuracy)
               - nprobe=50: high accuracy (~99%) but slower
           - Pros: Memory efficient, good for large datasets, faster than flat
           - Cons: Requires training, slightly lower recall than HNSW
           - Best for: Very large datasets (>1M), when memory is limited
        """
        if method == "flat":
            return faiss.IndexFlatL2(dimension)
        elif method == "hnsw":
            # M: number of connections per layer (higher = better recall, more memory)
            # efConstruction: quality of index construction (higher = better quality, slower build)
            M = 32  # Default: 32, good balance
            index = faiss.IndexHNSWFlat(dimension, M)
            index.hnsw.efConstruction = 40  # Default: 40
            index.hnsw.efSearch = 16  # Search-time parameter, can be adjusted later
            return index
        elif method == "ivf":
            # nlist: number of clusters/cells (sqrt(N) is a good heuristic for N docs)
            nlist = 100  # Will be adjusted based on dataset size during training
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            # Note: IVF index needs training before use (will be done during ingest)
            return index
        else:
            raise ValueError(f"Unknown vector index method: {method}. Choose 'flat', 'hnsw', or 'ivf'.")
    
    def _train_vector_index(self, index, embeddings: np.ndarray):
        """Train IVF index on embeddings if needed.
        
        IVF (Inverted File Index) requires a one-time training phase to:
        1. Cluster the embedding space into nlist regions using k-means
        2. Build an inverted index mapping cluster_id -> vector_ids
        
        After training, search works by:
        1. Finding the nprobe nearest cluster centroids to the query
        2. Searching only within those clusters (much faster than full scan)
        
        Note: In incremental scenarios, this trains on the FIRST batch only.
        Subsequent batches are assigned to existing clusters without retraining.
        For production systems handling continuous data growth, consider:
        - Periodic retraining when dataset size doubles
        - Using all accumulated data for retraining
        - Online clustering algorithms that adapt to new data
        
        Args:
            index: FAISS index (only IVF types need training)
            embeddings: Numpy array of embeddings to train on
        """
        import numpy as np
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Adjust nlist (number of clusters) based on dataset size
        # Rule of thumb: nlist = sqrt(N) to 4*sqrt(N)
        n_samples = len(embeddings)
        optimal_nlist = max(10, min(int(np.sqrt(n_samples) * 2), 1000))
        
        # Update nlist if different from default
        if optimal_nlist != self._index.nlist:
            print(f"Adjusting IVF nlist from {self._index.nlist} to {optimal_nlist} based on {n_samples} samples")
            # Need to recreate index with new nlist
            quantizer = faiss.IndexFlatL2(self._embedding_dimension)
            self._index = faiss.IndexIVFFlat(quantizer, self._embedding_dimension, optimal_nlist)
            # Update vector store's index
            self._vector_store.index = self._index
        
        print(f"Training IVF index on {n_samples} samples...")
        self._index.train(embeddings_array)
        
        # Set nprobe (number of clusters to search) for better accuracy
        self._index.nprobe = self._ivf_nprobe
        print(f"IVF index trained successfully with {self._index.nlist} clusters, nprobe={self._ivf_nprobe}")
        print(f"  → Will search {self._ivf_nprobe} clusters per query (~{100*self._ivf_nprobe/self._index.nlist:.1f}% of clusters)")
    
    def _get_embeddings_cache_path(self, passages_path: str) -> str:
        """Get the cache path for embeddings based on passages file path."""
        from pathlib import Path
        passages_path = Path(passages_path)
        # Replace extension with .emb.pkl
        cache_path = passages_path.with_suffix('.emb.pkl')
        return str(cache_path)
    
    def _save_embeddings_cache(self, embeddings: list, passages_path: str):
        """Save embeddings to a pickle file for reuse."""
        import os, pickle
        from pathlib import Path
        
        cache_path = self._get_embeddings_cache_path(passages_path)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        
        if os.path.exists(cache_path):
            print(f"Embeddings cache exists: {cache_path}")
            return

        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"💾 Saved embeddings cache to {cache_path}")
    
    def _load_embeddings_cache(self, passages_path: str) -> list:
        """Load embeddings from cache if available."""
        import pickle
        from pathlib import Path
        
        cache_path = self._get_embeddings_cache_path(passages_path)
        if not Path(cache_path).exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"✓ Loaded embeddings from cache: {cache_path}")
            return embeddings
        except Exception as e:
            print(f"⚠️  Failed to load embeddings cache: {e}")
            return None
    
    def _embed_documents_parallel(self, passages: List[str]) -> list:
        """Generate embeddings using multiple devices in parallel.
        
        Uses the device type from --device option and spawns multiple workers.
        
        Args:
            passages: List of text passages to embed
            
        Returns:
            List of embeddings (one per passage)
        """
        import torch
        import multiprocessing as mp
        
        # Use the device type already configured via --device option
        base_device = self._device  # e.g., 'xpu', 'cuda', 'cpu'
        
        # Determine number of available devices based on device type
        if base_device == 'cpu':
            # For CPU, use requested number as process count
            num_devices = self._num_embedding_devices
        elif base_device == 'xpu' and hasattr(torch, 'xpu'):
            num_devices = torch.xpu.device_count()
        elif base_device == 'cuda':
            num_devices = torch.cuda.device_count()
        else:
            # Fallback for unknown device types
            num_devices = 1
        
        num_workers = min(self._num_embedding_devices, num_devices, len(passages))
        
        if num_workers <= 1:
            # Fallback to single device
            return self._embedding_model.embed_documents(passages)
        
        # Set spawn method for device compatibility (required for XPU/CUDA)
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, ignore
            pass
        
        print(f"🚀 Parallel embedding on {num_workers} {base_device.upper()} device(s)...")
        
        # Split passages into chunks - one chunk per device
        chunk_size = (len(passages) + num_workers - 1) // num_workers
        chunks = [passages[i:i + chunk_size] for i in range(0, len(passages), chunk_size)]
        
        print(f"   Split {len(passages)} passages into {len(chunks)} chunks (~{chunk_size} passages/device)")
        
        # Create result queue and spawn one worker per device
        result_queue = mp.Queue()
        processes = []
        
        encode_kwargs = {'normalize_embeddings': True}
        
        # Spawn one worker per device (not per chunk)
        for device_id in range(num_workers):
            if device_id < len(chunks):
                # Each worker processes one chunk on one device
                p = mp.Process(target=_parallel_embed_worker, 
                           args=(device_id, [device_id], [chunks[device_id]], result_queue,
                                 self._retriever_model_name, encode_kwargs, base_device))
                p.start()
                processes.append(p)
        
        # Collect results
        results = {}
        for _ in range(min(num_workers, len(chunks))):
            chunk_idx, embeddings = result_queue.get()
            if embeddings is not None:
                results[chunk_idx] = embeddings
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Combine results in order
        all_embeddings = []
        for i in range(len(chunks)):
            if i in results:
                all_embeddings.extend(results[i])
        
        print(f"✓ Generated {len(all_embeddings)} embeddings across {num_workers} devices")
        
        return all_embeddings
    
    def _calculate_index_output_size(self):
        """Calculate the size of VectorDB output data (db file - metadata).
        
        Returns the total size in bytes of the serialized database file,
        excluding configuration metadata overhead.
        
        The .db file contains:
        - FAISS index (vectors)
        - Passages (docstore)
        - Metadata (small overhead)
        
        We estimate metadata size and subtract it from total file size.
        """
        from pathlib import Path
        
        # VectorDB uses serialize path, not _database_name like BM25
        if not hasattr(self, '_serialize_path') or not self._serialize_path:
            return 0
        
        db_path = Path(self._serialize_path)
        if not db_path.exists():
            return 0
        
        total_file_size = db_path.stat().st_size
        return total_file_size
    
    def ingest(self, passages: List[str], metadatas: List[dict], **kwargs):
        """Ingest passages with performance monitoring."""
        # Handle BM25-specific parameters gracefully
        if 'num_threads' in kwargs:
            print(f"Warning: num_threads parameter is not used in VectorDB, ignoring")
        
        # Extract passages source path for embeddings caching
        passages_path = kwargs.get('passages_path', None)
        
        # Start timing (works for both benchmark and non-benchmark modes)
        ingestion_start = self._start_ingestion_timer()
        
        total_chars = sum(len(passage) for passage in passages)
        
        # Handle embeddings: try to load from cache or generate new ones
        embeddings = None

        if self._load_embeddings and passages_path:
            embeddings = self._load_embeddings_cache(passages_path)

        # Generate embeddings if not cached
        if embeddings is None:
            if self._num_embedding_devices > 1:
                # Use parallel embedding generation across multiple devices
                embeddings = self._track_component("embedding_generation", total_chars, len(passages), 
                                                  lambda: self._embed_documents_parallel(passages),
                                                  is_pipeline_input=True)
            else:
                # Single device embedding generation
                embeddings = self._track_component("embedding_generation", total_chars, len(passages), 
                                                  lambda: self._embedding_model.embed_documents(passages),
                                                  is_pipeline_input=True)
        
        # Train IVF index if needed (before adding any embeddings)
        if self._vector_index_method == "ivf" and not self._index.is_trained:
            self._train_vector_index(self._index, embeddings)
        
        # Determine batch size: single batch for small datasets, multiple batches for scaling analysis
        track_incremental = self._benchmark and self._monitor and len(passages) >= 500
        if track_incremental:
            batch_size = max(1000, len(passages) // 10)  # 10 batches, minimum 1000 docs per batch
            print(f"🔬 Incremental indexing analysis: {len(passages)} docs in batches of {batch_size}")
        else:
            batch_size = len(passages)  # Single batch
        
        # Track total indexing time for component metrics
        import time
        indexing_component_start = time.perf_counter()
        
        # Process in batches
        for i in range(0, len(passages), batch_size):
            batch_end = min(i + batch_size, len(passages))
            self._ingest_single_batch(passages, metadatas, embeddings, i, batch_end, track_incremental)
        
        indexing_component_end = time.perf_counter()
        indexing_component_duration = indexing_component_end - indexing_component_start
        
        # Create component metrics for the entire indexing operation
        if not track_incremental:
            # For single batch, component was tracked inside _ingest_single_batch
            pass
        elif self._monitor:
            # For incremental, create component metrics here for the entire operation
            embedding_bytes = len(passages) * self._embedding_dimension * self._embedding_bytes_per_element
            
            from ingestion_monitor import ComponentMetrics
            self._monitor.components["faiss_indexing"] = ComponentMetrics(
                name="faiss_indexing",
                duration=indexing_component_duration,
                input_size_bytes=embedding_bytes,
                output_size_bytes=embedding_bytes,  # Vectors stored in FAISS index
                items_processed=len(passages),
                throughput_mb_per_sec=(embedding_bytes / (1024 * 1024)) / indexing_component_duration if indexing_component_duration > 0 else 0,
                throughput_items_per_sec=len(passages) / indexing_component_duration if indexing_component_duration > 0 else 0,
                is_pipeline_input=False,
                is_pipeline_output=True
            )
        
        # Store ingestion metrics for later reporting
        self._ingestion_start = ingestion_start
        self._ingestion_item_count = len(passages)
        self._ingestion_total_chars = total_chars
        
        # Save embeddings to cache 
        if self._load_embeddings and passages_path:
            self._save_embeddings_cache(embeddings, passages_path)

    def _ingest_single_batch(self, passages: List[str], metadatas: List[dict], embeddings: list,
                            batch_start: int, batch_end: int, track_incremental: bool):
        """Ingest a batch of passages. Can be used for single or incremental indexing.
        
        Args:
            passages: All passages
            metadatas: All metadata
            embeddings: All embeddings
            batch_start: Start index for this batch
            batch_end: End index for this batch (exclusive)
            track_incremental: Whether to track this batch for incremental analysis
        """
        import time
        
        # Extract batch data
        batch_passages = passages[batch_start:batch_end]
        batch_metadatas = metadatas[batch_start:batch_end] if metadatas else [{}] * (batch_end - batch_start)
        batch_embeddings = embeddings[batch_start:batch_end]
        
        # Track DB size before adding (for incremental tracking)
        db_size_before = len(self._doc_list) if track_incremental else 0
        
        # Calculate embedding size for this batch
        batch_embedding_bytes = len(batch_passages) * self._embedding_dimension * self._embedding_bytes_per_element
        
        # Time and execute indexing operation
        indexing_start = time.perf_counter()
        
        if track_incremental:
            # For incremental: just add embeddings without component tracking
            self._vector_store.add_embeddings(
                list(zip(batch_passages, batch_embeddings)), 
                batch_metadatas
            )
        else:
            # For single batch: use component tracking
            self._track_component("faiss_indexing", batch_embedding_bytes, len(batch_passages),
                                 lambda: self._vector_store.add_embeddings(
                                     list(zip(batch_passages, batch_embeddings)), batch_metadatas),
                                 is_pipeline_output=True)
        
        indexing_end = time.perf_counter()
        indexing_time = indexing_end - indexing_start
        
        # Update document list
        self._doc_list.extend(batch_passages)
        
        # Track for incremental analysis if requested
        if track_incremental and self._monitor:
            self._monitor.track_incremental_indexing(
                db_size_before=db_size_before,
                batch_size=len(batch_passages),
                indexing_time=indexing_time
            )
    
    def lookup(self, query: str, k: int):
        results = self._vector_store.similarity_search(query, k=k)
        return results
    
    def lookup_with_scores(self, query: str, k: int):
        """
        Lookup documents with similarity scores.
        Returns list of (document, score) tuples.
        
        Note: FAISS returns L2 distances (lower is better), but we convert to
        similarity scores (higher is better) for consistency with BM25.
        """
        results_with_scores = self._vector_store.similarity_search_with_score(query, k=k)
        
        # FAISS returns (document, distance) where distance is L2 distance (lower is better)
        # Convert to similarity score (higher is better) by negating
        # This makes it consistent with BM25 scores for filtering algorithms
        results_with_similarity = [(doc, -distance) for doc, distance in results_with_scores]
        
        return results_with_similarity

    def rerank(self, query: str, passages: List[str]):
        assert self._reranker_model is not None, "Reranker model not initialized"
        pairs = [[query, passage] for passage in passages]

        with torch.no_grad():
            inputs = self._reranker_tokenizer(pairs, padding=True, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            scores = self._reranker_model(**inputs).logits.view(-1).float()
        
        scored_passages = list(zip(passages, scores.cpu().tolist()))
        # Sort by score descending
        scored_passages.sort(key=lambda x: x[1], reverse=True)
        
        # Return passages in sorted order (optionally include scores)
        return [(p, s) for p, s in scored_passages]


    def serialize(self, path: str):
        # Store path for output size calculation
        self._serialize_path = path
        
        data = self._vector_store.serialize_to_bytes()
        with open(path, "wb") as f:
            f.write(data)
        
        # Update output size after serialization (now file exists)
        if self._benchmark and self._monitor:
            self._monitor.set_output_size_callback("faiss_indexing", self._calculate_index_output_size)
        
        # Report performance after serialization if benchmarking
        if self._benchmark and self._monitor and hasattr(self, '_ingestion_start'):
            # Determine db_type based on whether incremental was used
            db_type = "VectorDB (Incremental)" if hasattr(self._monitor, 'indexing_trend') and len(self._monitor.indexing_trend) > 0 else "VectorDB"
            self._report_performance(self._ingestion_start, self._ingestion_item_count, 
                                    self._ingestion_total_chars, db_type)

    def from_serialized(self, path: str):
        assert len(self._vector_store.index_to_docstore_id) == 0, "Vector store already has documents"
        with open(path, "rb") as f:
            data = f.read()
        self._vector_store = FAISS.deserialize_from_bytes(embeddings=self._embedding_model,
            serialized=data,
            allow_dangerous_deserialization=True) # <--- USE WITH CAUTION - Only deserialize files you trust
        
        # If it's an IVF index, restore nprobe setting
        if self._vector_index_method == "ivf" and hasattr(self._vector_store.index, 'nprobe'):
            self._vector_store.index.nprobe = self._ivf_nprobe
            print(f"Restored IVF index with nprobe={self._ivf_nprobe}")
