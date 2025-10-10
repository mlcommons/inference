import faiss
import torch
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .ragdb import RagDB

class VectorDB(RagDB):
    @classmethod
    def get_default_db_name(cls) -> str:
        """Get the default database filename for VectorDB."""
        return "vector.db"
    
    def __init__(self,
            retriever_model: str = None,
            reranker_model: str = None,
            device: str = "auto",
            benchmark: bool = False,
            **kwargs
        ):
        super().__init__(reranker_model, device, benchmark)
        self._retriever_model_name = retriever_model
        self._reranker_model_name = reranker_model

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

        # The index defines the algoriothm used for the similarity search
        self._index = faiss.IndexFlatL2(self._embedding_dimension)

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
        
        # Start timing (works for both benchmark and non-benchmark modes)
        ingestion_start = self._start_ingestion_timer()
        
        # Auto-enable incremental indexing for scaling analysis when benchmarking
        # and we have enough documents to make it meaningful
        if (self._benchmark and self._monitor and len(passages) >= 500):
            batch_size = max(1000, len(passages) // 10)  # 10 batches, minimum 1000 docs per batch
            return self._ingest_incremental(passages, metadatas, batch_size, ingestion_start)
        
        total_chars = sum(len(passage) for passage in passages)
        
        embeddings = self._track_component("embedding_generation", total_chars, len(passages), 
                                          lambda: self._embedding_model.embed_documents(passages),
                                          is_pipeline_input=True)
        
        # Calculate embedding size in bytes for accurate throughput measurement
        embedding_bytes = len(passages) * self._embedding_dimension * self._embedding_bytes_per_element
        self._track_component("faiss_indexing", embedding_bytes, len(passages),
                             lambda: self._vector_store.add_embeddings(
                                 list(zip(passages, embeddings)), metadatas),
                             is_pipeline_output=True)
        
        # Store ingestion metrics for later reporting (after serialization)
        self._ingestion_start = ingestion_start
        self._ingestion_item_count = len(passages)
        self._ingestion_total_chars = total_chars
        
        # Keep track of ingested documents
        self._doc_list.extend(passages)
    
    def _ingest_incremental(self, passages: List[str], metadatas: List[dict], batch_size: int, ingestion_start: float):
        """Ingest in batches to study indexing scaling trends."""
        import time
        
        print(f"🔬 Incremental indexing analysis: {len(passages)} docs in batches of {batch_size}")
        
        total_chars = sum(len(passage) for passage in passages)
        
        embeddings = self._track_component("embedding_generation", total_chars, len(passages),
                                         lambda: self._embedding_model.embed_documents(passages),
                                         is_pipeline_input=True)
        
        # Track total indexing time for benchmark component
        indexing_component_start = time.perf_counter()
        
        # Now add to index in batches to track scaling
        for i in range(0, len(passages), batch_size):
            batch_end = min(i + batch_size, len(passages))
            batch_passages = passages[i:batch_end]
            batch_metadatas = metadatas[i:batch_end] if metadatas else [{}] * (batch_end - i)
            batch_embeddings = embeddings[i:batch_end]
            
            # Track current DB size before adding this batch
            db_size_before = len(self._doc_list)
            
            # Time just the indexing operation
            indexing_start = time.perf_counter()
            self._vector_store.add_embeddings(
                list(zip(batch_passages, batch_embeddings)), 
                batch_metadatas
            )
            indexing_end = time.perf_counter()
            indexing_time = indexing_end - indexing_start
            
            # Track this batch's performance
            self._monitor.track_incremental_indexing(
                db_size_before=db_size_before,
                batch_size=len(batch_passages),
                indexing_time=indexing_time
            )
            
            # Update our document list
            self._doc_list.extend(batch_passages)
            
        indexing_component_end = time.perf_counter()
        indexing_component_duration = indexing_component_end - indexing_component_start
        
        if self._benchmark and self._monitor:
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
        
        # Store ingestion metrics for later reporting (after serialization)
        self._ingestion_start = ingestion_start
        self._ingestion_item_count = len(passages)
        self._ingestion_total_chars = total_chars
    
    def lookup(self, query: str, k: int):
        results = self._vector_store.similarity_search(query, k=k)
        return results

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