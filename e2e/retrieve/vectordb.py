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

        self._embedding_model = HuggingFaceEmbeddings(model_name=self._retriever_model_name) # Embedding model == retriever model
        self._embedding_dimension = len(self._embedding_model.embed_query("hello world"))

        self._reranker_model = None
        self._reranker_tokenizer = None
        if self._reranker_model_name:
            self._reranker_model = AutoModelForSequenceClassification.from_pretrained(self._reranker_model_name)
            self._reranker_tokenizer = AutoTokenizer.from_pretrained(self._reranker_model_name)
            self._reranker_model.eval()

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
    
    def ingest(self, passages: List[str], metadatas: List[dict], **kwargs):
        """Ingest passages with performance monitoring."""
        import time
        
        # Handle BM25-specific parameters gracefully
        if 'num_threads' in kwargs:
            print(f"Warning: num_threads parameter is not used in VectorDB, ignoring")
        
        total_chars = sum(len(passage) for passage in passages)
        start_time = time.perf_counter()
        
        # Unified processing with optional monitoring
        embeddings = self._track_component("embedding_generation", total_chars, len(passages), 
                                          lambda: self._embedding_model.embed_documents(passages))
        
        self._track_component("faiss_indexing", total_chars, len(passages),
                             lambda: self._vector_store.add_embeddings(
                                 list(zip(passages, embeddings)), metadatas))
        
        # Report performance
        self._report_performance(start_time, len(passages), total_chars, "VectorDB")
        
        # Keep track of ingested documents
        self._doc_list = passages
    
    def _track_component(self, name: str, total_chars: int, item_count: int, func):
        """Execute function with optional component tracking."""
        if self._benchmark and self._monitor:
            with self._monitor.track_component(name, input_size_bytes=total_chars, 
                                             items_count=item_count, text_only=True) as ctx:
                result = func()
                ctx.add_text_bytes(total_chars)
                return result
        else:
            return func()
    
    def _report_performance(self, start_time: float, item_count: int, total_chars: int, db_type: str):
        """Report performance metrics."""
        if self._benchmark and self._monitor:
            with self._monitor.track_ingestion() as ingestion_ctx:
                ingestion_ctx.set_item_count(item_count)
            print(f"\n=== {db_type} Performance ===")
            self._monitor.print_summary()
        else:
            end_time = time.perf_counter()
            duration = end_time - start_time
            docs_per_sec = item_count / duration if duration > 0 else 0
            chars_per_sec = total_chars / duration if duration > 0 else 0
            print(f"Vector ingestion: {item_count} docs, {total_chars:,} chars in {duration:.2f}s")
            print(f"  Performance: {docs_per_sec:.1f} docs/sec, {chars_per_sec/1024:.1f} KB/sec")
    
    def lookup(self, query: str, k: int):
        results = self._vector_store.similarity_search(query, k=k)
        return results

    def rerank(self, query: str, passages: List[str]):
        assert self._reranker_model is not None, "Reranker model not initialized"
        pairs = [[query, passage] for passage in passages]

        with torch.no_grad():
            inputs = self._reranker_tokenizer(pairs, padding=True, return_tensors='pt', truncation=True, max_length=512)
            scores = self._reranker_model(**inputs).logits.view(-1).float()
        
        scored_passages = list(zip(passages, scores.tolist()))
        # Sort by score descending
        scored_passages.sort(key=lambda x: x[1], reverse=True)
        
        # Return passages in sorted order (optionally include scores)
        return [(p, s) for p, s in scored_passages]


    def serialize(self, path: str):
        data = self._vector_store.serialize_to_bytes()
        with open(path, "wb") as f:
            f.write(data)

    def from_serialized(self, path: str):
        assert len(self._vector_store.index_to_docstore_id) == 0, "Vector store already has documents"
        with open(path, "rb") as f:
            data = f.read()
        self._vector_store = FAISS.deserialize_from_bytes(embeddings=self._embedding_model,
            serialized=data,
            allow_dangerous_deserialization=True) # <--- USE WITH CAUTION - Only deserialize files you trust