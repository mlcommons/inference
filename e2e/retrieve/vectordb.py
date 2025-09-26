import faiss
import torch
try:
    import intel_extension_for_pytorch as ipex
    XPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, 'xpu') else False
except ImportError:
    XPU_AVAILABLE = False
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class VectorDB:
    def __init__(self,
            retriever_model: str = None,
            reranker_model: str = None,
            device: str = "auto"
        ):
        self._retriever_model_name = retriever_model
        self._reranker_model_name = reranker_model
        
        # Determine device
        if device == "auto":
            if XPU_AVAILABLE:
                self._device = "xpu"
                print("Using XPU device for embeddings and reranking")
            elif torch.cuda.is_available():
                self._device = "cuda"
                print("Using CUDA device for embeddings and reranking")
            else:
                self._device = "cpu"
                print("Using CPU device for embeddings and reranking")
        else:
            self._device = device

        # Initialize embedding model
        model_kwargs = {'device': self._device}
        
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=self._retriever_model_name, # Embedding model == retriever model
            model_kwargs=model_kwargs
        )
        self._embedding_dimension = len(self._embedding_model.embed_query("hello world"))

        self._reranker_model = None
        self._reranker_tokenizer = None
        if self._reranker_model_name:
            self._reranker_model = AutoModelForSequenceClassification.from_pretrained(self._reranker_model_name)
            self._reranker_tokenizer = AutoTokenizer.from_pretrained(self._reranker_model_name)
            
            # Move model to device
            self._reranker_model = self._reranker_model.to(self._device)
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
    
    def ingest(self, passages: List[str], metadatas: List[dict]):
        self._vector_store.add_texts(passages, metadatas)

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