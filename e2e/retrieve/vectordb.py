import faiss
import torch
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class VectorDB:
    def __init__(self,
            retriever_model: str = None,
            reranker_model: str = None
        ):
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