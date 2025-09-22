import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from transformers import AutoModel, AutoTokenizer

class VectorDB:
    def __init__(self, retriever_model: str, reranker_model: str):
        self._retriever_model_name = retriever_model
        self._reranker_model_name = reranker_model

        self._embedding_model = HuggingFaceEmbeddings(model_name=self._retriever_model_name) # Embedding model == retriever model
        self._embedding_dimension = len(self._embedding_model.embed_query("hello world"))

        self._reranker_model = AutoModel.from_pretrained(self._reranker_model_name)
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
        pass

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