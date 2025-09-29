import bm25s
import pickle
from pathlib import Path
from typing import List, Dict, Any
from .ragdb import RagDB

class BM25DB(RagDB):
    """BM25 database implementation for lexical search."""
    
    DATA_DIR = "bm25_data"
    DB_NAME = "bm25.db"
    
    @classmethod
    def get_default_db_name(cls) -> str:
        """Get the default database filename for BM25DB."""
        return cls.DB_NAME
    
    def __init__(self, reranker_model: str = None, device: str = "auto"):
        super().__init__(reranker_model, device)
        self._bm25_retriever = None
        self._passages_list = []
        self._passages_metadata = []
        self._num_threads = 4
    
    def ingest(self, passages: List[str], metadatas: List[Dict[str, Any]], num_threads: int = 4):
        """Ingest passages using BM25 indexing."""
        self._passages_list = passages
        self._passages_metadata = metadatas
        self._num_threads = num_threads
        
        corpus_tokens = bm25s.tokenize(passages, stopwords="en")
        self._bm25_retriever = bm25s.BM25(corpus=passages)
        self._bm25_retriever.index(corpus_tokens)
    
    def lookup(self, query: str, k: int) -> List[Any]:
        """Retrieve top-k passages using BM25."""
        if self._bm25_retriever is None:
            raise ValueError("BM25 retriever not initialized. Call ingest() first.")
        
        query_tokens = bm25s.tokenize([query], stopwords="en")
        results_data, scores = self._bm25_retriever.retrieve(query_tokens, k=k, n_threads=self._num_threads)
        
        results = []
        
        for i in range(len(results_data[0])):
            result_item = results_data[0, i]
            score = scores[0, i]
            
            # Handle different result formats from BM25S
            if isinstance(result_item, dict) and 'id' in result_item:
                # New format: {'id': index, 'text': content}
                doc_idx = result_item['id']
                # Use the text from the dictionary, not from our stored list
                page_content = result_item.get('text', '')
            elif isinstance(result_item, str):
                # Old format: string content, find its index
                doc_idx = None
                for idx, passage in enumerate(self._passages_list):
                    if passage == result_item:
                        doc_idx = idx
                        break
                page_content = result_item if doc_idx is not None else ''
            else:
                # Old format: integer index
                try:
                    doc_idx = int(result_item)
                    page_content = self._passages_list[doc_idx] if 0 <= doc_idx < len(self._passages_list) else ''
                except (ValueError, TypeError):
                    continue
            
            # Create result object if valid index found
            if doc_idx is not None and 0 <= doc_idx < len(self._passages_metadata):
                result = type('Result', (), {
                    'page_content': page_content,
                    'metadata': self._passages_metadata[doc_idx]
                })()
                results.append(result)
        
        return results
    
    def serialize(self, path: str):
        """Save BM25 index and metadata."""
        if self._bm25_retriever is None:
            raise ValueError("BM25 retriever not initialized")
        
        # Save BM25 index to separate directory
        bm25_dir = Path(self.DATA_DIR)
        self._bm25_retriever.save(str(bm25_dir))
        
        # Save database file outside bm25_data
        db_path = Path(path)
        
        data = {
            'type': 'BM25DB',
            'bm25_directory': str(bm25_dir),
            'passages_metadata': self._passages_metadata,
            'num_passages': len(self._passages_list),
            'num_threads': getattr(self, '_num_threads', 4),
            'method': 'bm25'
        }
        
        with open(db_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        db_size = db_path.stat().st_size
        bm25_size = sum(f.stat().st_size for f in bm25_dir.rglob('*') if f.is_file())
        total_size = db_size + bm25_size
        print(f"BM25 database saved to {db_path} ({db_size / (1024**2):.1f} MB)")
        print(f"BM25 index saved to {bm25_dir} ({bm25_size / (1024**2):.1f} MB)")
        print(f"Total: {total_size / (1024**2):.1f} MB")

    def from_serialized(self, path: str):
        """Load BM25 index and metadata."""
        db_path = Path(path)
        
        with open(db_path, "rb") as f:
            data = pickle.load(f)
        
        self._passages_metadata = data['passages_metadata']
        self._num_threads = data.get('num_threads', 4)
        bm25_directory = data.get('bm25_directory', self.DATA_DIR)
        
        self._bm25_retriever = bm25s.BM25.load(bm25_directory, load_corpus=True)
        self._passages_list = list(self._bm25_retriever.corpus)
        
        print(f"BM25 database loaded from {db_path} ({len(self._passages_list)} passages)")