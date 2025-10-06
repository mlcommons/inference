import bm25s
import pickle
from pathlib import Path
from typing import List, Dict, Any
from .ragdb import RagDB

class BM25DB(RagDB):
    """BM25 database implementation for lexical search."""
    
    @classmethod
    def get_default_db_name(cls) -> str:
        """Get the default database filename for BM25DB."""
        return "bm25.db"
    
    def __init__(self, reranker_model: str = None, device: str = "auto", k1: float = None, b: float = None, method: str = None, 
                 database: str = None, delta: float = None, idf_method: str = None, dtype: str = None, 
                 backend: str = None, token_pattern: str = None, stopwords = None, stemmer = None, 
                 lower: bool = None, show_progress: bool = None, benchmark: bool = False, **kwargs):
        super().__init__(reranker_model, device, benchmark)
        self._bm25_retriever = None
        self._doc_list = []
        self._passages_metadata = []
        self._num_threads = 4
        
        # Set database name (use default if not provided)
        self._database_name = database if database is not None else "bm25"
        
        # Set BM25 parameters with defaults
        self._k1 = k1 if k1 is not None else 1.5
        self._b = b if b is not None else 0.75
        self._method = method if method is not None else "lucene"
        self._delta = delta if delta is not None else 0.5
        self._idf_method = idf_method if idf_method is not None else None  # Will default to method
        self._dtype = dtype if dtype is not None else "float32"
        self._backend = backend if backend is not None else "numba"  # Default to numba for speed
        
        # Tokenization parameters
        self._token_pattern = token_pattern if token_pattern is not None else r"(?u)\b\w\w+\b"
        self._stopwords = stopwords if stopwords is not None else "en"  # Default to English stopwords
        self._stemmer = self._create_stemmer_func(stemmer)  # Create stemmer function
        self._lower = lower if lower is not None else True
        self._show_progress = show_progress if show_progress is not None else True
        
        # Ignore vector-specific parameters
        ignored_params = []
        if 'retriever_model' in kwargs:
            ignored_params.append('retriever_model')
        if ignored_params:
            print(f"BM25DB: Ignoring vector-specific parameters: {ignored_params}")
    
    def _create_stemmer_func(self, stemmer: str):
        """Create stemmer function based on stemmer type."""
        self._original_stemmer_type = stemmer  # Store for serialization
        if stemmer is None:
            return None
        elif stemmer == "porter":
            try:
                from nltk.stem import PorterStemmer
                stemmer_obj = PorterStemmer()
                def porter_wrapper(tokens):
                    if isinstance(tokens, list):
                        return [stemmer_obj.stem(token) for token in tokens]
                    else:
                        return stemmer_obj.stem(tokens)
                return porter_wrapper
            except ImportError:
                print("Warning: NLTK not available, falling back to no stemming")
                return None
        elif stemmer == "snowball":
            try:
                from nltk.stem import SnowballStemmer
                stemmer_obj = SnowballStemmer("english")
                def snowball_wrapper(tokens):
                    if isinstance(tokens, list):
                        return [stemmer_obj.stem(token) for token in tokens]
                    else:
                        return stemmer_obj.stem(tokens)
                return snowball_wrapper
            except ImportError:
                print("Warning: NLTK not available, falling back to no stemming")
                return None
        elif stemmer == "lancaster":
            try:
                from nltk.stem import LancasterStemmer
                stemmer_obj = LancasterStemmer()
                def lancaster_wrapper(tokens):
                    if isinstance(tokens, list):
                        return [stemmer_obj.stem(token) for token in tokens]
                    else:
                        return stemmer_obj.stem(tokens)
                return lancaster_wrapper
            except ImportError:
                print("Warning: NLTK not available, falling back to no stemming")
                return None
        elif stemmer == "pystemmer":
            try:
                import Stemmer
                stemmer_obj = Stemmer.Stemmer("english")
                def pystemmer_wrapper(tokens):
                    if isinstance(tokens, list):
                        return stemmer_obj.stemWords(tokens)  # Note: stemWords for list
                    else:
                        return stemmer_obj.stemWord(tokens)   # Note: stemWord for single
                return pystemmer_wrapper
            except ImportError:
                print("Warning: PyStemmer not available, falling back to no stemming")
                return None
        else:
            print(f"Warning: Unknown stemmer '{stemmer}', falling back to no stemming")
            return None
    
    def _get_stemmer_type(self):
        """Get the stemmer type for serialization."""
        if self._stemmer is None:
            return None
        # Store the original stemmer type for reconstruction
        return getattr(self, '_original_stemmer_type', None)
    
    def ingest(self, passages: List[str], metadatas: List[Dict[str, Any]], num_threads: int = 4):
        """Ingest passages using BM25 indexing with performance monitoring."""
        import time
        
        self._doc_list = passages
        self._passages_metadata = metadatas
        self._num_threads = num_threads
        total_chars = sum(len(passage) for passage in passages)
        
        start_time = time.perf_counter()
        
        # Unified tokenization with optional monitoring
        corpus_tokens = self._track_component("bm25_tokenization", total_chars, len(passages),
            lambda: bm25s.tokenize(passages, stopwords=self._stopwords, 
                                 token_pattern=self._token_pattern, stemmer=self._stemmer,
                                 lower=self._lower, show_progress=self._show_progress))
        
        # Create BM25 retriever
        self._bm25_retriever = bm25s.BM25(
            k1=self._k1, b=self._b, method=self._method,
            delta=self._delta, idf_method=self._idf_method,
            dtype=self._dtype, backend=self._backend
        )
        
        # Unified indexing with optional monitoring
        self._track_component("bm25_indexing", total_chars, len(passages),
            lambda: self._bm25_retriever.index(corpus_tokens, show_progress=self._show_progress))
        
        # Report performance
        self._report_performance(start_time, len(passages), total_chars, "BM25DB")
    
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
            total_time = end_time - start_time
            docs_per_sec = item_count / total_time if total_time > 0 else 0
            chars_per_sec = total_chars / total_time if total_time > 0 else 0
            print(f"BM25 ingestion: {item_count} docs, {total_chars:,} chars in {total_time:.2f}s")
            print(f"  Performance: {docs_per_sec:.1f} docs/sec, {chars_per_sec/1024:.1f} KB/sec")
    
    def ingest_from_folder(self, folder_path: str, num_threads: int = 4):
        """Ingest whole txt files from a folder instead of passages"""
        from pathlib import Path
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import load_url_mapping
        
        folder_path = Path(folder_path)
        url_mapping = load_url_mapping(str(folder_path))
        
        doc_list = []
        passage_metadata = []
        
        # Process all .txt files in the folder
        txt_files = list(folder_path.glob("*.txt"))
        if not txt_files:
            print(f"Warning: No .txt files found in {folder_path}")
            return
        
        for txt_file in txt_files:
            try:
                # Read the file content
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    continue
                
                # Get base filename without extension for URL lookup
                base_filename = txt_file.stem
                original_url = url_mapping.get(base_filename, "")
                
                # Create passage and metadata
                doc_list.append(content)
                metadata = {
                    'pdf_filename': txt_file.name,
                    'original_url': original_url,
                    'base_filename': base_filename
                }
                passage_metadata.append(metadata)
                
            except Exception as e:
                print(f"Warning: Could not read {txt_file}: {e}")
                continue
        
        print(f"Loaded {len(doc_list)} documents from {folder_path}")
        
        # Call regular ingest method
        self.ingest(doc_list, passage_metadata, num_threads)

    def lookup(self, query: str, k: int) -> List[Any]:
        """Retrieve top-k passages using BM25."""
        if self._bm25_retriever is None:
            raise ValueError("BM25 retriever not initialized. Call ingest() first.")
        
        query_tokens = bm25s.tokenize([query], 
                                     stopwords=self._stopwords,
                                     token_pattern=self._token_pattern,
                                     stemmer=self._stemmer,
                                     lower=self._lower)
        results_data, scores = self._bm25_retriever.retrieve(query_tokens, k=k, n_threads=self._num_threads)
        
        results = []
        
        for i in range(len(results_data[0])):
            result_item = results_data[0, i]
            score = scores[0, i]
            
            # Handle different result formats from BM25S
            if isinstance(result_item, dict):
                # Dictionary format: {'id': idx, 'text': content}
                doc_idx = int(result_item['id'])
                page_content = result_item.get('text', '')
            else:
                # Index format: result_item is the document index (convert to int)
                doc_idx = int(result_item)
                page_content = self._doc_list[doc_idx] if doc_idx < len(self._doc_list) else ""
            
            # Create result object if valid index found
            if doc_idx is not None and 0 <= doc_idx < len(self._passages_metadata):
                result = type('Result', (), {
                    'page_content': page_content,
                    'metadata': self._passages_metadata[doc_idx]
                })()
                results.append(result)
        
        return results
    
    def lookup_with_scores(self, query: str, k: int):
        """Return results with BM25 scores for score-based filtering."""
        if self._bm25_retriever is None:
            raise ValueError("BM25 retriever not initialized. Call ingest() first.")
        
        query_tokens = bm25s.tokenize([query], 
                                     stopwords=self._stopwords,
                                     token_pattern=self._token_pattern,
                                     stemmer=self._stemmer,
                                     lower=self._lower)
        results_data, scores = self._bm25_retriever.retrieve(query_tokens, k=k, n_threads=self._num_threads)
        
        results_with_scores = []
        
        for i in range(len(results_data[0])):
            result_item = results_data[0, i]
            score = float(scores[0, i])
            
            # Handle different result formats from BM25S
            if isinstance(result_item, dict):
                doc_idx = int(result_item['id'])
                page_content = result_item.get('text', '')
            else:
                doc_idx = int(result_item)
                page_content = self._doc_list[doc_idx] if doc_idx < len(self._doc_list) else ""
            
            # Create result object if valid index found
            if doc_idx is not None and 0 <= doc_idx < len(self._passages_metadata):
                result = type('Result', (), {
                    'page_content': page_content,
                    'metadata': self._passages_metadata[doc_idx]
                })()
                results_with_scores.append((result, score))
        
        return results_with_scores
    
    def serialize(self, path: str):
        """Save BM25 index and metadata."""
        if self._bm25_retriever is None:
            raise ValueError("BM25 retriever not initialized")
        
        # Save BM25 index to separate directory based on database name
        bm25_dir = Path(self.get_data_dir(path))
        self._bm25_retriever.save(str(bm25_dir))
        
        # Save database file
        db_path = Path(path)
        
        data = {
            'type': 'BM25DB',
            'bm25_directory': str(bm25_dir),
            'passages_metadata': self._passages_metadata,
            'doc_list': self._doc_list,  # Save the actual document content
            'num_passages': len(self._doc_list),
            'num_threads': getattr(self, '_num_threads', 4),
            'k1': self._k1,
            'b': self._b,
            'method': self._method,
            'delta': self._delta,
            'idf_method': self._idf_method,
            'dtype': self._dtype,
            'backend': self._backend,
            'token_pattern': self._token_pattern,
            'stopwords': self._stopwords,
            'stemmer_type': self._get_stemmer_type(),  # Save stemmer type, not function
            'lower': self._lower,
            'show_progress': self._show_progress
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
        
        # Check if BM25 parameters match current instance
        saved_k1 = data.get('k1', 1.5)
        saved_b = data.get('b', 0.75) 
        saved_method = data.get('method', 'lucene')
        
        # Only warn about mismatches for explicitly specified parameters (not None)
        mismatches = [saved_k1 != self._k1, saved_b != self._b, saved_method != self._method]

        if any(mismatches):
            print(f"WARNING: Explicitly specified BM25 parameters don't match database:")
            for mismatch in mismatches:
                print(f"  {mismatch}")
            print(f"  Using database parameters. To use new parameters, recreate the database.")
        
        # Always use database parameters (whether there was a warning or not)
        self._k1, self._b, self._method = saved_k1, saved_b, saved_method
        
        # Load all tokenization parameters
        self._delta = data.get('delta', 0.5)
        self._idf_method = data.get('idf_method', None)
        self._dtype = data.get('dtype', 'float32')
        self._backend = data.get('backend', 'numba')  # Default to numba for speed
        self._token_pattern = data.get('token_pattern', r"(?u)\b\w\w+\b")
        self._stopwords = data.get('stopwords', 'en')
        self._lower = data.get('lower', True)
        self._show_progress = data.get('show_progress', True)
        
        # Recreate stemmer function from saved type
        stemmer_type = data.get('stemmer_type', None)
        self._stemmer = self._create_stemmer_func(stemmer_type)
        
        self._passages_metadata = data['passages_metadata']
        self._num_threads = data.get('num_threads', 4)
        # Use the data directory based on the database path, or fallback to saved directory
        default_bm25_dir = self.get_data_dir(path)
        bm25_directory = data.get('bm25_directory', default_bm25_dir)
        
        # Load BM25 without corpus to avoid the same issue we fixed in __init__
        self._bm25_retriever = bm25s.BM25.load(bm25_directory, load_corpus=False)
        
        # Load doc_list from saved data if available, otherwise use fallback
        if 'doc_list' in data:
            self._doc_list = data['doc_list']
        else:
            # Fallback for older databases that didn't save doc_list
            print("Warning: doc_list not found in database, using URLs as fallback")
            self._doc_list = [metadata.get('original_url', '') for metadata in self._passages_metadata]
        
        print(f"BM25 database loaded from {db_path} ({len(self._doc_list)} passages)")
        print(f"BM25 parameters: k1={self._k1}, b={self._b}, method='{self._method}'")