import abc
from typing import List, Dict, Any

class RagDB(abc.ABC):
    """Base class for retrieval-augmented generation databases."""
    
    def __init__(self, reranker_model: str = None, device: str = "auto"):
        self._reranker_model_name = reranker_model
        self._device = self._determine_device(device)
        self._reranker_model = None
        self._reranker_tokenizer = None
        
        # Initialize reranker if specified
        if self._reranker_model_name:
            self._init_reranker()
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use."""
        try:
            import intel_extension_for_pytorch as ipex
            import torch
            XPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, 'xpu') else False
        except ImportError:
            import torch
            XPU_AVAILABLE = False
        
        if device == "auto":
            if XPU_AVAILABLE:
                print("Using XPU device for reranking")
                return "xpu"
            elif torch.cuda.is_available():
                print("Using CUDA device for reranking")
                return "cuda"
            else:
                print("Using CPU device for reranking")
                return "cpu"
        else:
            return device
    
    def _init_reranker(self):
        """Initialize the reranker model."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        self._reranker_model = AutoModelForSequenceClassification.from_pretrained(self._reranker_model_name)
        self._reranker_tokenizer = AutoTokenizer.from_pretrained(self._reranker_model_name)
        
        self._reranker_model = self._reranker_model.to(self._device)
        self._reranker_model.eval()
    
    @abc.abstractmethod
    def ingest(self, passages: List[str], metadatas: List[Dict[str, Any]]):
        """Ingest passages and their metadata into the database."""
        pass
    
    @abc.abstractmethod
    def lookup(self, query: str, k: int) -> List[Any]:
        """Retrieve top-k relevant passages for a query."""
        pass
    
    @abc.abstractmethod
    def serialize(self, path: str):
        """Serialize the database to disk."""
        pass
    
    @abc.abstractmethod
    def from_serialized(self, path: str):
        """Load the database from disk."""
        pass
    
    def rerank(self, query: str, passages: List[str]):
        """Rerank passages using the reranker model."""
        if self._reranker_model is None:
            # If no reranker, return passages with dummy scores
            return [(p, 0.0) for p in passages]
        
        import torch
        
        pairs = [[query, passage] for passage in passages]
        
        with torch.no_grad():
            inputs = self._reranker_tokenizer(pairs, padding=True, return_tensors='pt', 
                                            truncation=True, max_length=512)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            scores = self._reranker_model(**inputs).logits.view(-1).float()
        
        scored_passages = list(zip(passages, scores.cpu().tolist()))
        # Sort by score descending
        scored_passages.sort(key=lambda x: x[1], reverse=True)
        
        return scored_passages
    
    def lookup_with_rerank(self, query: str, k: int, rerank_k: int = None) -> List[Any]:
        """Retrieve and rerank passages."""
        if rerank_k is None:
            rerank_k = k
            
        # Get initial results
        results = self.lookup(query, k=rerank_k)
        
        # If no reranker or fewer results than requested, return as-is
        if self._reranker_model is None or len(results) <= k:
            return results[:k]
        
        # Extract passages for reranking
        passages = [result.page_content for result in results]
        
        # Rerank
        reranked_passages = self.rerank(query, passages)
        
        # Map back to original results and return top-k
        reranked_results = []
        for passage, score in reranked_passages[:k]:
            for result in results:
                if result.page_content == passage:
                    reranked_results.append(result)
                    break
        
        return reranked_results

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device