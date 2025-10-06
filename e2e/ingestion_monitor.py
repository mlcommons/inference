#!/usr/bin/env python3
"""
Ingestion Performance Monitor - Real-time performance tracking for RAG ingestion pipeline.

Usage:
    from ingestion_monitor import IngestionMonitor
    
    monitor = IngestionMonitor()
    
    # Track document processing
    with monitor.track_component("html_parsing"):
        process_html_files(files)
    
    # Track embedding generation  
    with monitor.track_component("embedding_generation"):
        embeddings = generate_embeddings(texts)
        
    # Get performance report
    report = monitor.get_performance_report()
"""

import time
import json
import os
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class ComponentMetrics:
    """Metrics for a single pipeline component."""
    name: str
    duration: float
    input_size_bytes: int
    output_size_bytes: int
    items_processed: int
    throughput_mb_per_sec: float
    throughput_items_per_sec: float
    memory_usage_mb: Optional[float] = None

@dataclass
class IngestionReport:
    """Complete ingestion performance report."""
    total_duration: float
    total_input_bytes: int
    total_output_bytes: int
    total_items: int
    overall_throughput_mb_per_sec: float
    components: List[ComponentMetrics]
    bottleneck_component: str = "none"
    bottleneck_component: str

class IngestionMonitor:
    """Real-time ingestion performance monitoring."""
    
    def __init__(self):
        self.components: Dict[str, ComponentMetrics] = {}
        self.start_time = time.time()
        self.current_component = None
        self.component_start_time = None
        
    @contextmanager
    def track_component(self, component_name: str, input_size_bytes: int = 0, 
                       items_count: int = 0, text_only: bool = False):
        """Context manager to track performance of a pipeline component.
        
        Args:
            component_name: Name of the component being tracked
            input_size_bytes: Input data size in bytes
            items_count: Number of items processed
            text_only: If True, only count text content bytes (exclude metadata)
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        class ComponentContext:
            def __init__(self):
                self.input_size_bytes = input_size_bytes
                self.items_count = items_count
                self.text_only = text_only
                
            def set_input_size(self, size_bytes: int):
                self.input_size_bytes = size_bytes
                
            def set_item_count(self, count: int):
                self.items_count = count
                
            def add_text_bytes(self, text_bytes: int):
                """Add text-only bytes for passage tracking."""
                self.input_size_bytes += text_bytes
        
        context = ComponentContext()
        
        try:
            self.current_component = component_name
            self.component_start_time = start_time
            yield context
            
        finally:
            end_time = time.time()
            duration = end_time - start_time
            end_memory = self._get_memory_usage()
            
            # Calculate throughput  
            throughput_mb = (context.input_size_bytes / (1024 * 1024)) / duration if duration > 0 else 0
            throughput_items = context.items_count / duration if duration > 0 else 0
            
            # Store metrics
            self.components[component_name] = ComponentMetrics(
                name=component_name,
                duration=duration,
                input_size_bytes=context.input_size_bytes,
                output_size_bytes=0,  # Can be set later with set_output_size()
                items_processed=context.items_count,
                throughput_mb_per_sec=throughput_mb,
                throughput_items_per_sec=throughput_items,
                memory_usage_mb=end_memory - start_memory if start_memory and end_memory else None
            )
            
    def set_output_size(self, component_name: str, output_size_bytes: int):
        """Set the output size for a component after processing."""
        if component_name in self.components:
            self.components[component_name].output_size_bytes = output_size_bytes
    
    @contextmanager
    def track_ingestion(self):
        """Track overall ingestion performance."""
        start_time = time.time()
        self.ingestion_start_time = start_time
        
        class IngestionContext:
            def __init__(self):
                self.item_count = 0
                
            def set_item_count(self, count: int):
                self.item_count = count
        
        context = IngestionContext()
        
        try:
            yield context
        finally:
            end_time = time.time()
            self.total_ingestion_time = end_time - start_time
            
    def track_faiss_indexing(self, index, vectors_added: int, vector_bytes: int):
        """Special tracking for FAISS indexing performance."""
        component_name = "faiss_indexing"
        
        if hasattr(index, 'ntotal'):
            index_size = index.ntotal
        else:
            index_size = vectors_added
            
        # Estimate FAISS memory usage (rough approximation)
        if hasattr(index, 'd'):  # dimension
            estimated_memory = (index_size * index.d * 4) / (1024 * 1024)  # 4 bytes per float
        else:
            estimated_memory = None
            
        # Update or create FAISS metrics
        if component_name in self.components:
            metrics = self.components[component_name]
            metrics.output_size_bytes = index_size * 384 * 4  # Assume 384-dim vectors
            metrics.memory_usage_mb = estimated_memory
        else:
            # Create metrics if not tracked with context manager
            self.components[component_name] = ComponentMetrics(
                name=component_name,
                duration=0.1,  # Placeholder
                input_size_bytes=vector_bytes,
                output_size_bytes=index_size * 384 * 4,
                items_processed=vectors_added,
                throughput_mb_per_sec=0,
                throughput_items_per_sec=0,
                memory_usage_mb=estimated_memory
            )
            
    def get_performance_report(self) -> IngestionReport:
        """Generate comprehensive performance report."""
        total_duration = time.time() - self.start_time
        
        # Aggregate metrics
        total_input = sum(c.input_size_bytes for c in self.components.values())
        total_output = sum(c.output_size_bytes for c in self.components.values())
        total_items = sum(c.items_processed for c in self.components.values())
        
        overall_throughput = (total_input / (1024 * 1024)) / total_duration if total_duration > 0 else 0
        
        # Find bottleneck and calculate efficiency ratio
        bottleneck_name = "none"
        
        if self.components:
            bottleneck = min(self.components.values(), key=lambda x: x.throughput_mb_per_sec)
            fastest = max(self.components.values(), key=lambda x: x.throughput_mb_per_sec)
            bottleneck_name = bottleneck.name
            
        return IngestionReport(
            total_duration=total_duration,
            total_input_bytes=total_input,
            total_output_bytes=total_output,
            total_items=total_items,
            overall_throughput_mb_per_sec=overall_throughput,
            components=list(self.components.values()),
            bottleneck_component=bottleneck_name,
        )
        
    def save_report(self, filename: str = "ingestion_performance.json"):
        """Save performance report to JSON file."""
        report = self.get_performance_report()
        
        # Convert to serializable format
        report_dict = asdict(report)
        
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2)
            
        return report_dict
        
    def print_summary(self):
        """Print detailed performance summary with individual components."""
        report = self.get_performance_report()
        
        print("🚀 INGESTION PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"📊 Overall Metrics:")
        print(f"   Total duration: {report.total_duration:.2f}s")
        print(f"   Overall throughput: {report.overall_throughput_mb_per_sec:.2f} MB/s")
        print(f"   Items processed: {report.total_items:,}")
        print(f"   Data processed: {report.total_input_bytes / (1024*1024):.2f} MB")
        print(f"   Bottleneck component: {report.bottleneck_component}")
        
        print(f"\n🔧 Component Performance Details:")
        for component in sorted(report.components, key=lambda x: x.duration, reverse=True):
            percentage = (component.duration / report.total_duration) * 100 if report.total_duration > 0 else 0
            mb_processed = component.input_size_bytes / (1024*1024)
            print(f"   📈 {component.name}:")
            print(f"      ⏱️  Duration: {component.duration:.3f}s ({percentage:.1f}% of total)")
            print(f"      🚀 Throughput: {component.throughput_mb_per_sec:.2f} MB/s")
            print(f"      📦 Items: {component.items_processed:,}")
            print(f"      💾 Data: {mb_processed:.2f} MB")
            if component.memory_usage_mb:
                print(f"      🧠 Memory: {component.memory_usage_mb:.1f} MB")
            print()
                  
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return None

# Example usage and integration helpers
def benchmark_bm25_ingestion(bm25_db, documents: List[str]) -> IngestionReport:
    """Benchmark BM25 ingestion with detailed component tracking."""
    monitor = IngestionMonitor()
    
    # Calculate input size
    input_size = sum(len(doc.encode('utf-8')) for doc in documents)
    
    with monitor.track_component("bm25_tokenization", input_size, len(documents)):
        # Track tokenization if BM25DB exposes it
        pass
        
    with monitor.track_component("bm25_indexing", input_size, len(documents)):
        bm25_db.ingest_from_passages(documents)
        
    return monitor.get_performance_report()

def benchmark_vector_ingestion(vector_db, documents: List[str]) -> IngestionReport:
    """Benchmark Vector DB ingestion with detailed component tracking."""
    monitor = IngestionMonitor()
    
    input_size = sum(len(doc.encode('utf-8')) for doc in documents)
    
    with monitor.track_component("embedding_generation", input_size, len(documents)):
        # This would be tracked inside VectorDB if modified
        pass
        
    with monitor.track_component("vector_indexing", input_size, len(documents)):
        vector_db.ingest_from_passages(documents)
        
    # Track FAISS performance if accessible
    if hasattr(vector_db, '_index') and vector_db._index:
        monitor.track_faiss_indexing(vector_db._index, len(documents), input_size)
        
    return monitor.get_performance_report()

if __name__ == "__main__":
    # Example usage
    monitor = IngestionMonitor()
    
    # Simulate components
    with monitor.track_component("html_parsing", 1024*1024, 100):  # 1MB, 100 files
        time.sleep(0.1)
        
    with monitor.track_component("embedding_generation", 512*1024, 500):  # 512KB, 500 chunks  
        time.sleep(0.5)
        
    monitor.print_summary()
    monitor.save_report("example_performance.json")