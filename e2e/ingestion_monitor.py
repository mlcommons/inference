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
    is_pipeline_input: bool = False  # Mark if this is a pipeline input component
    is_pipeline_output: bool = False  # Mark if this is a pipeline output component

@dataclass
class IndexingTrendPoint:
    """Single data point for indexing performance trend."""
    db_size: int  # Number of items in DB at this point
    batch_size: int  # Number of items added in this batch
    indexing_time: float  # Time to add this batch (seconds)
    throughput_items_per_sec: float
    cumulative_time: float  # Total time so far
    
@dataclass
class IngestionReport:
    """Complete ingestion performance report."""
    total_duration: float
    total_input_bytes: int
    total_output_bytes: int
    total_items: int
    overall_throughput_mb_per_sec: float
    components: List[ComponentMetrics]
    bottleneck_component: str
    indexing_trend: List[IndexingTrendPoint] = None  # For scaling analysis = "none"
    bottleneck_component: str

class IngestionMonitor:
    """Real-time ingestion performance monitoring."""
    
    def __init__(self):
        self.components: Dict[str, ComponentMetrics] = {}
        self.start_time = None  # Will be set when ingestion starts
        self.current_component = None
        self.component_start_time = None
        self.indexing_trend: List[IndexingTrendPoint] = []
        self.cumulative_indexing_time = 0.0
        
    def start_ingestion(self):
        """Mark the start of ingestion. Should be called at the beginning of ingest()."""
        self.start_time = time.time()
        
    @contextmanager
    def track_component(self, component_name: str, input_size_bytes: int = 0, 
                       items_count: int = 0, text_only: bool = False,
                       is_pipeline_input: bool = False, is_pipeline_output: bool = False):
        """Context manager to track performance of a pipeline component.
        
        Args:
            component_name: Name of the component being tracked
            input_size_bytes: Input data size in bytes
            items_count: Number of items processed
            text_only: If True, only count text content bytes (exclude metadata)
            is_pipeline_input: If True, mark as pipeline input component for aggregation
            is_pipeline_output: If True, mark as pipeline output component for aggregation
        """
        start_time = time.time()
        
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
                is_pipeline_input=is_pipeline_input,
                is_pipeline_output=is_pipeline_output
            )
            
    def set_output_size(self, component_name: str, output_size_bytes: int):
        """Set the output size for a component after processing."""
        if component_name in self.components:
            self.components[component_name].output_size_bytes = output_size_bytes
    
    def set_output_size_callback(self, component_name: str, callback_fn):
        """Set the output size for a component using a callback function.
        
        This is useful when the output size calculation is complex or requires
        accessing class-specific data (e.g., BM25 index files).
        
        Args:
            component_name: Name of the component
            callback_fn: Function that returns the output size in bytes
        """
        if component_name in self.components:
            try:
                output_size = callback_fn()
                self.components[component_name].output_size_bytes = output_size
            except Exception as e:
                print(f"Warning: Failed to calculate output size for {component_name}: {e}")
    
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
            
    def track_incremental_indexing(self, db_size_before: int, batch_size: int, 
                                 indexing_time: float):
        """Track indexing performance for incremental batches to analyze scaling trends.
        
        Args:
            db_size_before: Number of items in DB before adding this batch
            batch_size: Number of items added in this batch  
            indexing_time: Time taken to index this batch (seconds)
        """
        db_size_after = db_size_before + batch_size
        throughput = batch_size / indexing_time if indexing_time > 0 else 0
        self.cumulative_indexing_time += indexing_time
        
        trend_point = IndexingTrendPoint(
            db_size=db_size_after,
            batch_size=batch_size,
            indexing_time=indexing_time,
            throughput_items_per_sec=throughput,
            cumulative_time=self.cumulative_indexing_time
        )
        
        self.indexing_trend.append(trend_point)
        
    def track_faiss_indexing(self, index, vectors_added: int, vector_bytes: int):
        """Special tracking for FAISS indexing performance."""
        component_name = "faiss_indexing"
        
        if hasattr(index, 'ntotal'):
            index_size = index.ntotal
        else:
            index_size = vectors_added
            
        # Update or create FAISS metrics
        if component_name in self.components:
            metrics = self.components[component_name]
            metrics.output_size_bytes = index_size * 384 * 4  # Assume 384-dim vectors
        else:
            # Create metrics if not tracked with context manager
            self.components[component_name] = ComponentMetrics(
                name=component_name,
                duration=0.1,  # Placeholder
                input_size_bytes=vector_bytes,
                output_size_bytes=index_size * 384 * 4,
                items_processed=vectors_added,
                throughput_mb_per_sec=0,
                throughput_items_per_sec=0
            )
            
    def get_performance_report(self) -> IngestionReport:
        """Generate comprehensive performance report."""
        # Calculate duration from when start_ingestion() was called
        if self.start_time is None:
            raise ValueError("start_ingestion() must be called before getting performance report")
        
        total_duration = time.time() - self.start_time
        
        # Aggregate metrics based on pipeline input/output flags
        # If no flags set, fall back to first component for input
        input_components = [c for c in self.components.values() if c.is_pipeline_input]
        output_components = [c for c in self.components.values() if c.is_pipeline_output]
        
        total_input = sum(c.input_size_bytes for c in input_components)
        total_items = sum(c.items_processed for c in input_components)
        total_output = sum(c.output_size_bytes for c in output_components)
        
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
            indexing_trend=self.indexing_trend if self.indexing_trend else None
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
        # Show input data size from report (aggregated from marked input components or first component)
        print(f"   Input data size: {report.total_input_bytes / (1024*1024):.2f} MB")
        
        # Show output size and expansion ratio if output data exists
        if report.total_output_bytes > 0:
            output_size_mb = report.total_output_bytes / (1024*1024)
            expansion_ratio = report.total_output_bytes / report.total_input_bytes if report.total_input_bytes > 0 else 0
            print(f"   Output data size: {output_size_mb:.2f} MB")
            print(f"   Output/Input ratio: {expansion_ratio:.1f}x")
        
        print(f"   Bottleneck component: {report.bottleneck_component}")
        
        print(f"\n🔧 Component Performance Details:")
        for component in sorted(report.components, key=lambda x: x.duration, reverse=True):
            percentage = (component.duration / report.total_duration) * 100 if report.total_duration > 0 else 0
            mb_processed = component.input_size_bytes / (1024*1024)
            avg_latency_ms = (component.duration * 1000 / component.items_processed) if component.items_processed > 0 else 0
            print(f"   📈 {component.name}:")
            print(f"      ⏱️  Duration: {component.duration:.3f}s ({percentage:.1f}% of total)")
            print(f"      🚀 Throughput: {component.throughput_mb_per_sec:.2f} MB/s")
            print(f"      📦 Items: {component.items_processed:,}")
            print(f"      💾 Data: {mb_processed:.2f} MB")
            print(f"      ⚡ Avg latency: {avg_latency_ms:.2f}ms per item")
            print()
            
        # Print indexing trend analysis if available
        if report.indexing_trend and len(report.indexing_trend) > 1:
            print("📈 VECTOR DB INDEXING SCALING ANALYSIS")
            print("=" * 60)
            print("DB Size → Batch Time (Throughput)")
            
            for i, point in enumerate(report.indexing_trend):
                db_size_k = point.db_size // 1000 if point.db_size >= 1000 else point.db_size
                size_unit = "K" if point.db_size >= 1000 else ""
                
                print(f"   {db_size_k:>4}{size_unit} docs → {point.indexing_time:>6.3f}s ({point.throughput_items_per_sec:>6.1f} docs/sec)")
            
            # Calculate scaling trend
            if len(report.indexing_trend) >= 3:
                first_point = report.indexing_trend[0]
                last_point = report.indexing_trend[-1]
                
                size_ratio = last_point.db_size / first_point.db_size if first_point.db_size > 0 else 0
                time_ratio = last_point.indexing_time / first_point.indexing_time if first_point.indexing_time > 0 else 0
                
                if size_ratio > 1:
                    scaling_factor = time_ratio / size_ratio
                    if scaling_factor > 1.5:
                        trend_desc = "📈 Super-linear scaling (indexing gets slower with size)"
                    elif scaling_factor > 0.8:
                        trend_desc = "📊 Linear scaling (time proportional to size)"
                    else:
                        trend_desc = "📉 Sub-linear scaling (indexing gets more efficient)"
                    
                    print(f"\n💡 Trend Analysis:")
                    print(f"   Size increased {size_ratio:.1f}x, time increased {time_ratio:.1f}x")
                    print(f"   {trend_desc}")
            print()

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