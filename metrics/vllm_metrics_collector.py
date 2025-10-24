#!/usr/bin/env python3
"""
vLLM Metrics Collector

A Python program that continuously queries vLLM metrics endpoint and stores
specified metrics using threading for non-blocking operation.
"""

import requests
import threading
import time
import json
import csv
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import argparse
import signal
import sys
import os


@dataclass
class MetricData:
    """Data class to represent a metric entry."""
    timestamp: str
    metric_name: str
    value: float
    labels: Dict[str, str]


class MetricsStorage:
    """Base class for metrics storage backends."""
    
    def store_metric(self, metric: MetricData) -> None:
        """Store a metric entry."""
        raise NotImplementedError
    
    def close(self) -> None:
        """Close the storage backend."""
        pass


class JSONStorage(MetricsStorage):
    """JSON file storage for metrics."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.metrics = []
    
    def store_metric(self, metric: MetricData) -> None:
        """Store metric in JSON format."""
        self.metrics.append({
            'timestamp': metric.timestamp,
            'metric_name': metric.metric_name,
            'value': metric.value,
            'labels': metric.labels
        })
    
    def close(self) -> None:
        """Write metrics to JSON file."""
        with open(self.filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)


class CSVStorage(MetricsStorage):
    """CSV file storage for metrics."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['timestamp', 'metric_name', 'value', 'labels'])
    
    def store_metric(self, metric: MetricData) -> None:
        """Store metric in CSV format."""
        self.writer.writerow([
            metric.timestamp,
            metric.metric_name,
            metric.value,
            json.dumps(metric.labels)
        ])
        self.file.flush()
    
    def close(self) -> None:
        """Close CSV file."""
        self.file.close()


class SQLiteStorage(MetricsStorage):
    """SQLite database storage for metrics."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metric_name TEXT,
                value REAL,
                labels TEXT
            )
        ''')
        self.conn.commit()
    
    def store_metric(self, metric: MetricData) -> None:
        """Store metric in SQLite database."""
        self.cursor.execute('''
            INSERT INTO metrics (timestamp, metric_name, value, labels)
            VALUES (?, ?, ?, ?)
        ''', (
            metric.timestamp,
            metric.metric_name,
            metric.value,
            json.dumps(metric.labels)
        ))
        self.conn.commit()
    
    def close(self) -> None:
        """Close SQLite connection."""
        self.conn.close()


class PrometheusStorage(MetricsStorage):
    """Prometheus storage for metrics - supports both file output and Pushgateway."""
    
    def __init__(self, 
                 output_path: Optional[str] = None,
                 pushgateway_url: Optional[str] = None,
                 job_name: str = "vllm_metrics",
                 instance: Optional[str] = None):
        """
        Initialize Prometheus storage.
        
        Args:
            output_path: Path to write Prometheus format metrics file (optional)
            pushgateway_url: URL of Prometheus Pushgateway (optional)
            job_name: Job name for Pushgateway
            instance: Instance name for Pushgateway
        """
        self.output_path = output_path
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.instance = instance or f"instance_{int(time.time())}"
        self.metrics_buffer = []
        
        # Create output directory if needed
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    
    def store_metric(self, metric: MetricData) -> None:
        """Store metric in Prometheus format."""
        # Format metric in Prometheus format
        prometheus_line = self._format_prometheus_metric(metric)
        self.metrics_buffer.append(prometheus_line)
        
        # If we have a file output, write immediately
        if self.output_path:
            self._write_to_file(prometheus_line)
    
    def _format_prometheus_metric(self, metric: MetricData) -> str:
        """Format a metric in Prometheus format with proper histogram handling."""
        # Clean metric name (replace colons with underscores, remove special chars)
        clean_name = metric.metric_name.replace(':', '_').replace('-', '_')
        
        # Format labels
        if metric.labels:
            labels_str = ','.join([f'{k}="{v}"' for k, v in metric.labels.items()])
            metric_line = f'{clean_name}{{{labels_str}}} {metric.value}'
        else:
            metric_line = f'{clean_name} {metric.value}'
        
        # Add timestamp for histogram metrics (required for proper Prometheus format)
        if metric.metric_name.endswith('_bucket') or metric.metric_name.endswith('_count') or metric.metric_name.endswith('_sum'):
            timestamp_ms = int(datetime.fromisoformat(metric.timestamp).timestamp() * 1000)
            return f'{metric_line} {timestamp_ms}'
        else:
            return metric_line
    
    def _write_to_file(self, line: str) -> None:
        """Write a single metric line to file."""
        try:
            with open(self.output_path, 'a') as f:
                f.write(line + '\n')
        except Exception as e:
            logging.error(f"Error writing to Prometheus file: {e}")
    
    def _push_to_gateway(self) -> None:
        """Push metrics to Prometheus Pushgateway."""
        if not self.pushgateway_url:
            return
        
        try:
            # Format all buffered metrics
            metrics_text = '\n'.join(self.metrics_buffer)
            
            # Push to gateway
            push_url = f"{self.pushgateway_url.rstrip('/')}/metrics/job/{self.job_name}/instance/{self.instance}"
            response = requests.post(push_url, data=metrics_text, timeout=30)
            response.raise_for_status()
            
            logging.info(f"Successfully pushed {len(self.metrics_buffer)} metrics to Pushgateway")
            
        except Exception as e:
            logging.error(f"Error pushing to Pushgateway: {e}")
    
    def close(self) -> None:
        """Close Prometheus storage and push final metrics."""
        if self.pushgateway_url and self.metrics_buffer:
            self._push_to_gateway()
        
        # Clear buffer
        self.metrics_buffer.clear()


class PrometheusFileStorage(MetricsStorage):
    """Simplified Prometheus file storage that writes metrics in Prometheus format."""
    
    def __init__(self, filename: str):
        """
        Initialize Prometheus file storage.
        
        Args:
            filename: Path to the output file
        """
        self.filename = filename
        self.file = open(filename, 'w')
        self.file.write('# Prometheus metrics from vLLM\n')
        self.file.write('# Generated by vLLM Metrics Collector\n\n')
    
    def store_metric(self, metric: MetricData) -> None:
        """Store metric in Prometheus format with proper histogram handling."""
        # Clean metric name
        clean_name = metric.metric_name.replace(':', '_').replace('-', '_')
        
        # Format labels
        if metric.labels:
            labels_str = ','.join([f'{k}="{v}"' for k, v in metric.labels.items()])
            metric_line = f'{clean_name}{{{labels_str}}} {metric.value}'
        else:
            metric_line = f'{clean_name} {metric.value}'
        
        # Add timestamp for histogram metrics (required for proper Prometheus format)
        if metric.metric_name.endswith('_bucket') or metric.metric_name.endswith('_count') or metric.metric_name.endswith('_sum'):
            timestamp_ms = int(datetime.fromisoformat(metric.timestamp).timestamp() * 1000)
            line = f'{metric_line} {timestamp_ms}\n'
        else:
            line = f'{metric_line}\n'
        
        self.file.write(line)
        self.file.flush()
    
    def close(self) -> None:
        """Close the Prometheus file."""
        self.file.close()


class VLLMMetricsCollector:
    """Main class for collecting vLLM metrics."""
    
    def __init__(self, 
                 metrics_endpoint: str,
                 storage: MetricsStorage,
                 metrics_to_collect: List[str],
                 collection_interval: int = 10,
                 timeout: int = 30):
        """
        Initialize the metrics collector.
        
        Args:
            metrics_endpoint: URL of the vLLM metrics endpoint
            storage: Storage backend for metrics
            metrics_to_collect: List of metric names to collect
            collection_interval: Time interval between collections (seconds)
            timeout: Request timeout (seconds)
        """
        self.metrics_endpoint = metrics_endpoint
        self.storage = storage
        self.metrics_to_collect = metrics_to_collect
        self.collection_interval = collection_interval
        self.timeout = timeout
        self.running = False
        self.thread = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def parse_metrics(self, metrics_text: str) -> List[MetricData]:
        """
        Parse Prometheus format metrics text into MetricData objects.
        Handles both regular metrics and histogram metrics properly.
        
        Args:
            metrics_text: Raw metrics text from vLLM endpoint
            
        Returns:
            List of MetricData objects
        """
        metrics = []
        current_timestamp = datetime.now().isoformat()
        
        # Track histogram metrics to group them properly
        histogram_metrics = {}
        
        for line in metrics_text.splitlines():
            line = line.strip()
            
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            
            # Parse metric line (format: metric_name{labels} value)
            if ' ' in line:
                metric_part, value_part = line.rsplit(' ', 1)
                try:
                    value = float(value_part)
                except ValueError:
                    continue
                
                # Extract metric name and labels
                if '{' in metric_part and '}' in metric_part:
                    metric_name = metric_part.split('{')[0]
                    labels_str = metric_part.split('{')[1].rstrip('}')
                    labels = {}
                    
                    # Parse labels (simple key=value parsing)
                    for label_pair in labels_str.split(','):
                        if '=' in label_pair:
                            key, val = label_pair.split('=', 1)
                            labels[key.strip()] = val.strip().strip('"')
                else:
                    metric_name = metric_part
                    labels = {}
                
                # Check if this metric is in our collection list
                if any(target_metric in metric_name for target_metric in self.metrics_to_collect):
                    # Handle histogram metrics specially
                    if self._is_histogram_metric(metric_name, labels):
                        self._process_histogram_metric(metric_name, labels, value, current_timestamp, histogram_metrics)
                    else:
                        # Regular metric
                        metrics.append(MetricData(
                            timestamp=current_timestamp,
                            metric_name=metric_name,
                            value=value,
                            labels=labels
                        ))
        
        # Add processed histogram metrics
        metrics.extend(self._finalize_histogram_metrics(histogram_metrics))
        
        return metrics
    
    def _is_histogram_metric(self, metric_name: str, labels: Dict[str, str]) -> bool:
        """Check if a metric is part of a histogram."""
        # Histogram metrics typically have _bucket, _count, or _sum suffixes
        return (metric_name.endswith('_bucket') or 
                metric_name.endswith('_count') or 
                metric_name.endswith('_sum') or
                'le=' in str(labels))  # le= label indicates histogram bucket
    
    def _process_histogram_metric(self, metric_name: str, labels: Dict[str, str], 
                                 value: float, timestamp: str, histogram_metrics: Dict) -> None:
        """Process histogram metric components."""
        # Extract base metric name (remove _bucket, _count, _sum suffixes)
        base_name = metric_name
        if metric_name.endswith('_bucket'):
            base_name = metric_name[:-7]  # Remove '_bucket'
        elif metric_name.endswith('_count'):
            base_name = metric_name[:-6]  # Remove '_count'
        elif metric_name.endswith('_sum'):
            base_name = metric_name[:-4]  # Remove '_sum'
        
        # Create histogram key (base name + non-le labels)
        histogram_key = base_name
        non_le_labels = {k: v for k, v in labels.items() if k != 'le'}
        if non_le_labels:
            label_str = ','.join([f'{k}="{v}"' for k, v in non_le_labels.items()])
            histogram_key = f"{base_name}{{{label_str}}}"
        
        if histogram_key not in histogram_metrics:
            histogram_metrics[histogram_key] = {
                'base_name': base_name,
                'labels': non_le_labels,
                'buckets': [],
                'count': None,
                'sum': None,
                'timestamp': timestamp
            }
        
        # Store the metric component
        if metric_name.endswith('_bucket'):
            histogram_metrics[histogram_key]['buckets'].append({
                'le': labels.get('le', '+Inf'),
                'value': value
            })
        elif metric_name.endswith('_count'):
            histogram_metrics[histogram_key]['count'] = value
        elif metric_name.endswith('_sum'):
            histogram_metrics[histogram_key]['sum'] = value
    
    def _finalize_histogram_metrics(self, histogram_metrics: Dict) -> List[MetricData]:
        """Convert processed histogram metrics to MetricData objects."""
        metrics = []
        
        for histogram_key, hist_data in histogram_metrics.items():
            # Add bucket metrics
            for bucket in hist_data['buckets']:
                bucket_labels = hist_data['labels'].copy()
                bucket_labels['le'] = bucket['le']
                
                metrics.append(MetricData(
                    timestamp=hist_data['timestamp'],
                    metric_name=f"{hist_data['base_name']}_bucket",
                    value=bucket['value'],
                    labels=bucket_labels
                ))
            
            # Add count metric
            if hist_data['count'] is not None:
                metrics.append(MetricData(
                    timestamp=hist_data['timestamp'],
                    metric_name=f"{hist_data['base_name']}_count",
                    value=hist_data['count'],
                    labels=hist_data['labels']
                ))
            
            # Add sum metric
            if hist_data['sum'] is not None:
                metrics.append(MetricData(
                    timestamp=hist_data['timestamp'],
                    metric_name=f"{hist_data['base_name']}_sum",
                    value=hist_data['sum'],
                    labels=hist_data['labels']
                ))
        
        return metrics
    
    def collect_metrics(self) -> None:
        """Main metrics collection loop."""
        self.logger.info(f"Starting metrics collection from {self.metrics_endpoint}")
        
        while self.running:
            try:
                # Fetch metrics from endpoint
                response = requests.get(
                    self.metrics_endpoint,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Parse and store metrics
                metrics = self.parse_metrics(response.text)
                
                for metric in metrics:
                    self.storage.store_metric(metric)
                    self.logger.debug(f"Stored metric: {metric.metric_name} = {metric.value}")
                
                self.logger.info(f"Collected {len(metrics)} metrics")
                
            except requests.RequestException as e:
                self.logger.error(f"Error fetching metrics: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
            
            # Wait for next collection
            time.sleep(self.collection_interval)
    
    def start(self) -> None:
        """Start the metrics collection thread."""
        if self.running:
            self.logger.warning("Metrics collection already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.collect_metrics, daemon=True)
        self.thread.start()
        self.logger.info("Metrics collection started")
    
    def stop(self) -> None:
        """Stop the metrics collection thread."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.storage.close()
        self.logger.info("Metrics collection stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\nReceived shutdown signal. Stopping metrics collection...")
    sys.exit(0)


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='vLLM Metrics Collector')
    parser.add_argument('--endpoint', 
                       default='http://localhost:8000/metrics',
                       help='vLLM metrics endpoint URL')
    parser.add_argument('--interval', 
                       type=int, 
                       default=10,
                       help='Collection interval in seconds')
    parser.add_argument('--timeout', 
                       type=int, 
                       default=30,
                       help='Request timeout in seconds')
    parser.add_argument('--storage-type', 
                       choices=['json', 'csv', 'sqlite', 'prometheus', 'prometheus-file'],
                       default='csv',
                       help='Storage backend type')
    parser.add_argument('--output', 
                       default='vllm_metrics',
                       help='Output file/database name (without extension)')
    parser.add_argument('--metrics', 
                       nargs='+',
                       default=[
                           'vllm:num_requests_running',
                           'vllm:generation_tokens_total',
                           'vllm:request_success_total',
                           'vllm:request_failure_total',
                           'vllm:request_latency',
                           'vllm:request_input_tokens',
                           'vllm:request_output_tokens',
                           'vllm:request_duration',
                           'vllm:request_ttft',
                           'vllm:request_itl'
                       ],
                       help='Metrics to collect')
    parser.add_argument('--verbose', 
                       action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--pushgateway-url',
                       help='Prometheus Pushgateway URL (for prometheus storage type)')
    parser.add_argument('--job-name',
                       default='vllm_metrics',
                       help='Job name for Prometheus Pushgateway')
    parser.add_argument('--instance',
                       help='Instance name for Prometheus Pushgateway')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create storage backend
    if args.storage_type == 'json':
        storage = JSONStorage(f"{args.output}.json")
    elif args.storage_type == 'csv':
        storage = CSVStorage(f"{args.output}.csv")
    elif args.storage_type == 'sqlite':
        storage = SQLiteStorage(f"{args.output}.db")
    elif args.storage_type == 'prometheus':
        storage = PrometheusStorage(
            output_path=f"{args.output}.prom",
            pushgateway_url=args.pushgateway_url,
            job_name=args.job_name,
            instance=args.instance
        )
    elif args.storage_type == 'prometheus-file':
        storage = PrometheusFileStorage(f"{args.output}.prom")
    
    # Create metrics collector
    collector = VLLMMetricsCollector(
        metrics_endpoint=args.endpoint,
        storage=storage,
        metrics_to_collect=args.metrics,
        collection_interval=args.interval,
        timeout=args.timeout
    )
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start metrics collection
        collector.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        collector.stop()


if __name__ == "__main__":
    main()
