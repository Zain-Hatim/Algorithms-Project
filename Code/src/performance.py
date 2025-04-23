import csv
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import networkx as nx
from .karger_stein import KargerStein
from .graph_builder import GraphBuilder

class PerformanceLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.fieldnames = [
            'timestamp', 'n', 'm', 'k', 'cut_weight', 'runtime',
            'algorithm', 'sparsified', 'm_sparse', 'sparsification_ratio'
        ]
        self._ensure_log_file()
        
    def _ensure_log_file(self):
        """Ensure the log file exists with proper headers."""
        try:
            with open(self.log_file, 'r') as f:
                # File exists, check if it has content
                if not f.read().strip():
                    self._write_headers()
        except FileNotFoundError:
            self._write_headers()
            
    def _write_headers(self):
        """Write headers to the log file."""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.fieldnames)
            
    def log_run(self, **metrics):
        """Log a single run with metrics."""
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Handle additional metrics
        additional_metrics = metrics.pop('additional_metrics', {})
        
        # Add any new fields from additional metrics to fieldnames
        new_fields = list(additional_metrics.keys())
        for field in new_fields:
            if field not in self.fieldnames:
                self.fieldnames.append(field)
        
        # Update metrics with additional metrics
        metrics.update(additional_metrics)
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics)

def benchmark_algorithm(
    graph: nx.Graph,
    k: int,
    recursive: bool = True,
    sparsify: bool = False,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Run a single benchmark of the algorithm."""
    start_time = time.time()
    
    # Create algorithm instance
    ks = KargerStein(graph, k, seed=seed)
    
    # Run the algorithm
    result = ks.find_min_k_cut()
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Prepare metrics
    metrics = {
        'n': graph.number_of_nodes(),
        'm': graph.number_of_edges(),
        'k': k,
        'cut_weight': result['weight'],
        'runtime': runtime,
        'algorithm': 'recursive' if recursive else 'basic',
        'sparsified': sparsify,
        'm_sparse': None,
        'sparsification_ratio': None
    }
    
    if sparsify:
        # Calculate sparsification metrics
        # Use the graph's properties for initialization
        builder = GraphBuilder(n=graph.number_of_nodes(), p=0.5)  # p=0.5 is a reasonable default
        builder.graph = graph  # Set the graph directly since we're not building a new one
        sparse_graph = builder.nagamochi_ibaraki_sparsification(k)
        metrics['m_sparse'] = sparse_graph.number_of_edges()
        metrics['sparsification_ratio'] = metrics['m_sparse'] / metrics['m']
    
    return {
        'result': result,
        'metrics': metrics
    }

def run_benchmark_suite(
    graph: nx.Graph,
    k_values: List[int],
    recursive: bool = True,
    sparsify: bool = False,
    seed: Optional[int] = None
) -> Dict[int, Dict[str, Any]]:
    """Run a suite of benchmarks for different k values."""
    results = {}
    
    for k in k_values:
        results[k] = benchmark_algorithm(
            graph, k, recursive, sparsify, seed
        )
    
    return results 