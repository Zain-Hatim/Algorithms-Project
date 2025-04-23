#!/usr/bin/env python3
import argparse
import logging
import networkx as nx
from src.utils import load_graph_from_file, save_graph_to_file
from src.graph_builder import GraphBuilder
from src.karger_stein import KargerStein
import time
import json
from pathlib import Path
from src.logger import PerformanceLogger
import math

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """Main entry point for the Karger-Stein algorithm."""
    parser = argparse.ArgumentParser(description='Find minimum k-cut using Karger-Stein algorithm')
    parser.add_argument('--input', type=str, required=True, help='Input graph file')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--k', type=int, required=True, help='Number of partitions')
    parser.add_argument('--variant', type=str, choices=['basic', 'recursive'],
                      default='recursive', help='Algorithm variant to use')
    parser.add_argument('--trials', type=int, help='Number of trials to run (default: n^2 log n)')
    parser.add_argument('--sparsify', action='store_true', help='Use Nagamochi-Ibaraki sparsification')
    parser.add_argument('--log-file', type=str, default='results/runtime_logs.csv',
                      help='Path to performance log file')
    
    args = parser.parse_args()
    setup_logging()
    
    # Load graph
    graph = load_graph_from_file(args.input)
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    
    # Initialize performance logger
    logger = PerformanceLogger(args.log_file)
    
    # Run algorithm
    start_time = time.time()
    karger = KargerStein(graph, args.k)
    
    if args.variant == 'basic':
        result = karger.find_min_k_cut(args.trials)
    else:  # recursive
        result = karger.find_min_k_cut_recursive(args.k, args.trials)
    
    runtime = time.time() - start_time
    
    # Log performance
    logger.log_trial({
        'graph_size': n,
        'edge_count': m,
        'k': args.k,
        'cut_weight': result['weight'],
        'trial_count': args.trials or int(n * n * math.log(n)),
        'runtime_ms': runtime * 1000,
        'algorithm': args.variant,
        'sparsified': args.sparsify,
        'parallel': False,
        'adaptive': False
    })
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'weight': result['weight'],
                'partitions': [list(part) for part in result['partitions']],
                'all_min_cuts': [[list(part) for part in cut] for cut in result['all_min_cuts']],
                'runtime': runtime
            }, f, indent=2)
    
    # Print summary
    logging.info(f"Found minimum {args.k}-cut with weight {result['weight']}")
    logging.info(f"Runtime: {runtime:.2f} seconds")
    logging.info(f"Number of minimum cuts found: {len(result['all_min_cuts'])}")

if __name__ == '__main__':
    main() 