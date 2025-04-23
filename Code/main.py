#!/usr/bin/env python3
import argparse
import logging
import networkx as nx
import matplotlib.pyplot as plt
from src.utils import load_graph_from_file, save_graph_to_file
from src.graph_builder import GraphBuilder
from src.karger_stein import KargerStein
import time
import json
from pathlib import Path
from src.logger import PerformanceLogger
import math
import os
import numpy as np
from datetime import datetime

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def calculate_complexity(n, k, variant, sparsify):
    """Calculate theoretical runtime complexity of the algorithm."""
    if variant == 'basic':
        # Basic Karger-Stein: O(n² log n)
        complexity = f"O(n² log n) = O({n}² log {n})"
        if sparsify:
            complexity += " with sparsification"
    else:  # recursive
        # Recursive Karger-Stein: O(n² log² n)
        complexity = f"O(n² log² n) = O({n}² log² {n})"
        if sparsify:
            complexity += " with sparsification"
    return complexity

def visualize_graph(graph, partitions=None, output_path=None, complexity=None):
    """Visualize the graph with optional partition coloring and cut edge highlighting."""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    
    # Draw regular edges
    regular_edges = []
    cut_edges = []
    
    if partitions:
        # Find edges that cross partitions
        for u, v in graph.edges():
            u_partition = next(i for i, part in enumerate(partitions) if u in part)
            v_partition = next(i for i, part in enumerate(partitions) if v in part)
            if u_partition != v_partition:
                cut_edges.append((u, v))
            else:
                regular_edges.append((u, v))
    else:
        regular_edges = list(graph.edges())
    
    # Draw regular edges
    nx.draw_networkx_edges(graph, pos, edgelist=regular_edges, 
                          width=1.0, alpha=0.5, edge_color='gray')
    
    # Draw cut edges with different style
    if cut_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=cut_edges,
                              width=2.0, alpha=0.8, edge_color='red',
                              style='dashed')
    
    # Draw nodes with partition colors if provided
    if partitions:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(partitions)))
        for i, partition in enumerate(partitions):
            nx.draw_networkx_nodes(graph, pos, nodelist=partition, 
                                 node_color=[colors[i]], node_size=500)
    else:
        nx.draw_networkx_nodes(graph, pos, node_size=500)
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos)
    
    # Add edge weights
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    # Add legend for cut edges
    if cut_edges:
        plt.plot([], [], 'r--', label='Cut Edges', linewidth=2)
        plt.legend()
    
    # Add complexity information to title
    title = "Graph Visualization with K-Cut Partitions" if partitions else "Graph Visualization"
    if complexity:
        title += f"\nRuntime Complexity: {complexity}"
    plt.title(title)
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

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
    parser.add_argument('--log-file', type=str, help='Path to performance log file')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    setup_logging()
    
    # Create unique run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(f'results/run_{timestamp}')
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images directory if visualization is requested
    if args.visualize:
        images_dir = run_dir / 'images'
        images_dir.mkdir(exist_ok=True)
    
    # Set up log file in the run directory
    if not args.log_file:
        args.log_file = run_dir / 'runtime_logs.csv'
    else:
        args.log_file = run_dir / Path(args.log_file).name
    
    # Load graph
    graph = load_graph_from_file(args.input)
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    
    # Calculate theoretical complexity
    complexity = calculate_complexity(n, args.k, args.variant, args.sparsify)
    logging.info(f"Theoretical Runtime Complexity: {complexity}")
    
    # Initialize performance logger
    logger = PerformanceLogger(str(args.log_file))
    
    # Visualize original graph if requested
    if args.visualize:
        visualize_graph(graph, output_path=str(images_dir / 'original_graph.png'))
    
    # Run algorithm
    start_time = time.time()
    karger = KargerStein(graph, args.k)
    
    if args.variant == 'basic':
        result = karger.find_min_k_cut(args.trials)
    else:  # recursive
        result = karger.find_min_k_cut_recursive(args.k, args.trials)
    
    runtime = time.time() - start_time
    
    # Visualize result if requested
    if args.visualize:
        visualize_graph(graph, partitions=result['partitions'], 
                       output_path=str(images_dir / 'k_cut_result.png'),
                       complexity=complexity)
    
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
        'adaptive': False,
        'complexity': complexity
    })
    
    # Save results
    if args.output:
        output_path = run_dir / Path(args.output).name
        with open(output_path, 'w') as f:
            json.dump({
                'weight': result['weight'],
                'partitions': [list(part) for part in result['partitions']],
                'all_min_cuts': [[list(part) for part in cut] for cut in result['all_min_cuts']],
                'runtime': runtime,
                'complexity': complexity
            }, f, indent=2)
    
    # Print summary
    logging.info(f"Found minimum {args.k}-cut with weight {result['weight']}")
    logging.info(f"Runtime: {runtime:.2f} seconds")
    logging.info(f"Number of minimum cuts found: {len(result['all_min_cuts'])}")
    if args.visualize:
        logging.info(f"Visualizations saved in {images_dir}")
    logging.info(f"Results saved in {run_dir}")

if __name__ == '__main__':
    main() 