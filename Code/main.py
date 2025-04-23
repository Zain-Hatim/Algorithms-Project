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
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
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
            # Find which partition contains each node (if any)
            u_partition = -1
            v_partition = -1
            for i, part in enumerate(partitions):
                if u in part:
                    u_partition = i
                if v in part:
                    v_partition = i
                if u_partition != -1 and v_partition != -1:
                    break
            
            # If either node is not in any partition, consider it a regular edge
            if u_partition == -1 or v_partition == -1:
                regular_edges.append((u, v))
            # If nodes are in different partitions, it's a cut edge
            elif u_partition != v_partition:
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
        colors = plt.cm.rainbow(np.linspace(0, 1, len(partitions) + 1))  # +1 for unassigned nodes
        # Draw nodes in partitions
        for i, partition in enumerate(partitions):
            if partition:  # Only draw non-empty partitions
                nx.draw_networkx_nodes(graph, pos, nodelist=list(partition), 
                                     node_color=[colors[i]], node_size=500)
        
        # Draw unassigned nodes in gray
        assigned_nodes = set().union(*[p for p in partitions if p])
        unassigned_nodes = set(graph.nodes()) - assigned_nodes
        if unassigned_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=list(unassigned_nodes),
                                 node_color='gray', node_size=500)
    else:
        nx.draw_networkx_nodes(graph, pos, node_size=500)
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos)
    
    # Add edge weights
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    # Add legend for cut edges and unassigned nodes
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
    
    logging.info("=== Starting K-Cut Algorithm Execution ===")
    logging.info(f"Algorithm variant: {args.variant}")
    logging.info(f"Number of partitions (k): {args.k}")
    logging.info(f"Using sparsification: {args.sparsify}")
    
    # Create unique run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(f'results/run_{timestamp}')
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created run directory: {run_dir}")
    
    # Create images directory if visualization is requested
    if args.visualize:
        images_dir = run_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        logging.info(f"Created visualization directory: {images_dir}")
    
    # Set up log file in the run directory
    if not args.log_file:
        args.log_file = run_dir / 'runtime_logs.csv'
    else:
        args.log_file = run_dir / Path(args.log_file).name
    logging.info(f"Performance logs will be written to: {args.log_file}")
    
    # Load graph
    logging.info(f"Loading input graph from: {args.input}")
    graph = load_graph_from_file(args.input)
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    logging.info(f"Graph loaded successfully:")
    logging.info(f"  - Number of vertices: {n}")
    logging.info(f"  - Number of edges: {m}")
    logging.info(f"  - Average degree: {2*m/n:.2f}")
    
    # Initialize performance logger
    logger = PerformanceLogger(str(args.log_file))
    logging.info("Performance logger initialized")
    
    # Visualize original graph if requested
    if args.visualize:
        logging.info("Generating visualization of original graph...")
        visualize_graph(graph, output_path=str(images_dir / 'original_graph.png'))
        logging.info("Original graph visualization saved")
    
    # Run Karger-Stein algorithm
    logging.info("\n=== Starting Karger-Stein Algorithm ===")
    complexity = calculate_complexity(n, args.k, args.variant, args.sparsify)
    logging.info(f"Theoretical Runtime Complexity: {complexity}")
    
    num_trials = args.trials or int(n * n * math.log(n))
    logging.info(f"Number of trials to run: {num_trials}")
    
    start_time = time.time()
    karger = KargerStein(graph, args.k)
    logging.info("KargerStein instance initialized")
    
    if args.variant == 'basic':
        logging.info("Running basic Karger-Stein algorithm...")
        result = karger.find_min_k_cut(args.trials)
    else:  # recursive
        logging.info("Running recursive Karger-Stein algorithm...")
        result = karger.find_min_k_cut_recursive(args.k, args.trials)
    
    runtime = time.time() - start_time
    logging.info("\n=== Karger-Stein Algorithm Complete ===")
    logging.info(f"Total runtime: {runtime:.2f} seconds")
    
    # Log performance
    perf_data = {
        'graph_size': n,
        'edge_count': m,
        'k': args.k,
        'cut_weight': result['weight'],
        'trial_count': num_trials,
        'runtime_ms': runtime * 1000,
        'algorithm': args.variant,
        'sparsified': args.sparsify,
        'parallel': False,
        'adaptive': False,
        'complexity': complexity
    }
    logger.log_trial(perf_data)
    logging.info("Performance data logged")
    
    # Visualize result if requested
    if args.visualize:
        logging.info("\nGenerating visualization of Karger-Stein result...")
        visualize_graph(graph, partitions=result['partitions'], 
                      output_path=str(images_dir / 'karger_stein_result.png'),
                      complexity=complexity)
        logging.info("Karger-Stein result visualization saved")
    
    # Print summary
    logging.info("\n=== Results Summary ===")
    logging.info(f"Cut weight: {result['weight']:.6f}")
    logging.info(f"Runtime: {runtime:.3f} seconds")
    logging.info(f"Theoretical Complexity: {complexity}")
    logging.info(f"Number of unique minimum cuts found: {len(result['all_min_cuts'])}")
    
    # Save results
    if args.output:
        output_path = run_dir / Path(args.output).name
        logging.info(f"\nSaving detailed results to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump({
                'results': {
                    'weight': result['weight'],
                    'partitions': [list(part) for part in result['partitions']],
                    'all_min_cuts': [[list(part) for part in cut] for cut in result['all_min_cuts']],
                    'runtime': runtime,
                    'complexity': complexity
                }
            }, f, indent=2)
        logging.info("Results saved successfully")
    
    logging.info(f"\nAll run artifacts saved in: {run_dir}")
    logging.info("=== Execution Complete ===\n")

if __name__ == '__main__':
    main() 