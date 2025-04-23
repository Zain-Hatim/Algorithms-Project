import networkx as nx
from src.karger_stein import KargerStein
import time
import random

def main():
    # Create a random graph with 8 nodes
    n = 8
    p = 0.5  # Probability of edge creation
    graph = nx.erdos_renyi_graph(n, p, seed=42)
    
    # Assign random weights to edges
    for u, v in graph.edges():
        # Random weight between 0.1 and 10.0
        graph[u][v]['weight'] = random.uniform(0.1, 10.0)
    
    # Number of desired partitions
    k = 3
    
    print(f"Running Karger-Stein on random graph with {n} nodes, p={p}, and k={k} partitions...")
    
    # Initialize KargerStein algorithm
    ks = KargerStein(graph, k, seed=42)
    
    # Run the algorithm and measure time
    start_time = time.time()
    result = ks.find_min_k_cut()
    end_time = time.time()
    
    # Print results
    print("\nResults:")
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"Minimum cut weight: {result['weight']}")
    print(f"Success rate: {result['success_rate']:.2%}")
    
    print("\nPartitions:")
    for i, partition in enumerate(result['partitions']):
        print(f"Partition {i+1}: {sorted(partition)}")
    
    # Print edge weights for reference
    print("\nEdge weights:")
    for u, v, data in graph.edges(data=True):
        print(f"Edge ({u}, {v}): weight = {data['weight']:.2f}")

if __name__ == "__main__":
    main() 