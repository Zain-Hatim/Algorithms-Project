import networkx as nx
from src.karger_stein import KargerStein
import time

def main():
    # Create a complete graph with 6 nodes
    n = 6
    graph = nx.complete_graph(n)
    
    # Assign random weights to edges
    for u, v in graph.edges():
        graph[u][v]['weight'] = 1.0  # Uniform weights for complete graph
    
    # Number of desired partitions
    k = 3
    
    print(f"Running Karger-Stein on complete graph with {n} nodes and k={k} partitions...")
    
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

if __name__ == "__main__":
    main() 