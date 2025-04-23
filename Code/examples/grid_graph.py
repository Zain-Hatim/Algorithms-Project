import networkx as nx
from src.karger_stein import KargerStein
import time

def main():
    # Create a 3x3 grid graph
    rows, cols = 3, 3
    graph = nx.grid_2d_graph(rows, cols)
    
    # Convert node labels from (x,y) to integers
    mapping = {(i,j): i*cols + j for i in range(rows) for j in range(cols)}
    graph = nx.relabel_nodes(graph, mapping)
    
    # Assign weights to edges
    # Horizontal edges have weight 1, vertical edges have weight 2
    for u, v in graph.edges():
        if abs(u - v) == 1:  # Horizontal edge
            graph[u][v]['weight'] = 1.0
        else:  # Vertical edge
            graph[u][v]['weight'] = 2.0
    
    # Number of desired partitions
    k = 2
    
    print(f"Running Karger-Stein on {rows}x{cols} grid graph with k={k} partitions...")
    
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