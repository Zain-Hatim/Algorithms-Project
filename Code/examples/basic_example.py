import networkx as nx
from src.karger_stein import KargerStein
import time

def print_graph_info(graph: nx.Graph):
    """Print information about the graph."""
    print("\nGraph Information:")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print("Edge weights:")
    for u, v, data in graph.edges(data=True):
        print(f"  Edge ({u}, {v}): weight = {data['weight']}")

def print_partition_info(graph: nx.Graph, partitions: list, cut_weight: float):
    """Print detailed information about the partitions."""
    print("\nPartition Details:")
    print(f"Total cut weight: {cut_weight}")
    for i, partition in enumerate(partitions):
        print(f"\nPartition {i+1}:")
        print(f"  Size: {len(partition)} nodes")
        print(f"  Nodes: {sorted(partition)}")
        
    # Print edges crossing partitions
    print("\nEdges crossing partitions:")
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            for u in partitions[i]:
                for v in partitions[j]:
                    if graph.has_edge(u, v):
                        print(f"  Edge ({u}, {v}): weight = {graph[u][v]['weight']}")

def main():
    # Create a sample graph (you can replace this with your own graph)
    graph = nx.Graph()
    
    # Add some nodes and edges with weights
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=2.0)
    graph.add_edge(2, 3, weight=1.0)
    graph.add_edge(3, 0, weight=2.0)
    graph.add_edge(0, 2, weight=3.0)
    graph.add_edge(1, 3, weight=3.0)
    
    # Print initial graph information
    print_graph_info(graph)
    
    # Number of desired partitions
    k = 2
    
    print(f"\nRunning Karger-Stein algorithm with k={k} partitions...")
    
    # Initialize KargerStein algorithm
    ks = KargerStein(graph, k, seed=42)  # seed for reproducibility
    
    # Run the algorithm and measure time
    start_time = time.time()
    result = ks.find_min_k_cut()
    end_time = time.time()
    
    # Print results
    print("\nAlgorithm Results:")
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"Minimum cut weight: {result['weight']}")
    print(f"Success rate: {result['success_rate']:.2%} ({result['num_successes']}/{result['num_trials']} trials found minimum cut)")
    
    # Print detailed partition information
    print_partition_info(graph, result['partitions'], result['weight'])
    
    # Print all minimum cuts found
    if len(result['all_min_cuts']) > 1:
        print(f"\nFound {len(result['all_min_cuts'])} distinct minimum cuts")
        for i, cut in enumerate(result['all_min_cuts'][1:], 1):
            print(f"\nAlternative minimum cut {i+1}:")
            for j, partition in enumerate(cut):
                print(f"  Partition {j+1}: {sorted(partition)}")

if __name__ == "__main__":
    main() 