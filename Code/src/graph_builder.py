import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Union, Set, Optional
import random

class GraphBuilder:
    def __init__(self, n: int, p: float, seed: int = None):
        """Initialize with number of vertices and edge probability."""
        self.n = n
        self.p = p
        self.seed = seed
        self.graph = None
        self._rng = np.random.default_rng(seed)
    
    def build_graph(self) -> nx.Graph:
        """Build a random graph using the Erdős-Rényi model."""
        self.graph = nx.erdos_renyi_graph(self.n, self.p, seed=self.seed)
        # Assign random weights to edges
        for u, v in self.graph.edges():
            self.graph.edges[u, v]['weight'] = self._rng.random()
        return self.graph
    
    def get_graph(self) -> nx.Graph:
        """Return the current graph."""
        return self.graph
    
    def nagamochi_ibaraki_sparsification(self, k: int) -> nx.Graph:
        """
        Implement Nagamochi-Ibaraki sparsification as described in the 2020 paper.
        Returns a graph with at most k(n-1) edges that preserves all cuts of size ≤ k.
        
        The algorithm:
        1. Start with graph G = (V, E)
        2. Initialize empty subgraph H
        3. For i = 1 to k:
           - Extract a maximal spanning forest Fᵢ from remaining edges
           - Add Fᵢ to H
           - Remove Fᵢ from working graph
        """
        if self.graph is None:
            raise ValueError("Graph not built yet")
            
        # Create a copy of the graph to work with
        G = self.graph.copy()
        n = G.number_of_nodes()
        
        # Initialize empty subgraph H
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        
        # Working graph that we'll remove edges from
        working_graph = G.copy()
        
        for i in range(k):
            # Extract a maximal spanning forest
            F_i = nx.Graph()
            F_i.add_nodes_from(working_graph.nodes())
            
            # Sort edges by weight in non-increasing order
            edges = sorted(working_graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
            
            # Add edges to F_i until it's a maximal forest
            for u, v, data in edges:
                if not nx.has_path(F_i, u, v):
                    F_i.add_edge(u, v, weight=data['weight'])
                    # Remove the edge from working graph
                    working_graph.remove_edge(u, v)
            
            # Add F_i to H
            H.add_edges_from(F_i.edges(data=True))
        
        return H
    
    def get_edge_weights(self) -> Dict[Tuple[int, int], float]:
        """
        Get all edge weights as a dictionary.
        
        Returns:
            Dictionary mapping edges to their weights
        """
        return {(u, v): data['weight'] for u, v, data in self.graph.edges(data=True)}
    
    def get_vertex_count(self) -> int:
        """
        Get the number of vertices in the graph.
        
        Returns:
            Number of vertices
        """
        return self.graph.number_of_nodes()
    
    def get_edge_count(self) -> int:
        """
        Get the number of edges in the graph.
        
        Returns:
            Number of edges
        """
        return self.graph.number_of_edges()
    
    def _nagamochi_ibaraki_sparsify(self, graph: nx.Graph, k: int) -> nx.Graph:
        """
        Apply Nagamochi-Ibaraki sparsification to reduce graph size while preserving minimum k-cuts.
        
        Args:
            graph: Input graph
            k: Number of partitions
            
        Returns:
            Sparsified graph
        """
        # Compute maximum edge connectivity
        max_connectivity = self._compute_max_connectivity(graph)
        
        # If graph is already sparse enough, return it
        if graph.number_of_edges() <= max_connectivity * graph.number_of_nodes():
            return graph
            
        # Sort edges by weight in descending order
        edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        
        # Initialize sparsified graph
        sparsified = nx.Graph()
        sparsified.add_nodes_from(graph.nodes())
        
        # Add edges until we reach the target number
        target_edges = max_connectivity * graph.number_of_nodes()
        added_edges = 0
        
        for u, v, data in edges:
            if added_edges >= target_edges:
                break
                
            # Check if adding this edge would create a cycle
            if not nx.has_path(sparsified, u, v):
                sparsified.add_edge(u, v, weight=data['weight'])
                added_edges += 1
                
        return sparsified
        
    def _compute_max_connectivity(self, graph: nx.Graph) -> int:
        """
        Compute the maximum edge connectivity of the graph.
        
        Args:
            graph: Input graph
            
        Returns:
            Maximum edge connectivity
        """
        if not nx.is_connected(graph):
            return 0
            
        # For small graphs, compute exact connectivity
        if graph.number_of_nodes() <= 10:
            return nx.edge_connectivity(graph)
            
        # For larger graphs, use approximation
        # Start with degree-based lower bound
        min_degree = min(dict(graph.degree()).values())
        max_connectivity = min_degree
        
        # Try to find a better lower bound
        for node in graph.nodes():
            if graph.degree(node) == min_degree:
                # Compute local connectivity
                local_connectivity = nx.edge_connectivity(graph, node, list(graph.neighbors(node))[0])
                max_connectivity = max(max_connectivity, local_connectivity)
                
        return max_connectivity
        
    def _preserve_min_k_cut(self, original: nx.Graph, sparsified: nx.Graph, k: int) -> bool:
        """
        Verify that the minimum k-cut is preserved in the sparsified graph.
        
        Args:
            original: Original graph
            sparsified: Sparsified graph
            k: Number of partitions
            
        Returns:
            True if minimum k-cut is preserved
        """
        # For small graphs, compute exact minimum k-cut
        if original.number_of_nodes() <= 10:
            original_cut = self._compute_min_k_cut(original, k)
            sparsified_cut = self._compute_min_k_cut(sparsified, k)
            return abs(original_cut - sparsified_cut) < 1e-6
            
        # For larger graphs, use sampling
        num_samples = min(100, original.number_of_nodes())
        for _ in range(num_samples):
            # Generate random k-partition
            partition = self._generate_random_partition(original, k)
            
            # Compute cut weights
            original_weight = self._compute_cut_weight(original, partition)
            sparsified_weight = self._compute_cut_weight(sparsified, partition)
            
            if abs(original_weight - sparsified_weight) > 1e-6:
                return False
                
        return True
        
    def _compute_min_k_cut(self, graph: nx.Graph, k: int) -> float:
        """
        Compute the minimum k-cut weight for a small graph.
        
        Args:
            graph: Input graph
            k: Number of partitions
            
        Returns:
            Minimum k-cut weight
        """
        from itertools import combinations
        
        min_weight = float('inf')
        n = graph.number_of_nodes()
        
        # Try all possible k-partitions
        for partition in self._generate_k_partitions(n, k):
            cut_weight = self._compute_cut_weight(graph, partition)
            if cut_weight < min_weight:
                min_weight = cut_weight
                
        return min_weight
        
    def _generate_random_partition(self, graph: nx.Graph, k: int) -> List[Set[int]]:
        """
        Generate a random k-partition of the graph.
        
        Args:
            graph: Input graph
            k: Number of partitions
            
        Returns:
            Random k-partition
        """
        nodes = list(graph.nodes())
        random.shuffle(nodes)
        
        partition = [set() for _ in range(k)]
        for i, node in enumerate(nodes):
            partition[i % k].add(node)
            
        return partition
        
    def _compute_cut_weight(self, graph: nx.Graph, partition: List[Set[int]]) -> float:
        """
        Compute the weight of edges crossing the partition.
        
        Args:
            graph: Input graph
            partition: k-partition of nodes
            
        Returns:
            Total weight of cut edges
        """
        cut_weight = 0.0
        for i in range(len(partition)):
            for j in range(i + 1, len(partition)):
                for u in partition[i]:
                    for v in partition[j]:
                        if graph.has_edge(u, v):
                            cut_weight += graph[u][v]['weight']
        return cut_weight 