import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import random
from .utils import contract_edge
import logging
import math
from .logger import PerformanceLogger

class KargerStein:
    def __init__(self, graph: nx.Graph, k: int, seed: Optional[int] = None):
        """
        Initialize the Karger-Stein algorithm.
        
        Args:
            graph: Input graph (must be weighted)
            k: Number of desired components
            seed: Random seed for reproducibility
        """
        self.original_graph = graph
        self.k = k
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._lambda_k = None  # Cache for λₖ estimation
        self.logger = PerformanceLogger()
        
        # Initialize supernodes tracking
        self.supernodes = {node: {node} for node in graph.nodes()}
        
        # Validate input
        if not isinstance(graph, nx.Graph):
            raise ValueError("Input must be a NetworkX graph")
        if not all('weight' in data for _, _, data in graph.edges(data=True)):
            raise ValueError("All edges must have weights")
        if k < 2:
            raise ValueError("k must be at least 2")
            
    def _select_random_edge(self, graph: nx.Graph) -> Tuple[int, int]:
        """
        Select an edge with probability proportional to its weight and the sum of its endpoint degrees.
        This favors edges connected to high-degree nodes to speed up contraction.
        
        Args:
            graph: Current graph state
            
        Returns:
            Tuple of (u, v) representing the selected edge
        """
        edges = list(graph.edges(data=True))
        
        # Compute degrees for all nodes
        degrees = dict(graph.degree())
        
        # Calculate weights incorporating degrees: weight * (deg[u] + deg[v])
        weights = []
        for u, v, data in edges:
            edge_weight = data['weight']
            degree_sum = degrees[u] + degrees[v]
            weights.append(edge_weight * degree_sum)
            
        # Normalize weights to get probabilities
        total_weight = sum(weights)
        if total_weight == 0:
            # If all weights are zero, fall back to uniform selection
            probabilities = [1/len(edges)] * len(edges)
        else:
            probabilities = [w/total_weight for w in weights]
        
        # Sample edge based on probabilities
        selected_idx = self._rng.choice(len(edges), p=probabilities)
        return edges[selected_idx][0], edges[selected_idx][1]
        
    def _contract_edge(self, graph: nx.Graph, u: int, v: int) -> nx.Graph:
        """
        Contract edge (u,v) by merging the two vertices.
        
        Args:
            graph: Current graph state
            u, v: Vertices to contract
            
        Returns:
            New graph with contracted edge
        """
        # Create a copy of the graph
        new_graph = graph.copy()
        
        # Use the utility function to contract the edge
        contract_edge(new_graph, (u, v))
        
        # Update supernodes tracking
        self.supernodes[u] = self.supernodes[u].union(self.supernodes[v])
        del self.supernodes[v]
        
        return new_graph
        
    def _calculate_cut_weight(self, graph: nx.Graph, partition: List[Set[int]]) -> float:
        """
        Calculate the total weight of edges crossing the partition.
        
        Args:
            graph: Original graph
            partition: List of k sets representing the partition
            
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
        
    def find_min_k_cut(self, num_trials: int = None) -> Dict:
        """
        Find minimum k-cut using repeated random contractions.
        
        Args:
            num_trials: Number of trials to run (default: n^2 log n)
            
        Returns:
            Dictionary containing the cut weight, best partition, and all minimum cuts
        """
        if num_trials is None:
            n = self.original_graph.number_of_nodes()
            num_trials = int(n * n * np.log(n))
            
        min_cut_weight = float('inf')
        best_partition = None
        all_min_cuts = []
        num_successes = 0  # Counter for successful trials
        
        for trial in range(num_trials):
            # Create a copy of the graph for this trial
            graph = self.original_graph.copy()
            
            # Reset supernodes tracking for this trial
            self.supernodes = {node: {node} for node in graph.nodes()}
            
            # Use recursive contraction strategy
            result = self._recursive_contraction(graph)
            
            # Build partition from supernodes
            # Each supernode is a set of original nodes that were contracted together
            partition = []
            for supernode in self.supernodes.values():
                if supernode:  # Only add non-empty supernodes
                    partition.append(supernode)
            
            # Ensure we have exactly k partitions
            # The recursive contraction might not always result in exactly k partitions
            # so we need to adjust the partition count
            
            if len(partition) > self.k:
                # If we have more than k partitions, merge the smallest ones
                # This helps maintain balanced partition sizes
                while len(partition) > self.k:
                    # Find two smallest partitions to minimize the impact on cut weight
                    partition.sort(key=len)
                    # Merge the two smallest partitions
                    merged = partition[0].union(partition[1])
                    partition = [merged] + partition[2:]
                    
            elif len(partition) < self.k:
                # If we have fewer than k partitions, split the largest one
                # This ensures we meet the k-partition requirement
                while len(partition) < self.k:
                    # Sort partitions by size (largest first) to find the best candidate for splitting
                    partition.sort(key=len, reverse=True)
                    # Split the largest partition into two roughly equal parts
                    largest = partition[0]
                    split_point = len(largest) // 2
                    # Create two new partitions from the split
                    partition = [set(list(largest)[:split_point]), 
                               set(list(largest)[split_point:])] + partition[1:]
            
            # Calculate cut weight for the current partition
            # This is the sum of weights of edges crossing between partitions
            cut_weight = self._calculate_cut_weight(self.original_graph, partition)
            
            # Update best results if we found a better cut
            if cut_weight < min_cut_weight:
                min_cut_weight = cut_weight
                best_partition = partition
                all_min_cuts = [partition]
                num_successes = 1  # Reset counter when new minimum found
            elif cut_weight == min_cut_weight:
                # Check if this is a new distinct cut
                if not self._is_duplicate_cut(partition, all_min_cuts):
                    all_min_cuts.append(partition)
                num_successes += 1  # Increment counter for successful trial
                
        # Calculate and log success rate
        success_rate = num_successes / num_trials
        self.logger.log(f"Success rate: {success_rate:.2%} ({num_successes}/{num_trials} trials found minimum cut)")
                
        return {
            'weight': min_cut_weight,
            'partitions': best_partition,
            'all_min_cuts': all_min_cuts,
            'success_rate': success_rate,
            'num_successes': num_successes,
            'num_trials': num_trials
        }
        
    def _recursive_contraction(self, graph: nx.Graph, threshold: int = 6) -> nx.Graph:
        """
        Recursive contraction strategy for Karger-Stein algorithm.
        
        Args:
            graph: Current graph state
            threshold: Number of nodes below which to compute cut directly
            
        Returns:
            Contracted graph
        """
        n = graph.number_of_nodes()
        
        # Base case: if graph has ≤ threshold nodes, compute cut directly
        if n <= threshold:
            return graph
            
        # Recursive case: contract down to t = ceil(n / √2) nodes
        t = math.ceil(n / math.sqrt(2))
        
        # Create two copies of the graph for independent contraction
        graph1 = graph.copy()
        graph2 = graph.copy()
        
        # Save current supernodes state
        original_supernodes = self.supernodes.copy()
        
        # Contract first copy
        while graph1.number_of_nodes() > t:
            u, v = self._select_random_edge(graph1)
            graph1 = self._contract_edge(graph1, u, v)
            
        # Save supernodes state after first contraction
        supernodes1 = self.supernodes.copy()
        
        # Restore original supernodes for second contraction
        self.supernodes = original_supernodes.copy()
        
        # Contract second copy
        while graph2.number_of_nodes() > t:
            u, v = self._select_random_edge(graph2)
            graph2 = self._contract_edge(graph2, u, v)
            
        # Save supernodes state after second contraction
        supernodes2 = self.supernodes.copy()
        
        # Recursively find cuts for both contracted graphs
        # Use first supernodes state
        self.supernodes = supernodes1
        result1 = self._recursive_contraction(graph1, threshold)
        cut1 = self._calculate_cut_weight(self.original_graph, list(self.supernodes.values()))
        
        # Use second supernodes state
        self.supernodes = supernodes2
        result2 = self._recursive_contraction(graph2, threshold)
        cut2 = self._calculate_cut_weight(self.original_graph, list(self.supernodes.values()))
        
        # Return the graph with the smaller cut and its corresponding supernodes state
        if cut1 <= cut2:
            self.supernodes = supernodes1
            return result1
        else:
            self.supernodes = supernodes2
            return result2
        
    def _is_duplicate_cut(self, partition: List[Set[int]], existing_cuts: List[List[Set[int]]]) -> bool:
        """
        Check if a partition is equivalent to any existing cut.
        
        Args:
            partition: Partition to check
            existing_cuts: List of existing cuts
            
        Returns:
            True if partition is equivalent to any existing cut
        """
        # Convert partitions to canonical form (sorted tuples of sorted sets)
        canonical_new = tuple(tuple(sorted(part)) for part in sorted(partition, key=lambda x: min(x)))
        
        for existing in existing_cuts:
            canonical_existing = tuple(tuple(sorted(part)) for part in sorted(existing, key=lambda x: min(x)))
            if canonical_new == canonical_existing:
                return True
                
        return False
        
    def _get_contraction_probability(self, t: float) -> float:
        """
        Compute the probability of contracting an edge at time t.
        
        Args:
            t: Current time step
            
        Returns:
            Probability of contraction
        """
        lambda_k = self._estimate_lambda_k()
        return 1 - math.exp(-t / lambda_k)
        
    def _estimate_lambda_k(self, num_samples: int = 10) -> float:
        """
        Estimate λₖ using multiple methods and return the best estimate.
        
        Args:
            num_samples: Number of samples to use for estimation
            
        Returns:
            Estimated value of λₖ
        """
        if self._lambda_k is not None:
            return self._lambda_k
            
        n = self.original_graph.number_of_nodes()
        
        # For small k, use brute force
        if self.k <= 3 and n <= 10:
            self._lambda_k = self._brute_force_lambda_k()
            return self._lambda_k
            
        # For k=2, use min cut
        if self.k == 2:
            self._lambda_k = self._min_cut_weight()
            return self._lambda_k
            
        # Otherwise, use multiple runs of basic Karger-Stein
        estimates = []
        for _ in range(num_samples):
            result = self.find_min_k_cut(num_trials=1)
            estimates.append(result['weight'])  # result['weight'] is the cut weight
            
        # Take the minimum estimate
        self._lambda_k = min(estimates)
        return self._lambda_k
        
    def _brute_force_lambda_k(self) -> float:
        """
        Compute λₖ exactly for small graphs using brute force.
        
        Returns:
            Exact value of λₖ
        """
        from itertools import combinations
        
        n = self.original_graph.number_of_nodes()
        min_weight = float('inf')
        
        # Try all possible k-partitions
        for partition in self._generate_k_partitions(n, self.k):
            cut_weight = self._calculate_cut_weight(self.original_graph, partition)
            if cut_weight < min_weight:
                min_weight = cut_weight
                
        return min_weight
        
    def _generate_k_partitions(self, n: int, k: int) -> List[List[Set[int]]]:
        """
        Generate all possible k-partitions of n elements.
        
        Args:
            n: Number of elements
            k: Number of parts
            
        Returns:
            List of all possible k-partitions
        """
        from itertools import combinations
        
        # Base case: k=1
        if k == 1:
            return [[set(range(n))]]
            
        # Base case: k=n
        if k == n:
            return [[{i} for i in range(n)]]
            
        # Recursive case
        partitions = []
        for m in range(1, n - k + 2):
            # Choose m elements for the first part
            for first_part in combinations(range(n), m):
                # Generate all (k-1)-partitions of the remaining elements
                remaining = set(range(n)) - set(first_part)
                for sub_partition in self._generate_k_partitions(len(remaining), k - 1):
                    # Map the sub-partition to the original indices
                    mapping = {i: x for i, x in enumerate(remaining)}
                    mapped_sub = [{mapping[i] for i in part} for part in sub_partition]
                    partitions.append([set(first_part)] + mapped_sub)
                    
        return partitions
        
    def _min_cut_weight(self) -> float:
        """
        Compute the minimum 2-cut weight using Karger's algorithm.
        
        Returns:
            Minimum cut weight
        """
        min_weight = float('inf')
        n = self.original_graph.number_of_nodes()
        num_trials = int(n * n * math.log(n))
        
        for _ in range(num_trials):
            graph = self.original_graph.copy()
            
            # Reset supernodes tracking for this trial
            self.supernodes = {node: {node} for node in graph.nodes()}
            
            while graph.number_of_nodes() > 2:
                u, v = self._select_random_edge(graph)
                graph = self._contract_edge(graph, u, v)
                
            # The remaining edge represents the cut
            cut_weight = sum(data['weight'] for _, _, data in graph.edges(data=True))
            if cut_weight < min_weight:
                min_weight = cut_weight
                
        return min_weight
        
    def find_min_k_cut_recursive(self, k: int, num_trials: int = None) -> Dict:
        """
        Find minimum k-cut using recursive Karger-Stein algorithm with Nagamochi-Ibaraki sparsification.
        
        Args:
            k: Number of partitions
            num_trials: Number of trials to run (default: n^2 log n)
            
        Returns:
            Dictionary containing the cut weight, best partition, and all minimum cuts
        """
        if num_trials is None:
            n = self.original_graph.number_of_nodes()
            num_trials = int(n * n * math.log(n))
            
        # Apply Nagamochi-Ibaraki sparsification
        sparsified_graph = self._nagamochi_ibaraki_sparsify(self.original_graph, k)
        
        # Run basic Karger-Stein on sparsified graph
        result = self.find_min_k_cut(num_trials)
        
        return result
        
    def _nagamochi_ibaraki_sparsify(self, graph: nx.Graph, k: int) -> nx.Graph:
        """
        Apply Nagamochi-Ibaraki sparsification to the graph.
        
        Args:
            graph: Input graph
            k: Number of partitions
            
        Returns:
            Sparsified graph
        """
        # Compute maximum connectivity
        max_connectivity = self._compute_max_connectivity(graph)
        
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
                
        # Verify that minimum k-cut is preserved
        if not self._preserve_min_k_cut(graph, sparsified, k):
            # If not preserved, return original graph
            return graph
            
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
            
        # For larger graphs, use sampling
        num_samples = min(100, graph.number_of_nodes())
        min_connectivity = float('inf')
        
        for _ in range(num_samples):
            # Choose a random node
            u = random.choice(list(graph.nodes()))
            
            # Find minimum cut to another random node
            v = random.choice(list(graph.nodes()))
            while v == u:
                v = random.choice(list(graph.nodes()))
                
            connectivity = nx.edge_connectivity(graph, u, v)
            if connectivity < min_connectivity:
                min_connectivity = connectivity
                
        return min_connectivity
        
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
            original_weight = self._calculate_cut_weight(original, partition)
            sparsified_weight = self._calculate_cut_weight(sparsified, partition)
            
            if abs(original_weight - sparsified_weight) > 1e-6:
                return False
                
        return True
        
    def _compute_min_k_cut(self, graph: nx.Graph, k: int) -> float:
        """
        Compute the exact minimum k-cut for small graphs.
        
        Args:
            graph: Input graph
            k: Number of partitions
            
        Returns:
            Minimum k-cut weight
        """
        min_weight = float('inf')
        
        # Try all possible k-partitions
        for partition in self._generate_k_partitions(graph.number_of_nodes(), k):
            cut_weight = self._calculate_cut_weight(graph, partition)
            if cut_weight < min_weight:
                min_weight = cut_weight
                
        return min_weight
        
    def _generate_random_partition(self, graph: nx.Graph, k: int) -> List[Set[int]]:
        """
        Generate a random k-partition of the graph's nodes.
        
        Args:
            graph: Input graph
            k: Number of partitions
            
        Returns:
            List of k sets representing the partition
        """
        nodes = list(graph.nodes())
        random.shuffle(nodes)
        
        # Distribute nodes among partitions
        partition = [set() for _ in range(k)]
        for i, node in enumerate(nodes):
            partition[i % k].add(node)
            
        return partition 