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
        # Set up the algorithm with input graph and number of desired components
        self.original_graph = graph
        self.k = k
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._lambda_k = None  # Cache for min cut value estimation
        self.logger = PerformanceLogger()
        
        # Track which original nodes are merged into each supernode
        self.supernodes = {node: {node} for node in graph.nodes()}
        
        # Validate input parameters
        if not isinstance(graph, nx.Graph):
            raise ValueError("Input must be a NetworkX graph")
        if not all('weight' in data for _, _, data in graph.edges(data=True)):
            raise ValueError("All edges must have weights")
        if k < 2:
            raise ValueError("k must be at least 2")
            
    def _select_random_edge(self, graph: nx.Graph) -> Tuple[int, int]:
        # Select edge with probability proportional to weight and endpoint degrees
        # This helps speed up contraction by prioritizing high-degree nodes
        edges = list(graph.edges(data=True))
        
        degrees = dict(graph.degree())
        
        weights = []
        for u, v, data in edges:
            edge_weight = data['weight']
            degree_sum = degrees[u] + degrees[v]
            weights.append(edge_weight * degree_sum)
            
        total_weight = sum(weights)
        if total_weight == 0:
            probabilities = [1/len(edges)] * len(edges)
        else:
            probabilities = [w/total_weight for w in weights]
        
        selected_idx = self._rng.choice(len(edges), p=probabilities)
        return edges[selected_idx][0], edges[selected_idx][1]
        
    def _contract_edge(self, graph: nx.Graph, u: int, v: int) -> nx.Graph:
        # Merge vertices u and v into a single vertex
        new_graph = graph.copy()
        
        contract_edge(new_graph, (u, v))
        
        # Update which original nodes are in each supernode
        self.supernodes[u] = self.supernodes[u].union(self.supernodes[v])
        del self.supernodes[v]
        
        return new_graph
        
    def _calculate_cut_weight(self, graph: nx.Graph, partition: List[Set[int]]) -> float:
        # Calculate total weight of edges crossing between different partitions
        cut_weight = 0.0
        for i in range(len(partition)):
            for j in range(i + 1, len(partition)):
                for u in partition[i]:
                    for v in partition[j]:
                        if graph.has_edge(u, v):
                            cut_weight += graph[u][v]['weight']
        return cut_weight
        
    def _is_partition_connected(self, graph: nx.Graph, partition: List[Set[int]]) -> bool:
        # Check if each partition forms a connected subgraph
        for part in partition:
            subgraph = graph.subgraph(part)
            if not nx.is_connected(subgraph):
                return False
        return True
        
    def _merge_disconnected_components(self, graph: nx.Graph, partition: List[Set[int]]) -> List[Set[int]]:
        # Ensure each partition is connected by merging disconnected components
        new_partition = []
        for part in partition:
            subgraph = graph.subgraph(part)
            components = list(nx.connected_components(subgraph))
            
            if len(components) == 1:
                new_partition.append(part)
            else:
                # Keep merging components until only one remains
                while len(components) > 1:
                    min_distance = float('inf')
                    merge_pair = (0, 1)
                    
                    # Find closest components to merge based on edge weights
                    for i in range(len(components)):
                        for j in range(i + 1, len(components)):
                            for u in components[i]:
                                for v in components[j]:
                                    if graph.has_edge(u, v):
                                        weight = graph[u][v]['weight']
                                        if weight < min_distance:
                                            min_distance = weight
                                            merge_pair = (i, j)
                    
                    # Merge the components with minimum distance between them
                    i, j = merge_pair
                    merged = components[i].union(components[j])
                    components = [c for idx, c in enumerate(components) if idx not in (i, j)]
                    components.append(merged)
                
                new_partition.append(components[0])
        
        return new_partition

    def find_min_k_cut(self, num_trials: int = None) -> Dict:
        # Main method to find minimum k-cut using repeated random contractions
        if num_trials is None:
            n = self.original_graph.number_of_nodes()
            num_trials = int(n * n * np.log(n))
            
        min_cut_weight = float('inf')
        best_partition = None
        all_min_cuts = []
        num_successes = 0
        
        for trial in range(num_trials):
            # Create a fresh copy of the graph for this trial
            graph = self.original_graph.copy()
            
            # Reset supernode tracking
            self.supernodes = {node: {node} for node in graph.nodes()}
            
            # Use recursive contraction strategy (the core of Karger-Stein algorithm)
            result = self._recursive_contraction(graph)
            
            # Build partition from supernodes
            partition = []
            for supernode in self.supernodes.values():
                if supernode:
                    partition.append(supernode)
            
            # Ensure we have exactly k partitions
            if len(partition) > self.k:
                # Too many partitions - merge some
                while len(partition) > self.k:
                    min_distance = float('inf')
                    merge_pair = (0, 1)
                    
                    # Find closest partitions to merge
                    for i in range(len(partition)):
                        for j in range(i + 1, len(partition)):
                            merged = partition[i].union(partition[j])
                            if nx.is_connected(self.original_graph.subgraph(merged)):
                                weight = sum(self.original_graph[u][v]['weight']
                                          for u in partition[i]
                                          for v in partition[j]
                                          if self.original_graph.has_edge(u, v))
                                if weight < min_distance:
                                    min_distance = weight
                                    merge_pair = (i, j)
                    
                    # Merge the closest partitions
                    i, j = merge_pair
                    merged = partition[i].union(partition[j])
                    partition = [p for idx, p in enumerate(partition) if idx not in (i, j)]
                    partition.append(merged)
                    
            elif len(partition) < self.k:
                # Too few partitions - split some using spectral clustering
                while len(partition) < self.k:
                    partition.sort(key=len, reverse=True)
                    largest = partition[0]
                    subgraph = self.original_graph.subgraph(largest)
                    
                    # Use spectral clustering to split the component
                    laplacian = nx.laplacian_matrix(subgraph).todense()
                    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
                    # Use Fiedler vector for splitting
                    fiedler = eigenvectors[:, 1]
                    
                    # Split based on the sign of the Fiedler vector
                    nodes = list(subgraph.nodes())
                    part1 = {nodes[i] for i in range(len(nodes)) if fiedler[i] >= 0}
                    part2 = {nodes[i] for i in range(len(nodes)) if fiedler[i] < 0}
                    
                    # Update partition
                    partition = [part1, part2] + partition[1:]
            
            # Ensure connectivity within each partition
            partition = self._merge_disconnected_components(self.original_graph, partition)
            
            # Calculate cut weight
            cut_weight = self._calculate_cut_weight(self.original_graph, partition)
            
            # Update best results if we found a better or equal cut
            if (cut_weight < min_cut_weight and 
                self._is_partition_connected(self.original_graph, partition)):
                min_cut_weight = cut_weight
                best_partition = partition
                all_min_cuts = [partition]
                num_successes = 1
            elif (cut_weight == min_cut_weight and 
                  self._is_partition_connected(self.original_graph, partition)):
                if not self._is_duplicate_cut(partition, all_min_cuts):
                    all_min_cuts.append(partition)
                num_successes += 1
                
        # Log success statistics
        success_rate = num_successes / num_trials
        self.logger.log(f"Success rate: {success_rate:.2%} ({num_successes}/{num_trials} trials found minimum cut)")
                
        return {
            'weight': min_cut_weight,
            'partitions': best_partition,
            'all_min_cuts': all_min_cuts
        }
        
    def _recursive_contraction(self, graph: nx.Graph, threshold: int = 6) -> nx.Graph:
        # Core of Karger-Stein algorithm: recursively contract graph and keep the best cut
        n = graph.number_of_nodes()
        
        # Base case: if graph is small enough, compute cut directly
        if n <= threshold:
            return graph
            
        # Contract graph down to t = ceil(n / √2) nodes
        t = math.ceil(n / math.sqrt(2))
        
        # Create two independent copies for parallel contraction attempts
        graph1 = graph.copy()
        graph2 = graph.copy()
        
        # Save current supernodes state
        original_supernodes = self.supernodes.copy()
        
        # Contract first copy
        while graph1.number_of_nodes() > t:
            u, v = self._select_random_edge(graph1)
            graph1 = self._contract_edge(graph1, u, v)
            
        # Save supernodes after first contraction
        supernodes1 = self.supernodes.copy()
        
        # Restore original supernodes for second contraction
        self.supernodes = original_supernodes.copy()
        
        # Contract second copy
        while graph2.number_of_nodes() > t:
            u, v = self._select_random_edge(graph2)
            graph2 = self._contract_edge(graph2, u, v)
            
        # Save supernodes after second contraction
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
        
        # Return the graph with the smaller cut
        if cut1 <= cut2:
            self.supernodes = supernodes1
            return result1
        else:
            self.supernodes = supernodes2
            return result2
        
    def _is_duplicate_cut(self, partition: List[Set[int]], existing_cuts: List[List[Set[int]]]) -> bool:
        # Check if a partition is equivalent to any existing cut
        # Convert to canonical form for comparison
        canonical_new = tuple(tuple(sorted(part)) for part in sorted(partition, key=lambda x: min(x)))
        
        for existing in existing_cuts:
            canonical_existing = tuple(tuple(sorted(part)) for part in sorted(existing, key=lambda x: min(x)))
            if canonical_new == canonical_existing:
                return True
                
        return False
        
    def _get_contraction_probability(self, t: float) -> float:
        # Calculate probability of contracting an edge at time t
        lambda_k = self._estimate_lambda_k()
        return 1 - math.exp(-t / lambda_k)
        
    def _estimate_lambda_k(self, num_samples: int = 10) -> float:
        # Estimate λₖ (min k-cut value) using multiple methods
        if self._lambda_k is not None:
            return self._lambda_k
            
        n = self.original_graph.number_of_nodes()
        
        # For small problems, use brute force
        if self.k <= 3 and n <= 10:
            self._lambda_k = self._brute_force_lambda_k()
            return self._lambda_k
            
        # For k=2, use standard min cut
        if self.k == 2:
            self._lambda_k = self._min_cut_weight()
            return self._lambda_k
            
        # Otherwise, use sampling
        estimates = []
        for _ in range(num_samples):
            result = self.find_min_k_cut(num_trials=1)
            estimates.append(result['weight'])
            
        # Take the minimum estimate
        self._lambda_k = min(estimates)
        return self._lambda_k
        
    def _brute_force_lambda_k(self) -> float:
        # Find exact min k-cut for small graphs by trying all partitions
        from itertools import combinations
        
        n = self.original_graph.number_of_nodes()
        min_weight = float('inf')
        
        for partition in self._generate_k_partitions(n, self.k):
            cut_weight = self._calculate_cut_weight(self.original_graph, partition)
            if cut_weight < min_weight:
                min_weight = cut_weight
                
        return min_weight
        
    def _generate_k_partitions(self, n: int, k: int) -> List[List[Set[int]]]:
        # Generate all possible ways to partition n elements into k groups
        from itertools import combinations
        
        # Base cases
        if k == 1:
            return [[set(range(n))]]
            
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
        # Find minimum 2-cut weight using basic Karger's algorithm
        min_weight = float('inf')
        n = self.original_graph.number_of_nodes()
        num_trials = int(n * n * math.log(n))
        
        for _ in range(num_trials):
            graph = self.original_graph.copy()
            
            self.supernodes = {node: {node} for node in graph.nodes()}
            
            # Contract until only 2 nodes remain
            while graph.number_of_nodes() > 2:
                u, v = self._select_random_edge(graph)
                graph = self._contract_edge(graph, u, v)
                
            # The remaining edge represents the cut
            cut_weight = sum(data['weight'] for _, _, data in graph.edges(data=True))
            if cut_weight < min_weight:
                min_weight = cut_weight
                
        return min_weight
        
    def find_min_k_cut_recursive(self, k: int, num_trials: int = None) -> Dict:
        # Enhanced version using Nagamochi-Ibaraki sparsification
        if num_trials is None:
            n = self.original_graph.number_of_nodes()
            num_trials = int(n * n * math.log(n))
            
        # Apply sparsification to reduce edge count while preserving min k-cut
        sparsified_graph = self._nagamochi_ibaraki_sparsify(self.original_graph, k)
        
        # Run algorithm on sparsified graph
        result = self.find_min_k_cut(num_trials)
        
        return result
        
    def _nagamochi_ibaraki_sparsify(self, graph: nx.Graph, k: int) -> nx.Graph:
        # Reduce graph size while preserving min k-cut
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
                
            # Only add edge if it doesn't create a cycle
            if not nx.has_path(sparsified, u, v):
                sparsified.add_edge(u, v, weight=data['weight'])
                added_edges += 1
                
        # Verify that minimum k-cut is preserved
        if not self._preserve_min_k_cut(graph, sparsified, k):
            # If not preserved, use original graph
            return graph
            
        return sparsified
        
    def _compute_max_connectivity(self, graph: nx.Graph) -> int:
        # Estimate maximum edge connectivity of the graph
        if not nx.is_connected(graph):
            return 0
            
        # For small graphs, compute exact connectivity
        if graph.number_of_nodes() <= 10:
            return nx.edge_connectivity(graph)
            
        # For larger graphs, use sampling
        num_samples = min(100, graph.number_of_nodes())
        min_connectivity = float('inf')
        
        for _ in range(num_samples):
            # Find minimum cut between random pairs of nodes
            u = random.choice(list(graph.nodes()))
            v = random.choice(list(graph.nodes()))
            while v == u:
                v = random.choice(list(graph.nodes()))
                
            connectivity = nx.edge_connectivity(graph, u, v)
            if connectivity < min_connectivity:
                min_connectivity = connectivity
                
        return min_connectivity
        
    def _preserve_min_k_cut(self, original: nx.Graph, sparsified: nx.Graph, k: int) -> bool:
        # Check if sparsified graph preserves the minimum k-cut
        # For small graphs, compare exact min cuts
        if original.number_of_nodes() <= 10:
            original_cut = self._compute_min_k_cut(original, k)
            sparsified_cut = self._compute_min_k_cut(sparsified, k)
            return abs(original_cut - sparsified_cut) < 1e-6
            
        # For larger graphs, test with random partitions
        num_samples = min(100, original.number_of_nodes())
        for _ in range(num_samples):
            # Generate random k-partition
            partition = self._generate_random_partition(original, k)
            
            # Compare cut weights in both graphs
            original_weight = self._calculate_cut_weight(original, partition)
            sparsified_weight = self._calculate_cut_weight(sparsified, partition)
            
            if abs(original_weight - sparsified_weight) > 1e-6:
                return False
                
        return True
        
    def _compute_min_k_cut(self, graph: nx.Graph, k: int) -> float:
        # Compute exact minimum k-cut for small graphs
        min_weight = float('inf')
        
        # Try all possible k-partitions
        for partition in self._generate_k_partitions(graph.number_of_nodes(), k):
            cut_weight = self._calculate_cut_weight(graph, partition)
            if cut_weight < min_weight:
                min_weight = cut_weight
                
        return min_weight
        
    def _generate_random_partition(self, graph: nx.Graph, k: int) -> List[Set[int]]:
        # Create a random k-partition of the graph's nodes
        nodes = list(graph.nodes())
        random.shuffle(nodes)
        
        # Distribute nodes evenly among partitions
        partition = [set() for _ in range(k)]
        for i, node in enumerate(nodes):
            partition[i % k].add(node)
            
        return partition