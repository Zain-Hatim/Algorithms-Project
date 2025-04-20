import unittest
import networkx as nx
import numpy as np
from src.graph_builder import GraphBuilder

class TestGraphBuilder(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.n = 10
        self.p = 0.5
        self.seed = 42
        self.builder = GraphBuilder(self.n, self.p, self.seed)
        
    def test_build_graph(self):
        """Test graph construction."""
        graph = self.builder.build_graph()
        
        # Check graph type
        self.assertIsInstance(graph, nx.Graph)
        
        # Check number of nodes
        self.assertEqual(graph.number_of_nodes(), self.n)
        
        # Check edge weights
        for _, _, data in graph.edges(data=True):
            self.assertIn('weight', data)
            self.assertGreaterEqual(data['weight'], 0)
            self.assertLessEqual(data['weight'], 1)
            
    def test_nagamochi_ibaraki_sparsification(self):
        """Test Nagamochi-Ibaraki sparsification."""
        graph = self.builder.build_graph()
        k = 3
        
        # Get original edge count
        original_edges = graph.number_of_edges()
        
        # Apply sparsification
        sparsified = self.builder.nagamochi_ibaraki_sparsification(k)
        
        # Check edge count reduction
        self.assertLessEqual(sparsified.number_of_edges(), k * (self.n - 1))
        
        # Check connectivity preservation
        self.assertEqual(nx.is_connected(graph), nx.is_connected(sparsified))
        
        # Check node preservation
        self.assertEqual(set(graph.nodes()), set(sparsified.nodes()))
        
    def test_preserve_min_k_cut(self):
        """Test that minimum k-cut is preserved after sparsification."""
        graph = self.builder.build_graph()
        k = 2
        
        # Apply sparsification
        sparsified = self.builder.nagamochi_ibaraki_sparsification(k)
        
        # Check that minimum k-cut is preserved
        self.assertTrue(self.builder._preserve_min_k_cut(graph, sparsified, k))
        
    def test_compute_max_connectivity(self):
        """Test maximum connectivity computation."""
        graph = self.builder.build_graph()
        
        # Ensure graph is connected
        if not nx.is_connected(graph):
            graph = nx.complete_graph(self.n)
            for u, v in graph.edges():
                graph[u][v]['weight'] = np.random.random()
                
        # Compute maximum connectivity
        max_conn = self.builder._compute_max_connectivity(graph)
        
        # Check bounds
        self.assertGreaterEqual(max_conn, 0)
        self.assertLessEqual(max_conn, self.n - 1)
        
    def test_compute_cut_weight(self):
        """Test cut weight computation."""
        graph = self.builder.build_graph()
        k = 2
        
        # Create a random partition
        partition = self.builder._generate_random_partition(graph, k)
        
        # Compute cut weight
        weight = self.builder._compute_cut_weight(graph, partition)
        
        # Check weight is non-negative
        self.assertGreaterEqual(weight, 0)
        
        # Check weight matches manual computation
        manual_weight = 0
        for i in range(k):
            for j in range(i + 1, k):
                for u in partition[i]:
                    for v in partition[j]:
                        if graph.has_edge(u, v):
                            manual_weight += graph[u][v]['weight']
        self.assertEqual(weight, manual_weight)
        
    def test_generate_random_partition(self):
        """Test random partition generation."""
        graph = self.builder.build_graph()
        k = 3
        
        # Generate partition
        partition = self.builder._generate_random_partition(graph, k)
        
        # Check partition properties
        self.assertEqual(len(partition), k)
        
        # Check all nodes are included
        all_nodes = set()
        for part in partition:
            all_nodes.update(part)
        self.assertEqual(all_nodes, set(graph.nodes()))
        
        # Check no node is in multiple parts
        for i in range(k):
            for j in range(i + 1, k):
                self.assertFalse(partition[i] & partition[j])
                
if __name__ == '__main__':
    unittest.main() 