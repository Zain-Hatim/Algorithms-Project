import unittest
import networkx as nx
import numpy as np
from src.karger_stein import KargerStein
from src.graph_builder import GraphBuilder

class TestKargerStein(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.n = 10
        self.p = 0.5
        self.seed = 42
        self.builder = GraphBuilder(self.n, self.p, self.seed)
        self.graph = self.builder.build_graph()
        
    def test_initialization(self):
        """Test KargerStein initialization."""
        k = 3
        ks = KargerStein(self.graph, k, self.seed)
        
        # Check initialization
        self.assertEqual(ks.k, k)
        self.assertEqual(ks.seed, self.seed)
        self.assertEqual(ks.original_graph, self.graph)
        
    def test_find_min_k_cut(self):
        """Test basic minimum k-cut finding."""
        k = 2
        ks = KargerStein(self.graph, k, self.seed)
        
        # Find minimum k-cut
        result = ks.find_min_k_cut()
        
        # Check result structure
        self.assertIn('weight', result)
        self.assertIn('partitions', result)
        self.assertIn('all_min_cuts', result)
        
        # Check partition properties
        partitions = result['partitions']
        self.assertEqual(len(partitions), k)
        
        # Check all nodes are included
        all_nodes = set()
        for part in partitions:
            all_nodes.update(part)
        self.assertEqual(all_nodes, set(self.graph.nodes()))
        
        # Check no node is in multiple parts
        for i in range(k):
            for j in range(i + 1, k):
                self.assertFalse(partitions[i] & partitions[j])
                
    def test_find_min_k_cut_recursive(self):
        """Test recursive minimum k-cut finding."""
        k = 3
        ks = KargerStein(self.graph, k, self.seed)
        
        # Find minimum k-cut
        result = ks.find_min_k_cut_recursive(k)
        
        # Check result structure
        self.assertIn('weight', result)
        self.assertIn('partitions', result)
        self.assertIn('all_min_cuts', result)
        
        # Check partition properties
        partitions = result['partitions']
        self.assertEqual(len(partitions), k)
        
        # Check all nodes are included
        all_nodes = set()
        for part in partitions:
            all_nodes.update(part)
        self.assertEqual(all_nodes, set(self.graph.nodes()))
        
    def test_find_min_k_cut_parallel(self):
        """Test parallel minimum k-cut finding."""
        k = 2
        ks = KargerStein(self.graph, k, self.seed)
        
        # Find minimum k-cut
        result = ks.find_min_k_cut_parallel(k)
        
        # Check result structure
        self.assertIn('weight', result)
        self.assertIn('partitions', result)
        self.assertIn('all_min_cuts', result)
        
        # Check partition properties
        partitions = result['partitions']
        self.assertEqual(len(partitions), k)
        
    def test_find_min_k_cut_adaptive(self):
        """Test adaptive minimum k-cut finding."""
        k = 3
        ks = KargerStein(self.graph, k, self.seed)
        
        # Find minimum k-cut
        result = ks.find_min_k_cut_adaptive(k)
        
        # Check result structure
        self.assertIn('weight', result)
        self.assertIn('partitions', result)
        self.assertIn('all_min_cuts', result)
        self.assertIn('trial_count', result)
        
        # Check trial count is reasonable
        self.assertGreaterEqual(result['trial_count'], 10)  # min_trials
        self.assertLessEqual(result['trial_count'], self.n * self.n * np.log(self.n))
        
    def test_estimate_lambda_k(self):
        """Test λₖ estimation."""
        k = 2
        ks = KargerStein(self.graph, k, self.seed)
        
        # Estimate λₖ
        lambda_k = ks._estimate_lambda_k()
        
        # Check bounds
        self.assertGreaterEqual(lambda_k, 0)
        
        # For k=2, λₖ should be the minimum cut weight
        if k == 2:
            result = ks.find_min_k_cut()
            self.assertAlmostEqual(lambda_k, result['weight'])
            
    def test_is_duplicate_cut(self):
        """Test duplicate cut detection."""
        k = 2
        ks = KargerStein(self.graph, k, self.seed)
        
        # Create two identical partitions
        partition1 = [{0, 1, 2}, {3, 4, 5, 6, 7, 8, 9}]
        partition2 = [{3, 4, 5, 6, 7, 8, 9}, {0, 1, 2}]
        
        # Check that they are considered duplicates
        self.assertTrue(ks._is_duplicate_cut(partition1, [partition2]))
        
        # Create a different partition
        partition3 = [{0, 1, 2, 3}, {4, 5, 6, 7, 8, 9}]
        
        # Check that it's not considered a duplicate
        self.assertFalse(ks._is_duplicate_cut(partition1, [partition3]))
        
    def test_contraction_probability(self):
        """Test contraction probability computation."""
        k = 2
        ks = KargerStein(self.graph, k, self.seed)
        
        # Compute contraction probability
        t = 0.5 * np.log(self.n)
        p = ks._get_contraction_probability(t)
        
        # Check bounds
        self.assertGreaterEqual(p, 0)
        self.assertLessEqual(p, 1)
        
if __name__ == '__main__':
    unittest.main() 