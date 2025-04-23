import unittest
import tempfile
import os
import networkx as nx
from src.performance import PerformanceLogger, benchmark_algorithm, run_benchmark_suite
from src.graph_builder import GraphBuilder
from src.karger_stein import KargerStein

class TestPerformanceLogger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test_log.csv')
        self.logger = PerformanceLogger(self.log_file)
        
    def test_log_run(self):
        """Test logging a single run."""
        metrics = {
            'n': 10,
            'm': 20,
            'k': 3,
            'cut_weight': 5.0,
            'runtime': 0.1,
            'algorithm': 'basic',
            'sparsified': False
        }
        
        self.logger.log_run(**metrics)
        
        # Verify file was created with correct header
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # header + data
            self.assertTrue('timestamp' in lines[0])
            self.assertTrue('n,m,k,cut_weight,runtime,algorithm,sparsified' in lines[0])
            
    def test_log_run_with_additional_metrics(self):
        """Test logging with additional metrics."""
        metrics = {
            'n': 10,
            'm': 20,
            'k': 3,
            'cut_weight': 5.0,
            'runtime': 0.1,
            'algorithm': 'basic',
            'sparsified': False,
            'additional_metrics': {'extra1': 1, 'extra2': 2}
        }
        
        self.logger.log_run(**metrics)
        
        # Verify additional metrics were logged
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
            self.assertTrue('extra1,extra2' in lines[0])
            self.assertTrue('1,2' in lines[1])

class TestBenchmarkAlgorithm(unittest.TestCase):
    def setUp(self):
        # Create a simple test graph
        self.graph = nx.Graph()
        self.graph.add_edges_from([
            (0, 1, {'weight': 1.0}),
            (1, 2, {'weight': 2.0}),
            (2, 0, {'weight': 3.0})
        ])
        
    def test_benchmark_basic(self):
        """Test basic benchmarking without sparsification."""
        result = benchmark_algorithm(self.graph, k=2, recursive=False, sparsify=False)
        
        self.assertIn('result', result)
        self.assertIn('metrics', result)
        self.assertIn('cut_weight', result['metrics'])
        self.assertIn('runtime', result['metrics'])
        
    def test_benchmark_with_sparsification(self):
        """Test benchmarking with sparsification."""
        result = benchmark_algorithm(self.graph, k=2, recursive=False, sparsify=True)
        
        self.assertIn('m_sparse', result['metrics'])
        self.assertIn('sparsification_ratio', result['metrics'])
        self.assertLessEqual(result['metrics']['m_sparse'], self.graph.number_of_edges())
        
    def test_benchmark_recursive(self):
        """Test benchmarking with recursive algorithm."""
        result = benchmark_algorithm(self.graph, k=2, recursive=True, sparsify=False)
        
        self.assertEqual(result['metrics']['algorithm'], 'recursive')

class TestBenchmarkSuite(unittest.TestCase):
    def setUp(self):
        # Create a simple test graph
        self.graph = nx.Graph()
        self.graph.add_edges_from([
            (0, 1, {'weight': 1.0}),
            (1, 2, {'weight': 2.0}),
            (2, 0, {'weight': 3.0}),
            (2, 3, {'weight': 4.0}),
            (3, 4, {'weight': 5.0})
        ])
        self.k_values = [2, 3]
        
    def test_run_benchmark_suite(self):
        """Test running a suite of benchmarks."""
        results = run_benchmark_suite(
            self.graph, 
            k_values=self.k_values,
            recursive=False,
            sparsify=False
        )
        
        self.assertEqual(len(results), len(self.k_values))
        for k in self.k_values:
            self.assertIn(k, results)
            self.assertIn('result', results[k])
            self.assertIn('metrics', results[k])
            
    def test_run_benchmark_suite_with_sparsification(self):
        """Test running a suite with sparsification."""
        results = run_benchmark_suite(
            self.graph,
            k_values=self.k_values,
            recursive=False,
            sparsify=True
        )
        
        for k in self.k_values:
            self.assertIn('m_sparse', results[k]['metrics'])
            self.assertIn('sparsification_ratio', results[k]['metrics'])

if __name__ == '__main__':
    unittest.main() 