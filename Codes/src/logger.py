import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class PerformanceLogger:
    def __init__(self, log_dir: str = "results"):
        """
        Initialize the performance logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up CSV logging
        self.csv_path = self.log_dir / "runtime_logs.csv"
        self._init_csv()
        
        # Set up debug logging
        logging.basicConfig(
            filename=self.log_dir / "debug.log",
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Configure basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger('KargerStein')
        
    def _init_csv(self):
        """Initialize the CSV file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'graph_size',
                    'edge_count',
                    'k',
                    'cut_weight',
                    'trial_count',
                    'runtime_ms',
                    'algorithm',
                    'sparsified',
                    'parallel',
                    'adaptive'
                ])
                
    def log_trial(self, metrics: Dict[str, Any]):
        """
        Log metrics for a single trial.
        
        Args:
            metrics: Dictionary containing trial metrics
        """
        # Ensure required fields are present
        required_fields = ['graph_size', 'edge_count', 'k', 'cut_weight']
        for field in required_fields:
            if field not in metrics:
                raise ValueError(f"Missing required metric: {field}")
                
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Write to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.get('timestamp', ''),
                metrics.get('graph_size', 0),
                metrics.get('edge_count', 0),
                metrics.get('k', 0),
                metrics.get('cut_weight', 0.0),
                metrics.get('trial_count', 0),
                metrics.get('runtime_ms', 0),
                metrics.get('algorithm', ''),
                metrics.get('sparsified', False),
                metrics.get('parallel', False),
                metrics.get('adaptive', False)
            ])
            
        # Log debug information
        logging.debug(f"Trial completed: {metrics}")
        
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Log an error that occurred during execution.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
        """
        error_msg = f"Error: {str(error)}"
        if context:
            error_msg += f" Context: {context}"
        logging.error(error_msg)
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the logged trials.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.csv_path.exists():
            return {}
            
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            trials = list(reader)
            
        if not trials:
            return {}
            
        # Calculate summary statistics
        cut_weights = [float(t['cut_weight']) for t in trials]
        runtimes = [float(t['runtime_ms']) for t in trials]
        
        return {
            'total_trials': len(trials),
            'min_cut_weight': min(cut_weights),
            'max_cut_weight': max(cut_weights),
            'avg_cut_weight': sum(cut_weights) / len(cut_weights),
            'min_runtime_ms': min(runtimes),
            'max_runtime_ms': max(runtimes),
            'avg_runtime_ms': sum(runtimes) / len(runtimes)
        }

    def log(self, message: str) -> None:
        """
        Log a message with timestamp.
        
        Args:
            message: Message to log
        """
        self.logger.info(message) 