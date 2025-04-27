# Karger-Stein Algorithm Implementation

This repository contains an implementation of the Karger-Stein algorithm for finding minimum k-cuts in graphs. The implementation includes both basic and recursive variants of the algorithm, along with optional Nagamochi-Ibaraki sparsification.

## Repository Structure

```
.
├── Code/
│   ├── src/              # Source code for the algorithm implementation
│   ├── data/             # Input graph data files
│   ├── results/          # Output results and visualizations
│   ├── main.py          # Main execution script
│   ├── requirements.txt  # Python dependencies
│   └── setup.py         # Package setup configuration
├── Documentation/        # Additional documentation
├── Project-Resources/    # Project related resources
└── results/             # Global results directory
```

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Dependencies

The project requires the following Python packages:
- networkx >= 2.6.3
- matplotlib >= 3.5.1
- numpy >= 1.21.0
- pandas >= 1.3.3
- scipy >= 1.7.1
- tqdm >= 4.62.3
- pytest >= 7.1.2
- jupyter >= 1.0.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Algorithms-Project.git
cd Algorithms-Project
```

2. Install the required dependencies:
```bash
cd Code
pip install -r requirements.txt
```

## Usage

The main script (`main.py`) supports various command-line arguments for customizing the algorithm's execution:

```bash
python main.py --input <input_graph_file> --k <num_partitions> [options]
```

### Required Arguments:
- `--input`: Path to the input graph file
- `--k`: Number of partitions (k-cut value)

### Optional Arguments:
- `--variant`: Algorithm variant to use ('basic' or 'recursive', default: 'recursive')
- `--trials`: Number of trials to run (default: n²log n)
- `--sparsify`: Enable Nagamochi-Ibaraki sparsification
- `--output`: Output file for results
- `--log-file`: Path to performance log file
- `--visualize`: Generate visualizations

### Example Usage:

```bash
# Run recursive variant with 3 partitions and visualization
python main.py --input data/sample_graph.txt --k 3 --variant recursive --visualize

# Run basic variant with sparsification
python main.py --input data/large_graph.txt --k 4 --variant basic --sparsify
```

## Output

The algorithm generates several outputs:
- Partition information for the minimum k-cut found
- Cut value and execution statistics
- Performance logs (if specified)
- Graph visualizations (if enabled) showing:
  - Original graph
  - Final partitions with highlighted cut edges
  - Runtime complexity information

Results are stored in the `results/` directory, organized by timestamp for each run.

## Documentation

Additional documentation and implementation details can be found in the `Documentation/` directory.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
