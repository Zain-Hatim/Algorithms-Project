# Karger-Stein Algorithm Implementation

This project implements the Karger-Stein algorithm for finding minimum k-cuts in graphs, as described in the 2020 paper *"The Karger-Stein Algorithm Is Optimal for k-Cut."*

## Features

- Basic randomized contraction algorithm
- Recursive Karger-Stein algorithm
- λₖ estimation for optimal contraction probabilities
- Nagamochi-Ibaraki sparsification
- Performance logging and analysis

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The main script `main.py` provides a command-line interface for running the algorithm:

```bash
python main.py --input <graph_file> --k <k> [options]
```

### Required Arguments

- `--input`: Path to input graph file
- `--k`: Number of partitions for k-cut

### Optional Arguments

- `--output`: Path to output file for results
- `--variant`: Algorithm variant to use (`basic` or `recursive`, default: `recursive`)
- `--trials`: Number of trials to run (default: n² log n)
- `--sparsify`: Use Nagamochi-Ibaraki sparsification
- `--log-file`: Path to performance log file (default: `results/runtime_logs.csv`)

### Example

```bash
python main.py --input data/graph.txt --k 3 --variant recursive --sparsify
```

## Implementation Details

The implementation follows the 2020 paper closely, with the following key components:

1. **Graph Builder**: Constructs random graphs and handles sparsification
2. **Karger-Stein Algorithm**: Implements both basic and recursive variants
3. **λₖ Estimation**: Computes optimal contraction probabilities
4. **Performance Logger**: Tracks runtime metrics and algorithm performance

## Testing

Run the test suite with:

```bash
python -m unittest discover tests
```

## License

MIT License 