import networkx as nx
import json
from typing import Dict, List, Tuple, Union
import random
import logging

def contract_edge(graph: nx.Graph, edge: Tuple[int, int]) -> None:
    """
    Contract an edge in the graph by merging its endpoints.
    
    Args:
        graph: NetworkX graph
        edge: Tuple representing the edge to contract
    """
    u, v = edge
    if u == v:
        return
    
    # Merge edges from v to u
    for neighbor, data in list(graph[v].items()):
        if neighbor != u:
            if graph.has_edge(u, neighbor):
                graph[u][neighbor]['weight'] += data['weight']
            else:
                graph.add_edge(u, neighbor, weight=data['weight'])
    
    # Remove the contracted vertex
    graph.remove_node(v)

def load_graph_from_file(filepath: str) -> nx.Graph:
    """
    Load a graph from a file. Supports multiple formats:
    - .txt: Edge list format (u v weight)
    - .json: JSON format with nodes and edges
    
    Args:
        filepath: Path to the graph file
        
    Returns:
        NetworkX graph with weights
    """
    if filepath.endswith('.txt'):
        return _load_edge_list(filepath)
    elif filepath.endswith('.json'):
        return _load_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

def _load_edge_list(filepath: str) -> nx.Graph:
    """
    Load graph from edge list format.
    Each line: u v weight
    
    Args:
        filepath: Path to edge list file
        
    Returns:
        NetworkX graph with weights
    """
    G = nx.Graph()
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                try:
                    u, v, weight = map(float, line.strip().split())
                    G.add_edge(int(u), int(v), weight=weight)
                except ValueError as e:
                    logging.warning(f"Skipping invalid line: {line.strip()}")
    return G

def _load_json(filepath: str) -> nx.Graph:
    """
    Load graph from JSON format.
    Expected format:
    {
        "nodes": [...],
        "edges": [[u, v, weight], ...]
    }
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        NetworkX graph with weights
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    G = nx.Graph()
    G.add_nodes_from(data.get('nodes', []))
    for u, v, weight in data.get('edges', []):
        G.add_edge(u, v, weight=weight)
    return G

def save_graph_to_file(graph: nx.Graph, filepath: str) -> None:
    """
    Save a graph to a file. Supports multiple formats:
    - .txt: Edge list format
    - .json: JSON format
    
    Args:
        graph: NetworkX graph to save
        filepath: Path to save the graph
    """
    if filepath.endswith('.txt'):
        _save_edge_list(graph, filepath)
    elif filepath.endswith('.json'):
        _save_json(graph, filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

def _save_edge_list(graph: nx.Graph, filepath: str) -> None:
    """
    Save graph to edge list format.
    
    Args:
        graph: NetworkX graph to save
        filepath: Path to save the graph
    """
    with open(filepath, 'w') as f:
        for u, v, data in graph.edges(data=True):
            f.write(f"{u} {v} {data['weight']}\n")

def _save_json(graph: nx.Graph, filepath: str) -> None:
    """
    Save graph to JSON format.
    
    Args:
        graph: NetworkX graph to save
        filepath: Path to save the graph
    """
    data = {
        'nodes': list(graph.nodes()),
        'edges': [[u, v, data['weight']] for u, v, data in graph.edges(data=True)]
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def generate_random_graph(n: int, p: float, weight_range: Tuple[float, float] = (1.0, 10.0)) -> nx.Graph:
    """
    Generate a random weighted graph.
    
    Args:
        n: Number of vertices
        p: Probability of edge creation
        weight_range: Tuple of (min_weight, max_weight)
        
    Returns:
        Randomly generated NetworkX graph
    """
    graph = nx.erdos_renyi_graph(n, p)
    for u, v in graph.edges():
        weight = random.uniform(weight_range[0], weight_range[1])
        graph[u][v]['weight'] = weight
    return graph 