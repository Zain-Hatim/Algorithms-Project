```markdown
# Implementation of the Optimal Karger-Stein Algorithm for k-Cut (Based on Gupta, Lee, and Li)

This README.md details the low-level implementation of the Karger-Stein algorithm for the k-cut problem, following the approach presented in "The Karger-Stein Algorithm Is Optimal for k-Cut" by Gupta, Lee, and Li. This implementation aims to achieve the near-optimal $\hat{O}(n^k)$ time complexity by employing a recursive contraction strategy.

## 1. Algorithm Overview

The algorithm is a randomized recursive procedure based on repeatedly contracting edges of the graph. The probability of contracting an edge is proportional to its weight. The recursion continues with adjusted parameters until a small number of vertices remain. By carefully analyzing the probability of a minimum k-cut surviving these contractions and employing a specific recursive structure, the algorithm finds a minimum k-cut with high probability.

## 2. Data Structures

The following data structures are chosen for an efficient implementation:

*   Weighted Graph: The graph $G = (V, E, w)$ will be represented using an **adjacency list**. For each vertex $u \in V$, we will maintain a list of its neighbors $v$ and the weight $w(u, v)$ of the edge connecting them. This representation is suitable for iterating over edges incident to a vertex and for implementing edge contraction. To handle **multi-edges** that arise during contraction, the adjacency list will store pairs of (neighbor, weight). If contracting an edge creates a new edge between two vertices that already have an edge, their weights will be summed.

*   Vertex Representation: Vertices will be represented by unique integer identifiers (from 0 to $n-1$ initially). When vertices are contracted, a **disjoint-set data structure (Union-Find)** will be used to keep track of which original vertices belong to the current super-vertex. Each set in the Union-Find structure will represent a contracted vertex, and the representative of the set will serve as the identifier for the super-vertex.

*   Edge List (Auxiliary): An auxiliary list of all edges in the current graph, along with their weights and original endpoints, will be maintained to facilitate the **weighted random edge sampling**. This list will need to be updated after each contraction.

## 3. Implementation Steps

The implementation will follow a recursive structure:

### 3.1. Initialization

1.  Read the input graph with $n$ vertices and weighted edges.
2.  Initialize the adjacency list representation of the graph.
3.  Create an initial edge list containing all edges and their weights.
4.  Initialize the Union-Find data structure, where each original vertex is in its own set.

### 3.2. Recursive Function `RecursiveKCut(graph, vertex_count, iteration_number)`

This function will perform the core recursive steps:

1.  Base Case: If `vertex_count` is less than or equal to a small constant (related to $k$, as discussed in the paper's analysis for small vertex counts), run a basic k-cut finding procedure (or continue with more contractions until $k$ super-vertices remain) and return the resulting cut.

2.  Determine Contraction Probability: Based on the current `vertex_count` (let's denote it as $n_{current}$) and the `iteration_number`, calculate the probability $p = 1 - e^{-t/\lambda_k}$ for contracting each edge, where $t$ is a parameter set as $t = \frac{1}{2} \ln(n_{current})$ (following the paper's strategy). Since $\lambda_k$ (the minimum k-cut value) is generally unknown, we will need to either:
    *   **Assume a known or estimated $\lambda_k$:** For initial implementation, one could start by assuming $\lambda_k = 1$ or using a known approximation if available. A more robust implementation would involve techniques discussed in the paper, such as guessing $\lambda_k$ within a range.
    *   **Adapt the probability directly based on the total edge weight:** Alternatively, one could sample edges with probability proportional to their weight, as in the original Karger-Stein algorithm, and the recursive analysis of Gupta, Lee, and Li provides the theoretical justification for its optimality when applied with the specific recursive structure.

3.  Perform Multiple Trials: The recursive step involves performing multiple independent trials of edge contraction. The number of trials in iteration $i$ is suggested to be around $(n̂_{i-1})^{k/2} \cdot (1 - (n̂_{i-1})^{-3k/2})$, where $n̂_i$ is a target number of vertices in iteration $i$ (approximately $(n̂_{i-1})^{1/2}$).

4.  Edge Contraction in Each Trial: For each trial:
    a. Create a copy of the current graph and the Union-Find structure.
    b. Iterate through the edges in the edge list.
    c. For each edge $(u, v)$ with weight $w(u, v)$, decide whether to contract it based on the calculated probability $p$ (or by weighted sampling).
    d. **Contract Operation:** If an edge $(u, v)$ is to be contracted:
        i. Find the representatives of the sets containing $u$ and $v$ in the Union-Find structure (say, $rep\_u$ and $rep\_v$).
        ii. If $rep\_u$ and $rep\_v$ are different, merge their sets in the Union-Find structure. Let the new representative be $rep_{new}$.
        iii. Update the adjacency list of $rep_{new}$. Iterate through the neighbors of $rep\_u$ and $rep\_v$. For each neighbor $w$:
            *   If $w$ is not $rep\_u$ or $rep\_v$, the edge $(rep_{new}, w)$ is formed. If an edge between $rep_{new}$ and $w$ already exists, add the weight of the new edge to the existing weight.
            *   Remove any self-loops (edges where both endpoints are the same after merging).
        iv. Update the edge list by removing the contracted edge and potentially adding/updating edges between the newly formed super-vertex and others.

5.  Recursive Call: After performing the contractions in a trial, let $H$ be the contracted graph with $n'$ vertices. If $n'$ is less than or equal to a target number $n̂_i$, recursively call `RecursiveKCut(H, n', iteration_number + 1)`.

6.  Keep Track of Best Cut: Maintain a variable to store the minimum weight k-cut found across all recursive calls. When the base case is reached (or when the number of remaining super-vertices equals $k$), determine the k-cut by examining the edges in the original graph that now have their endpoints in different connected components (represented by the $k$ super-vertices). Update the minimum weight k-cut if a better one is found.

7.  Return Result: After all trials and recursive calls complete, return the minimum weight k-cut found.

### 3.3. Top-Level Function

1.  Initialize the process by calling `RecursiveKCut(initial_graph, initial_vertex_count, 0)`.
2.  The function will return the minimum weight set of edges that form a k-cut.

