# Algorithms-Project

This paper focuses on the k-Cut problem: given a weighted graph and an integer k, the goal is to remove a set of edges with the smallest possible total weight such that the graph breaks into at least k connected components. The problem generalizes the standard min-cut (where k = 2).


The authors show that the Karger-Stein algorithm, which uses random edge contractions, is actually optimal for solving the k-Cut problem for any fixed k. They prove that it outputs a correct solution with high probability and runs in Õ (nᵏ) time, which matches known lower bounds up to logarithmic factors. The paper also includes a detailed analysis of how the graph structure evolves during the contractions and gives new bounds on the number of minimum k-cuts.
