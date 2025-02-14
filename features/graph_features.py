"""Graph-based feature extraction"""

import networkx as nx
import numpy as np


class GraphFeatureExtractor:
    def __init__(self, graph, max_hops=3):
        self.graph = graph
        self.max_hops = max_hops

    def compute_node_features(self, node):
        """Compute features for a single node"""
        subgraph = nx.ego_graph(
            self.graph, node, radius=self.max_hops, undirected=False
        )

        features = {
            "num_nodes": subgraph.number_of_nodes(),
            "num_edges": subgraph.number_of_edges(),
            "in_degree": subgraph.in_degree(node),
            "out_degree": subgraph.out_degree(node),
            "degree_centrality": nx.degree_centrality(subgraph)[node],
            "in_degree_centrality": nx.in_degree_centrality(subgraph)[node],
            "out_degree_centrality": nx.out_degree_centrality(subgraph)[node],
            "pagerank": nx.pagerank(subgraph)[node],
            "clustering_coeff": nx.clustering(subgraph.to_undirected(), node),
            "local_clustering_coeff": nx.average_clustering(subgraph.to_undirected()),
            "average_neighbor_degree": self._calculate_avg_neighbor_degree(
                subgraph, node
            ),
            "connectivity_ratio": self._calculate_connectivity_ratio(subgraph),
        }

        # Add safe centrality calculations
        features.update(self._calculate_safe_centralities(subgraph, node))

        return features

    def _calculate_avg_neighbor_degree(self, subgraph, node):
        """Calculate average neighbor degree safely"""
        neighbors = list(subgraph.neighbors(node))
        if not neighbors:
            return 0
        return np.mean([subgraph.degree(n) for n in neighbors])

    def _calculate_connectivity_ratio(self, subgraph):
        """Calculate connectivity ratio safely"""
        n = subgraph.number_of_nodes()
        if n <= 1:
            return 0
        return subgraph.number_of_edges() / (n * (n - 1) + 1)

    def _calculate_safe_centralities(self, subgraph, node):
        """Calculate various centrality measures with error handling"""
        centralities = {}

        # Harmonic centrality
        try:
            centralities["harmonic_centrality"] = nx.harmonic_centrality(subgraph)[node]
        except:
            centralities["harmonic_centrality"] = 0

        # Eigenvector centrality
        try:
            centralities["eigenvector_centrality"] = nx.eigenvector_centrality(
                subgraph, max_iter=100, tol=1e-2
            ).get(node, 0)
        except:
            centralities["eigenvector_centrality"] = 0

        # Betweenness centrality
        try:
            centralities["betweenness_centrality"] = nx.betweenness_centrality(
                subgraph
            )[node]
        except:
            centralities["betweenness_centrality"] = 0

        return centralities

    def extract_features_for_class(self, merged_df, node_class):
        """Extract features for all nodes of a specific class"""
        nodes = merged_df[merged_df["class_source"] == str(node_class)][
            "source"
        ].tolist()
        features = {node: self.compute_node_features(node) for node in nodes}
        return features
