
from typing import Dict, List, Optional
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression


class DependencyGraphEngine:
    """
    Maps variable relationship networks and computes topological centrality metrics.

    Provides static utilities to construct weighted directed association graphs
    from mining rules, derive structural graph centrality measures, and compute
    pairwise continuous dependency metrics including Spearman's rank correlation
    and mutual information.
    """

    @staticmethod
    def build_association_graph(rules_df: pd.DataFrame) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """
        Constructs a directed graph weighted by lift from association rules.

        Iterates over antecedent-consequent pairs extracted from association rules,
        adding directed edges weighted by lift, confidence, and support. If an edge
        already exists, its attributes are updated if a higher lift value is found.

        Args:
            rules_df (pd.DataFrame): DataFrame containing association rules. Expected
                columns are `'antecedents'`, `'consequents'`, `'lift'`,
                `'confidence'`, and `'support'`.

        Returns:
            Tuple[nx.DiGraph, pd.DataFrame]: A tuple containing:
                - **G** (`nx.DiGraph`): Directed network graph where nodes represent
                  items and edges store association metric attributes.
                - **edges_df** (`pd.DataFrame`): DataFrame containing deduplicated edge
                  records with columns `'source'`, `'target'`, `'lift'`,
                  `'confidence'`, and `'support'`.
        """
        G = nx.DiGraph()
        edge_rows = []

        if rules_df.empty:
            return G, pd.DataFrame()

        for _, r in rules_df.iterrows():
            for u in list(r["antecedents"]):
                for v in list(r["consequents"]):

                    if u == v:
                        continue

                    lift = float(r["lift"])
                    conf = float(r["confidence"])
                    supp = float(r["support"])

                    if G.has_edge(u, v):
                        if lift > G[u][v]["lift"]:
                            G[u][v]["lift"] = lift
                            G[u][v]["confidence"] = conf
                            G[u][v]["support"] = supp
                    else:
                        G.add_edge(u, v, lift=lift, confidence=conf, support=supp)

                    edge_rows.append({"source": u, "target": v, "lift": lift, "confidence": conf, "support": supp})

        edges_df = pd.DataFrame(edge_rows).drop_duplicates(subset=["source", "target"])

        return G, edges_df


    @staticmethod
    def compute_centrality_metrics(G: nx.DiGraph) -> pd.DataFrame:
        """
        Computes network centrality metrics for nodes in a directed graph.

        Evaluates weighted in-degree, weighted out-degree, betweenness centrality,
        and closeness centrality for each node in the network.

        Args:
            G (nx.DiGraph): Directed association graph with weighted edges.

        Returns:
            pd.DataFrame: DataFrame sorted by `'betweenness_centrality'` in
                descending order, containing the following columns:
                - `'node'`: Identifier of the node.
                - `'in_degree_weighted'`: Sum of incoming edge lift weights.
                - `'out_degree_weighted'`: Sum of outgoing edge lift weights.
                - `'betweenness_centrality'`: Normalized betweenness centrality score.
                - `'closeness_centrality'`: Closeness centrality score.
                Returns an empty DataFrame if `G` has no nodes.
        """
        if G.number_of_nodes() == 0:
            return pd.DataFrame()

        in_deg = dict(G.in_degree(weight="lift"))
        out_deg = dict(G.out_degree(weight="lift"))
        betweenness = nx.betweenness_centrality(G, weight="lift", normalized=True)
        closeness = nx.closeness_centrality(G)

        rows = []

        for node in G.nodes():
            rows.append({
                "node": node,
                "in_degree_weighted": float(in_deg.get(node, 0.0)),
                "out_degree_weighted": float(out_deg.get(node, 0.0)),
                "betweenness_centrality": float(betweenness.get(node, 0.0)),
                "closeness_centrality": float(closeness.get(node, 0.0)),
            })

        return pd.DataFrame(rows).sort_values(by="betweenness_centrality", ascending=False)


    @staticmethod
    def compute_continuous_dependencies(df: pd.DataFrame, features: List[str], seed: int = 42) -> pd.DataFrame:
        """
        Computes pairwise continuous multivariate dependencies between features.

        Calculates Spearman's rank correlation coefficient (rho) and non-linear
        mutual information for all unique feature pairs.

        Args:
            df (pd.DataFrame): Dataset containing continuous numerical variables.
            features (List[str]): List of column names to evaluate pairwise.
            seed (int, optional): Random seed used for mutual information
                regression estimation. Defaults to 42.

        Returns:
            pd.DataFrame: DataFrame containing pairwise dependency measures with columns:
                - `'var_x'`: First variable name.
                - `'var_y'`: Second variable name.
                - `'spearman_rho'`: Monotonic correlation coefficient in [-1.0, 1.0].
                - `'mutual_information'`: Estimated non-negative mutual information score.
        """
        rows = []

        for i, col1 in enumerate(features):
            for j, col2 in enumerate(features):
                
                if i >= j:
                    continue
                
                rho, _ = spearmanr(df[col1], df[col2])
                mi = mutual_info_regression(df[[col1]], df[col2], random_state=seed, n_neighbors=5)[0]
                
                rows.append({
                    "var_x": col1,
                    "var_y": col2,
                    "spearman_rho": float(rho),
                    "mutual_information": float(mi),
                })

        return pd.DataFrame(rows)