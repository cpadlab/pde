
from typing import Any, Dict, List, Optional, Tuple
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
    def _upsert_edge(graph: nx.DiGraph, u: Any, v: Any, lift: float, conf: float, supp: float) -> None:
        """
        Inserts a directed edge or updates its attributes if a higher lift is observed.

        Checks whether an edge from node `u` to node `v` already exists in `graph`.
        If present, updates its `'lift'`, `'confidence'`, and `'support'` attributes
        only when the incoming `lift` exceeds the existing edge's lift. If the edge
        does not exist, adds it with the specified metrics.

        Args:
            graph (nx.DiGraph): Target directed graph to modify in place.
            u (Any): Source node identifier representing the rule antecedent.
            v (Any): Target node identifier representing the rule consequent.
            lift (float): Association rule lift value.
            conf (float): Association rule confidence value.
            supp (float): Association rule support value.
        """
        if graph.has_edge(u, v):
            if lift > graph[u][v]["lift"]:
                graph[u][v]["lift"] = lift
                graph[u][v]["confidence"] = conf
                graph[u][v]["support"] = supp
        else:
            graph.add_edge(u, v, lift=lift, confidence=conf, support=supp)


    @classmethod
    def _process_rule_edges(
        cls,
        graph: nx.DiGraph,
        edge_rows: List[Dict[str, Any]],
        rule_series: pd.Series,
    ) -> None:
        """
        Processes antecedent-consequent pairs for a single association rule.

        Iterates through the Cartesian product of antecedents and consequents
        for a given rule, discarding self-loops (`u == v`), updating the graph
        via `_upsert_edge`, and appending an edge attribute dictionary to
        `edge_rows`.

        Args:
            graph (nx.DiGraph): Target directed association graph modified in place.
            edge_rows (List[Dict[str, Any]]): Accumulator list of edge dictionary records
                to which each valid directed connection is appended.
            rule_series (pd.Series): Pandas Series containing rule attributes with expected
                keys `'antecedents'`, `'consequents'`, `'lift'`, `'confidence'`, and
                `'support'`.
        """
        lift = float(rule_series["lift"])
        conf = float(rule_series["confidence"])
        supp = float(rule_series["support"])

        for u in rule_series["antecedents"]:
            for v in rule_series["consequents"]:
                if u == v:
                    continue
                cls._upsert_edge(graph, u, v, lift, conf, supp)
                edge_rows.append({"source": u, "target": v, "lift": lift, "confidence": conf, "support": supp})


    @classmethod
    def build_association_graph(cls, rules_df: pd.DataFrame) -> Tuple[nx.DiGraph, pd.DataFrame]:
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
                - **graph** (`nx.DiGraph`): Directed network graph where nodes represent
                  items and edges store association metric attributes.
                - **edges_df** (`pd.DataFrame`): DataFrame containing deduplicated edge
                  records with columns `'source'`, `'target'`, `'lift'`,
                  `'confidence'`, and `'support'`.
        """
        graph = nx.DiGraph()
        edge_rows: List[Dict[str, Any]] = []

        if rules_df.empty:
            return graph, pd.DataFrame()

        for _, r in rules_df.iterrows():
            cls._process_rule_edges(graph, edge_rows, r)

        edges_df = pd.DataFrame(edge_rows).drop_duplicates(subset=["source", "target"])

        return graph, edges_df


    @staticmethod
    def compute_centrality_metrics(graph: nx.DiGraph) -> pd.DataFrame:
        """
        Computes network centrality metrics for nodes in a directed graph.

        Evaluates weighted in-degree, weighted out-degree, betweenness centrality,
        and closeness centrality for each node in the network.

        Args:
            graph (nx.DiGraph): Directed association graph with weighted edges.

        Returns:
            pd.DataFrame: DataFrame sorted by `'betweenness_centrality'` in
                descending order, containing the following columns:
                - `'node'`: Identifier of the node.
                - `'in_degree_weighted'`: Sum of incoming edge lift weights.
                - `'out_degree_weighted'`: Sum of outgoing edge lift weights.
                - `'betweenness_centrality'`: Normalized betweenness centrality score.
                - `'closeness_centrality'`: Closeness centrality score.
                Returns an empty DataFrame if `graph` has no nodes.
        """
        if graph.number_of_nodes() == 0:
            return pd.DataFrame()

        in_deg = dict(graph.in_degree(weight="lift"))
        out_deg = dict(graph.out_degree(weight="lift"))
        betweenness = nx.betweenness_centrality(graph, weight="lift", normalized=True)
        closeness = nx.closeness_centrality(graph)

        rows = []

        for node in graph.nodes():
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