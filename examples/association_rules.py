"""
ASSOCIATION RULES, QUANTILE DISCRETIZATION, AND DEPENDENCY NETWORKS.

Demonstrates symbolic co-occurrence discovery, rule-violation contextual anomaly
scoring, and topological directed dependency modeling from continuous multimodal spaces.

Evaluated workflow:
1. Multimodal synthetic continuous data generation with physical feature correlations.
2. Leakage-free quantile discretization strictly fitted on training data (QuantileDiscretizer).
3. Frequent association rule mining (Apriori) pruned by minimum confidence and lift.
4. Logical rule violation scoring (AssociationRuleScorer) and diagnostic explainability.
5. Contextual anomaly detection bypassing pure distance-based metrics.
6. Directed dependency graph construction and network centrality profiling (DependencyGraphEngine).
7. Continuous non-parametric dependency analysis (Spearman rho and mutual information).
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from pde.association.discretizer import QuantileDiscretizer
from pde.association.graph import DependencyGraphEngine
from pde.association.rules import AssociationRuleScorer


def generate_association_data(
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates multimodal continuous datasets alongside contextual logic anomalies.

    Simulates normal observations drawn from three correlated clusters alongside
    contextual anomalies whose individual feature values are nominal but whose
    joint association violates domain-level co-occurrence logic (e.g., low sessions
    paired with extremely high duration and action counts).

    Args:
        seed (int, optional): Random seed for reproducible generation. Defaults
            to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - **df_train**: 360 in-distribution normal training observations.
            - **df_test_norm**: 90 in-distribution normal test observations.
            - **df_anom**: 15 contextual logic anomaly observations.
    """
    centers = [[10.0, 100.0, 20.0], [50.0, 500.0, 100.0], [90.0, 900.0, 180.0]]
    stds = [[3.0, 25.0, 5.0], [5.0, 40.0, 10.0], [4.0, 35.0, 8.0]]

    X_normal, _ = make_blobs(
        n_samples=450,
        centers=centers,
        cluster_std=stds,
        random_state=seed,
    )
    df_train = pd.DataFrame(
        X_normal[:360], columns=["sessions", "duration", "actions"]
    )
    df_test_norm = pd.DataFrame(
        X_normal[360:], columns=["sessions", "duration", "actions"]
    )

    rng = np.random.default_rng(seed + 303)
    anom_contextual = np.column_stack(
        [
            rng.normal(10.0, 3.0, 15),
            rng.normal(900.0, 35.0, 15),
            rng.normal(180.0, 9.0, 15),
        ]
    )
    df_anom = pd.DataFrame(
        anom_contextual, columns=["sessions", "duration", "actions"]
    )

    return df_train, df_test_norm, df_anom


def main() -> None:
    """
    Executes the association rule mining and dependency network workflow.
    """
    print("Association Rule Mining And Dependency Networks")
    print("=" * 48)

    print("\n[INFO] Generating Multimodal Continuous Data And Contextual Anomalies...")
    df_train, df_test_norm, df_anom = generate_association_data(seed=42)
    df_test_all = pd.concat([df_test_norm, df_anom], ignore_index=True)

    print(f"\t• Normal training samples:     {len(df_train)}")
    print(f"\t• Normal test samples:         {len(df_test_norm)}")
    print(f"\t• Contextual anomaly samples:  {len(df_anom)}")

    print("\n[INFO] Discretizing Continuous Features Strictly On Training Data...")
    discretizer = QuantileDiscretizer(n_bins=3, labels=["Low", "Medium", "High"])
    discretizer.fit(df_train)

    train_trans = discretizer.transform(df_train)
    test_trans = discretizer.transform(df_test_all)

    print(f"\t• Quantile bins for 'duration': {np.round(discretizer.bin_edges_['duration'], 2)}")
    print(f"\t• Quantile bins for 'actions':  {np.round(discretizer.bin_edges_['actions'], 2)}")
    print(f"\t• Generated transactional columns: {train_trans.columns.tolist()[:5]}...")

    print("\n[INFO] Mining Frequent Association Rules And Pruning By Lift...")
    rule_scorer = AssociationRuleScorer(
        min_support=0.05,
        min_confidence=0.50,
        min_lift=1.10,
        max_len=3,
    )
    rule_scorer.fit(train_trans)
    rules_df = rule_scorer.rules_df_

    print(f"\t• Extracted valid rules meeting criteria: {len(rules_df)}")
    print("\n\t• Top 5 association rules ranked by lift:")
    top_rules = rules_df.nlargest(5, "lift")
    for _, r in top_rules.iterrows():
        print(
            f"\t  - {r['rule_repr']:42s} | Lift: {r['lift']:.2f} | "
            f"Conf: {r['confidence']:.2f} | Supp: {r['support']:.2f}"
        )

    print("\n[INFO] Evaluating Rule Violations And Diagnostic Explainability...")
    scores, explanations = rule_scorer.score(test_trans)

    norm_scores = scores[:len(df_test_norm)]
    anom_scores = scores[len(df_test_norm):]

    print(f"\t• Mean violation score (normal samples):     {norm_scores.mean():.4f}")
    print(f"\t• Mean violation score (contextual anomalies): {anom_scores.mean():.4f}")

    print("\n\t• Contextual anomaly explainability diagnostics (sample detections):")
    for i in range(len(df_test_norm), len(df_test_norm) + 3):
        sample = df_test_all.iloc[i]
        broken_rules = explanations[i]
        print(
            f"\n\t  Record #{i} [sessions={sample['sessions']:.1f}, "
            f"duration={sample['duration']:.1f}, actions={sample['actions']:.1f}]:"
        )
        print(f"\t  • Violation score: {scores[i]:.4f}")
        print(f"\t  • Broken rules ({len(broken_rules)}):")
        for br in broken_rules[:2]:
            print(f"\t    - Violated: {br}")

    print("\n[INFO] Constructing Directed Dependency Graph And Centrality Analysis...")
    G, edge_df = DependencyGraphEngine.build_association_graph(rules_df)
    centrality_df = DependencyGraphEngine.compute_centrality_metrics(G)

    print(f"\t• Association graph nodes:        {G.number_of_nodes()}")
    print(f"\t• Directed implication edges:     {G.number_of_edges()}")

    print("\n\t• Top 4 nodes ranked by betweenness centrality:")
    for _, row in centrality_df.head(4).iterrows():
        print(
            f"\t  - Node: {row['node']:18s} | Betweenness: {row['betweenness_centrality']:.4f} | "
            f"In-Degree: {row['in_degree_weighted']:5.1f} | Out-Degree: {row['out_degree_weighted']:5.1f}"
        )

    print("\n[INFO] Continuous Multivariate Dependencies (Spearman Rho & Mutual Information):")
    cont_deps = DependencyGraphEngine.compute_continuous_dependencies(
        df_train, ["sessions", "duration", "actions"]
    )
    for _, cd in cont_deps.iterrows():
        print(
            f"\t• Pair {cd['var_x']:8s} - {cd['var_y']:8s} | "
            f"Spearman rho = {cd['spearman_rho']:.4f} | "
            f"Mutual Info = {cd['mutual_information']:.4f}"
        )


if __name__ == "__main__":
    main()
