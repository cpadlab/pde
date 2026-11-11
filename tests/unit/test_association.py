
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import pytest

import pde.association
from pde.association import (
    AssociationRuleScorer,
    DependencyGraphEngine,
    QuantileDiscretizer,
)


@pytest.mark.unit
class TestAssociationModuleExports:
    """
    Validates package-level exports and __all__ definitions in pde.association.
    """

    def test_all_exports(self) -> None:
        """
        Ensures __all__ contains QuantileDiscretizer, AssociationRuleScorer, and DependencyGraphEngine.
        """
        expected = [
            "QuantileDiscretizer",
            "AssociationRuleScorer",
            "DependencyGraphEngine",
        ]
        assert sorted(pde.association.__all__) == sorted(expected)
        assert pde.association.QuantileDiscretizer is QuantileDiscretizer
        assert pde.association.AssociationRuleScorer is AssociationRuleScorer
        assert pde.association.DependencyGraphEngine is DependencyGraphEngine


@pytest.mark.unit
class TestQuantileDiscretizer:
    """
    Unit test suite for QuantileDiscretizer fitting, transformation, and serialization.
    """

    def test_init_default_parameters(self) -> None:
        """
        Verifies default initialization parameters and initial state.
        """
        discretizer = QuantileDiscretizer()
        assert discretizer.n_bins == 3
        assert discretizer.labels == ["Low", "Medium", "High"]
        assert discretizer.bin_edges_ == {}
        assert discretizer.feature_names_ == []
        assert not discretizer.is_fitted_

    def test_init_custom_labels_and_bins(self) -> None:
        """
        Verifies custom bins count and custom label list.
        """
        labels = ["VeryLow", "Low", "Neutral", "High", "VeryHigh"]
        discretizer = QuantileDiscretizer(n_bins=5, labels=labels)
        assert discretizer.n_bins == 5
        assert discretizer.labels == labels

    def test_init_raises_value_error_on_label_mismatch(self) -> None:
        """
        Verifies that a ValueError is raised if the number of labels does not match n_bins.
        """
        with pytest.raises(ValueError, match="must match n_bins"):
            QuantileDiscretizer(n_bins=4, labels=["A", "B", "C"])

    def test_fit_auto_detects_numeric_columns(self) -> None:
        """
        Ensures that if features=None, all numeric columns are automatically selected and non-numeric ignored.
        """
        df = pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "cat_col": ["a", "b", "c", "d", "e", "f"],
                "num2": [10, 20, 30, 40, 50, 60],
            }
        )
        discretizer = QuantileDiscretizer(n_bins=3)
        discretizer.fit(df)

        assert discretizer.is_fitted_
        assert discretizer.feature_names_ == ["num1", "num2"]
        assert "num1" in discretizer.bin_edges_
        assert "num2" in discretizer.bin_edges_

        edges1 = discretizer.bin_edges_["num1"]
        assert edges1[0] == -np.inf
        assert edges1[-1] == np.inf
        assert len(edges1) == 4

    def test_fit_explicit_features(self) -> None:
        """
        Ensures that explicit feature selection only fits specified columns.
        """
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        discretizer = QuantileDiscretizer(n_bins=2, labels=["Low", "High"])
        discretizer.fit(df, features=["b"])

        assert discretizer.feature_names_ == ["b"]
        assert "b" in discretizer.bin_edges_
        assert "a" not in discretizer.bin_edges_

    def test_fit_fallback_on_collapsed_quantiles(self) -> None:
        """
        Verifies that when quantiles collapse due to identical values,
        the fallback to np.linspace(min, max, n_bins + 1) executes cleanly.
        """
        df = pd.DataFrame({"constant_val": [5.0] * 50 + [10.0]})
        discretizer = QuantileDiscretizer(n_bins=3)
        discretizer.fit(df)

        edges = discretizer.bin_edges_["constant_val"]
        assert len(edges) == 4
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf
        assert edges[1] > -np.inf
        assert edges[2] < np.inf

    def test_transform_raises_runtime_error_if_not_fitted(self) -> None:
        """
        Ensures calling transform before fit raises a RuntimeError.
        """
        discretizer = QuantileDiscretizer()
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="call .fit\\(\\) before transforming"):
            discretizer.transform(df)

    def test_transform_produces_boolean_dummies(self) -> None:
        """
        Verifies that transform generates one-hot encoded boolean columns for each bin label.
        """
        df = pd.DataFrame({"x": [10.0, 50.0, 90.0]})
        discretizer = QuantileDiscretizer(n_bins=3, labels=["L", "M", "H"])
        discretizer.fit(df)
        transformed = discretizer.transform(df)

        expected_columns = ["x_L", "x_M", "x_H"]
        assert list(transformed.columns) == expected_columns
        assert transformed.dtypes["x_L"] == bool
        assert transformed.dtypes["x_M"] == bool
        assert transformed.dtypes["x_H"] == bool

        row_sums = transformed.sum(axis=1).to_numpy()
        assert np.all(row_sums == 1)

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        """
        Verifies complete serialization and reconstruction fidelity.
        """
        df = pd.DataFrame({"signal": np.linspace(0, 100, 30)})
        discretizer = QuantileDiscretizer(n_bins=3, labels=["L", "M", "H"])
        discretizer.fit(df)

        serialized = discretizer.to_dict()
        assert serialized["n_bins"] == 3
        assert serialized["labels"] == ["L", "M", "H"]
        assert serialized["is_fitted"] is True
        assert isinstance(serialized["bin_edges"]["signal"], list)

        reconstructed = QuantileDiscretizer.from_dict(serialized)
        assert reconstructed.is_fitted_ is True
        assert reconstructed.n_bins == discretizer.n_bins
        assert reconstructed.labels == discretizer.labels
        assert reconstructed.feature_names_ == discretizer.feature_names_
        assert np.allclose(
            reconstructed.bin_edges_["signal"],
            discretizer.bin_edges_["signal"],
        )

        orig_out = discretizer.transform(df)
        recon_out = reconstructed.transform(df)
        pd.testing.assert_frame_equal(orig_out, recon_out)


@pytest.mark.unit
class TestAssociationRuleScorer:
    """
    Unit test suite for AssociationRuleScorer mining, violation scoring, and explainability.
    """

    def test_init_parameters(self) -> None:
        """
        Verifies default parameter initialization.
        """
        scorer = AssociationRuleScorer(
            min_support=0.1, min_confidence=0.6, min_lift=1.2, max_len=2
        )
        assert scorer.min_support == 0.1
        assert scorer.min_confidence == 0.6
        assert scorer.min_lift == 1.2
        assert scorer.max_len == 2
        assert scorer.rules_df_ is None
        assert not scorer.is_fitted_

    def test_fit_empty_frequent_itemsets(self) -> None:
        """
        Ensures that if no itemset meets min_support, rules_df_ is set to empty DataFrame.
        """
        transactions = pd.DataFrame(
            {
                "A": [True] + [False] * 99,
                "B": [False] * 100,
            }
        )
        scorer = AssociationRuleScorer(min_support=0.50)
        scorer.fit(transactions)

        assert scorer.is_fitted_
        assert isinstance(scorer.rules_df_, pd.DataFrame)
        assert scorer.rules_df_.empty

    def test_fit_no_rules_above_confidence(self) -> None:
        """
        Ensures that if frequent itemsets exist but none meet min_confidence,
        rules_df_ is empty and fitted flag is set.
        """
        rng = np.random.default_rng(42)
        transactions = pd.DataFrame(
            rng.choice([True, False], size=(100, 4)),
            columns=["item_1", "item_2", "item_3", "item_4"],
        )
        scorer = AssociationRuleScorer(
            min_support=0.01, min_confidence=0.999, min_lift=1.0
        )
        scorer.fit(transactions)

        assert scorer.is_fitted_
        assert isinstance(scorer.rules_df_, pd.DataFrame)
        assert scorer.rules_df_.empty

    def test_fit_and_rule_extraction_success(self) -> None:
        """
        Verifies successful rule discovery with metadata string columns.
        """
        data = {
            "apple": [True] * 40 + [False] * 10,
            "banana": [True] * 40 + [False] * 10,
            "milk": [False] * 40 + [True] * 10,
        }
        transactions = pd.DataFrame(data)

        scorer = AssociationRuleScorer(
            min_support=0.20, min_confidence=0.70, min_lift=1.05
        )
        scorer.fit(transactions)

        assert scorer.is_fitted_
        assert not scorer.rules_df_.empty
        assert "antecedents_str" in scorer.rules_df_.columns
        assert "consequents_str" in scorer.rules_df_.columns
        assert "rule_repr" in scorer.rules_df_.columns

        rule_reprs = scorer.rules_df_["rule_repr"].tolist()
        assert any("apple => banana" in r or "banana => apple" in r for r in rule_reprs)

    def test_score_raises_runtime_error_if_not_fitted(self) -> None:
        """
        Ensures score raises a RuntimeError if called before fit.
        """
        scorer = AssociationRuleScorer()
        df = pd.DataFrame({"item_a": [True, False]})
        with pytest.raises(RuntimeError, match="call .fit\\(\\) before scoring"):
            scorer.score(df)

    def test_score_with_empty_rules_df(self) -> None:
        """
        Verifies that an empty rules model returns zero scores and empty explanation lists.
        """
        scorer = AssociationRuleScorer(min_support=0.99)
        transactions = pd.DataFrame({"A": [True, False], "B": [False, True]})
        scorer.fit(transactions)

        scores, explanations = scorer.score(transactions)
        assert len(scores) == 2
        assert np.all(scores == 0.0)
        assert explanations == [[], []]

    def test_score_and_explanations_detection(self) -> None:
        """
        Verifies that a record with antecedent present but consequent absent
        receives a non-zero violation score and an informative explanation.
        """
        train_data = pd.DataFrame(
            {
                "login": [True] * 40 + [False] * 5,
                "auth": [True] * 40 + [False] * 5,
            }
        )
        scorer = AssociationRuleScorer(
            min_support=0.50, min_confidence=0.80, min_lift=1.01
        )
        scorer.fit(train_data)

        test_data = pd.DataFrame(
            {
                "login": [True, True, False],
                "auth": [True, False, False],
            }
        )
        scores, explanations = scorer.score(test_data)

        assert scores[0] == 0.0
        assert explanations[0] == []

        assert scores[1] > 0.0
        assert len(explanations[1]) > 0
        assert any("login => auth" in r for r in explanations[1])

        assert scores[2] == 0.0
        assert explanations[2] == []

    def test_score_with_missing_columns_in_test(self) -> None:
        """
        Verifies robustness when test data does not contain all rule columns.
        """
        train_data = pd.DataFrame(
            {
                "A": [True] * 30 + [False] * 10,
                "B": [True] * 30 + [False] * 10,
            }
        )
        scorer = AssociationRuleScorer(
            min_support=0.5, min_confidence=0.8, min_lift=1.01
        )
        scorer.fit(train_data)

        test_missing_col = pd.DataFrame({"A": [True, False]})
        scores, explanations = scorer.score(test_missing_col)

        assert len(scores) == 2
        assert isinstance(explanations, list)


@pytest.mark.unit
class TestDependencyGraphEngine:
    """
    Unit test suite for DependencyGraphEngine graph construction, centrality, and dependencies.
    """

    def test_build_association_graph_empty_rules(self) -> None:
        """
        Verifies handling of an empty rules DataFrame.
        """
        empty_df = pd.DataFrame()
        G, edges_df = DependencyGraphEngine.build_association_graph(empty_df)

        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0
        assert edges_df.empty

    def test_build_association_graph_with_valid_rules(self) -> None:
        """
        Verifies graph construction, edge weights, and deduplication.
        """
        rules_data = pd.DataFrame(
            {
                "antecedents": [frozenset(["A"]), frozenset(["B"]), frozenset(["A"])],
                "consequents": [frozenset(["B"]), frozenset(["C"]), frozenset(["B"])],
                "lift": [2.5, 1.8, 3.0],
                "confidence": [0.8, 0.7, 0.9],
                "support": [0.4, 0.3, 0.5],
            }
        )
        G, edges_df = DependencyGraphEngine.build_association_graph(rules_data)

        assert G.has_edge("A", "B")
        assert G.has_edge("B", "C")
        assert not G.has_edge("C", "A")

        assert G["A"]["B"]["lift"] == 3.0
        assert G["A"]["B"]["confidence"] == 0.9

        assert len(edges_df) == 2
        assert set(edges_df["source"]) == {"A", "B"}
        assert set(edges_df["target"]) == {"B", "C"}

    def test_build_association_graph_ignores_self_loops(self) -> None:
        """
        Verifies that rules where antecedent item equals consequent item (u == v) are ignored.
        """
        rules_data = pd.DataFrame(
            {
                "antecedents": [frozenset(["A"])],
                "consequents": [frozenset(["A"])],
                "lift": [2.0],
                "confidence": [0.9],
                "support": [0.5],
            }
        )
        G, edges_df = DependencyGraphEngine.build_association_graph(rules_data)

        assert not G.has_edge("A", "A")
        assert G.number_of_edges() == 0
        assert edges_df.empty

    def test_compute_centrality_metrics_empty_graph(self) -> None:
        """
        Verifies that an empty graph returns an empty DataFrame.
        """
        G = nx.DiGraph()
        df_centrality = DependencyGraphEngine.compute_centrality_metrics(G)
        assert isinstance(df_centrality, pd.DataFrame)
        assert df_centrality.empty

    def test_compute_centrality_metrics_valid_graph(self) -> None:
        """
        Verifies calculation of betweenness, in-degree, and out-degree centrality.
        """
        G = nx.DiGraph()
        G.add_edge("A", "B", lift=2.0, confidence=0.8, support=0.4)
        G.add_edge("B", "C", lift=3.0, confidence=0.9, support=0.5)

        centrality_df = DependencyGraphEngine.compute_centrality_metrics(G)

        assert not centrality_df.empty
        assert set(centrality_df.columns) == {
            "node",
            "in_degree_weighted",
            "out_degree_weighted",
            "betweenness_centrality",
            "closeness_centrality",
        }
        assert len(centrality_df) == 3

        node_b_row = centrality_df[centrality_df["node"] == "B"].iloc[0]
        assert node_b_row["in_degree_weighted"] == 2.0
        assert node_b_row["out_degree_weighted"] == 3.0
        assert node_b_row["betweenness_centrality"] > 0.0

    def test_compute_continuous_dependencies_pairwise(self) -> None:
        """
        Verifies pairwise Spearman rho and mutual information calculation.
        """
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + rng.normal(0, 0.5, 50)
        z = rng.normal(0, 1.0, 50)

        df = pd.DataFrame({"x": x, "y": y, "z": z})
        features = ["x", "y", "z"]

        deps_df = DependencyGraphEngine.compute_continuous_dependencies(
            df, features, seed=42
        )

        assert len(deps_df) == 3
        expected_cols = ["var_x", "var_y", "spearman_rho", "mutual_information"]
        assert list(deps_df.columns) == expected_cols

        xy_row = deps_df[(deps_df["var_x"] == "x") & (deps_df["var_y"] == "y")].iloc[0]
        assert xy_row["spearman_rho"] > 0.90
        assert xy_row["mutual_information"] > 0.50

    def test_compute_continuous_dependencies_fewer_than_two_features(self) -> None:
        """
        Verifies that providing fewer than 2 features returns an empty dependencies DataFrame.
        """
        df = pd.DataFrame({"single": [1.0, 2.0, 3.0]})
        deps_df = DependencyGraphEngine.compute_continuous_dependencies(
            df, ["single"], seed=42
        )
        assert isinstance(deps_df, pd.DataFrame)
        assert deps_df.empty
