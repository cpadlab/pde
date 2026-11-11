
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest

import pde.clustering
from pde.clustering import GowerDistanceMetric, KneeLocator, RobustKMedoids


@pytest.mark.unit
class TestClusteringModuleExports:
    """
    Validates package-level exports and __all__ definitions in pde.clustering.
    """

    def test_all_exports(self) -> None:
        """
        Ensures __all__ contains GowerDistanceMetric, RobustKMedoids, and KneeLocator.
        """
        expected = ["GowerDistanceMetric", "RobustKMedoids", "KneeLocator"]
        assert sorted(pde.clustering.__all__) == sorted(expected)
        assert pde.clustering.GowerDistanceMetric is GowerDistanceMetric
        assert pde.clustering.RobustKMedoids is RobustKMedoids
        assert pde.clustering.KneeLocator is KneeLocator


@pytest.mark.unit
class TestGowerDistanceMetric:
    """
    Unit test suite for GowerDistanceMetric range calculation, distance matrices, and serialization.
    """

    def test_init_parameters(self) -> None:
        """
        Verifies default initialization attributes.
        """
        metric = GowerDistanceMetric()
        assert metric.numeric_features == []
        assert metric.categorical_features == []
        assert metric.ranges_ == {}
        assert metric.mins_ == {}
        assert not metric.is_fitted_

    def test_fit_auto_detects_numeric_and_categorical(self) -> None:
        """
        Verifies automatic schema inference when features are not explicitly passed.
        """
        df = pd.DataFrame(
            {
                "num_a": [10.0, 20.0, 30.0],
                "cat_b": ["red", "blue", "green"],
                "num_c": [1, 2, 3],
            }
        )
        metric = GowerDistanceMetric()
        metric.fit(df)

        assert metric.is_fitted_
        assert set(metric.numeric_features) == {"num_a", "num_c"}
        assert set(metric.categorical_features) == {"cat_b"}
        assert np.isclose(metric.ranges_["num_a"], 20.0)
        assert np.isclose(metric.mins_["num_a"], 10.0)

    def test_fit_explicit_features(self) -> None:
        """
        Verifies that explicit feature specifications take precedence.
        """
        df = pd.DataFrame(
            {
                "feat1": [1.0, 5.0],
                "feat2": ["A", "B"],
                "feat3": [100.0, 200.0],
            }
        )
        metric = GowerDistanceMetric(
            numeric_features=["feat1"], categorical_features=["feat2"]
        )
        metric.fit(df)

        assert metric.numeric_features == ["feat1"]
        assert metric.categorical_features == ["feat2"]
        assert "feat3" not in metric.ranges_

    def test_fit_handles_constant_numeric_feature_safe_range(self) -> None:
        """
        Ensures that a constant numeric column has range bounded by 1e-9 to prevent division by zero.
        """
        df = pd.DataFrame({"constant": [42.0, 42.0, 42.0]})
        metric = GowerDistanceMetric().fit(df)
        assert metric.ranges_["constant"] == 1e-9

    def test_compute_distance_matrix_raises_runtime_error_if_not_fitted(self) -> None:
        """
        Ensures calling compute_distance_matrix before fit raises RuntimeError.
        """
        metric = GowerDistanceMetric()
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(RuntimeError, match="must call .fit\\(\\)"):
            metric.compute_distance_matrix(df)

    def test_compute_distance_matrix_raises_value_error_if_no_features_defined(self) -> None:
        """
        Ensures compute_distance_matrix raises ValueError when feature set is empty.
        """
        metric = GowerDistanceMetric()
        metric.is_fitted_ = True
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="no defined characteristics"):
            metric.compute_distance_matrix(df)

    def test_compute_distance_matrix_intra_dataset_symmetry_and_diagonal(self) -> None:
        """
        Verifies square distance matrix properties: D(i, i) = 0 and D(i, j) == D(j, i).
        """
        df = pd.DataFrame(
            {
                "age": [20.0, 40.0, 60.0],
                "city": ["NY", "SF", "NY"],
            }
        )
        metric = GowerDistanceMetric(
            numeric_features=["age"], categorical_features=["city"]
        ).fit(df)

        D = metric.compute_distance_matrix(df)

        assert D.shape == (3, 3)
        assert np.allclose(np.diag(D), 0.0)
        assert np.allclose(D, D.T)
        assert np.all((D >= 0.0) & (D <= 1.0))

        assert np.isclose(D[0, 1], 0.75)

    def test_compute_distance_matrix_inter_dataset_rectangular(self) -> None:
        """
        Verifies distance computation between query set df_a (N=2) and reference set df_b (M=3).
        """
        df_train = pd.DataFrame(
            {
                "val": [0.0, 50.0, 100.0],
                "kind": ["A", "B", "A"],
            }
        )
        metric = GowerDistanceMetric().fit(df_train)

        df_query = pd.DataFrame(
            {
                "val": [0.0, 100.0],
                "kind": ["A", "B"],
            }
        )
        D_inter = metric.compute_distance_matrix(df_query, df_train)

        assert D_inter.shape == (2, 3)
        assert np.isclose(D_inter[0, 0], 0.0)

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        """
        Verifies serialization and faithful reconstruction of GowerDistanceMetric.
        """
        df = pd.DataFrame(
            {
                "score": [10.0, 20.0, 30.0],
                "group": ["x", "y", "x"],
            }
        )
        metric = GowerDistanceMetric().fit(df)
        serialized = metric.to_dict()

        reconstructed = GowerDistanceMetric.from_dict(serialized)
        assert reconstructed.is_fitted_ is True
        assert reconstructed.numeric_features == metric.numeric_features
        assert reconstructed.categorical_features == metric.categorical_features
        assert reconstructed.ranges_ == metric.ranges_
        assert reconstructed.mins_ == metric.mins_

        d_orig = metric.compute_distance_matrix(df)
        d_recon = reconstructed.compute_distance_matrix(df)
        assert np.allclose(d_orig, d_recon)


@pytest.mark.unit
class TestRobustKMedoids:
    """
    Unit test suite for RobustKMedoids k-medoids++ initialization, fitting, and prediction.
    """

    def test_init_parameters(self) -> None:
        """
        Verifies default parameter initialization.
        """
        model = RobustKMedoids(
            n_clusters=4, n_init=5, max_iter=50, random_state=42
        )
        assert model.n_clusters == 4
        assert model.n_init == 5
        assert model.max_iter == 50
        assert model.random_state == 42
        assert model.labels_ is None
        assert model.medoid_indices_ is None
        assert model.inertia_ == float("inf")

    def test_k_medoids_plus_plus_init(self) -> None:
        """
        Verifies that k-medoids++ initialization selects valid, unique medoid indices.
        """
        rng = np.random.default_rng(42)
        dist_matrix = np.array(
            [
                [0.0, 1.0, 5.0, 6.0],
                [1.0, 0.0, 5.0, 6.0],
                [5.0, 5.0, 0.0, 1.0],
                [6.0, 6.0, 1.0, 0.0],
            ]
        )
        medoids = RobustKMedoids._k_medoids_plus_plus_init(
            dist_matrix, n_clusters=2, rng=rng
        )

        assert len(medoids) == 2
        assert len(set(medoids)) == 2
        assert all(0 <= idx < 4 for idx in medoids)

    def test_k_medoids_plus_plus_fallback_on_zero_distances(self) -> None:
        """
        Verifies uniform fallback in _k_medoids_plus_plus_init when distance sum is zero.
        """
        rng = np.random.default_rng(42)
        dist_matrix = np.zeros((4, 4), dtype=float)
        medoids = RobustKMedoids._k_medoids_plus_plus_init(
            dist_matrix, n_clusters=2, rng=rng
        )
        assert len(medoids) == 2

    def test_fit_converges_to_known_clusters(self) -> None:
        """
        Verifies that fitting partitions two well-separated point clusters accurately.
        """
        dist_matrix = np.array(
            [
                [0.0, 0.1, 5.0, 5.1],
                [0.1, 0.0, 5.0, 5.0],
                [5.0, 5.0, 0.0, 0.2],
                [5.1, 5.0, 0.2, 0.0],
            ]
        )
        model = RobustKMedoids(n_clusters=2, n_init=5, random_state=42)
        model.fit(dist_matrix)

        assert model.labels_ is not None
        assert model.medoid_indices_ is not None
        assert len(model.medoid_indices_) == 2
        assert model.inertia_ < 1.0

        assert model.labels_[0] == model.labels_[1]
        assert model.labels_[2] == model.labels_[3]
        assert model.labels_[0] != model.labels_[2]

    def test_predict_raises_runtime_error_if_not_fitted(self) -> None:
        """
        Ensures predict raises RuntimeError if called prior to fit.
        """
        model = RobustKMedoids()
        query_dist = np.array([[0.1, 0.2]])
        with pytest.raises(RuntimeError, match="must be fitted with .fit\\(\\)"):
            model.predict(query_dist)

    def test_predict_inductive_assignment(self) -> None:
        """
        Verifies inductive assignment of query points to learned medoids.
        """
        dist_matrix = np.array(
            [
                [0.0, 0.1, 5.0, 5.0],
                [0.1, 0.0, 5.0, 5.0],
                [5.0, 5.0, 0.0, 0.1],
                [5.0, 5.0, 0.1, 0.0],
            ]
        )
        model = RobustKMedoids(n_clusters=2, n_init=5, random_state=42).fit(dist_matrix)

        dist_query_to_train = np.array(
            [
                [0.05, 0.05, 5.0, 5.0],
                [5.0, 5.0, 0.05, 0.05],
            ]
        )
        clusters, min_dists = model.predict(dist_query_to_train)

        assert len(clusters) == 2
        assert len(min_dists) == 2
        assert clusters[0] != clusters[1]
        assert np.all(min_dists < 0.2)


@pytest.mark.unit
class TestKneeLocator:
    """
    Unit test suite for KneeLocator geometric elbow point detection.
    """

    def test_detect_knee_point_raises_value_error_if_fewer_than_three_points(self) -> None:
        """
        Ensures detect_knee_point raises ValueError if len(k_values) < 3.
        """
        with pytest.raises(ValueError, match="At least 3 values of k"):
            KneeLocator.detect_knee_point([2, 3], [100.0, 50.0])

    def test_detect_knee_point_identifies_clear_elbow(self) -> None:
        """
        Verifies identification of an obvious elbow point on an inertia curve.
        """
        k_values = [2, 3, 4, 5, 6]
        inertias = [100.0, 20.0, 15.0, 12.0, 10.0]

        best_k, distances = KneeLocator.detect_knee_point(k_values, inertias)

        assert best_k == 3
        assert len(distances) == 5
        assert np.isclose(distances[0], 0.0)
        assert np.isclose(distances[-1], 0.0)
        assert distances[1] == max(distances)

    def test_detect_knee_point_collinear_curve(self) -> None:
        """
        Verifies that for a straight line, all orthogonal distances are 0.0.
        """
        k_values = [1, 2, 3, 4]
        inertias = [10.0, 8.0, 6.0, 4.0]

        best_k, distances = KneeLocator.detect_knee_point(k_values, inertias)
        assert np.allclose(distances, 0.0)
        assert best_k in k_values
