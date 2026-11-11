
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest

import pde.core
from pde.core import AutoPipelineRouter, MetaDataProfiler


@pytest.mark.unit
class TestCoreModuleExports:
    """
    Validates package-level exports and __all__ definitions in pde.core.
    """

    def test_all_exports(self) -> None:
        """
        Ensures __all__ contains MetaDataProfiler and AutoPipelineRouter.
        """
        expected = ["MetaDataProfiler", "AutoPipelineRouter"]
        assert sorted(pde.core.__all__) == sorted(expected)
        assert pde.core.MetaDataProfiler is MetaDataProfiler
        assert pde.core.AutoPipelineRouter is AutoPipelineRouter


@pytest.mark.unit
class TestMetaDataProfiler:
    """
    Unit test suite for MetaDataProfiler heuristic profiling across all data modalities.
    """

    def test_profile_sequential_temporal_fixture(
        self,
        sequential_temporal_data: Tuple[
            List[Tuple[List[int], List[float]]],
            List[Tuple[List[int], List[float]]],
            List[Tuple[List[int], List[float], str]],
        ],
    ) -> None:
        """
        Verifies profiling of sequential temporal trajectories using the shared conftest fixture.
        """
        train_seqs, _, _ = sequential_temporal_data
        profile = MetaDataProfiler.profile(train_seqs)

        assert profile["data_modality"] == "sequential_temporal"
        assert profile["n_samples"] == len(train_seqs)
        assert profile["n_features"] == 2
        assert profile["numeric_features"] == []
        assert profile["categorical_features"] == []
        assert profile["recommended_pipeline"] == "Semi_Markov_Conformal_Engine"

    def test_profile_sequential_discrete_fixture(
        self,
        sequential_discrete_data: Tuple[List[List[int]], List[List[int]]],
    ) -> None:
        """
        Verifies profiling of discrete sequential sequences using the shared conftest fixture.
        """
        train_events, _ = sequential_discrete_data
        profile = MetaDataProfiler.profile(train_events)

        assert profile["data_modality"] == "sequential_discrete"
        assert profile["n_samples"] == len(train_events)
        assert profile["n_features"] == 1
        assert profile["numeric_features"] == []
        assert profile["categorical_features"] == []
        assert profile["recommended_pipeline"] == "Semi_Markov_Conformal_Engine"

    def test_profile_continuous_tabular_fixture(
        self,
        continuous_tabular_splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """
        Verifies profiling of continuous tabular data using the shared conftest fixture.
        """
        df_train, _, _ = continuous_tabular_splits
        profile = MetaDataProfiler.profile(df_train)

        assert profile["data_modality"] == "tabular_continuous"
        assert profile["n_samples"] == len(df_train)
        assert profile["n_features"] == df_train.shape[1]
        assert profile["numeric_features"] == list(df_train.columns)
        assert profile["categorical_features"] == []
        assert profile["recommended_pipeline"] == "Continuous_Conformal_Geometric_Engine"

    def test_profile_heterogeneous_tabular_fixture(
        self,
        heterogeneous_tabular_data: Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]],
    ) -> None:
        """
        Verifies profiling of mixed-type tabular data using the shared conftest fixture.
        """
        train_df, _, feature_types = heterogeneous_tabular_data
        profile = MetaDataProfiler.profile(train_df)

        assert profile["data_modality"] == "tabular_heterogeneous"
        assert profile["n_samples"] == len(train_df)
        assert profile["n_features"] == train_df.shape[1]
        assert set(profile["numeric_features"]) == set(feature_types["numeric"])
        assert set(profile["categorical_features"]) == set(feature_types["categorical"])
        assert profile["recommended_pipeline"] == "Heterogeneous_Gower_KMedoids_Engine"

    def test_profile_sequential_discrete_tuple_single_element(self) -> None:
        """
        Verifies that a tuple with length != 2 is classified as sequential_discrete.
        """
        data = [
            ([0, 1, 2],),
            ([1, 2],),
        ]
        profile = MetaDataProfiler.profile(data)

        assert profile["data_modality"] == "sequential_discrete"
        assert profile["n_samples"] == 2
        assert profile["n_features"] == 1
        assert profile["recommended_pipeline"] == "Semi_Markov_Conformal_Engine"

    def test_profile_sequential_discrete_tuple_three_elements(self) -> None:
        """
        Verifies that a tuple with 3 elements (len != 2) defaults to sequential_discrete.
        """
        data = [
            ([0, 1], [1.0, 2.0], "label_a"),
            ([1, 2], [2.0, 3.0], "label_b"),
        ]
        profile = MetaDataProfiler.profile(data)

        assert profile["data_modality"] == "sequential_discrete"
        assert profile["n_samples"] == 2
        assert profile["n_features"] == 1
        assert profile["recommended_pipeline"] == "Semi_Markov_Conformal_Engine"

    def test_profile_tabular_continuous_small_sample_low_cardinality(self) -> None:
        """
        Verifies that low cardinality (<= 5) does NOT trigger categorical reclassification
        when sample size <= 50 (e.g. n_rows = 20).
        """
        n_samples = 20
        df = pd.DataFrame(
            {
                "flag_num": [0, 1] * 10,
                "score": np.linspace(1.0, 5.0, n_samples),
            }
        )
        profile = MetaDataProfiler.profile(df)

        assert profile["data_modality"] == "tabular_continuous"
        assert profile["n_samples"] == 20
        assert profile["n_features"] == 2
        assert "flag_num" in profile["numeric_features"]
        assert profile["categorical_features"] == []
        assert profile["recommended_pipeline"] == "Continuous_Conformal_Geometric_Engine"

    def test_profile_tabular_heterogeneous_low_cardinality_numeric_heuristic(
        self,
        rng: np.random.Generator,
    ) -> None:
        """
        Verifies that discrete numeric columns with <= 5 unique values when n_rows > 50
        are reclassified as categorical.
        """
        n_samples = 70
        df = pd.DataFrame(
            {
                "continuous_signal": rng.uniform(0.0, 100.0, n_samples),
                "discrete_code": rng.choice([1, 2, 3], size=n_samples),
                "binary_indicator": rng.choice([0.0, 1.0], size=n_samples),
            }
        )
        profile = MetaDataProfiler.profile(df)

        assert profile["data_modality"] == "tabular_heterogeneous"
        assert profile["n_samples"] == 70
        assert profile["n_features"] == 3
        assert profile["numeric_features"] == ["continuous_signal"]
        assert set(profile["categorical_features"]) == {"discrete_code", "binary_indicator"}
        assert profile["recommended_pipeline"] == "Heterogeneous_Gower_KMedoids_Engine"

    def test_profile_numpy_array_input(self) -> None:
        """
        Verifies that a 2D NumPy array is safely cast to DataFrame and profiled.
        """
        arr = np.array(
            [
                [1.2, 3.4, 5.6],
                [7.8, 9.0, 1.2],
                [3.4, 5.6, 7.8],
            ]
        )
        profile = MetaDataProfiler.profile(arr)

        assert profile["data_modality"] == "tabular_continuous"
        assert profile["n_samples"] == 3
        assert profile["n_features"] == 3
        assert profile["numeric_features"] == ["0", "1", "2"]
        assert profile["categorical_features"] == []
        assert profile["recommended_pipeline"] == "Continuous_Conformal_Geometric_Engine"

    def test_profile_dictionary_input(self) -> None:
        """
        Verifies that a dictionary of lists/arrays is safely converted and profiled.
        """
        data = {
            "x": [1.0, 2.0, 3.0, 4.0],
            "city": ["London", "Paris", "Berlin", "Madrid"],
        }
        profile = MetaDataProfiler.profile(data)

        assert profile["data_modality"] == "tabular_heterogeneous"
        assert profile["n_samples"] == 4
        assert profile["n_features"] == 2
        assert profile["numeric_features"] == ["x"]
        assert profile["categorical_features"] == ["city"]
        assert profile["recommended_pipeline"] == "Heterogeneous_Gower_KMedoids_Engine"

    def test_profile_list_of_dictionaries_fallback(self) -> None:
        """
        Verifies that a list of records (where first item is dict, not tuple/list)
        converts to DataFrame rather than sequential modality.
        """
        records = [
            {"val": 10.0, "status": "ok"},
            {"val": 20.0, "status": "alert"},
        ]
        profile = MetaDataProfiler.profile(records)

        assert profile["data_modality"] == "tabular_heterogeneous"
        assert profile["n_samples"] == 2
        assert profile["n_features"] == 2
        assert profile["numeric_features"] == ["val"]
        assert profile["categorical_features"] == ["status"]
        assert profile["recommended_pipeline"] == "Heterogeneous_Gower_KMedoids_Engine"

    def test_profile_empty_list_fallback(self) -> None:
        """
        Verifies handling of an empty list, falling back to an empty DataFrame.
        """
        profile = MetaDataProfiler.profile([])

        assert profile["data_modality"] == "tabular_continuous"
        assert profile["n_samples"] == 0
        assert profile["n_features"] == 0
        assert profile["numeric_features"] == []
        assert profile["categorical_features"] == []
        assert profile["recommended_pipeline"] == "Continuous_Conformal_Geometric_Engine"


@pytest.mark.unit
class TestAutoPipelineRouter:
    """
    Unit test suite for AutoPipelineRouter dispatching and state persistence.
    """

    def test_init_state(self) -> None:
        """
        Verifies that a newly initialized router holds an empty last_profile_ attribute.
        """
        router = AutoPipelineRouter()
        assert isinstance(router.last_profile_, dict)
        assert len(router.last_profile_) == 0

    def test_route_sequential_temporal(
        self,
        sequential_temporal_data: Tuple[
            List[Tuple[List[int], List[float]]],
            List[Tuple[List[int], List[float]]],
            List[Tuple[List[int], List[float], str]],
        ],
    ) -> None:
        """
        Verifies routing of sequential temporal data to Semi_Markov_Conformal_Engine.
        """
        train_seqs, _, _ = sequential_temporal_data
        router = AutoPipelineRouter()
        pipeline = router.route(train_seqs)

        assert pipeline == "Semi_Markov_Conformal_Engine"
        assert router.last_profile_["data_modality"] == "sequential_temporal"
        assert router.last_profile_["recommended_pipeline"] == "Semi_Markov_Conformal_Engine"
        assert router.last_profile_["n_samples"] == len(train_seqs)

    def test_route_sequential_discrete(
        self,
        sequential_discrete_data: Tuple[List[List[int]], List[List[int]]],
    ) -> None:
        """
        Verifies routing of sequential discrete data to Semi_Markov_Conformal_Engine.
        """
        train_events, _ = sequential_discrete_data
        router = AutoPipelineRouter()
        pipeline = router.route(train_events)

        assert pipeline == "Semi_Markov_Conformal_Engine"
        assert router.last_profile_["data_modality"] == "sequential_discrete"
        assert router.last_profile_["recommended_pipeline"] == "Semi_Markov_Conformal_Engine"

    def test_route_tabular_continuous(
        self,
        continuous_tabular_splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """
        Verifies routing of pure continuous data to Continuous_Conformal_Geometric_Engine.
        """
        df_train, _, _ = continuous_tabular_splits
        router = AutoPipelineRouter()
        pipeline = router.route(df_train)

        assert pipeline == "Continuous_Conformal_Geometric_Engine"
        assert router.last_profile_["data_modality"] == "tabular_continuous"
        assert router.last_profile_["recommended_pipeline"] == "Continuous_Conformal_Geometric_Engine"

    def test_route_tabular_heterogeneous(
        self,
        heterogeneous_tabular_data: Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]],
    ) -> None:
        """
        Verifies routing of mixed data to Heterogeneous_Gower_KMedoids_Engine.
        """
        train_df, _, _ = heterogeneous_tabular_data
        router = AutoPipelineRouter()
        pipeline = router.route(train_df)

        assert pipeline == "Heterogeneous_Gower_KMedoids_Engine"
        assert router.last_profile_["data_modality"] == "tabular_heterogeneous"
        assert router.last_profile_["recommended_pipeline"] == "Heterogeneous_Gower_KMedoids_Engine"

    def test_state_persistence_across_consecutive_calls(self) -> None:
        """
        Ensures that consecutive route calls update last_profile_ consistently without side-effects.
        """
        router = AutoPipelineRouter()

        pipe1 = router.route([[1, 2, 3]])
        assert pipe1 == "Semi_Markov_Conformal_Engine"
        assert router.last_profile_["data_modality"] == "sequential_discrete"

        pipe2 = router.route(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
        assert pipe2 == "Continuous_Conformal_Geometric_Engine"
        assert router.last_profile_["data_modality"] == "tabular_continuous"

        pipe3 = router.route(pd.DataFrame({"cat": ["x", "y"]}))
        assert pipe3 == "Heterogeneous_Gower_KMedoids_Engine"
        assert router.last_profile_["data_modality"] == "tabular_heterogeneous"
