
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest

from pde import PatternDiscoveryEngine


@pytest.mark.integration
class TestRoundtripFidelity:
    """
    Verifies 100% roundtrip reconstruction fidelity between pre-serialization and post-serialization states.
    """

    def test_continuous_roundtrip_fidelity(
        self,
        continuous_tabular_splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        tmp_model_path: Path,
    ) -> None:
        """
        Validates exact numerical identity (atol=1e-9) for continuous tabular models.
        """
        df_train, df_cal, df_test = continuous_tabular_splits

        engine_orig = PatternDiscoveryEngine(
            alpha=0.05, review_boundary=0.20, auto_adapt=True, random_state=42
        )
        engine_orig.fit(df_train, cal_data=df_cal)
        preds_orig = engine_orig.predict(df_test)

        saved_path = engine_orig.save(tmp_model_path)
        assert saved_path == tmp_model_path
        assert tmp_model_path.exists()

        engine_reloaded = PatternDiscoveryEngine.load(tmp_model_path)
        assert engine_reloaded.is_fitted is True
        assert engine_reloaded.modality == "tabular_continuous"
        assert engine_reloaded.pipeline_name == engine_orig.pipeline_name
        assert engine_reloaded.feature_names == engine_orig.feature_names

        preds_load = engine_reloaded.predict(df_test)

        assert np.allclose(preds_orig["scores"], preds_load["scores"], atol=1e-9)
        assert np.allclose(preds_orig["p_values"], preds_load["p_values"], atol=1e-9)
        assert np.array_equal(preds_orig["decisions"], preds_load["decisions"])
        assert np.array_equal(preds_orig["is_anomaly"], preds_load["is_anomaly"])

    def test_heterogeneous_roundtrip_fidelity(
        self,
        heterogeneous_tabular_data: Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]],
        tmp_path: Path,
    ) -> None:
        """
        Validates exact numerical identity (atol=1e-9) for heterogeneous mixed-type models.
        """
        train_df, test_df, _ = heterogeneous_tabular_data
        model_file = tmp_path / "heterogeneous_roundtrip.json"

        engine_orig = PatternDiscoveryEngine(
            alpha=0.05, review_boundary=0.20, random_state=42
        )
        engine_orig.fit(train_df)
        preds_orig = engine_orig.predict(test_df)

        engine_orig.save(model_file)
        assert model_file.exists()

        engine_reloaded = PatternDiscoveryEngine.load(model_file)
        assert engine_reloaded.is_fitted is True
        assert engine_reloaded.modality == "tabular_heterogeneous"

        preds_load = engine_reloaded.predict(test_df)

        assert np.allclose(preds_orig["scores"], preds_load["scores"], atol=1e-9)
        assert np.allclose(preds_orig["p_values"], preds_load["p_values"], atol=1e-9)
        assert np.array_equal(preds_orig["decisions"], preds_load["decisions"])
        assert np.array_equal(preds_orig["is_anomaly"], preds_load["is_anomaly"])

    def test_sequential_temporal_roundtrip_fidelity(
        self,
        sequential_temporal_data: Tuple[
            List[Tuple[List[int], List[float]]],
            List[Tuple[List[int], List[float]]],
            List[Tuple[List[int], List[float], str]],
        ],
        tmp_path: Path,
    ) -> None:
        """
        Validates exact numerical identity (atol=1e-9) for sequential temporal models.
        """
        train_seqs, test_normal, anomalies = sequential_temporal_data
        test_queries = test_normal + [(a[0], a[1]) for a in anomalies]
        model_file = tmp_path / "sequential_roundtrip.json"

        engine_orig = PatternDiscoveryEngine(
            alpha=0.05, review_boundary=0.20, random_state=42
        )
        engine_orig.fit(train_seqs)
        preds_orig = engine_orig.predict(test_queries)

        engine_orig.save(model_file)
        assert model_file.exists()

        engine_reloaded = PatternDiscoveryEngine.load(model_file)
        assert engine_reloaded.is_fitted is True
        assert engine_reloaded.modality == "sequential_temporal"

        preds_load = engine_reloaded.predict(test_queries)

        assert np.allclose(preds_orig["scores"], preds_load["scores"], atol=1e-9)
        assert np.allclose(preds_orig["p_values"], preds_load["p_values"], atol=1e-9)
        assert np.array_equal(preds_orig["decisions"], preds_load["decisions"])
        assert np.array_equal(preds_orig["is_anomaly"], preds_load["is_anomaly"])
