from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs


def pytest_configure(config: pytest.Config) -> None:
    """
    Registers custom test markers to satisfy --strict-markers.
    """
    config.addinivalue_line("markers", "unit: Unit test marker.")
    config.addinivalue_line("markers", "integration: Integration test marker.")
    config.addinivalue_line("markers", "slow: Computationally intensive test marker.")

    warnings.filterwarnings("ignore", category=UserWarning, module="hdbscan")


@pytest.fixture(scope="session")
def global_seed() -> int:
    """
    Returns the standard random seed for deterministic test reproducibility.
    """
    return 42


@pytest.fixture(scope="function")
def rng(global_seed: int) -> np.random.Generator:
    """
    Provides an independent NumPy random number generator for test functions.
    """
    return np.random.default_rng(global_seed)


@pytest.fixture(scope="session")
def continuous_tabular_splits(
    global_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates continuous Gaussian tabular data partitioned into train, cal, and test.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - **df_train**: 360 normal observations across 3 features.
            - **df_cal**: 100 normal calibration observations.
            - **df_test**: 90 observations (60 normal, 30 out-of-distribution anomalies).
    """
    centers = [[10.0, 100.0, 20.0], [50.0, 500.0, 100.0], [90.0, 900.0, 180.0]]
    stds = [[3.0, 25.0, 5.0], [5.0, 40.0, 10.0], [4.0, 35.0, 8.0]]

    X_norm, _ = make_blobs(
        n_samples=520,
        centers=centers,
        cluster_std=stds,
        random_state=global_seed,
    )
    cols = ["feat_x1", "feat_x2", "feat_x3"]

    df_train = pd.DataFrame(X_norm[:360], columns=cols)
    df_cal = pd.DataFrame(X_norm[360:460], columns=cols)
    df_test_norm = pd.DataFrame(X_norm[460:], columns=cols)

    rng = np.random.default_rng(global_seed + 101)
    df_test_anom = pd.DataFrame(
        rng.uniform(130.0, 200.0, size=(30, 3)), columns=cols
    )
    df_test = pd.concat([df_test_norm, df_test_anom], ignore_index=True)

    return df_train, df_cal, df_test


@pytest.fixture(scope="session")
def continuous_numpy_partitions(
    global_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Provides pure NumPy arrays for geometric models and conformal ensemble suites.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - **X_train**: 360 normal observations.
            - **X_cal**: 1000 normal calibration observations.
            - **X_test_norm**: 90 normal evaluation observations.
            - **X_anom**: 45 anomalous evaluation observations.
    """
    centers = [[10.0, 100.0, 20.0], [50.0, 500.0, 100.0], [90.0, 900.0, 180.0]]
    stds = [[3.0, 25.0, 5.0], [5.0, 40.0, 10.0], [4.0, 35.0, 8.0]]

    X_norm, _ = make_blobs(
        n_samples=1450,
        centers=centers,
        cluster_std=stds,
        random_state=global_seed,
    )

    X_train = X_norm[:360]
    X_cal = X_norm[360:1360]
    X_test_norm = X_norm[1360:]

    rng = np.random.default_rng(global_seed + 999)
    X_anom = np.column_stack(
        [
            rng.uniform(130.0, 180.0, 45),
            rng.uniform(1300.0, 1800.0, 45),
            rng.uniform(260.0, 360.0, 45),
        ]
    )

    return X_train, X_cal, X_test_norm, X_anom


@pytest.fixture(scope="session")
def heterogeneous_tabular_data(
    global_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    Generates synthetic mixed continuous and categorical tabular datasets.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
            - **train_df**: 300 mixed-type training records.
            - **test_df**: 60 test records (nominal + anomalies).
            - **feature_types**: Dict with 'numeric' and 'categorical' column lists.
    """
    rng = np.random.default_rng(global_seed)

    def sample_mixed(n: int, anomalous: bool = False) -> pd.DataFrame:
        if not anomalous:
            dur = np.maximum(1.0, rng.normal(300.0, 50.0, size=n))
            cost = np.maximum(0.1, rng.normal(100.0, 20.0, size=n))
            acts = np.maximum(0, rng.poisson(25, size=n))
            dev = rng.choice(["mobile", "desktop", "tablet"], size=n, p=[0.6, 0.3, 0.1])
            net = rng.choice(["local", "vpn"], size=n, p=[0.8, 0.2])
        else:
            dur = rng.uniform(2500.0, 5000.0, size=n)
            cost = rng.uniform(800.0, 2000.0, size=n)
            acts = rng.poisson(300, size=n)
            dev = rng.choice(["bot_agent", "datacenter"], size=n)
            net = rng.choice(["tor_exit", "proxy_pool"], size=n)

        return pd.DataFrame(
            {
                "duration": dur,
                "cost": cost,
                "actions": acts,
                "device_type": dev,
                "network_zone": net,
            }
        )

    train_df = sample_mixed(300, anomalous=False)
    test_normal = sample_mixed(50, anomalous=False)
    test_anom = sample_mixed(10, anomalous=True)
    test_df = pd.concat([test_normal, test_anom], ignore_index=True)

    feature_types = {
        "numeric": ["duration", "cost", "actions"],
        "categorical": ["device_type", "network_zone"],
    }

    return train_df, test_df, feature_types


@pytest.fixture(scope="session")
def sequential_temporal_data(
    global_seed: int,
) -> Tuple[
    List[Tuple[List[int], List[float]]],
    List[Tuple[List[int], List[float]]],
    List[Tuple[List[int], List[float], str]],
]:
    """
    Generates synthetic Semi-Markov trajectories with states and holding times.

    Returns:
        Tuple[List, List, List]:
            - **train_seqs**: 120 normal training trajectories.
            - **test_normal**: 30 in-distribution normal test trajectories.
            - **anomalies**: 15 anomalous trajectories with descriptive pathology labels.
    """
    rng = np.random.default_rng(global_seed)
    p_trans = np.array(
        [
            [0.1, 0.7, 0.1, 0.1],
            [0.0, 0.2, 0.7, 0.1],
            [0.0, 0.0, 0.2, 0.8],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    def sample_seq() -> Tuple[List[int], List[float]]:
        curr = 0
        events = [curr]
        durations = []
        for _ in range(5):
            if curr == 3:
                break
            dt = float(rng.lognormal(1.2, 0.3))
            durations.append(max(0.01, dt))
            nxt = int(rng.choice(4, p=p_trans[curr]))
            events.append(nxt)
            curr = nxt
        return events, durations

    train_seqs = [sample_seq() for _ in range(120)]
    test_normal = [sample_seq() for _ in range(30)]

    anomalies: List[Tuple[List[int], List[float], str]] = [
        ([0, 3], [0.001], "Illegal Fast Jump"),
        ([0, 1, 2, 3], [500.0, 600.0, 450.0], "Stalled Dwell Time"),
    ] * 8

    return train_seqs, test_normal, anomalies[:15]


@pytest.fixture(scope="session")
def sequential_discrete_data(
    sequential_temporal_data: Tuple[
        List[Tuple[List[int], List[float]]],
        List[Tuple[List[int], List[float]]],
        List[Tuple[List[int], List[float], str]],
    ],
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Extracts pure discrete state sequences (without duration intervals).

    Returns:
        Tuple[List[List[int]], List[List[int]]]:
            - **train_events**: Discrete event sequence lists for training.
            - **test_events**: Discrete event sequence lists for evaluation.
    """
    train_temporal, test_temporal, _ = sequential_temporal_data
    train_events = [item[0] for item in train_temporal]
    test_events = [item[0] for item in test_temporal]
    return train_events, test_events


@pytest.fixture(scope="session")
def conformal_score_distributions(
    global_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates standard non-conformity score distributions across calibration and test.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - **cal_scores**: 500 in-distribution calibration scores.
            - **test_scores**: 150 evaluation scores (100 normal, 50 anomalous).
            - **y_test**: Ground truth labels (0=normal, 1=anomaly).
    """
    rng = np.random.default_rng(global_seed)

    cal_scores = rng.chisquare(df=3, size=500)
    test_normal = rng.chisquare(df=3, size=100)
    test_anomaly = np.maximum(0.1, rng.normal(loc=12.0, scale=2.5, size=50))

    test_scores = np.concatenate([test_normal, test_anomaly])
    y_test = np.concatenate([np.zeros(100, dtype=int), np.ones(50, dtype=int)])

    return cal_scores, test_scores, y_test


@pytest.fixture(scope="function")
def tmp_model_path(tmp_path: Path) -> Path:
    """
    Provides an isolated temporary JSON file path for serialization roundtrips.
    """
    return tmp_path / "test_pde_model.json"
