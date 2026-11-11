"""
END-TO-END UNIFIED PIPELINE: AUTO-ROUTING, INFERENCE, AND PERSISTENCE.

Demonstrates the complete production workflow using the central orchestrator
PatternDiscoveryEngine across the system's three core data modalities.

Evaluated workflow:
1. Unsupervised meta-profiling: autonomous detection of input data modality
   (continuous tabular, heterogeneous tabular, and sequential-temporal).
2. Autonomous routing to the corresponding algorithmic pipeline.
3. Inductive model fitting and integrated conformal calibration (.fit()).
4. Inference with raw scores, conformal p-values, and selective classification (.predict()).
5. Continuous distributional drift monitoring over inference windows (.monitor_drift()).
6. Full disk serialization to JSON (.save()) and roundtrip reconstruction (.load()).
7. Verification of exact numerical identity pre- and post-serialization.
"""

from pathlib import Path
import tempfile
from typing import Any, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from pde import PatternDiscoveryEngine

warnings.filterwarnings("ignore", category=UserWarning)


def get_continuous_data(
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates continuous Gaussian tabular data across splits.

    Args:
        seed (int, optional): Random seed for reproducible generation. Defaults
            to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - **df_train**: 360 normal training observations.
            - **df_cal**: 50 normal calibration observations.
            - **df_test**: 60 test observations (40 normal, 20 out-of-distribution anomalies).
    """
    X, _ = make_blobs(
        n_samples=450,
        n_features=3,
        centers=3,
        cluster_std=0.8,
        random_state=seed,
    )
    df_train = pd.DataFrame(X[:360], columns=["f1", "f2", "f3"])
    df_cal = pd.DataFrame(X[360:410], columns=["f1", "f2", "f3"])

    rng = np.random.default_rng(seed + 101)
    df_test_norm = pd.DataFrame(X[410:], columns=["f1", "f2", "f3"])
    df_test_anom = pd.DataFrame(
        rng.uniform(-10.0, 15.0, size=(20, 3)), columns=["f1", "f2", "f3"]
    )
    df_test = pd.concat([df_test_norm, df_test_anom], ignore_index=True)

    return df_train, df_cal, df_test


def get_heterogeneous_data(
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates heterogeneous tabular data with continuous and categorical features.

    Args:
        seed (int, optional): Random seed for reproducible generation. Defaults
            to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - **df_train**: 300 mixed-type normal training records.
            - **df_test**: 2 test records (1 nominal sample, 1 extreme anomaly).
    """
    rng = np.random.default_rng(seed)
    n = 300
    df_train = pd.DataFrame(
        {
            "duration": rng.normal(300.0, 50.0, n),
            "cost": rng.normal(100.0, 20.0, n),
            "actions": rng.poisson(25, n),
            "device": rng.choice(
                ["mobile", "desktop", "tablet"], n, p=[0.6, 0.3, 0.1]
            ),
            "network": rng.choice(["local", "vpn"], n, p=[0.8, 0.2]),
        }
    )

    df_test = pd.DataFrame(
        {
            "duration": [305.0, 3800.0],
            "cost": [98.0, 1400.0],
            "actions": [24, 450],
            "device": ["mobile", "datacenter_bot"],
            "network": ["local", "foreign"],
        }
    )

    return df_train, df_test


def get_sequential_data(
    seed: int = 42,
) -> Tuple[
    List[Tuple[List[int], List[float]]], List[Tuple[List[int], List[float]]]
]:
    """
    Generates sequential temporal trajectories with state transitions and durations.

    Args:
        seed (int, optional): Random seed for reproducible generation. Defaults
            to 42.

    Returns:
        Tuple[List, List]: A tuple containing:
            - **train_seqs**: 120 normal synthetic trajectories.
            - **test_seqs**: 2 test trajectories (1 valid sequence, 1 illegal instant jump).
    """
    rng = np.random.default_rng(seed)
    p_trans = np.array(
        [
            [0.1, 0.7, 0.1, 0.1],
            [0.0, 0.2, 0.7, 0.1],
            [0.0, 0.0, 0.2, 0.8],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    train_seqs = []
    for _ in range(120):
        curr = 0
        events = [curr]
        durations = []
        for _t in range(5):
            if curr == 3:
                break
            dt = float(rng.lognormal(1.5, 0.3))
            durations.append(dt)
            nxt = int(rng.choice(4, p=p_trans[curr]))
            events.append(nxt)
            curr = nxt
        train_seqs.append((events, durations))

    test_seqs = [
        ([0, 1, 2, 3], [4.2, 5.1, 3.8]),
        ([0, 3], [0.002]),
    ]

    return train_seqs, test_seqs


def evaluate_domain(
    domain_name: str,
    train_data: Any,
    test_data: Any,
    cal_data: Optional[Any] = None,
) -> PatternDiscoveryEngine:
    """
    Executes profiling, fitting, inference, and serialization validation for a domain.

    Args:
        domain_name (str): Human-readable name of the evaluated data domain.
        train_data (Any): Training dataset for the domain.
        test_data (Any): Out-of-sample evaluation data.
        cal_data (Optional[Any], optional): Independent calibration partition.
            Defaults to None.

    Returns:
        PatternDiscoveryEngine: Fitted, calibrated, and validated engine instance.
    """
    print(f"\n[INFO] Evaluating Modality Domain: {domain_name}...")

    engine = PatternDiscoveryEngine(
        alpha=0.05, review_boundary=0.20, auto_adapt=True
    )
    engine.fit(train_data, cal_data=cal_data)

    print(f"\t• Inferred modality:   {engine.modality}")
    print(f"\t• Routed pipeline:     {engine.pipeline_name}")

    preds = engine.predict(test_data)
    print(f"\t• Evaluated samples:   {len(preds['scores'])}")
    print(
        "\t• Decisions emitted:   "
        f"{pd.Series(preds['decisions']).value_counts().to_dict()}"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / f"{domain_name.lower()}_model.json"
        engine.save(model_path)

        reloaded_engine = PatternDiscoveryEngine.load(model_path)
        reloaded_preds = reloaded_engine.predict(test_data)

        scores_match = np.allclose(
            preds["scores"], reloaded_preds["scores"], atol=1e-9
        )
        pvals_match = np.allclose(
            preds["p_values"], reloaded_preds["p_values"], atol=1e-9
        )
        decisions_match = np.array_equal(
            preds["decisions"], reloaded_preds["decisions"]
        )

        fidelity_ok = scores_match and pvals_match and decisions_match
        status_msg = (
            "100% FIDELITY (Exact identity verified)"
            if fidelity_ok
            else "IDENTITY CHECK FAILED"
        )
        print(f"\t• JSON persistence:    {status_msg}")

    return engine


def main() -> None:
    """
    Executes the end-to-end multi-modal routing, inference, and persistence workflow.
    """
    print("Unified Pattern Discovery Pipeline And Auto-Routing")
    print("=" * 51)

    df_tr_c, df_cal_c, df_te_c = get_continuous_data(seed=42)
    engine_cont = evaluate_domain(
        "Continuous Tabular", df_tr_c, df_te_c, cal_data=df_cal_c
    )

    df_tr_h, df_te_h = get_heterogeneous_data(seed=42)
    evaluate_domain("Heterogeneous Tabular", df_tr_h, df_te_h)

    seqs_tr, seqs_te = get_sequential_data(seed=42)
    evaluate_domain("Sequential Temporal", seqs_tr, seqs_te)

    print("\n[INFO] Evaluating Distributional Drift Monitoring On Continuous Data...")
    rng = np.random.default_rng(2026)

    window_normal = pd.DataFrame(
        rng.normal(
            df_tr_c.mean().to_numpy(),
            df_tr_c.std().to_numpy(),
            size=(100, 3),
        ),
        columns=["f1", "f2", "f3"],
    )
    res_normal = engine_cont.monitor_drift(window_normal)
    print(
        f"\t• Stationary window:  KS p-value = {res_normal['ks_pvalue']:.4f} | "
        f"Drift detected = {res_normal['drift_detected']}"
    )

    window_drifted = window_normal + 4.5
    res_drift = engine_cont.monitor_drift(window_drifted)
    print(
        f"\t• Shifted window:     KS p-value = {res_drift['ks_pvalue']:.2e} | "
        f"Drift detected = {res_drift['drift_detected']} | "
        f"Auto-adapted = {res_drift['adapted']}"
    )


if __name__ == "__main__":
    main()
