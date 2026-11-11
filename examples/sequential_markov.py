"""
SEQUENTIAL DYNAMICS AND SEMI-MARKOV PROCESSES.

Demonstrates joint modeling of discrete state transition sequences and
continuous inter-event holding times using Semi-Markov processes.

Evaluated workflow:
1. Sequence generation with stochastic transition matrices and Log-Normal durations.
2. Laplace-regularized transition probability estimation on training data.
3. Empirical state-conditional duration parameter estimation (mu, sigma).
4. Dual non-conformity scoring: Transition NLL and Duration NLL.
5. Two-way Bonferroni conformal calibration (alpha = 0.05) for balanced detection.
6. Simultaneous detection of 3 structural pathologies:
   - Illegal topological jumps with typical holding times.
   - Ultra-fast automated bots with valid syntax but millisecond durations.
   - Stalled sessions exhibiting abnormally long dwell times.
"""

from typing import List, Sequence, Tuple
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pde.calibration.conformal import ConformalCalibrator
from pde.calibration.selective import SelectiveClassifier
from pde.sequential.markov import SemiMarkovEngine


def generate_semi_markov_dataset(
    seed: int = 42,
) -> Tuple[
    List[Tuple[List[int], List[float]]],
    List[Tuple[List[int], List[float]]],
    List[Tuple[List[int], List[float]]],
    List[Tuple[List[int], List[float], str]],
]:
    """
    Generates Semi-Markov synthetic sequence trajectories across partitions.

    Simulates realistic web application navigation workflows across 5 states:
    AUTH (0), VIEW (1), INTERACT (2), PAYMENT (3), and COMPLETE (4). Creates
    nominal sequences with state-dependent Log-Normal holding times alongside
    three distinct anomaly categories.

    Args:
        seed (int, optional): Random seed for reproducible generation. Defaults
            to 42.

    Returns:
        Tuple[List, List, List, List]: A tuple containing:
            - **train_seqs**: 360 normal training trajectories.
            - **cal_seqs**: 1000 independent normal calibration trajectories.
            - **test_normal**: 90 in-distribution normal test trajectories.
            - **anomalies**: 45 anomalous trajectories (15 illegal jumps,
              15 high-speed bots, 15 stalled sessions) with descriptive labels.
    """
    rng = np.random.default_rng(seed)

    p_matrix = np.array(
        [
            [0.05, 0.75, 0.15, 0.00, 0.05],  # AUTH
            [0.00, 0.30, 0.55, 0.00, 0.15],  # VIEW
            [0.00, 0.25, 0.25, 0.40, 0.10],  # INTERACT
            [0.00, 0.05, 0.05, 0.00, 0.90],  # PAYMENT
            [0.00, 0.00, 0.00, 0.00, 1.00],  # COMPLETE (terminal)
        ]
    )

    dur_params = {
        0: (1.0, 0.35),  # AUTH: ~2.7s
        1: (2.5, 0.40),  # VIEW: ~12.2s
        2: (2.0, 0.40),  # INTERACT: ~7.4s
        3: (3.2, 0.35),  # PAYMENT: ~24.5s
    }

    def sample_normal_seq(
        max_len: int = 12,
    ) -> Tuple[List[int], List[float]]:
        curr = 0
        events = [curr]
        durations = []
        for _ in range(max_len - 1):
            if curr == 4:
                break
            mu, sigma = dur_params[curr]
            dt = float(rng.lognormal(mu, sigma))
            durations.append(max(0.01, dt))
            nxt = int(rng.choice(5, p=p_matrix[curr]))
            events.append(nxt)
            curr = nxt
        return events, durations

    train_seqs = [sample_normal_seq() for _ in range(360)]
    cal_seqs = [sample_normal_seq() for _ in range(1000)]
    test_normal = [sample_normal_seq() for _ in range(90)]

    anomalies: List[Tuple[List[int], List[float], str]] = []

    for _ in range(15):
        anomalies.append(
            ([0, 3, 3, 4], [2.7, 22.0, 25.0], "Illegal Jump (Topological)")
        )

    for _ in range(15):
        anomalies.append(
            ([0, 1, 2, 3, 4], [0.01, 0.02, 0.01, 0.02], "Ultra-Fast Bot (Temporal)")
        )

    for _ in range(15):
        anomalies.append(
            ([0, 1, 4], [450.0, 600.0], "Stalled Session (Linger Time)")
        )

    return train_seqs, cal_seqs, test_normal, anomalies


def main() -> None:
    """
    Executes the Semi-Markov modeling and conformal verification workflow.
    """
    print("Sequential Dynamics And Semi-Markov Modeling")
    print("=" * 44)

    print("\n[INFO] Generating Sequence Streams And Inter-Event Holding Times...")
    train_seqs, cal_seqs, test_normal, anomalies = generate_semi_markov_dataset(seed=42)

    test_all = [(item[0], item[1]) for item in test_normal] + [
        (a[0], a[1]) for a in anomalies
    ]
    y_test = np.concatenate(
        [np.zeros(len(test_normal), dtype=int), np.ones(len(anomalies), dtype=int)]
    )
    anomaly_types = ["Normal"] * len(test_normal) + [a[2] for a in anomalies]

    print(f"\t• Training trajectories:    {len(train_seqs)}")
    print(f"\t• Calibration trajectories: {len(cal_seqs)}")
    print(
        f"\t• Test trajectories:        {len(test_all)} ({len(test_normal)} normal, {len(anomalies)} anomalous)"
    )

    print("\n[INFO] Estimating Discrete Transitions And Duration Densities On Train...")
    engine = SemiMarkovEngine(n_states=5, laplace_alpha=0.01)
    engine.fit(train_seqs)

    print("\t• Estimated transition matrix P_hat (with Laplace smoothing):")
    states = ["AUTH", "VIEW", "INTERACT", "PAYMENT", "COMPLETE"]
    if engine.p_hat_ is not None:
        for i, row in enumerate(engine.p_hat_):
            row_str = " ".join([f"{v:6.3f}" for v in row])
            print(f"\t  [{states[i]:8s}] -> {row_str}")

    print("\n\t• Estimated LogNormal duration parameters:")
    for state_id, (mu, sigma) in engine.duration_params_.items():
        median_sec = np.exp(mu)
        print(
            f"\t  - {states[state_id]:8s}: mu_log = {mu:.3f}, sigma_log = {sigma:.3f} (Median: {median_sec:5.1f} s)"
        )

    print("\n[INFO] Computing Transition And Duration NLL Scores Across Calibration And Test...")
    cal_t_nll, cal_d_nll = engine.score(cal_seqs)
    test_t_nll, test_d_nll = engine.score(test_all)

    calibrator_trans = ConformalCalibrator().fit(cal_t_nll)
    calibrator_dur = ConformalCalibrator().fit(cal_d_nll)

    p_trans = calibrator_trans.compute_pvalue(test_t_nll)
    p_dur = calibrator_dur.compute_pvalue(test_d_nll)

    p_matrix = np.column_stack([p_trans, p_dur])
    p_semi = ConformalCalibrator.bonferroni_combine(p_matrix)

    classifier = SelectiveClassifier(alpha=0.05, review_boundary=0.20)

    print("\n[INFO] Performance Comparison And False Positive Rate Control:")
    print("-" * 76)
    print(
        f"{'Evaluated Model':32s} | {'ROC-AUC':8s} | {'AP':8s} | {'Pop FPR':8s} | {'Op Recall':9s}"
    )
    print("-" * 76)

    models = {
        "Transition Only (Markov NLL)": p_trans,
        "Duration Only (Time NLL)": p_dur,
        "Semi-Markov Bonferroni Ensemble": p_semi,
    }

    for name, p_vals in models.items():
        decisions = classifier.decide(p_vals)
        metrics = classifier.evaluate_metrics(y_test, decisions)
        auc = roc_auc_score(y_test, -p_vals)
        ap = average_precision_score(y_test, -p_vals)
        print(
            f"{name:32s} | {auc:.4f}   | {ap:.4f}   | "
            f"{metrics['population_fpr']:.4f}   | {metrics['operational_detection_rate']:.4f}"
        )

    print("-" * 76)
    print("\n[INFO] Semi-Markov Ensemble Detection Breakdown By Pathology:")
    semi_decisions = classifier.decide(p_semi)

    target_pathologies = [
        "Illegal Jump (Topological)",
        "Ultra-Fast Bot (Temporal)",
        "Stalled Session (Linger Time)",
    ]

    for anomaly_type in target_pathologies:
        indices = [i for i, t in enumerate(anomaly_types) if t == anomaly_type]
        detected = int(np.sum(semi_decisions[indices] == "anomaly"))
        reviewed = int(np.sum(semi_decisions[indices] == "review"))
        total = len(indices)
        rate = (detected + reviewed) / total if total > 0 else 0.0
        print(
            f"\t• {anomaly_type:30s}: Detected = {detected:2d}/{total} | "
            f"Review = {reviewed:2d} | Operational Recall = {rate * 100:.1f}%"
        )


if __name__ == "__main__":
    main()