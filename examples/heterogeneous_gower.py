"""
HETEROGENEOUS DATA, GOWER DISTANCE, AND ROBUST K-MEDOIDS++.

Demonstrates clustering and out-of-sample inductive assignment on mixed data spaces
(continuous numerical and nominal categorical variables) using Gower distance
metric and robust k-medoids++ with geometric elbow selection.

Evaluated workflow:
1. Generation of heterogeneous synthetic profiles (duration, cost, actions, device, network).
2. Calculation of Gower ranges without data leakage (fitted strictly on train).
3. Unsupervised sweep of k in [2, 6] and autonomous knee point detection (KneeLocator).
4. Fitting robust k-medoids++ clustering with multi-start (n_init=10) for minimum inertia.
5. Out-of-sample inductive assignment on an unseen test set.
6. Cluster quality evaluation (Inertia, Silhouette Score, and ARI).
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from pde.clustering.gower import GowerDistanceMetric
from pde.clustering.kmedoids import RobustKMedoids
from pde.clustering.knee import KneeLocator


def generate_heterogeneous_data(
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic mixed-type datasets across train and test partitions.

    Simulates realistic heterogeneous user behavior spanning 3 demographic profiles:
    Consumer Mobile (0), Enterprise Desktop (1), and Automated Datacenter (2).
    Features combine continuous variables (duration, cost, actions) and nominal
    categorical attributes (device type, network zone).

    Args:
        seed (int, optional): Random seed for reproducible generation. Defaults
            to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - **train_df** (`pd.DataFrame`): Training partition with 360 balanced
              records across the 3 profiles.
            - **test_df** (`pd.DataFrame`): Held-out test partition with 90 balanced
              records across the 3 profiles.
    """
    rng = np.random.default_rng(seed)

    profiles = {
        0: {
            "name": "Consumer_Mobile",
            "duration": (120.0, 30.0),
            "cost": (25.0, 5.0),
            "actions_lambda": 8,
            "devices": ["mobile", "tablet", "desktop"],
            "device_probs": [0.80, 0.15, 0.05],
            "networks": ["local", "vpn", "datacenter"],
            "network_probs": [0.90, 0.09, 0.01],
        },
        1: {
            "name": "Enterprise_Desktop",
            "duration": (600.0, 80.0),
            "cost": (150.0, 20.0),
            "actions_lambda": 45,
            "devices": ["mobile", "tablet", "desktop"],
            "device_probs": [0.05, 0.05, 0.90],
            "networks": ["local", "vpn", "datacenter"],
            "network_probs": [0.60, 0.38, 0.02],
        },
        2: {
            "name": "Automated_Datacenter",
            "duration": (1200.0, 150.0),
            "cost": (500.0, 60.0),
            "actions_lambda": 180,
            "devices": ["mobile", "tablet", "desktop"],
            "device_probs": [0.02, 0.03, 0.95],
            "networks": ["local", "vpn", "datacenter"],
            "network_probs": [0.05, 0.15, 0.80],
        },
    }

    def sample_profile(profile_id: int, count: int) -> pd.DataFrame:
        p = profiles[profile_id]
        dur = np.maximum(
            1.0, rng.normal(p["duration"][0], p["duration"][1], size=count)
        )
        cst = np.maximum(
            0.1, rng.normal(p["cost"][0], p["cost"][1], size=count)
        )
        act = np.maximum(0, rng.poisson(p["actions_lambda"], size=count))
        dev = rng.choice(p["devices"], size=count, p=p["device_probs"])
        net = rng.choice(p["networks"], size=count, p=p["network_probs"])

        return pd.DataFrame(
            {
                "duration": dur,
                "cost": cst,
                "actions": act,
                "device_type": dev,
                "network_zone": net,
                "profile_label": profile_id,
            }
        )

    dfs_train = [sample_profile(i, 120) for i in range(3)]
    dfs_test = [sample_profile(i, 30) for i in range(3)]

    train_df = (
        pd.concat(dfs_train, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat(dfs_test, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    return train_df, test_df


def main() -> None:
    """
    Executes the heterogeneous Gower metric clustering and inductive assignment workflow.
    """
    print("Robust Heterogeneous Clustering And Gower Distance")
    print("=" * 50)

    print("\n[INFO] Generating Heterogeneous Dataset Across Mixed Spaces...")
    train_df, test_df = generate_heterogeneous_data(seed=42)

    numeric_cols = ["duration", "cost", "actions"]
    categorical_cols = ["device_type", "network_zone"]

    print(f"\t• Training samples:     {len(train_df)}")
    print(f"\t• Test samples:         {len(test_df)}")
    print(f"\t• Continuous features:  {numeric_cols}")
    print(f"\t• Categorical features: {categorical_cols}")

    print("\n[INFO] Fitting Gower Distance Metric Exclusively On Train...")
    gower = GowerDistanceMetric(
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
    )
    gower.fit(train_df)
    train_dist_matrix = gower.compute_distance_matrix(train_df)

    print(f"\t• Computed Gower pairwise distance matrix: {train_dist_matrix.shape}")
    print(f"\t• Learned normalization range for 'duration': {gower.ranges_['duration']:.2f}")
    print(f"\t• Learned normalization range for 'cost':     {gower.ranges_['cost']:.2f}")

    print("\n[INFO] Evaluating Cluster Inertia And Silhouette Across Candidate k:")
    print("-" * 52)
    print(f"{'Clusters (k)':14s} | {'Gower Inertia':15s} | {'Silhouette':12s}")
    print("-" * 52)

    k_values = [2, 3, 4, 5, 6]
    inertias = []
    silhouettes = []

    for k in k_values:
        kmedoids = RobustKMedoids(n_clusters=k, n_init=10, random_state=42)
        kmedoids.fit(train_dist_matrix)
        sil = silhouette_score(
            train_dist_matrix, kmedoids.labels_, metric="precomputed"
        )
        inertias.append(kmedoids.inertia_)
        silhouettes.append(sil)
        print(f"k = {k:<10d} | {kmedoids.inertia_:<15.4f} | {sil:<12.4f}")

    print("-" * 52)

    best_k, _ = KneeLocator.detect_knee_point(k_values, inertias)
    print(f"\n\t• Automatically detected geometric knee point: k* = {best_k}")

    print(f"\n[INFO] Fitting Final Model With Optimal k* = {best_k} On Training Data...")
    final_model = RobustKMedoids(n_clusters=best_k, n_init=10, random_state=42)
    final_model.fit(train_dist_matrix)

    y_train_true = train_df["profile_label"].to_numpy()
    train_ari = adjusted_rand_score(y_train_true, final_model.labels_)
    train_sil = silhouette_score(
        train_dist_matrix, final_model.labels_, metric="precomputed"
    )

    print(f"\t• Final minimum inertia:        {final_model.inertia_:.4f}")
    print(f"\t• Training Silhouette score:    {train_sil:.4f}")
    print(f"\t• Ground truth alignment (ARI): {train_ari:.4f}")

    print("\n\t• Discovered medoid exemplars in training set:")
    if final_model.medoid_indices_ is not None:
        for cluster_id, medoid_idx in enumerate(final_model.medoid_indices_):
            medoid_data = train_df.iloc[medoid_idx]
            print(
                f"\t  - Cluster {cluster_id}: Record #{medoid_idx:3d} | "
                f"Duration: {medoid_data['duration']:6.1f}s | Cost: {medoid_data['cost']:5.1f} | "
                f"Device: {medoid_data['device_type']:7s} | Network: {medoid_data['network_zone']}"
            )

    print("\n[INFO] Performing Out-Of-Sample Inductive Assignment On Test Data...")
    test_to_train_dist = gower.compute_distance_matrix(test_df, train_df)
    test_clusters, test_distances_to_medoids = final_model.predict(
        test_to_train_dist
    )

    y_test_true = test_df["profile_label"].to_numpy()
    test_ari = adjusted_rand_score(y_test_true, test_clusters)

    print(f"\t• Test inductive assignment ARI:        {test_ari:.4f}")
    print(f"\t• Mean distance to assigned medoid:     {test_distances_to_medoids.mean():.4f}")

    print("\n[INFO] Inductive Test Sample Assignment Inspection:")
    for i in [0, 30, 60]:
        sample = test_df.iloc[i]
        assigned_c = test_clusters[i]
        dist_m = test_distances_to_medoids[i]
        print(
            f"\t• Sample #{i:2d} ({sample['device_type']:7s}, {sample['network_zone']:10s}) "
            f"-> Assigned Cluster {assigned_c} (Gower Medoid Distance: {dist_m:.4f})"
        )


if __name__ == "__main__":
    main()
