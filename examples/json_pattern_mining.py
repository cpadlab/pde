"""
AUTOMATIC PATTERN DISCOVERY IN RAW JSON DATA.

Demonstrates how to process a list of raw JSON dictionaries containing mixed
data types (continuous numeric and boolean/categorical) to:
1. Extract frequent co-occurrence patterns and logical association rules
   (Apriori + Lift).
2. Segment records into natural demographic profiles via Gower distance and
   robust k-medoids clustering.
3. Audit incoming observations, flagging contextual anomalies and providing
   human-readable violation explanations.
"""

import json
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from pde.association.discretizer import QuantileDiscretizer
from pde.association.rules import AssociationRuleScorer
from pde.clustering.gower import GowerDistanceMetric
from pde.clustering.kmedoids import RobustKMedoids


def load_raw_json_data() -> List[Dict[str, Any]]:
    """
    Simulates the ingestion of a raw JSON payload with mixed-type attributes.

    Returns:
        List[Dict[str, Any]]: List of dictionary records, each containing an
            `'age'` (int) and `'car'` (bool) field.
    """
    return [
        {"age": 20, "car": True},
        {"age": 19, "car": True},
        {"age": 31, "car": True},
        {"age": 20, "car": True},
        {"age": 28, "car": True},
        {"age": 17, "car": False},
        {"age": 15, "car": False},
    ]


def main() -> None:
    """
    Executes the mixed-type pattern discovery and rule violation audit workflow.
    """
    print("Discovering Patterns In JSON Data")
    print("=" * 33)

    print("\n[INFO] Parsing JSON Payload...")
    raw_data = load_raw_json_data()
    df = pd.DataFrame(raw_data)
    print(df.to_string(index=True))

    print("\n[INFO] Discretizing Numeric Variables Without Data Leakage...")
    discretizer = QuantileDiscretizer(n_bins=2, labels=["Young", "Adult"])
    df_age = discretizer.fit(df, features=["age"]).transform(df)

    df_car = pd.get_dummies(df["car"], prefix="car", dtype=bool)
    transactions = pd.concat([df_age, df_car], axis=1)

    print(f"\t• Learned cutoff threshold for 'age': {discretizer.bin_edges_['age'][1]:.1f} years")
    print(f"\t• Transactional space generated: {list(transactions.columns)}")

    print("\n[INFO] Extracting Logical co-occurrence Patterns (Apriori + Lift)...")
    scorer = AssociationRuleScorer(
        min_support=0.20,
        min_confidence=0.80,
        min_lift=1.10
    )
    scorer.fit(transactions)
    rules_df = scorer.rules_df_

    print("\n" + "-" * 37)
    print("Discovered Logical Rules And Patterns")
    print("-" * 37)
    if rules_df is not None and not rules_df.empty:
        for idx, (_, r) in enumerate(rules_df.iterrows(), start=1):
            ant = list(r["antecedents"])
            con = list(r["consequents"])
            conf = r["confidence"] * 100
            supp = r["support"] * 100
            lift = r["lift"]
            print(f"\t• Pattern #{idx}: If {ant}  ==>  then  {con}")
            print(f"\t\t- Confidence: {conf:.1f}% | Support: {supp:.1f}% | Lift: {lift:.2f}")
    else:
        print("\t• No rules were found that exceeded the support/confidence thresholds.")

    print("\n[INFO] Discovering Natural Profiles (k-Medoids based on Gower distance)...")
    df_clustering = df.copy()
    df_clustering["car"] = df_clustering["car"].astype(str)

    gower = GowerDistanceMetric(numeric_features=["age"], categorical_features=["car"])
    gower.fit(df_clustering)
    dist_matrix = gower.compute_distance_matrix(df_clustering)

    kmedoids = RobustKMedoids(n_clusters=2, n_init=10, random_state=42)
    kmedoids.fit(dist_matrix)

    df["cluster"] = kmedoids.labels_
    print("\t• Representative medoids discovered:")
    if kmedoids.medoid_indices_ is not None:
        for cluster_id, medoid_idx in enumerate(kmedoids.medoid_indices_):
            row = df.iloc[medoid_idx]
            print(f"\t- Profile {cluster_id}: Age = {row['age']} years, Car = {row['car']} (Record #{medoid_idx})")

    print("\n" + "=" * 53)
    print("Audit Of New Incoming Cases (Detection Of Violations)")
    print("=" * 53)

    new_cases = pd.DataFrame([
        {"age": 22, "car": True},
        {"age": 14, "car": False},
        {"age": 15, "car": True},
    ])

    new_age = discretizer.transform(new_cases)
    new_car = pd.get_dummies(new_cases["car"], prefix="car", dtype=bool)

    for col in transactions.columns:
        if col not in new_car.columns and "car" in col:
            new_car[col] = False

    new_trans = pd.concat([new_age, new_car], axis=1)[transactions.columns]
    scores, explanations = scorer.score(new_trans)

    for idx, (_, row) in enumerate(new_cases.iterrows(), start=1):

        violations = explanations[idx - 1]
        is_anomaly = len(violations) > 0
        status = "[ALERT: ANOMALY]" if is_anomaly else "[PASSED: NORMAL]"

        print(f"\nCase #{idx}: Age = {row['age']}, Car = {row['car']} -> {status}")
        if is_anomaly:
            print(f"\t• Weighted violation score: {scores[idx - 1]:.4f}")
            print("\t• Breached rules:")
            for viol in violations:
                print(f"\t\t- {viol}")
        else:
            print("\t• Record strictly complies with historical association rules.")


if __name__ == "__main__":
    main()