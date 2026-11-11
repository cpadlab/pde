# Pattern Discovery Engine (PDE)

A unified library for classical machine learning, pattern discovery, and statistical anomaly detection with distribution-free false positive rate control.

---

## Overview

Pattern Discovery Engine (PDE) is a classical machine learning library designed to discover underlying structures, quantify evidence, and detect anomalies across multiple data modalities.

Unlike heuristic detectors that output incomparable raw scores, PDE integrates inductive conformal prediction to ensure strict Type I error bounds (FPR <= alpha) and supports tripartite selective decisions (`anomaly`, `review`, `no_alert`).

---

## Core Capabilities

- Continuous Tabular Modeling: Geometric detector quartet (HDBSCAN, Isolation Forest, Local Outlier Factor, and One-Class SVM) combined via conservative Bonferroni conformal ensembles.
- Heterogeneous Data: Leakage-free Gower distance metric computation coupled with Robust K-Medoids++ clustering and autonomous geometric elbow selection (KneeLocator).
- Sequential and Temporal Dynamics: Semi-Markov modeling combining discrete transition probability matrices (Laplace smoothed) and continuous state-conditional Log-Normal holding times.
- Symbolic Association and Rule Mining: Quantile discretization, Apriori rule mining pruned by lift and confidence, rule violation anomaly scoring, and directed dependency graph analysis.
- Conformal Calibration and Selective Classification: Exact non-parametric p-values against calibration quantiles with formal false positive rate control and audited manual review zones.
- Continuous Drift Monitoring: Kolmogorov-Smirnov goodness-of-fit testing on conformal p-values tracking deviations from standard uniformity U(0, 1), with optional autonomous re-adaptation.
- Production Engine API: Central `PatternDiscoveryEngine` orchestrator that autonomously profiles input modality, routes to optimal pipelines, and persists to JSON with 100% numerical fidelity.

---

## Project Structure

```text
pde/
├── src/pde/
│   ├── core/           # Meta-profiler and automatic pipeline router
│   ├── geometry/       # Continuous models suite and Bonferroni ensemble
│   ├── clustering/     # Gower distance, Robust K-Medoids++, and knee locator
│   ├── sequential/     # Semi-Markov engine for trajectories and durations
│   ├── association/    # Discretization, association rules, and dependency graphs
│   ├── calibration/    # Conformal calibrator and selective classifier
│   ├── drift/          # Kolmogorov-Smirnov distribution drift monitor
│   ├── persistence/    # JSON serialization and reconstruction utilities
│   └── engine.py       # Central PatternDiscoveryEngine production API
├── examples/           # Standalone terminal workflows and demonstrations
├── tests/              # Unit and integration test suites
├── pyproject.toml      # Poetry package configuration and test options
└── README.md
```

---

## Installation

Requirements: Python >= 3.12 (excluding 3.14.1 due to upstream wheel compatibility).

Using Poetry:

```powershell
poetry install
```

---

## Quickstart

```python
import numpy as np
import pandas as pd
from pde import PatternDiscoveryEngine

# 1. Prepare data
X_train = pd.DataFrame(np.random.normal(0, 1, size=(300, 3)), columns=["a", "b", "c"])
X_test = pd.DataFrame(np.random.normal(0, 1, size=(20, 3)), columns=["a", "b", "c"])

# 2. Fit and calibrate
engine = PatternDiscoveryEngine(alpha=0.05, review_boundary=0.20)
engine.fit(X_train)

# 3. Predict with conformal certainty
predictions = engine.predict(X_test)

print("Decisions:", predictions["decisions"])
print("Conformal p-values:", predictions["p_values"])
```

---

## Running Examples

The repository includes runnable end-to-end examples in the `examples/` directory:

```powershell
poetry run python examples/continuous_suite.py
poetry run python examples/heterogeneous_gower.py
poetry run python examples/sequential_markov.py
poetry run python examples/association_rules.py
poetry run python examples/conformal_inference.py
poetry run python examples/drift_monitoring.py
poetry run python examples/end_to_end_pipeline.py
```

---

## Running Tests

Run the full unit and integration test suite:

```powershell
poetry run pytest
```
