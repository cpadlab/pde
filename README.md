# Pattern Discovery Engine

A production-ready Python library for classical machine learning, pattern discovery, and statistical anomaly detection with distribution-free false positive rate control.

---

## Overview

**Pattern Discovery Engine (`cp-pde`)** discovers underlying patterns and detects anomalies across diverse data modalities without relying on arbitrary heuristic score thresholds. 

By integrating **inductive conformal prediction**, `cp-pde` provides rigorous statistical Type I error bounds (False Positive Rate $\le \alpha$) and delivers a structured tripartite decision framework (`anomaly`, `review`, and `no_alert`) to safely manage borderline cases.

---

## Core Capabilities

- **Continuous Tabular Data**: Geometric ensemble combining HDBSCAN, Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM with Bonferroni-corrected conformal bounds.
- **Heterogeneous / Mixed-Type Data**: Leakage-free Gower distance computation paired with robust k-Medoids++ clustering and automated elbow detection (`KneeLocator`).
- **Sequential & Temporal Dynamics**: Semi-Markovian modeling coupling Laplace-smoothed transition matrices with state-conditional log-normal holding time distributions.
- **Symbolic Association & Rule Mining**: Quantile-based feature discretization, Apriori rule extraction, contextual violation scoring, and dependency graphs.
- **Conformal Calibration**: Distribution-free, non-parametric p-value inference with mathematically bounded false positive rates.
- **Distribution Drift Monitoring**: Real-time Kolmogorov-Smirnov goodness-of-fit testing tracking shifts from uniform p-value distributions.
- **Lossless Persistence**: Native JSON serialization and restoration preserving exact numerical state across save/load cycles.

---

## Installation

Install the package via `pip`:

```bash
pip install cp-pde

```

### Requirements

* Python `>= 3.12` (excluding `3.14.1`)

---

## Package Structure

| Module | Core Responsibility |
| --- | --- |
| `pde.geometry` | Continuous geometric anomaly detection suites and Bonferroni ensembles |
| `pde.clustering` | Gower distance metrics, robust k-Medoids++, and optimal $k$ selection |
| `pde.sequential` | Semi-Markovian trajectory modeling (state transitions and holding times) |
| `pde.association` | Discretization, association rule mining, and contextual violation scoring |
| `pde.calibration` | Non-parametric conformal calibration and selective tripartite classification |
| `pde.drift` | Streaming Kolmogorov-Smirnov distribution drift monitoring |
| `pde.persistence` | Full engine serialization and deserialization via JSON |

---

## Development

Clone the repository and install development dependencies using [Poetry](https://python-poetry.org/):

```bash
git clone [https://github.com/cpadlab/pde.git](https://github.com/cpadlab/pde.git)
cd pde
poetry install

```

Run test suite and coverage reports:

```bash
poetry run pytest

```

---

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.