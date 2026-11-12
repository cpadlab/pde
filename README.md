# Pattern Discovery Engine

A Python laboratory and library exploring classical machine learning, pattern discovery, and conformal anomaly detection from first principles—without LLMs or black-box shortcuts.

> Read the full research notes and experimental breakdown in the [development article](https://cpadlab.github.io/posts/pattern-discovery-engine).

---

## Overview

**Pattern Discovery Engine (`cp-pde`)** evolved from a standalone clustering script into a modular Python package designed to test how classical algorithms behave, fail, and calibrate under controlled synthetic conditions.

Instead of relying on arbitrary score thresholds, the lab grounds every detector in **inductive conformal prediction**, providing distribution-free false positive control ($\text{FPR} \le \alpha$) and a tripartite decision framework (`anomaly`, `review`, `no_alert`) to abstain when statistical evidence is insufficient.

---

## Lab Modules & Capabilities

- **Geometric Detectors (`pde.geometry`)**: Benchmarks HDBSCAN, Isolation Forest, LOF, and One-Class SVM across distinct anomaly geometries, combining them via Bonferroni-corrected conformal bounds.
- **Symbolic Rules & Graphs (`pde.association`)**: Quantile-based feature discretization, Apriori rule mining, contextual rule-violation scoring, and NetworkX co-occurrence graphs.
- **Heterogeneous Clustering (`pde.clustering`)**: Leakage-free Gower distance paired with multi-start k-Medoids++ and automated $k$ selection via elbow detection.
- **Temporal Sequences (`pde.sequential`)**: Semi-Markov models combining Laplace-smoothed transition matrices with log-normal inter-event holding times.
- **Conformal Calibration (`pde.calibration`)**: Empirical p-value calibration on held-out partitions for scale-free ensembling and selective abstention (`review`).
- **Label-Free Drift Monitoring (`pde.drift`)**: Streaming Kolmogorov-Smirnov tests detecting distribution shifts via deviations from uniform p-values, triggering automatic recalibration.
- **Lossless Persistence (`pde.persistence`)**: Deterministic JSON serialization preserving exact engine state across `.save()` and `.load()` cycles.

---

## Quickstart

Install via `pip` (requires Python `>= 3.12`, excluding `3.14.1`):

```bash
pip install cp-pde
```

---

## Laboratory Methodology

All experiments in this repository follow three strict rules to prevent data leakage and p-hacking:

1. **Controlled Seeds**: Evaluated across 5 fixed seeds (`42, 7, 123, 2026, 99`) to measure variance rather than best-case runs.
2. **Strict Partitions**: Scalers, discretizers, and models are fitted exclusively on `train`; p-values are calibrated on an independent `calibration` split.
3. **Diagnostic Honesty**: Metrics are validated on synthetic generators with negative controls to verify statistical properties, not as production guarantees.

---

## Development

Clone the repository and install dependencies with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/cpadlab/pde.git
cd pde
poetry install
poetry run pytest
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.