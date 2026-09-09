# Contrail Detection

Research-oriented AI and aviation project focused on automatic detection of aircraft condensation trails (contrails) in GOES-16 ABI satellite imagery using the OpenContrails dataset.

## Motivation: aviation and climate context

Persistent aviation contrails can affect Earth’s radiative balance and are an important topic in aviation-climate science. Reliable, automated contrail segmentation from geostationary satellite imagery can support large-scale monitoring, impact assessment workflows, and future operational decision support.

## Problem definition

This project targets **binary semantic segmentation**:
- **Input**: GOES-16 ABI image data (spatiotemporal satellite observations).
- **Output**: pixel-level mask with two classes: `contrail` vs `background`.

## GOES-16 and ABI

GOES-16 (Geostationary Operational Environmental Satellite) provides continuous observations from geostationary orbit. Its Advanced Baseline Imager (ABI) captures multi-spectral measurements that are suitable for analyzing cloud structures and contrail evolution over time.

## OpenContrails dataset

OpenContrails is the planned dataset foundation for model development and evaluation in this repository. Dataset files are intentionally not stored in this repository.

## Research ambitions

- Reproducible scientific experimentation with clear configuration and versioning.
- Strong literature review and explicit comparison with prior work.
- Robust baseline reproduction before introducing new modeling ideas.
- Systematic investigation of temporal context for contrail segmentation.
- Uncertainty-aware learning that accounts for human annotator disagreement.
- Calibration-focused analysis of predictive uncertainty.
- Controlled temporal ablation studies and structured error analysis.
- Robustness and generalization assessment across scene conditions.
- Model efficiency analysis with potential operational relevance.
- Hypothesis-driven experiments with controlled comparisons and transparent reporting.

## Research questions (provisional)

1. How much does temporal context improve contrail segmentation compared with single-frame approaches?
2. Which temporal frames contribute the most useful information?
3. Can disagreement between human annotators be used to model and calibrate predictive uncertainty?
4. Does model uncertainty correlate with areas of human disagreement?
5. Can a lightweight architecture achieve competitive segmentation performance with lower computational cost?
6. Which scene characteristics are associated with false positives and false negatives?

## Planned workflow

1. **Data exploration**: inspect imagery, masks, metadata quality, and class balance.
2. **Baseline reproduction**: implement and validate a strong U-Net baseline under controlled settings.
3. **Temporal experiments**: evaluate temporal context strategies and temporal frame selection.
4. **Uncertainty and disagreement modeling**: analyze uncertainty behavior, calibration, and annotator disagreement signals.
5. **Ablation, robustness, and error analysis**: run systematic ablations and identify dominant failure modes.
6. **Final evaluation and comparison**: report reproducible results and compare against relevant prior approaches.

## Planned evaluation metrics

Core segmentation metrics:
- **Dice coefficient**
- **IoU / Jaccard index**
- **Precision**
- **Recall**

Uncertainty/calibration metrics (where relevant to the experiment design):
- Calibration-oriented metrics (for example, ECE-style reliability analysis)
- Uncertainty-quality analyses linked to disagreement and error patterns

## Installation

```bash
git clone https://github.com/Rocherelle0003/Contrail-detection.git
cd Contrail-detection
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Repository structure

```text
contrail-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_baseline_results.ipynb
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── scripts/
├── reports/
│   └── figures/
└── tests/
```

## Reproducibility principles

- Keep experiments configuration-driven (`configs/`) and version-controlled.
- Separate data handling, modeling, training, and evaluation logic (`src/`).
- Track metrics, artifacts, and figures in a structured, auditable workflow (`reports/`).
- Use explicit seeds and deterministic settings when feasible.
- Record hypotheses, protocol decisions, and evaluation settings for each experiment.
- Avoid committing datasets, checkpoints, or secrets.

## Current scope

This repository currently contains project scaffolding and documentation only:
- no model implementation,
- no training pipeline,
- no dataset download or storage in-repo,
- no claims of novelty, state-of-the-art performance, publication, or validated scientific contribution at this stage.
