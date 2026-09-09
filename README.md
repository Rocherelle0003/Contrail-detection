# Contrail Detection

Research-oriented Master's project on binary semantic segmentation of aircraft condensation trails (contrails) in GOES-16 ABI satellite imagery using the OpenContrails dataset.

## Motivation: aviation and climate context

Persistent aviation contrails can influence Earth’s radiative balance and are an important topic in aviation-climate research. Building reliable automatic contrail detection pipelines from geostationary satellite imagery supports scalable monitoring and downstream climate-impact studies.

## Problem definition

This project targets **binary semantic segmentation**:
- **Input**: GOES-16 ABI image data (spatiotemporal satellite observations).
- **Output**: pixel-level mask with two classes: `contrail` vs `background`.

## GOES-16 and ABI

GOES-16 (Geostationary Operational Environmental Satellite) provides continuous Earth observations from geostationary orbit. Its Advanced Baseline Imager (ABI) captures multi-spectral imagery suitable for identifying cloud and contrail patterns over time.

## OpenContrails dataset

OpenContrails is the dataset planned for this project’s experiments. This repository intentionally does **not** include dataset files. Data access, storage, and preprocessing scripts will be added in later milestones.

## Planned workflow

1. **Data exploration**: inspect imagery, masks, quality, and class balance.
2. **U-Net baseline**: establish a reproducible segmentation baseline.
3. **Temporal experiments**: evaluate methods that leverage time context.
4. **Uncertainty and error analysis**: analyze failure modes and prediction confidence.
5. **Final evaluation**: compare approaches and summarize findings.

## Planned evaluation metrics

- **Dice coefficient**
- **IoU / Jaccard index**
- **Precision**
- **Recall**

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
- Track metrics and visual outputs in a structured way (`reports/`).
- Prefer deterministic training/evaluation settings when possible.
- Avoid committing datasets, checkpoints, and secrets.

## Current scope

This commit provides project scaffolding and documentation only:
- no model implementation,
- no training pipeline,
- no dataset download or storage in-repo,
- no experimental claims or fabricated results.
