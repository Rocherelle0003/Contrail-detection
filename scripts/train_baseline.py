"""CLI pour lancer une expérience de baseline à partir d'un fichier de config YAML.

Usage :
    python scripts/train_baseline.py configs/baseline_unet_scratch.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import find_project_root, run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Chemin vers un fichier de config YAML")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    project_root = find_project_root(Path.cwd())
    print(f"Racine du projet : {project_root}")
    print(f"Expérience : {config['name']} (modèle={config['model']}, epochs={config['epochs']})")

    result = run_experiment(config, project_root=project_root)

    print(f"Meilleure epoch : {result['best_epoch']} (Dice val = {result['best_val_dice']:.4f})")
    print("Métriques finales (validation) :", result["final_val_metrics"])


if __name__ == "__main__":
    main()
