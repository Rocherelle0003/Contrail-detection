"""Boucle d'entraînement générique + orchestration d'une expérience complète.

Une expérience = une config (dict) décrivant seed, modèle, hyperparamètres et
chemins. `run_experiment` fait tourner l'entraînement, évalue sur le split
validation, sauvegarde le meilleur checkpoint (par Dice de validation) et les
métriques, puis retourne tout ce qu'il faut pour l'analyse (historique,
métriques par record, prédictions).
"""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np import torch
from torch.utils.data import DataLoader

from src.data.dataset import ContrailAshRGBDataset
from src.evaluation.metrics import compute_all_metrics
from src.models.factory import build_model
from src.training.losses import BCEDiceLoss


def find_project_root(start: Path) -> Path:
    markers = ("requirements.txt", ".git")
    start = start.resolve()
    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    raise FileNotFoundError(f"Racine du projet introuvable à partir de {start}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_one_epoch(model, loader, criterion, device, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    per_record: list[dict[str, Any]] = []

    for images, masks, record_ids in loader:
        images = images.to(device)
        probs = torch.sigmoid(model(images)).cpu().numpy()
        preds = probs >= threshold
        targets = masks.numpy() >= 0.5

        for i, record_id in enumerate(record_ids):
            metrics = compute_all_metrics(preds[i, 0], targets[i, 0])
            per_record.append({"record_id": record_id, **metrics})

    aggregate = {
        key: float(np.mean([r[key] for r in per_record])) for key in ("dice", "iou", "precision", "recall")
    }
    return aggregate, per_record


def run_experiment(config: dict[str, Any], project_root: Path | None = None) -> dict[str, Any]:
    if project_root is None:
        project_root = find_project_root(Path.cwd())

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = project_root / config["data_dir"]
    train_ds = ContrailAshRGBDataset(data_dir / "train")
    val_ds = ContrailAshRGBDataset(data_dir / "validation")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    model = build_model(config["model"], in_channels=config.get("in_channels", 3)).to(device)
    criterion = BCEDiceLoss(dice_weight=config.get("dice_weight", 0.5))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    checkpoint_dir = project_root / config.get("checkpoint_dir", "checkpoints") / config["name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"

    threshold = config.get("threshold", 0.5)
    history: list[dict[str, float]] = []
    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, config["epochs"] + 1):
        train_loss = _run_one_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_one_epoch(model, val_loader, criterion, device, optimizer=None)
        val_metrics, _ = evaluate(model, val_loader, device, threshold)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    # Recharge le meilleur checkpoint pour l'évaluation finale
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    final_val_metrics, final_val_per_record = evaluate(model, val_loader, device, threshold)

    reports_dir = project_root / config.get("reports_dir", "reports") / config["name"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "config": config,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "final_val_metrics": final_val_metrics,
        "final_val_per_record": final_val_per_record,
        "checkpoint_path": str(checkpoint_path),
    }

    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result
