"""Métriques de segmentation binaire : Dice, IoU, Precision, Recall.

Toutes les fonctions prennent des masques binaires (bool ou {0,1}) de même
forme et retournent un float. Convention : quand l'union prédiction/vérité
terrain est vide (pas de contrail prédit ni annoté), la métrique vaut 1.0
plutôt que d'être indéfinie — un pixel vide correctement prédit vide est un
succès, pas une absence de signal.
"""

import numpy as np


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool)


def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = _as_bool(pred), _as_bool(target)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(pred, target).sum() / denom


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = _as_bool(pred), _as_bool(target)
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return np.logical_and(pred, target).sum() / union


def precision_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = _as_bool(pred), _as_bool(target)
    predicted_positive = pred.sum()
    if predicted_positive == 0:
        return 1.0 if target.sum() == 0 else 0.0
    return np.logical_and(pred, target).sum() / predicted_positive


def recall_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = _as_bool(pred), _as_bool(target)
    actual_positive = target.sum()
    if actual_positive == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return np.logical_and(pred, target).sum() / actual_positive


def compute_all_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "dice": dice_coefficient(pred, target),
        "iou": iou_score(pred, target),
        "precision": precision_score(pred, target),
        "recall": recall_score(pred, target),
    }
