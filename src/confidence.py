"""Fit and apply loop-closure acceptance gates without test-set tuning."""

from __future__ import annotations

import math
from typing import Any

import torch


FEATURE_NAMES = (
    "retrieval_score",
    "sequence_score",
    "inlier_ratio",
    "log_num_matches",
)


def _confusion(
    accepted: torch.Tensor,
    top1_correct: torch.Tensor,
    has_match: torch.Tensor,
) -> dict[str, float]:
    accepted = accepted.bool()
    top1_correct = top1_correct.bool()
    has_match = has_match.bool()

    tp = int((accepted & top1_correct).sum())
    fp = int((accepted & ~top1_correct).sum())
    fn = int((~accepted & has_match).sum())
    tn = int((~accepted & ~has_match).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(int(has_match.sum()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def _candidate_thresholds(values: torch.Tensor) -> list[float]:
    unique = sorted({float(value) for value in values.tolist()})
    if not unique:
        return [float("inf")]
    return unique + [math.nextafter(unique[-1], float("inf"))]


def _best_threshold(
    scores: torch.Tensor,
    top1_correct: torch.Tensor,
    has_match: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    best_threshold = float("inf")
    best_metrics = _confusion(
        torch.zeros_like(scores, dtype=torch.bool), top1_correct, has_match
    )
    best_key = (best_metrics["f1"], best_metrics["precision"], best_metrics["recall"])

    for threshold in _candidate_thresholds(scores):
        metrics = _confusion(scores >= threshold, top1_correct, has_match)
        key = (metrics["f1"], metrics["precision"], metrics["recall"])
        if key > best_key:
            best_threshold, best_metrics, best_key = threshold, metrics, key
    return best_threshold, best_metrics


def signal_matrix(signals: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([signals[name].float() for name in FEATURE_NAMES], dim=1)


def fit_gate(
    mode: str,
    signals: dict[str, torch.Tensor],
    top1_correct: torch.Tensor,
    has_match: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Fit a gate on a validation split. The returned config is test-ready JSON."""
    if mode == "similarity":
        threshold, metrics = _best_threshold(
            signals["sequence_score"], top1_correct, has_match
        )
        return {"version": 1, "gate_mode": mode, "sequence_threshold": threshold}, metrics

    if mode == "geometry":
        threshold, metrics = _best_threshold(
            signals["inlier_ratio"], top1_correct, has_match
        )
        return {"version": 1, "gate_mode": mode, "inlier_ratio_threshold": threshold}, metrics

    if mode == "joint":
        best_config: dict[str, Any] = {}
        best_metrics = {"f1": -1.0, "precision": -1.0, "recall": -1.0}
        best_key = (-1.0, -1.0, -1.0)
        for sequence_threshold in _candidate_thresholds(signals["sequence_score"]):
            sequence_ok = signals["sequence_score"] >= sequence_threshold
            for ratio_threshold in _candidate_thresholds(signals["inlier_ratio"]):
                accepted = sequence_ok & (signals["inlier_ratio"] >= ratio_threshold)
                metrics = _confusion(accepted, top1_correct, has_match)
                key = (metrics["f1"], metrics["precision"], metrics["recall"])
                if key > best_key:
                    best_key = key
                    best_metrics = metrics
                    best_config = {
                        "version": 1,
                        "gate_mode": mode,
                        "sequence_threshold": sequence_threshold,
                        "inlier_ratio_threshold": ratio_threshold,
                    }
        return best_config, best_metrics

    if mode == "logistic":
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover - dependency error is user-facing
            raise RuntimeError("logistic gate requires scikit-learn") from exc

        labels = top1_correct.int().numpy()
        if len(set(labels.tolist())) < 2:
            raise ValueError("logistic gate needs both correct and incorrect validation examples")
        model = LogisticRegression(class_weight="balanced", random_state=0, max_iter=1000)
        features = signal_matrix(signals).numpy()
        model.fit(features, labels)
        probabilities = torch.from_numpy(model.predict_proba(features)[:, 1]).float()
        threshold, metrics = _best_threshold(probabilities, top1_correct, has_match)
        return {
            "version": 1,
            "gate_mode": mode,
            "feature_names": list(FEATURE_NAMES),
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "probability_threshold": threshold,
        }, metrics

    raise ValueError(f"unknown gate mode: {mode}")


def gate_scores(config: dict[str, Any], signals: dict[str, torch.Tensor]) -> torch.Tensor:
    mode = config["gate_mode"]
    if mode in {"similarity", "joint"}:
        return signals["sequence_score"]
    if mode == "geometry":
        return signals["inlier_ratio"]
    if mode == "logistic":
        coefficients = torch.tensor(config["coefficients"], dtype=torch.float32)
        logits = signal_matrix(signals) @ coefficients + float(config["intercept"])
        return torch.sigmoid(logits)
    raise ValueError(f"unknown gate mode: {mode}")


def apply_gate(config: dict[str, Any], signals: dict[str, torch.Tensor]) -> torch.Tensor:
    mode = config["gate_mode"]
    if mode == "similarity":
        return signals["sequence_score"] >= float(config["sequence_threshold"])
    if mode == "geometry":
        return signals["inlier_ratio"] >= float(config["inlier_ratio_threshold"])
    if mode == "joint":
        return (
            (signals["sequence_score"] >= float(config["sequence_threshold"]))
            & (signals["inlier_ratio"] >= float(config["inlier_ratio_threshold"]))
        )
    if mode == "logistic":
        return gate_scores(config, signals) >= float(config["probability_threshold"])
    raise ValueError(f"unknown gate mode: {mode}")

