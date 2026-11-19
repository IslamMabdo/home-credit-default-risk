"""Evaluation metrics, bootstrap confidence intervals, and slice error analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.15,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute core probabilistic classification metrics and bootstrap 95% ROC-AUC CI."""
    auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    rng = np.random.default_rng(seed)
    boot_aucs = np.empty(n_bootstrap, dtype=np.float64)
    n = len(y_true)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_aucs[i] = roc_auc_score(y_true[idx], y_prob[idx])
    ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])

    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int))
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    return {
        "roc_auc": auc,
        "roc_auc_ci95_low": float(ci_low),
        "roc_auc_ci95_high": float(ci_high),
        "pr_auc": pr_auc,
        "brier_score": brier,
        "threshold": threshold,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def slice_error_analysis(
    raw_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """Evaluate ROC-AUC across key business and data-completeness subpopulations."""
    ext_cnt = raw_df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].notna().sum(axis=1)
    slices = {
        "Contract: Cash loans": raw_df["NAME_CONTRACT_TYPE"] == "Cash loans",
        "Contract: Revolving loans": raw_df["NAME_CONTRACT_TYPE"] == "Revolving loans",
        "Gender: Female": raw_df["CODE_GENDER"] == "F",
        "Gender: Male": raw_df["CODE_GENDER"] == "M",
        "EXT_SOURCE count == 3 (complete)": ext_cnt == 3,
        "EXT_SOURCE count == 2": ext_cnt == 2,
        "EXT_SOURCE count <= 1 (sparse)": ext_cnt <= 1,
        "Employment: Pensioner/Anom": raw_df["DAYS_EMPLOYED"] == 365243,
        "Employment: Active Worker": raw_df["DAYS_EMPLOYED"] != 365243,
    }
    rows = []
    for name, mask in slices.items():
        m = mask.values
        if m.sum() == 0 or len(np.unique(y_true[m])) < 2:
            continue
        rows.append(
            {
                "slice": name,
                "count": int(m.sum()),
                "default_rate": float(y_true[m].mean()),
                "roc_auc": float(roc_auc_score(y_true[m], y_prob[m])),
            }
        )
    return pd.DataFrame(rows)
