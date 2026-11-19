"""Production CLI entrypoint for training, evaluation, and batch prediction.

Usage:
    python -m credit_risk.pipeline train --config configs/config.yaml
    python -m credit_risk.pipeline predict --config configs/config.yaml --input data/test.csv --output outputs/submission.csv
    python -m credit_risk.pipeline evaluate --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .evaluate import compute_metrics, slice_error_analysis
from .features import ID_COLUMN, TARGET_COLUMN
from .model import CreditRiskBundle, train_cv_bundle


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.model_type:
        cfg.model.type = args.model_type
    if args.gpu:
        cfg.model.use_gpu = True

    train_path = Path(args.train) if args.train else cfg.paths.train_csv
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else cfg.paths.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading training data from %s", train_path)
    train_df = pd.read_csv(train_path)
    y = train_df[TARGET_COLUMN].values.astype(int)

    bundle, oof_prob, cv_summary = train_cv_bundle(train_df, cfg)
    metrics = compute_metrics(
        y_true=y,
        y_prob=oof_prob,
        threshold=cfg.evaluation.decision_threshold,
        n_bootstrap=cfg.evaluation.bootstrap_samples,
        seed=cfg.seed,
    )
    slices_df = slice_error_analysis(train_df, y, oof_prob)

    bundle_path = artifact_dir / "model_bundle.joblib"
    bundle.save(bundle_path)

    oof_df = pd.DataFrame({ID_COLUMN: train_df[ID_COLUMN].values, TARGET_COLUMN: y, "PRED_PROB": oof_prob})
    oof_df.to_csv(artifact_dir / "oof_predictions.csv", index=False)
    slices_df.to_csv(artifact_dir / "slice_metrics.csv", index=False)

    report = {"cv_summary": cv_summary, "metrics": metrics}
    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info(
        "Training complete | OOF ROC-AUC = %.5f (95%% CI [%.5f, %.5f]) | PR-AUC = %.5f | Brier = %.5f",
        metrics["roc_auc"],
        metrics["roc_auc_ci95_low"],
        metrics["roc_auc_ci95_high"],
        metrics["pr_auc"],
        metrics["brier_score"],
    )


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else cfg.paths.artifact_dir
    bundle_path = artifact_dir / "model_bundle.joblib"
    input_path = Path(args.input) if args.input else cfg.paths.test_csv
    output_path = Path(args.output) if args.output else cfg.paths.output_csv

    logging.info("Loading bundle from %s", bundle_path)
    bundle = CreditRiskBundle.load(bundle_path)

    logging.info("Scoring %s...", input_path)
    df = pd.read_csv(input_path)
    probs = bundle.predict_proba(df)

    if not np.isfinite(probs).all():
        raise RuntimeError("Non-finite prediction values produced during inference.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID_COLUMN: df[ID_COLUMN].values, TARGET_COLUMN: probs})
    sub.to_csv(output_path, index=False)
    logging.info(
        "Wrote %d predictions to %s (mean=%.5f, std=%.5f, min=%.5f, max=%.5f)",
        len(sub),
        output_path,
        float(probs.mean()),
        float(probs.std()),
        float(probs.min()),
        float(probs.max()),
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else cfg.paths.artifact_dir
    metrics_path = artifact_dir / "metrics.json"
    slices_path = artifact_dir / "slice_metrics.csv"

    with metrics_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    print(json.dumps(report, indent=2))
    if slices_path.exists():
        print("\nSlice-Based Error Analysis:")
        print(pd.read_csv(slices_path).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit-risk",
        description="Stage-4 Home Credit Default Risk Training & Batch Inference CLI",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train CV model bundle and save artifacts")
    p_train.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    p_train.add_argument("--train", default=None, help="Override path to train.csv")
    p_train.add_argument("--artifact-dir", default=None, help="Override artifact output directory")
    p_train.add_argument("--model-type", choices=["xgboost", "lightgbm", "blend"], default=None)
    p_train.add_argument("--gpu", action="store_true", help="Enable GPU training for XGBoost")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Run standalone batch inference from saved bundle")
    p_pred.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    p_pred.add_argument("--artifact-dir", default=None, help="Artifact directory containing model_bundle.joblib")
    p_pred.add_argument("--input", default=None, help="Input CSV path (e.g., data/test.csv)")
    p_pred.add_argument("--output", default=None, help="Output predictions CSV path")
    p_pred.set_defaults(func=cmd_predict)

    p_eval = sub.add_parser("evaluate", help="Display saved CV metrics and slice report")
    p_eval.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    p_eval.add_argument("--artifact-dir", default=None, help="Artifact directory")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
