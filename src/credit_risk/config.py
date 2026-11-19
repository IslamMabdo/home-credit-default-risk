"""Typed configuration loader for the credit risk pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    train_csv: Path = Path("data/train.csv")
    test_csv: Path = Path("data/test.csv")
    artifact_dir: Path = Path("artifacts")
    output_csv: Path = Path("outputs/submission.csv")


@dataclass
class FeaturesConfig:
    use_ext_imputation: bool = True
    ext_imputer_params: dict[str, Any] = field(default_factory=dict)
    te_smoothing: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelConfig:
    type: str = "xgboost"
    use_gpu: bool = False
    xgboost: dict[str, Any] = field(default_factory=dict)
    lightgbm: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    bootstrap_samples: int = 500
    decision_threshold: float = 0.15


@dataclass
class PipelineConfig:
    seed: int = 42
    n_splits: int = 5
    paths: PathsConfig = field(default_factory=PathsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(path: str | Path = "configs/config.yaml") -> PipelineConfig:
    """Load YAML pipeline configuration into typed dataclasses."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        train_csv=Path(paths_raw.get("train_csv", "data/train.csv")),
        test_csv=Path(paths_raw.get("test_csv", "data/test.csv")),
        artifact_dir=Path(paths_raw.get("artifact_dir", "artifacts")),
        output_csv=Path(paths_raw.get("output_csv", "outputs/submission.csv")),
    )

    feat_raw = raw.get("features", {})
    features = FeaturesConfig(
        use_ext_imputation=bool(feat_raw.get("use_ext_imputation", True)),
        ext_imputer_params=dict(feat_raw.get("ext_imputer_params", {})),
        te_smoothing={k: float(v) for k, v in feat_raw.get("te_smoothing", {}).items()},
    )

    mod_raw = raw.get("model", {})
    model = ModelConfig(
        type=str(mod_raw.get("type", "xgboost")).lower(),
        use_gpu=bool(mod_raw.get("use_gpu", False)),
        xgboost=dict(mod_raw.get("xgboost", {})),
        lightgbm=dict(mod_raw.get("lightgbm", {})),
    )

    eval_raw = raw.get("evaluation", {})
    evaluation = EvaluationConfig(
        bootstrap_samples=int(eval_raw.get("bootstrap_samples", 500)),
        decision_threshold=float(eval_raw.get("decision_threshold", 0.15)),
    )

    return PipelineConfig(
        seed=int(raw.get("seed", 42)),
        n_splits=int(raw.get("n_splits", 5)),
        paths=paths,
        features=features,
        model=model,
        evaluation=evaluation,
    )
