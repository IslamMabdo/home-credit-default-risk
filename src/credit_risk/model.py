"""Stage-4 model training, fold bundling, and standalone production inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import PipelineConfig
from .features import (
    DEFAULT_TE_SMOOTHING,
    TARGET_COLUMN,
    ExtSourceImputer,
    FoldEncoder,
    align_categoricals,
    build_domain_features,
    make_discrete_keys,
    validate_schema,
)

logger = logging.getLogger(__name__)


@dataclass
class CreditRiskBundle:
    """Self-contained, serializable production inference bundle.

    Holds every artifact required to score unseen loan applications standalone
    (without access to training data at inference time):
      - Frozen categorical vocabularies (`categories_map`)
      - Fitted target-free `ExtSourceImputer` (optional)
      - Per-fold `(FoldEncoder, model)` pairs for fold-averaged predictions
      - Feature schema & training metadata
    """

    model_type: str
    categories_map: dict[str, list[str]]
    feature_names: list[str]
    ext_imputer: ExtSourceImputer | None
    fold_encoders: list[FoldEncoder]
    xgb_models: list[xgb.XGBClassifier] = field(default_factory=list)
    lgb_models: list[lgb.LGBMClassifier] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Generate calibrated default probabilities for unseen loan applications."""
        validate_schema(raw_df, require_target=False)

        X_dom = build_domain_features(raw_df)
        X_dom, _ = align_categoricals(X_dom, self.categories_map)
        keys = make_discrete_keys(raw_df)

        if self.ext_imputer is not None:
            ext_feats = self.ext_imputer.transform(X_dom)
            X_base = pd.concat([X_dom, ext_feats], axis=1)
        else:
            X_base = X_dom

        n_folds = len(self.fold_encoders)
        xgb_pred = np.zeros(len(raw_df), dtype=np.float64)
        lgb_pred = np.zeros(len(raw_df), dtype=np.float64)

        for fold, enc in enumerate(self.fold_encoders):
            X_f = enc.transform(X_base, keys)[self.feature_names]
            if self.xgb_models:
                xgb_pred += self.xgb_models[fold].predict_proba(X_f)[:, 1] / n_folds
            if self.lgb_models:
                lgb_pred += self.lgb_models[fold].predict_proba(X_f)[:, 1] / n_folds

        if self.model_type == "xgboost":
            probs = xgb_pred
        elif self.model_type == "lightgbm":
            probs = lgb_pred
        elif self.model_type == "blend":
            l_x = logit(np.clip(xgb_pred, 1e-6, 1.0 - 1e-6))
            l_l = logit(np.clip(lgb_pred, 1e-6, 1.0 - 1e-6))
            probs = expit(0.5 * l_x + 0.5 * l_l)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        return np.clip(probs, 1e-6, 1.0 - 1e-6)

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path, compress=3)
        logger.info("Saved model bundle to %s", out_path)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "CreditRiskBundle":
        return joblib.load(path)


def _fit_xgb_fold(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cfg: PipelineConfig,
    seed: int,
) -> xgb.XGBClassifier:
    params = dict(cfg.model.xgboost)
    params.setdefault("objective", "binary:logistic")
    params.setdefault("eval_metric", "auc")
    params.setdefault("tree_method", "hist")
    params.setdefault("enable_categorical", True)
    params.setdefault("max_cat_to_onehot", 4)
    params.setdefault("max_cat_threshold", 32)
    params.setdefault("learning_rate", 0.015)
    params.setdefault("max_depth", 4)
    params.setdefault("min_child_weight", 100)
    params.setdefault("max_bin", 512)
    params.setdefault("subsample", 0.8)
    params.setdefault("colsample_bytree", 0.25)
    params.setdefault("colsample_bylevel", 0.7)
    params.setdefault("reg_alpha", 8.0)
    params.setdefault("reg_lambda", 50.0)
    params.setdefault("gamma", 0.2)
    params.setdefault("n_estimators", 6000)
    params.setdefault("early_stopping_rounds", 150)
    params["device"] = "cuda" if cfg.model.use_gpu else "cpu"
    params["random_state"] = seed
    params["n_jobs"] = -1

    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def _fit_lgb_fold(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cfg: PipelineConfig,
    seed: int,
) -> lgb.LGBMClassifier:
    params = dict(cfg.model.lightgbm)
    early_stop = int(params.pop("early_stopping_rounds", 120))
    params.setdefault("objective", "binary")
    params.setdefault("metric", "auc")
    params.setdefault("learning_rate", 0.02)
    params.setdefault("num_leaves", 20)
    params.setdefault("max_depth", 5)
    params.setdefault("max_bin", 511)
    params.setdefault("min_child_samples", 250)
    params.setdefault("subsample", 0.8)
    params.setdefault("subsample_freq", 1)
    params.setdefault("colsample_bytree", 0.25)
    params.setdefault("cat_smooth", 120.0)
    params.setdefault("reg_alpha", 10.0)
    params.setdefault("reg_lambda", 50.0)
    params.setdefault("min_split_gain", 0.05)
    params.setdefault("n_estimators", 5000)
    params["random_state"] = seed
    params["n_jobs"] = -1
    params["verbose"] = -1

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(early_stop, verbose=False)],
    )
    return model


def train_cv_bundle(
    train_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> tuple[CreditRiskBundle, np.ndarray, dict[str, Any]]:
    """Run leak-free Stratified K-Fold CV and return a production-ready bundle + OOF predictions."""
    validate_schema(train_df, require_target=True)
    y = train_df[TARGET_COLUMN].values.astype(int)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    splits = list(skf.split(train_df, y))

    logger.info("Building stateless domain features for %d rows...", len(train_df))
    X_dom = build_domain_features(train_df)
    X_dom, categories_map = align_categoricals(X_dom)
    keys_all = make_discrete_keys(train_df)

    ext_imputer: ExtSourceImputer | None = None
    if cfg.features.use_ext_imputation:
        logger.info("Fitting target-free OOF EXT_SOURCE_1/2/3 LightGBM imputer...")
        ext_imputer = ExtSourceImputer(params=cfg.features.ext_imputer_params, seed=cfg.seed)
        ext_oof = ext_imputer.fit_transform_oof(X_dom, splits)
        logger.info("EXT_SOURCE OOF Pearson r: %s", {k: round(v, 4) for k, v in ext_imputer.correlations_.items()})
        X_base = pd.concat([X_dom, ext_oof], axis=1)
    else:
        X_base = X_dom

    te_smoothing = cfg.features.te_smoothing or dict(DEFAULT_TE_SMOOTHING)
    model_type = cfg.model.type
    if model_type not in {"xgboost", "lightgbm", "blend"}:
        raise ValueError(f"model.type must be one of 'xgboost', 'lightgbm', 'blend'; got '{model_type}'")

    fold_encoders: list[FoldEncoder] = []
    xgb_models: list[xgb.XGBClassifier] = []
    lgb_models: list[lgb.LGBMClassifier] = []
    oof_xgb = np.zeros(len(train_df), dtype=np.float64)
    oof_lgb = np.zeros(len(train_df), dtype=np.float64)
    fold_aucs: list[float] = []
    feature_names: list[str] = []

    for fold, (tr_i, va_i) in enumerate(splits):
        enc = FoldEncoder(te_smoothing=te_smoothing, seed=cfg.seed + 100 + fold)
        X_tr = enc.fit_transform_train(X_base.iloc[tr_i], keys_all.iloc[tr_i], y[tr_i])
        X_va = enc.transform(X_base.iloc[va_i], keys_all.iloc[va_i])
        if not feature_names:
            feature_names = list(X_tr.columns)
        X_tr = X_tr[feature_names]
        X_va = X_va[feature_names]
        fold_encoders.append(enc)

        if model_type in {"xgboost", "blend"}:
            m_xgb = _fit_xgb_fold(X_tr, y[tr_i], X_va, y[va_i], cfg, seed=cfg.seed + 200 + fold)
            xgb_models.append(m_xgb)
            oof_xgb[va_i] = m_xgb.predict_proba(X_va)[:, 1]

        if model_type in {"lightgbm", "blend"}:
            m_lgb = _fit_lgb_fold(X_tr, y[tr_i], X_va, y[va_i], cfg, seed=cfg.seed + 300 + fold)
            lgb_models.append(m_lgb)
            oof_lgb[va_i] = m_lgb.predict_proba(X_va)[:, 1]

        if model_type == "xgboost":
            p_va = oof_xgb[va_i]
        elif model_type == "lightgbm":
            p_va = oof_lgb[va_i]
        else:
            p_va = expit(0.5 * logit(np.clip(oof_xgb[va_i], 1e-6, 1 - 1e-6)) + 0.5 * logit(np.clip(oof_lgb[va_i], 1e-6, 1 - 1e-6)))

        f_auc = float(roc_auc_score(y[va_i], p_va))
        fold_aucs.append(f_auc)
        logger.info("Fold %d/%d ROC-AUC = %.5f (%d features)", fold + 1, cfg.n_splits, f_auc, len(feature_names))

    if model_type == "xgboost":
        oof_final = oof_xgb
    elif model_type == "lightgbm":
        oof_final = oof_lgb
    else:
        oof_final = expit(
            0.5 * logit(np.clip(oof_xgb, 1e-6, 1.0 - 1e-6))
            + 0.5 * logit(np.clip(oof_lgb, 1e-6, 1.0 - 1e-6))
        )

    cv_summary = {
        "model_type": model_type,
        "n_splits": cfg.n_splits,
        "n_features": len(feature_names),
        "oof_roc_auc": float(roc_auc_score(y, oof_final)),
        "fold_roc_auc_mean": float(np.mean(fold_aucs)),
        "fold_roc_auc_std": float(np.std(fold_aucs)),
        "fold_roc_aucs": fold_aucs,
        "ext_imputation_pearson_r": ext_imputer.correlations_ if ext_imputer else {},
    }
    if model_type == "blend":
        cv_summary["xgb_oof_roc_auc"] = float(roc_auc_score(y, oof_xgb))
        cv_summary["lgb_oof_roc_auc"] = float(roc_auc_score(y, oof_lgb))

    bundle = CreditRiskBundle(
        model_type=model_type,
        categories_map=categories_map,
        feature_names=feature_names,
        ext_imputer=ext_imputer,
        fold_encoders=fold_encoders,
        xgb_models=xgb_models,
        lgb_models=lgb_models,
        metadata=cv_summary,
    )
    return bundle, oof_final, cv_summary
