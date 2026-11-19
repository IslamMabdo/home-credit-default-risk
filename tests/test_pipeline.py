"""Unit tests for schema validation, leak-free feature encoding, and end-to-end serving."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import PipelineConfig
from credit_risk.features import (
    FoldEncoder,
    align_categoricals,
    build_domain_features,
    make_discrete_keys,
    validate_schema,
)
from credit_risk.model import CreditRiskBundle, train_cv_bundle


def _make_synthetic_dataset(n_rows: int = 300, seed: int = 42, include_target: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(1000, 1000 + n_rows),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], size=n_rows, p=[0.9, 0.1]),
            "CODE_GENDER": rng.choice(["F", "M"], size=n_rows),
            "CNT_CHILDREN": rng.integers(0, 4, size=n_rows),
            "AMT_INCOME_TOTAL": rng.choice([90000.0, 135000.0, 180000.0, 225000.0], size=n_rows),
            "AMT_CREDIT": rng.choice([225000.0, 450000.0, 545040.0, 675000.0], size=n_rows),
            "AMT_ANNUITY": rng.choice([12500.0, 22500.0, 26640.0, 31500.0], size=n_rows),
            "AMT_GOODS_PRICE": rng.choice([225000.0, 450000.0, 454500.0, 675000.0], size=n_rows),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "Commercial associate"], size=n_rows),
            "NAME_EDUCATION_TYPE": rng.choice(
                ["Secondary / secondary special", "Higher education", "Lower secondary"], size=n_rows
            ),
            "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married"], size=n_rows),
            "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "Rented apartment"], size=n_rows),
            "REGION_POPULATION_RELATIVE": rng.choice([0.006852, 0.01885, 0.072508], size=n_rows),
            "DAYS_BIRTH": rng.integers(-23000, -8000, size=n_rows),
            "DAYS_EMPLOYED": np.where(rng.random(n_rows) < 0.18, 365243, rng.integers(-6000, -100, size=n_rows)),
            "DAYS_REGISTRATION": rng.integers(-12000, -100, size=n_rows).astype(float),
            "DAYS_ID_PUBLISH": rng.integers(-5000, -10, size=n_rows),
            "OWN_CAR_AGE": np.where(rng.random(n_rows) < 0.65, np.nan, rng.choice([3.0, 10.0, 64.0], size=n_rows)),
            "FLAG_MOBIL": np.ones(n_rows, dtype=int),
            "FLAG_EMP_PHONE": rng.integers(0, 2, size=n_rows),
            "FLAG_WORK_PHONE": rng.integers(0, 2, size=n_rows),
            "OCCUPATION_TYPE": rng.choice(["Laborers", "Core staff", "Managers", "Other"], size=n_rows),
            "CNT_FAM_MEMBERS": rng.integers(1, 5, size=n_rows).astype(float),
            "REGION_RATING_CLIENT": rng.choice([1, 2, 3], size=n_rows),
            "ORGANIZATION_TYPE": rng.choice(["Business Entity Type 3", "XNA", "Self-employed"], size=n_rows),
            "EXT_SOURCE_1": np.where(rng.random(n_rows) < 0.65, np.nan, rng.uniform(0.1, 0.9, size=n_rows)),
            "EXT_SOURCE_2": rng.uniform(0.1, 0.85, size=n_rows),
            "EXT_SOURCE_3": np.where(rng.random(n_rows) < 0.30, np.nan, rng.uniform(0.1, 0.85, size=n_rows)),
            "DAYS_LAST_PHONE_CHANGE": np.where(rng.random(n_rows) < 0.12, 0.0, rng.integers(-2500, -1, size=n_rows)),
            "AMT_REQ_CREDIT_BUREAU_HOUR": np.where(rng.random(n_rows) < 0.14, np.nan, 0.0),
            "AMT_REQ_CREDIT_BUREAU_MON": np.where(rng.random(n_rows) < 0.14, np.nan, rng.integers(0, 3, size=n_rows)),
            "AMT_REQ_CREDIT_BUREAU_QRT": np.where(rng.random(n_rows) < 0.14, np.nan, rng.integers(0, 3, size=n_rows)),
            "AMT_REQ_CREDIT_BUREAU_YEAR": np.where(rng.random(n_rows) < 0.14, np.nan, rng.integers(0, 5, size=n_rows)),
        }
    )
    if include_target:
        # Synthetic signal correlated with EXT_SOURCE_2 so CV is non-trivial
        logit_p = -2.2 - 2.0 * (df["EXT_SOURCE_2"] - 0.5)
        probs = 1.0 / (1.0 + np.exp(-logit_p))
        df["TARGET"] = (rng.random(n_rows) < probs).astype(int)
    return df


def test_schema_validation_catches_missing_columns() -> None:
    df = _make_synthetic_dataset(20)
    validate_schema(df, require_target=True)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(df.drop(columns=["EXT_SOURCE_2"]), require_target=True)


def test_domain_features_clean_anomalies_and_are_deterministic() -> None:
    df = _make_synthetic_dataset(50)
    f1 = build_domain_features(df)
    f2 = build_domain_features(df)
    pd.testing.assert_frame_equal(f1, f2)
    assert "FLAG_MOBIL" not in f1.columns
    assert (f1.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED_ANOM"] == 1).all()
    assert f1.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED_CLEAN"].isna().all()
    assert f1.loc[df["OWN_CAR_AGE"] >= 60, "OWN_CAR_AGE_CLEAN"].isna().all()


def test_fold_encoder_handles_unseen_categories_without_nans() -> None:
    df = _make_synthetic_dataset(120)
    X_dom, cats = align_categoricals(build_domain_features(df))
    keys = make_discrete_keys(df)
    enc = FoldEncoder(seed=42)
    X_tr = enc.fit_transform_train(X_dom.iloc[:100], keys.iloc[:100], df["TARGET"].values[:100])

    # Mutate unseen category in inference slice
    unseen_df = df.iloc[100:].copy()
    unseen_df["AMT_GOODS_PRICE"] = 9999999.0
    X_u, _ = align_categoricals(build_domain_features(unseen_df), cats)
    X_out = enc.transform(X_u, make_discrete_keys(unseen_df))

    assert list(X_tr.columns) == list(X_out.columns)
    assert np.isfinite(X_out["TE_EXACT_GOODS"].values).all()
    assert np.isclose(X_out["TE_EXACT_GOODS"].values, enc.prior_).all()


def test_end_to_end_train_save_load_predict(tmp_path: Path) -> None:
    train_df = _make_synthetic_dataset(240, seed=1, include_target=True)
    test_df = _make_synthetic_dataset(40, seed=2, include_target=False)

    cfg = PipelineConfig(seed=42, n_splits=3)
    cfg.features.use_ext_imputation = True
    cfg.features.ext_imputer_params = {"n_estimators": 20, "min_child_samples": 10, "early_stopping_rounds": 10}
    cfg.model.type = "xgboost"
    cfg.model.xgboost = {
        "n_estimators": 30,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "early_stopping_rounds": 10,
    }

    bundle, oof_prob, summary = train_cv_bundle(train_df, cfg)
    assert len(oof_prob) == len(train_df)
    assert np.isfinite(oof_prob).all()
    assert 0.5 < summary["oof_roc_auc"] <= 1.0

    # Save & reload standalone bundle and score unseen batch (no train_df needed!)
    bundle_path = bundle.save(tmp_path / "model_bundle.joblib")
    loaded = CreditRiskBundle.load(bundle_path)
    preds = loaded.predict_proba(test_df)

    assert preds.shape == (len(test_df),)
    assert np.isfinite(preds).all()
    assert (preds > 0.0).all() and (preds < 1.0).all()
