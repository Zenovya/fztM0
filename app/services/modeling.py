from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
from datetime import datetime
from joblib import dump

from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
import warnings

from app.services.dataset_builder import DatasetBundle
from app.core.paths import PROJECT_ROOT

# Silenciar avisos de convergencia de la logística calibrada (ruido en entrenamiento)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Recall@K: proporción de positivos reales capturados dentro del Top-K por score.
    Maneja K > n y el caso y_true.sum()==0.
    """
    k = max(1, min(k, len(y_score)))
    order = np.argsort(-y_score)
    topk = order[:k]
    denom = int(y_true.sum())
    if denom == 0:
        return 0.0
    return float(y_true[topk].sum() / denom)


def _fit_baseline(X_numeric: np.ndarray) -> np.ndarray:
    """
    Baseline interno (regla fija suavizada con sigmoide).
    X_numeric = [age_months, failures_prev, days_since_last_failure]
    """
    age = X_numeric[:, 0]
    fails = X_numeric[:, 1]
    age_n = (age - age.mean()) / (age.std() + 1e-6)
    fails_n = (fails - fails.mean()) / (fails.std() + 1e-6)
    score = 0.6 * age_n + 0.4 * fails_n
    return 1.0 / (1.0 + np.exp(-score))


@dataclass
class ModelArtifacts:
    version_id: str
    path: Path
    metrics: Dict[str, Any]
    feature_columns_numeric: list[str]
    feature_columns_all: list[str]


def train_and_save(
    data: DatasetBundle,
    k_top: int,
    random_state: int = 42,
    n_estimators: int = 400,
) -> ModelArtifacts:
    """
    Entrena:
      - Baseline (interno para comparar),
      - Regresión logística calibrada,
      - Random Forest,
    y guarda artefactos + métrica comparativa (PR-AUC y Recall@K).
    """
    # === Baseline matrices (numéricas mínimas del proyecto)
    Xtr_num, ytr = data.X_train, data.y_train
    Xte_num, yte = data.X_test, data.y_test

    # === DataFrames completos + config de features del builder
    Xtr_full = data.meta["X_train_full"]
    Xte_full = data.meta["X_test_full"]
    feat_cfg = data.meta["feature_config"]
    cat_cols = feat_cfg["categorical"]
    num_cols = feat_cfg["numeric"]

    # ---------- Baseline ----------
    base_tr = _fit_baseline(Xtr_num)
    base_te = _fit_baseline(Xte_num)

    # ---------- Preprocesamiento ----------
    # Nota: en sklearn >=1.7, OneHotEncoder usa 'sparse_output' (True por defecto).
    cat_pipe = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="constant", fill_value="NA")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )

    # ---------- Regresión Logística calibrada ----------
    logit = LogisticRegression(
        max_iter=5000,
        solver="saga",
        penalty="l2",
        class_weight="balanced",
        n_jobs=-1,
        tol=1e-4,
        random_state=random_state,
    )
    logit_pipe = Pipeline(steps=[("pre", pre), ("clf", logit)])
    logit_cal = CalibratedClassifierCV(logit_pipe, method="isotonic", cv=3)
    logit_cal.fit(Xtr_full, ytr)
    logit_te = logit_cal.predict_proba(Xte_full)[:, 1]

    # ---------- Random Forest ----------
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf_pipe = Pipeline(steps=[("pre", pre), ("clf", rf)])
    rf_pipe.fit(Xtr_full, ytr)
    rf_te = rf_pipe.predict_proba(Xte_full)[:, 1]

    # ---------- Métricas ----------
    metrics = {
        "horizon_days": data.meta["horizon_days"],
        "cutoff_train": data.meta["cutoff_train"],
        "cutoff_test": data.meta["cutoff_test"],
        "pr_auc": {
            "baseline": float(average_precision_score(yte, base_te)),
            "logistic": float(average_precision_score(yte, logit_te)),
            "random_forest": float(average_precision_score(yte, rf_te)),
        },
        "recall_at_k": {
            "k": int(k_top),
            "baseline": float(recall_at_k(yte, base_te, k_top)),
            "logistic": float(recall_at_k(yte, logit_te, k_top)),
            "random_forest": float(recall_at_k(yte, rf_te, k_top)),
        },
        "features": {"numeric": num_cols, "categorical": cat_cols},
    }

    # ---------- Selección y guardado ----------
    best_name = max(metrics["pr_auc"], key=metrics["pr_auc"].get)
    best_model = {"logistic": logit_cal, "random_forest": rf_pipe}.get(best_name, None)

    version_id = "m" + datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = MODELS_DIR / version_id
    out_dir.mkdir(parents=True, exist_ok=False)

    # Persistir artefactos
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "feature_schema.json").write_text(
        json.dumps({"numeric": num_cols, "categorical": cat_cols}, indent=2)
    )
    if best_model is not None:
        dump(best_model, out_dir / f"{best_name}.joblib")
        (out_dir / "model_selected.txt").write_text(best_name)
    else:
        # Si por alguna razón ninguno supera el baseline (poco probable), quedarse con baseline
        (out_dir / "model_selected.txt").write_text("baseline")

    return ModelArtifacts(
        version_id=version_id,
        path=out_dir,
        metrics=metrics,
        feature_columns_numeric=num_cols,
        feature_columns_all=cat_cols + num_cols,
    )
