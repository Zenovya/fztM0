# app/api/risk.py
from __future__ import annotations
from fastapi import APIRouter
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime
import pandas as pd
from joblib import load
import json

from app.core.paths import PROJECT_ROOT
from app.services.dataset_builder import _read_active_csv, _features_core_at_ref

router = APIRouter(prefix="/risk", tags=["risk"])
MODELS_DIR = PROJECT_ROOT / "models"


def _load_selected_model() -> Optional[tuple[str, Path, Dict]]:
    if not MODELS_DIR.exists():
        return None
    subdirs = sorted([p for p in MODELS_DIR.iterdir() if p.is_dir()])
    if not subdirs:
        return None
    latest = subdirs[-1]
    sel_path = latest / "model_selected.txt"
    schema_path = latest / "feature_schema.json"
    if not sel_path.exists() or not schema_path.exists():
        return None
    name = sel_path.read_text().strip()
    schema = json.loads(schema_path.read_text())
    if name == "baseline":
        return ("baseline", latest, schema)
    model_file = latest / f"{name}.joblib"
    return (name, model_file, schema) if model_file.exists() else None


def _features_now(ref_date: pd.Timestamp) -> pd.DataFrame:
    """
    Construye features al corte 'ref_date' usando solo:
      - assets.csv (requerido)
      - incidents.csv (requerido; acepta 'incident_date' o 'date')
    """
    assets = _read_active_csv("assets.csv")
    incidents = _read_active_csv("incidents.csv")
    return _features_core_at_ref(assets, incidents, ref_date)


@router.post("/train", summary="Entrenar y registrar artefactos con datos activos")
def train(horizon_days: int = 90, k_top: int = 50):
    from app.services.dataset_builder import build_dataset
    from app.services.modeling import train_and_save

    ds = build_dataset(horizon_days=horizon_days)
    arts = train_and_save(ds, k_top=k_top)
    return {
        "ok": True,
        "version_id": arts.version_id,
        "metrics": arts.metrics,
        "models_dir": str(arts.path),
    }


@router.get("/top", summary="Top-K activos por riesgo", response_model=List[Dict])
def top(k: int = 50):
    sel = _load_selected_model()
    if sel is None:
        return []
    name, loc, schema = sel
    now = pd.Timestamp(datetime.now().date())
    feats = _features_now(now)

    if name == "baseline":
        from app.services.modeling import _fit_baseline

        cols = ["age_months", "failures_prev", "days_since_last_failure"]
        for c in cols:
            if c not in feats:
                feats[c] = 0.0
        scores = _fit_baseline(feats[cols].to_numpy())
    else:
        model = load(loc)
        cats = schema.get("categorical", [])
        nums = schema.get("numeric", [])
        for c in cats:
            if c not in feats:
                feats[c] = "NA"
        for c in nums:
            if c not in feats:
                feats[c] = 0.0
        scores = model.predict_proba(feats[cats + nums])[:, 1]

    out = feats.copy()
    out["score"] = scores
    out = out.sort_values("score", ascending=False).head(k)

    keep = [
        c
        for c in [
            "asset_id",
            "score",
            "model",
            "school",
            "age_months",
            "failures_prev",
            "days_since_last_failure",
        ]
        if c in out.columns
    ]
    return out[keep].to_dict(orient="records")


@router.get("/predict", summary="Riesgo para un asset_id específico")
def predict(asset_id: str):
    sel = _load_selected_model()
    if sel is None:
        return {
            "asset_id": asset_id,
            "score": None,
            "detail": "No hay modelo entrenado",
        }

    name, loc, schema = sel
    now = pd.Timestamp(datetime.now().date())
    feats = _features_now(now)

    row = feats[feats["asset_id"] == asset_id]
    if row.empty:
        return {
            "asset_id": asset_id,
            "score": None,
            "detail": "asset_id no encontrado en assets.csv",
        }

    if name == "baseline":
        from app.services.modeling import _fit_baseline

        score = float(
            _fit_baseline(
                row[
                    ["age_months", "failures_prev", "days_since_last_failure"]
                ].to_numpy()
            )[0]
        )
        model_name = "baseline"
    else:
        model = load(loc)
        cats = schema.get("categorical", [])
        nums = schema.get("numeric", [])
        for c in cats:
            if c not in row:
                row[c] = "NA"
        for c in nums:
            if c not in row:
                row[c] = 0.0
        score = float(model.predict_proba(row[cats + nums])[:, 1][0])
        model_name = Path(loc).stem

    return {
        "asset_id": asset_id,
        "score": score,
        "features": row.drop(columns=["asset_id"]).to_dict(orient="records")[0],
        "model": model_name,
    }
