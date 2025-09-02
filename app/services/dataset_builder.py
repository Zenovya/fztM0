from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

from app.core.paths import ACTIVE_DIR


# ----------------------------
# Utilidades de lectura/fechas
# ----------------------------
def _read_active_csv(name: str) -> pd.DataFrame:
    path = ACTIVE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Activa una versión primero.")
    return pd.read_csv(path)


def _coerce_date_any(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")  # ISO y varios comunes
    if s.isna().all():
        s = pd.to_datetime(s.astype(str), dayfirst=True, errors="coerce")
    return s


def _norm_incident_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "incident_date" in df.columns:
        pass
    elif "date" in df.columns:
        df = df.rename(columns={"date": "incident_date"})
    else:
        raise ValueError("incidents.csv debe contener 'incident_date' o 'date'")
    df["incident_date"] = _coerce_date_any(df["incident_date"])
    return df


# ----------------------------
# Etiquetas y features core
# ----------------------------
def _labels_in_horizon(
    incidents: pd.DataFrame, ref_date: pd.Timestamp, horizon_days: int
) -> pd.DataFrame:
    future_end = ref_date + pd.Timedelta(days=horizon_days)
    incidents = _norm_incident_columns(incidents)
    inc_future = incidents[
        (incidents["incident_date"] > ref_date)
        & (incidents["incident_date"] <= future_end)
    ]
    y = (inc_future.groupby("asset_id").size() > 0).astype(int)
    return y.rename("y").to_frame()


def _features_core_at_ref(
    assets: pd.DataFrame, incidents: pd.DataFrame, ref_date: pd.Timestamp
) -> pd.DataFrame:
    assets = assets.copy()
    incidents = _norm_incident_columns(incidents)

    # Fechas
    assets["delivery_date"] = _coerce_date_any(assets["delivery_date"])

    inc_past = incidents[incidents["incident_date"] <= ref_date].copy()
    agg = inc_past.groupby("asset_id").agg(
        failures_prev=("incident_date", "count"),
        last_failure=("incident_date", "max"),
    )

    # Base de features (num + cat mínimas)
    if "total_services" not in assets.columns:
        assets["total_services"] = 0
    out = assets[
        ["asset_id", "model", "school", "delivery_date", "total_services"]
    ].copy()
    out = out.merge(agg, on="asset_id", how="left")

    # Derivadas temporales
    out["age_days"] = (ref_date - out["delivery_date"]).dt.days.clip(lower=0)
    out["age_months"] = out["age_days"] / 30.0
    out["days_since_last_failure"] = (ref_date - out["last_failure"]).dt.days
    out["days_since_last_failure"] = (
        out["days_since_last_failure"].fillna(out["age_days"]).clip(lower=0)
    )
    out["failures_prev"] = out["failures_prev"].fillna(0).astype(int)

    # Ventanas móviles
    def _w(df, days):
        start = ref_date - pd.Timedelta(days=days)
        m = (df["incident_date"] > start) & (df["incident_date"] <= ref_date)
        return df.loc[m].groupby("asset_id").size()

    for days, name in [
        (30, "inc_last30"),
        (90, "inc_last90"),
        (180, "inc_last180"),
        (365, "inc_last365"),
    ]:
        out = out.merge(_w(inc_past, days).rename(name), on="asset_id", how="left")

    # Ratios
    out["failures_per_year"] = out["failures_prev"] / np.maximum(
        out["age_days"] / 365.25, 0.1
    )
    out["services_per_year"] = out["total_services"] / np.maximum(
        out["age_days"] / 365.25, 0.1
    )

    # Limpieza
    num_cols = [
        "age_months",
        "failures_prev",
        "days_since_last_failure",
        "inc_last30",
        "inc_last90",
        "inc_last180",
        "inc_last365",
        "failures_per_year",
        "services_per_year",
        "total_services",
    ]
    for c in num_cols:
        if c not in out:
            out[c] = 0.0
    out[num_cols] = out[num_cols].fillna(0.0)
    out[["model", "school"]] = out[["model", "school"]].astype(str)
    return out


# ----------------------------
# Bundle de dataset
# ----------------------------
@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    columns: List[str]
    meta: Dict


def build_dataset(
    horizon_days: int = 90, train_window_days: int = 365 * 2
) -> DatasetBundle:
    assets = _read_active_csv("assets.csv")
    incidents = _read_active_csv("incidents.csv")

    inc_tmp = _norm_incident_columns(incidents)
    max_incident_date = inc_tmp["incident_date"].max()
    if pd.isna(max_incident_date):
        raise ValueError("incidents.csv no contiene fechas válidas")

    cutoff_test = max_incident_date - pd.Timedelta(days=horizon_days)
    cutoff_train = cutoff_test - pd.Timedelta(days=train_window_days)

    Xtr_core = _features_core_at_ref(assets, inc_tmp, cutoff_train)
    Xte_core = _features_core_at_ref(assets, inc_tmp, cutoff_test)

    ytr_df = _labels_in_horizon(inc_tmp, cutoff_train, horizon_days)
    yte_df = _labels_in_horizon(inc_tmp, cutoff_test, horizon_days)
    train = Xtr_core.merge(ytr_df, on="asset_id", how="left").fillna({"y": 0})
    test = Xte_core.merge(yte_df, on="asset_id", how="left").fillna({"y": 0})

    baseline_num = ["age_months", "failures_prev", "days_since_last_failure"]

    # Numéricas (core + ventanas)
    num_core = [
        "age_months",
        "failures_prev",
        "days_since_last_failure",
        "failures_per_year",
        "services_per_year",
        "total_services",
        "inc_last30",
        "inc_last90",
        "inc_last180",
        "inc_last365",
    ]
    num_cols_all = sorted(list(set(num_core)))

    # Categóricas mínimas
    cat_cols = ["model", "school"]
    for c in cat_cols:
        if c not in train:
            train[c] = "NA"
        if c not in test:
            test[c] = "NA"

    X_train_num = train[baseline_num].to_numpy(dtype=float)
    X_test_num = test[baseline_num].to_numpy(dtype=float)

    return DatasetBundle(
        X_train=X_train_num,
        y_train=train["y"].to_numpy(dtype=int),
        X_test=X_test_num,
        y_test=test["y"].to_numpy(dtype=int),
        columns=baseline_num,
        meta={
            "cutoff_train": cutoff_train.strftime("%Y-%m-%d"),
            "cutoff_test": cutoff_test.strftime("%Y-%m-%d"),
            "horizon_days": horizon_days,
            "n_assets": int(assets["asset_id"].nunique()),
            "X_train_full": train[cat_cols + num_cols_all],
            "X_test_full": test[cat_cols + num_cols_all],
            "asset_index_train": train[["asset_id"]],
            "asset_index_test": test[["asset_id"]],
            "feature_config": {
                "categorical": cat_cols,
                "numeric": num_cols_all,
                "baseline_numeric": baseline_num,
            },
        },
    )
