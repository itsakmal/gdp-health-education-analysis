"""
Preprocessing pipeline for the World Bank wide-format dataset.

What it does:
1) Loads config + selected indicators.
2) Reads raw wide CSV (years as columns), keeps relevant metadata.
3) Melts to long format: (country, year, indicator, value).
4) Filters to chosen year window and chosen indicators (if provided).
5) Pivots to panel: one row per (country, year), indicators as columns.
6) Adds lagged features (1/3/5 years by default) and multi-year deltas.
7) Saves:
   - processed panel parquet (without imputation/scaling)
   - coverage QA: per-indicator and per-country completeness CSVs
   - country metadata CSV

**Important:** The imputation and standardization steps have been removed from this script
to prevent data leakage. These transformations should now be handled inside `train.py`
after the data has been split into training and validation sets.

Run from project root:
    python src/preprocess.py
Optional overrides:
    python src/preprocess.py --start_year 1980 --end_year 2023 --no_scale
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import yaml


# ---------- Paths & Config ----------

# find project root dynamically (two levels up from src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_SELECTED_INDICATORS = PROJECT_ROOT / "configs" / "selected_indicators.csv"

# columns we expect in the raw WB file (best-effort, we only keep those present)
META_CANDIDATES = [
    "FREQ",
    "REF_AREA",
    "REF_AREA_LABEL",
    "INDICATOR",
    "INDICATOR_LABEL",
    "UNIT_MEASURE",
]


@dataclass
class Settings:
    raw_csv: Path
    processed_parquet: Path
    selected_indicators_csv: Optional[Path]
    country_metadata_csv: Path
    start_year: int
    end_year: int
    lags: List[int]
    add_deltas_for: List[int]
    do_scale: bool = True


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load YAML config and produce Settings with sensible fallbacks."""
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    paths = cfg.get("paths", {})
    time = cfg.get("time", {})

    raw_csv = PROJECT_ROOT / paths.get("raw_csv", "data/raw/WB_GS_WIDEF.csv")
    processed_parquet = PROJECT_ROOT / paths.get("processed_parquet", "data/processed/panel_clean.parquet")
    selected_indicators_csv = PROJECT_ROOT / paths.get("selected_indicators_csv", "configs/selected_indicators.csv")
    if not (PROJECT_ROOT / "configs" / "selected_indicators.csv").exists():
        # allow None if user hasn't curated a list yet
        selected_indicators_csv = None
    country_meta = PROJECT_ROOT / paths.get("country_metadata_csv", "data/processed/country_meta.csv")

    start_year = int(time.get("start_year", 1980))
    end_year = int(time.get("end_year", 2023))
    lags = list(map(int, time.get("lag_years", [1, 3, 5, 10])))
    add_deltas_for = list(map(int, time.get("delta_years", [1, 3, 5, 10])))

    return Settings(
        raw_csv=raw_csv,
        processed_parquet=processed_parquet,
        selected_indicators_csv=Path(selected_indicators_csv) if selected_indicators_csv else None,
        country_metadata_csv=country_meta,
        start_year=start_year,
        end_year=end_year,
        lags=lags,
        add_deltas_for=add_deltas_for,
        do_scale=True,
    )


# ---------- Utilities ----------

def find_year_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that look like 4-digit years."""
    return [c for c in df.columns if c.isdigit() and len(c) == 4]


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _select_meta_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in META_CANDIDATES if c in df.columns]


def _sanitize_numeric(x):
    """Convert to float, coerce bad strings (e.g., '..') to NaN."""
    try:
        return float(x)
    except Exception:
        return np.nan


# ---------- Pipeline Steps ----------

def load_raw_dataframe(raw_csv: Path) -> pd.DataFrame:
    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw CSV not found at {raw_csv}. Place the dataset there.")
    df = pd.read_csv(raw_csv)
    # drop unnamed columns if any
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    return df


def melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = _select_meta_columns(df)
    year_cols = find_year_columns(df)
    if not year_cols:
        raise ValueError("No year columns detected (e.g., 1960..2023). Check your file.")
    long = df.melt(
        id_vars=meta_cols,
        value_vars=year_cols,
        var_name="YEAR",
        value_name="VALUE",
    )
    long["YEAR"] = long["YEAR"].astype(int)
    long["VALUE"] = long["VALUE"].apply(_sanitize_numeric).astype(float)
    return long


def filter_years(long: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    return long[(long["YEAR"] >= start_year) & (long["YEAR"] <= end_year)].copy()


def filter_selected_indicators(long: pd.DataFrame, selected_csv: Optional[Path]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    If a curated indicator list exists, inner-join to it.
    Returns (filtered_long, selected_table_or_None)
    """
    if selected_csv and selected_csv.exists():
        sel = pd.read_csv(selected_csv)
        need_cols = {"INDICATOR"}
        if not need_cols.issubset(set(sel.columns)):
            raise ValueError(f"{selected_csv} must contain at least the column INDICATOR.")
        filtered = long.merge(sel[["INDICATOR"]].drop_duplicates(), on="INDICATOR", how="inner")
        return filtered, sel
    return long, None


def pivot_to_panel(long: pd.DataFrame) -> pd.DataFrame:
    """
    index: (REF_AREA, REF_AREA_LABEL, YEAR)
    columns: INDICATOR (values: VALUE)
    """
    keep_cols = ["REF_AREA", "REF_AREA_LABEL", "YEAR", "INDICATOR", "VALUE"]
    missing = [c for c in keep_cols if c not in long.columns]
    if missing:
        raise ValueError(f"Long dataframe is missing required columns: {missing}")

    pivot = (
        long[keep_cols]
        .pivot_table(
            index=["REF_AREA", "REF_AREA_LABEL", "YEAR"],
            columns="INDICATOR",
            values="VALUE",
            aggfunc="mean",
        )
        .reset_index()
    )

    pivot = pivot.sort_values(["REF_AREA", "YEAR"]).reset_index(drop=True)
    return pivot


def compute_coverage_reports(pivot: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Coverage by indicator and by country (percent non-null)."""
    value_cols = [c for c in pivot.columns if c not in ("REF_AREA", "REF_AREA_LABEL", "YEAR")]
    by_indicator = (
        pivot[value_cols]
        .notna()
        .mean()
        .sort_values(ascending=False)
        .rename("coverage")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "INDICATOR"})
    )
    by_country = (
        pivot.assign(non_null=pivot[value_cols].notna().sum(axis=1))
        .groupby(["REF_AREA", "REF_AREA_LABEL"], as_index=False)["non_null"]
        .mean()
        .rename(columns={"non_null": "avg_non_null_features"})
    )
    return by_indicator, by_country


def add_lags_and_deltas(
    pivot: pd.DataFrame,
    lags: List[int],
    deltas: List[int],
) -> pd.DataFrame:
    """Add lagged features and multi-year deltas for every indicator column."""
    base_cols = ["REF_AREA", "REF_AREA_LABEL", "YEAR"]
    value_cols = [c for c in pivot.columns if c not in base_cols]
    out = pivot.copy()

    # Lags
    for L in sorted(set(lags)):
        lagged = (
            pivot.groupby("REF_AREA")[value_cols]
            .shift(L)
            .add_suffix(f"_lag{L}")
        )
        out = pd.concat([out, lagged], axis=1)

    # Deltas (current minus value L years ago)
    for D in sorted(set(deltas)):
        prev = pivot.groupby("REF_AREA")[value_cols].shift(D)
        delta = (pivot[value_cols] - prev).add_suffix(f"_chg{D}")
        out = pd.concat([out, delta], axis=1)

    return out


def save_country_metadata(long: pd.DataFrame, output_csv: Path) -> None:
    meta = long[["REF_AREA", "REF_AREA_LABEL"]].drop_duplicates().sort_values("REF_AREA_LABEL")
    ensure_dir(output_csv)
    meta.to_csv(output_csv, index=False)


def save_coverage_reports(
    by_indicator: pd.DataFrame,
    by_country: pd.DataFrame,
) -> None:
    ind_path = PROJECT_ROOT / "data" / "processed" / "coverage_by_indicator.csv"
    cty_path = PROJECT_ROOT / "data" / "processed" / "coverage_by_country.csv"
    ensure_dir(ind_path)
    by_indicator.to_csv(ind_path, index=False)
    by_country.to_csv(cty_path, index=False)
    print(f"[QA] Saved coverage_by_indicator → {ind_path}")
    print(f"[QA] Saved coverage_by_country  → {cty_path}")


# ---------- Orchestration ----------

def run_pipeline(settings: Settings) -> Path:
    print("=== Preprocessing Pipeline ===")
    print(f"Config: {DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else '(defaults)'}")
    print(f"Raw CSV: {settings.raw_csv}")

    raw = load_raw_dataframe(settings.raw_csv)
    print(f"[Load] Raw shape: {raw.shape}")

    long = melt_to_long(raw)
    print(f"[Melt] Long shape: {long.shape}")

    # optional filter to selected indicators
    long, sel_tbl = filter_selected_indicators(long, settings.selected_indicators_csv)
    if sel_tbl is not None:
        print(f"[Select] Using curated indicator list with {sel_tbl['INDICATOR'].nunique()} indicators.")

    # restrict years
    long = filter_years(long, settings.start_year, settings.end_year)
    print(f"[Years] Filtered to {settings.start_year}-{settings.end_year}, shape: {long.shape}")

    # save country metadata now (so the app can use it even if model not trained)
    save_country_metadata(long, settings.country_metadata_csv)
    print(f"[Meta] Wrote country metadata to {settings.country_metadata_csv}")

    # pivot to panel
    panel = pivot_to_panel(long)
    print(f"[Pivot] Panel shape: {panel.shape}")

    # coverage QA before imputation
    ind_cov, cty_cov = compute_coverage_reports(panel)
    save_coverage_reports(ind_cov, cty_cov)

    # lags & deltas
    enriched = add_lags_and_deltas(panel, lags=settings.lags, deltas=settings.add_deltas_for)
    print(f"[Features] Added lags {settings.lags} and deltas {settings.add_deltas_for}.")
    print(f"[Features] Final shape: {enriched.shape}")

    # save
    ensure_dir(settings.processed_parquet)
    enriched.to_parquet(settings.processed_parquet, index=False, engine="fastparquet")
    print(f"[Save] Wrote processed panel → {settings.processed_parquet}")

    return settings.processed_parquet


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess World Bank wide CSV into modeling panel.")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to configs/config.yaml")
    p.add_argument("--start_year", type=int, default=None, help="Override start year")
    p.add_argument("--end_year", type=int, default=None, help="Override end year")
    p.add_argument("--no_scale", action="store_true", help="Disable within-country standardization")
    return p.parse_args()


def main():
    args = parse_args()
    settings = load_config(Path(args.config) if args.config else DEFAULT_CONFIG_PATH)
    # apply cli overrides
    if args.start_year is not None:
        settings.start_year = int(args.start_year)
    if args.end_year is not None:
        settings.end_year = int(args.end_year)

    run_pipeline(settings)


if __name__ == "__main__":
    main()



