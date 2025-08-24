"""
Reusable plotting utilities (matplotlib only).

Usage inside notebooks/01_eda.ipynb:

    from src.visualization.plots import (
        plot_missing_bar,
        plot_indicator_coverage,
        lineplot_country_timeseries,
        scatter_xy,
        corr_heatmap,
        feature_importance_bar,
    )

    plot_missing_bar(df)  # raw/long/panel dfs supported
    plot_indicator_coverage(panel_df)

    lineplot_country_timeseries(panel_df, country="India",
                                columns=["GDP growth (annual %)", "Life expectancy at birth, total (years)"])

    scatter_xy(panel_df, x="Life expectancy at birth, total (years)_lag3",
               y="GDP growth (annual %)", annotate="REF_AREA_LABEL")

    corr_heatmap(panel_df, cols=[...], method="pearson")

    feature_importance_bar("data/processed/feature_importance.csv", top_n=25)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIG_DIR = Path("reports/figures")


# ---------- Helpers ----------

def _ensure_fig_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _title(t: Optional[str], default: str) -> str:
    return t if t else default


# ---------- Missingness / Coverage ----------

def plot_missing_bar(df: pd.DataFrame, top_n: int = 40, title: Optional[str] = None, save_as: Optional[str] = "missing_bar.png"):
    """
    Bar chart of columns with missing values (top_n by count).
    Works with raw/long/panel dataframes.
    """
    _ensure_fig_dir()
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, max(3, len(miss) * 0.25)))
    plt.barh(miss.index[::-1], miss.values[::-1])
    plt.xlabel("Missing values")
    plt.title(_title(title, "Missing values by column"))

    if save_as:
        path = FIG_DIR / save_as
        plt.tight_layout()
        plt.savefig(path, dpi=150)
    plt.show()


def plot_indicator_coverage(panel_df: pd.DataFrame, top_n: int = 40, title: Optional[str] = None, save_as: Optional[str] = "indicator_coverage.png"):
    """
    For a panel with indicator columns, show percent non-null by indicator.
    """
    _ensure_fig_dir()
    value_cols = [c for c in panel_df.columns if c not in ("REF_AREA", "REF_AREA_LABEL", "YEAR")]
    cov = panel_df[value_cols].notna().mean().sort_values(ascending=False)
    cov = cov.head(top_n)

    plt.figure(figsize=(10, max(3, len(cov) * 0.25)))
    plt.barh(cov.index[::-1], (cov.values * 100)[::-1])
    plt.xlabel("Coverage (%)")
    plt.title(_title(title, "Coverage by indicator (non-null %)"))

    if save_as:
        path = FIG_DIR / save_as
        plt.tight_layout()
        plt.savefig(path, dpi=150)
    plt.show()


# ---------- Time Series ----------

def lineplot_country_timeseries(panel_df: pd.DataFrame, country: str, columns: Iterable[str], title: Optional[str] = None, save_as: Optional[str] = None):
    """
    Plot one or more series over time for a selected country.
    """
    _ensure_fig_dir()
    sub = panel_df[panel_df["REF_AREA_LABEL"] == country].sort_values("YEAR")
    if sub.empty:
        raise ValueError(f"No rows for country '{country}'. Check REF_AREA_LABEL values.")

    plt.figure(figsize=(10, 5))
    for col in columns:
        if col not in sub.columns:
            continue
        plt.plot(sub["YEAR"].values, sub[col].values, label=col)

    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.title(_title(title, f"{country}: selected indicators over time"))
    plt.legend(loc="best")
    plt.tight_layout()

    if save_as:
        path = FIG_DIR / save_as
        plt.savefig(path, dpi=150)
    plt.show()


# ---------- Scatter & Correlation ----------

def scatter_xy(df: pd.DataFrame, x: str, y: str, annotate: Optional[str] = None, title: Optional[str] = None, save_as: Optional[str] = None):
    """
    Scatter plot of y vs x. Optionally annotate points with a column (e.g., 'REF_AREA_LABEL').
    """
    _ensure_fig_dir()
    sub = df[[x, y] + ([annotate] if annotate else [])].dropna()

    plt.figure(figsize=(7, 6))
    plt.scatter(sub[x].values, sub[y].values)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(_title(title, f"{y} vs {x}"))
    if annotate:
        # light annotation (skip if too many points)
        if len(sub) <= 150:
            for _, r in sub.iterrows():
                plt.annotate(str(r[annotate]), (r[x], r[y]), fontsize=8, alpha=0.7)
    plt.tight_layout()

    if save_as:
        path = FIG_DIR / (save_as if save_as.endswith(".png") else f"{save_as}.png")
        plt.savefig(path, dpi=150)
    plt.show()


def corr_heatmap(df: pd.DataFrame, cols: Optional[List[str]] = None, method: str = "pearson", title: Optional[str] = None, save_as: Optional[str] = "corr_heatmap.png"):
    """
    Correlation heatmap (matplotlib imshow). Provide 'cols' to focus on a subset.
    """
    _ensure_fig_dir()
    if cols is None:
        # auto-pick numeric columns (avoid REF_AREA/YEAR)
        cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ("YEAR",)]
    data = df[cols].astype(float)
    corr = data.corr(method=method)

    plt.figure(figsize=(max(6, len(cols) * 0.35), max(5, len(cols) * 0.35)))
    im = plt.imshow(corr.values, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title(_title(title, f"Correlation heatmap ({method})"))
    plt.tight_layout()

    if save_as:
        path = FIG_DIR / save_as
        plt.savefig(path, dpi=150)
    plt.show()


# ---------- Model Artifacts ----------

def feature_importance_bar(feature_importance_csv: str, top_n: int = 30, title: Optional[str] = None, save_as: Optional[str] = "feature_importance.png"):
    """
    Plot top-N feature importances from a CSV produced by src/models/train.py
    with columns: ['feature', 'importance'].
    """
    _ensure_fig_dir()
    path = Path(feature_importance_csv)
    if not path.exists():
        raise FileNotFoundError(f"Feature importance file not found: {path}")
    fi = pd.read_csv(path)
    fi = fi.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, max(3, top_n * 0.3)))
    plt.barh(fi["feature"].values[::-1], fi["importance"].values[::-1])
    plt.xlabel("Importance")
    plt.title(_title(title, "Top feature importances"))
    plt.tight_layout()

    if save_as:
        path = FIG_DIR / save_as
        plt.savefig(path, dpi=150)
    plt.show()
