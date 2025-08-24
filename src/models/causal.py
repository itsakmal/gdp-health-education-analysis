"""
causal.py

Panel causal analysis (two-way fixed effects) on the processed World Bank panel.

Model:
    y_{i,t} = beta*T_{i,t} + gamma'X_{i,t} + alpha_i + delta_t + e_{i,t}
where:
    - y is the target (default: WB_GS_NY_GDP_PCAP_KD)
    - T is/are treatment(s) you specify (e.g., education/health/spending indicators)
    - X are optional controls
    - alpha_i are country fixed effects (C(REF_AREA))
    - delta_t are year fixed effects (C(YEAR))
    - Clustered standard errors by country.

Outputs:
    - models/causal_params_<target>.csv  (coef table for treatments + controls)
    - models/causal_summary_<target>.txt (full regression summary)
    - models/causal_config_<timestamp>.yaml (what was actually used)

Usage examples:
    python src/causal.py --treatments WB_ED_SE_PRM_UNA_TOT --controls WB_SP_POP_TOTL,WB_NY_GDP_DEFL_KD_ZG
    python src/causal.py --target NY.GDP.PCAP.KD --treatments WB_SH_XPD_CHEX_PCAP_PP_CD
    python src/causal.py --treatments WB_SH_XPD_CHEX_PCAP_PP_CD --start_year 1990 --end_year 2020
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

import statsmodels.api as sm
import statsmodels.formula.api as smf

from datetime import datetime, timezone
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ---------- Paths & Config ----------
# NOTE: We mirror the path behavior you used in train.py so this drops in cleanly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]        # likely .../src
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data/processed/panel_clean.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


# ---------- Helpers ----------
def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict:
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_data(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed panel not found at {data_path}. "
            f"Run preprocess.py first."
        )
    df = pd.read_parquet(data_path)
    print(f"[Data] Loaded processed panel: {df.shape}")
    return df


def normalize_target_alias(name: str) -> str:
    """
    Match the alias logic from train.py so users can pass older names safely.
    """
    alias_map = {
        "GDP per capita (constant 2010 US$)": "WB_GS_NY_GDP_PCAP_KD",
        "GDP per capita": "WB_GS_NY_GDP_PCAP_KD",
        "NY.GDP.PCAP.KD": "WB_GS_NY_GDP_PCAP_KD",
    }
    return alias_map.get(name, name)


def parse_csv_list(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    return [s.strip() for s in arg.split(",") if s.strip()]


def years_window(df: pd.DataFrame, start: Optional[int], end: Optional[int]) -> pd.DataFrame:
    if start is not None:
        df = df[df["YEAR"] >= int(start)]
    if end is not None:
        df = df[df["YEAR"] <= int(end)]
    return df


def ensure_required_columns(df: pd.DataFrame, cols: List[str]) -> Tuple[List[str], List[str]]:
    """Return (present, missing) preserving the original order."""
    present, missing = [], []
    for c in cols:
        if c in df.columns:
            present.append(c)
        else:
            missing.append(c)
    return present, missing


def drop_small_panels(df: pd.DataFrame, min_years_per_country: int = 3) -> pd.DataFrame:
    counts = df.groupby("REF_AREA")["YEAR"].nunique()
    keep_ids = counts[counts >= min_years_per_country].index
    kept = df[df["REF_AREA"].isin(keep_ids)].copy()
    dropped = len(df["REF_AREA"].unique()) - len(kept["REF_AREA"].unique())
    if dropped > 0:
        print(f"[Filter] Dropped {dropped} countries with < {min_years_per_country} distinct years.")
    return kept


def safe_impute_controls(df: pd.DataFrame, controls: List[str]) -> pd.DataFrame:
    """
    For controls only, impute with global mean to avoid losing rows.
    Treatments and target are not imputed (rows with NaN in those are dropped).
    """
    if not controls:
        return df
    for c in controls:
        if c in df.columns:
            m = df[c].mean(skipna=True)
            df[c] = df[c].fillna(m)
    return df


def build_formula(target: str, treatments: List[str], controls: List[str]) -> str:
    if not treatments:
        raise ValueError("No treatments provided. Use --treatments or set 'causal.treatments' in config.yaml.")
    rhs_vars = treatments + controls
    rhs = " + ".join(rhs_vars) if rhs_vars else "1"
    # Country & Year fixed effects:
    fe = " + C(REF_AREA) + C(YEAR)"
    return f"{target} ~ {rhs}{fe}"


def cluster_fit(formula: str, data: pd.DataFrame, cluster_col: str = "REF_AREA"):
    """
    Fit OLS with two-way fixed effects via dummies and clustered SE by country.
    """
    print(f"[Model] Fitting OLS with FE and clustered SE (cluster={cluster_col})")
    model = smf.ols(formula=formula, data=data)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]}, use_t=True)
    return fit


def save_outputs(fit, target: str, used: Dict):
    # --- FIX: Format the timestamp into a filename-safe string ---
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Params table (coef + SE + t + p + CI)
    params_df = fit.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    
    # --- FIX: Use the formatted string in the filename ---
    params_path = MODEL_DIR / f"causal_params_{target}_{timestamp_str}.csv"
    params_df.to_csv(params_path, index=False)
    print(f"[Save] Params → {params_path}")

    # Full summary
    # --- FIX: Use the formatted string in the filename ---
    summ_path = MODEL_DIR / f"causal_summary_{target}_{timestamp_str}.txt"
    with open(summ_path, "w", encoding="utf-8") as f:
        f.write(str(fit.summary()))
    print(f"[Save] Summary → {summ_path}")

    # Config actually used
    # --- FIX: Use the formatted string in the filename ---
    cfg_path = MODEL_DIR / f"causal_config_{target}_{timestamp_str}.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(used, f, sort_keys=False)
    print(f"[Save] Config used → {cfg_path}")


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Causal (TWFE) analysis on processed panel.")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to configs/config.yaml")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to processed panel parquet")
    p.add_argument("--target", type=str, default=None, help="Target column (aliases handled like in train.py)")
    p.add_argument("--treatments", type=str, default=None, help="Comma-separated treatment columns")
    p.add_argument("--controls", type=str, default=None, help="Comma-separated control columns")
    p.add_argument("--start_year", type=int, default=None, help="Restrict to start year (optional)")
    p.add_argument("--end_year", type=int, default=None, help="Restrict to end year (optional)")
    p.add_argument("--min_years_per_country", type=int, default=3, help="Drop countries with < this many distinct years")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config (optional)
    cfg = load_config(Path(args.config))

    # Figure out target
    target = args.target or (cfg.get("targets", {}) or {}).get("economic_target", "WB_GS_NY_GDP_PCAP_KD")
    target = normalize_target_alias(target)

    # Treatments / Controls from CLI first; otherwise config; otherwise empty list
    cli_treatments = parse_csv_list(args.treatments)
    cli_controls = parse_csv_list(args.controls)

    cfg_causal = (cfg.get("causal", {}) if isinstance(cfg.get("causal", {}), dict) else {})
    cfg_treatments = cfg_causal.get("treatments", []) if cfg_causal else []
    cfg_controls = cfg_causal.get("controls", []) if cfg_causal else []

    treatments = cli_treatments or cfg_treatments
    controls = cli_controls or cfg_controls

    # Load data
    df = load_data(Path(args.data))

    # Restrict years if requested
    df = years_window(df, args.start_year, args.end_year)
    print(f"[Years] Using years {df['YEAR'].min()}–{df['YEAR'].max()} (post-filter)")

    # Basic sanity checks
    need_base = ["REF_AREA", "YEAR", target]
    present, missing = ensure_required_columns(df, need_base)
    if missing:
        raise ValueError(f"Missing required columns {missing} in processed panel. "
                         f"Available cols e.g.: {list(df.columns[:12])} ...")

    # Check treatments/controls existence and warn if some missing
    tr_present, tr_missing = ensure_required_columns(df, treatments)
    ct_present, ct_missing = ensure_required_columns(df, controls)

    if tr_missing:
        print(f"[Warn] Treatments not found and will be ignored: {tr_missing}")
    if ct_missing:
        print(f"[Warn] Controls not found and will be ignored: {ct_missing}")

    treatments = tr_present
    controls = ct_present

    if not treatments:
        # Give a helpful message with a quick peek at columns
        sample_cols = [c for c in df.columns if c not in ("REF_AREA", "REF_AREA_LABEL", "YEAR")][:30]
        raise ValueError(
            "No valid treatments provided or found.\n"
            "Pass them with --treatments or in config under causal.treatments.\n"
            f"Sample available columns:\n{sample_cols}"
        )

    # Keep only the columns we actually need for the regression
    use_cols = list(set(["REF_AREA", "YEAR", target] + treatments + controls))
    work = df[use_cols].copy()

    # Optionally drop tiny panels
    work = drop_small_panels(work, min_years_per_country=args.min_years_per_country)

    # Impute controls only (global mean). Target/treatments must be observed.
    work = safe_impute_controls(work, controls)

    # Drop rows with missing in target or treatments (we don't impute those)
    before = len(work)
    non_na_cols = [target] + treatments
    work = work.dropna(subset=non_na_cols)
    after = len(work)
    if before != after:
        print(f"[NA] Dropped {before - after} rows with NaN in target/treatments.")

    # Build formula and fit
    formula = build_formula(target, treatments, controls)
    print(f"[Formula] {formula}")

    fit = cluster_fit(formula, work, cluster_col="REF_AREA")

    # Report top-line treatment effects (pretty print)
    print("\n--- Treatment Coefficients (cluster-robust SE) ---")
    table = fit.summary2().tables[1].copy()
    # Print only rows for treatments (exact match on term names)
    for tr in treatments:
        if tr in table.index:
            row = table.loc[tr]
            coef, se, pval = row["Coef."], row["Std.Err."], row["P>|t|"]
            print(f"{tr:<40}  beta={coef:12.4f}  SE={se:10.4f}  p={pval:7.4f}")
        else:
            print(f"{tr:<40}  (not in model table—check naming)")

    # Save outputs
    used_config = {
        "data": str(Path(args.data).resolve()),
        "target": target,
        "treatments_used": treatments,
        "controls_used": controls,
        "years_range": [int(work["YEAR"].min()), int(work["YEAR"].max())],
        "min_years_per_country": args.min_years_per_country,
        "n_obs": int(len(work)),
        "n_countries": int(work["REF_AREA"].nunique()),
        "n_years": int(work["YEAR"].nunique()),
        "formula": formula,
        "cluster": "REF_AREA",
        "cov_type": "cluster",
    }
    save_outputs(fit, target, used_config)

    print("\n[Done] Causal analysis complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Make errors easy to spot/debug
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)
