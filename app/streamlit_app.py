# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

st.set_page_config(page_title="Education & Healthcare → GDP per capita (what-if)", layout="wide")
st.title("📈 Education & Healthcare → GDP per capita (what-if)")
st.caption(
    "Adjust life expectancy at 60 and tertiary enrollment to see the estimated change in "
    "GDP per capita (constant US$) based on your causal model coefficients."
)

# --- Indicator codes (from your pipeline/config) ---
TARGET = "WB_GS_NY_GDP_PCAP_KD"       # GDP per capita (constant US$)
TREAT_LE60 = "WB_GS_SP_DYN_LE60_IN"   # Life expectancy at 60 (years)
TREAT_TER = "WB_GS_SE_TER_ENRR"       # Tertiary enrollment (gross, %)

# --- Load config and paths ---
CONFIG = yaml.safe_load(open("configs/config.yaml", "r"))
panel_path = Path(CONFIG["paths"]["processed_parquet"])
sel_indicators_path = Path(CONFIG["paths"].get("selected_indicators_csv", "configs/selected_indicators.csv"))

# Try both locations for saved causal params
PROJECT_ROOT = Path.cwd()
CANDIDATE_MODELS_DIRS = [
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "src" / "models",
]

# -------- Helpers --------
@st.cache_data
def load_panel(parquet_path: Path) -> Optional[pd.DataFrame]:
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    st.warning(f"Processed panel not found at: {parquet_path}. Run `python src/preprocess.py`.")
    return None

def _pick_latest_params_file(target_code: str) -> Optional[Path]:
    candidates = []
    for d in CANDIDATE_MODELS_DIRS:
        if d.exists():
            candidates += list(d.glob(f"causal_params_{target_code}_*.csv"))
            # also allow any params csv as fallback
            candidates += list(d.glob("causal_params_*.csv"))
    if not candidates:
        return None
    # choose most recent by modified time
    return max(candidates, key=lambda p: p.stat().st_mtime)

@st.cache_data
def load_causal_params(target_code: str) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    f = _pick_latest_params_file(target_code)
    if f is None:
        return None, None
    try:
        df = pd.read_csv(f)
        return df, f
    except Exception as e:
        st.warning(f"Could not read params file {f}: {e}")
        return None, f

def _find_columns_case_insensitive(df: pd.DataFrame, options) -> Optional[str]:
    """Find a column by a set of candidate substrings (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for opt in options:
        for lc, orig in cols_lower.items():
            if opt in lc:
                return orig
    return None

def extract_betas(params_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Try to extract coefficients (and SE if available) for the two treatments.
    Expected that params_df has rows keyed by variable/term names and columns for coef/beta and se.
    """
    # Heuristic: figure out name columns and coeff/SE columns
    term_col = _find_columns_case_insensitive(params_df, ["term", "variable", "name", "param"])
    beta_col = _find_columns_case_insensitive(params_df, ["beta", "coef", "coefficient", "estimate"])
    se_col   = _find_columns_case_insensitive(params_df, ["se", "std", "stderr", "std.err"])

    if term_col is None or beta_col is None:
        # Can't parse; return empty dict
        return {}

    # Normalize term names (strip spaces just in case)
    df = params_df.copy()
    df[term_col] = df[term_col].astype(str).str.strip()

    def _row_for(ind_code: str) -> Optional[pd.Series]:
        # exact match
        hit = df[df[term_col] == ind_code]
        if not hit.empty:
            return hit.iloc[0]
        # sometimes terms are quoted or prefixed; try contains as fallback
        hit = df[df[term_col].str.contains(ind_code, na=False)]
        if not hit.empty:
            return hit.iloc[0]
        return None

    out = {}
    for code in [TREAT_LE60, TREAT_TER]:
        row = _row_for(code)
        if row is not None:
            out[code] = {
                "beta": float(row[beta_col]),
                "se": float(row[se_col]) if (se_col and pd.notna(row.get(se_col))) else None,
            }

    return out

def default_betas_if_missing() -> Dict[str, Dict[str, Any]]:
    """
    Fallback to the example numbers you printed from your last run:
      LE60 beta ≈ 2031.5035 (USD per capita per +1 year at age 60)
      TER  beta ≈   61.4820 (USD per capita per +1 percentage point)
    """
    return {
        TREAT_LE60: {"beta": 2031.5035, "se": 413.3171},
        TREAT_TER:  {"beta":   61.4820, "se":  26.5416},
    }

def nearest_non_null(series: pd.Series, year: int) -> Optional[float]:
    """Get value at year if present, otherwise nearest previous then next."""
    s = series.dropna()
    if year in s.index:
        return float(s.loc[year])
    before = s[s.index < year]
    after  = s[s.index > year]
    if not before.empty:
        return float(before.iloc[-1])
    if not after.empty:
        return float(after.iloc[0])
    return None

def q_range(series: pd.Series, qlow=0.05, qhigh=0.95, fallback: Tuple[float,float]=(0.0, 100.0)) -> Tuple[float, float]:
    s = series.dropna()
    if s.empty:
        return fallback
    return float(s.quantile(qlow)), float(s.quantile(qhigh))

# --------- Load data ---------
df = load_panel(panel_path)

if df is None:
    st.stop()

# Basic safety checks
needed = [ "REF_AREA", "REF_AREA_LABEL", "YEAR", TARGET, TREAT_LE60, TREAT_TER ]
missing = [c for c in needed if c not in df.columns]
if missing:
    st.error(f"Your processed panel is missing required columns: {missing}")
    st.stop()

# Keep only relevant columns for speed
keep_cols = needed
df_small = df[keep_cols].copy()
df_small["YEAR"] = df_small["YEAR"].astype(int)

# --------- Load causal params (betas) ----------
params_df, params_path = load_causal_params(TARGET)
betas = extract_betas(params_df) if params_df is not None else {}
if not betas:
    # fallback to example numbers from user's last causal run
    betas = default_betas_if_missing()
    st.info("Using fallback treatment coefficients from your last printed run (hard-coded). "
            "Place the latest `causal_params_*.csv` in `models/` or `src/models/` to use saved values.")

# Sidebar: show coefficients
with st.sidebar:
    st.header("Model coefficients")
    if params_path:
        st.caption(f"Loaded from: `{params_path}`")
    st.write("Point estimates used:")
    st.metric("β (Life expectancy @60, years)", f"{betas[TREAT_LE60]['beta']:.2f} USD / +1 year")
    st.metric("β (Tertiary enrollment, %)",     f"{betas[TREAT_TER]['beta']:.2f} USD / +1 pp")
    st.caption("These come from an OLS with country & year fixed effects. "
               "Counterfactuals assume other controls stay fixed and no re-equilibration effects.")

# --------- UI controls ----------
countries = sorted(df_small["REF_AREA_LABEL"].unique().tolist())
c_default = countries.index("India") if "India" in countries else 0
c_sel = st.selectbox("Country", countries, index=c_default)

sub = df_small[df_small["REF_AREA_LABEL"] == c_sel].sort_values("YEAR")
years = sub["YEAR"].unique().tolist()
y_default = int(np.nanmax(years)) if len(years) else None
y_sel = st.slider("Year", int(min(years)), int(max(years)), value=y_default, step=1)

# Baseline (closest available value around chosen year)
sub_idx = sub.set_index("YEAR")
x1_base = nearest_non_null(sub_idx[TREAT_LE60], y_sel)
x2_base = nearest_non_null(sub_idx[TREAT_TER],  y_sel)
y_base  = nearest_non_null(sub_idx[TARGET],     y_sel)

# Ranges for sliders (global quantiles)
x1_min, x1_max = q_range(df_small[TREAT_LE60], 0.05, 0.95, (10.0, 30.0))  # years at age 60 ~ [~14, ~26]
x2_min, x2_max = q_range(df_small[TREAT_TER],  0.05, 0.95, (0.0, 120.0))  # tertiary gross % can exceed 100

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("Set your what-if values")

    if x1_base is None or x2_base is None or y_base is None:
        st.warning(
            "This country/year is missing one or more baseline values. "
            "We’ll fall back to the nearest available year."
        )

    x1_new = st.number_input(
        "Life expectancy at 60 (years)",
        min_value=float(np.floor(x1_min)),
        max_value=float(np.ceil(x1_max)),
        value=float(x1_base) if x1_base is not None else float(np.clip((x1_min+x1_max)/2, x1_min, x1_max)),
        step=0.1,
        format="%.1f",
    )
    x2_new = st.number_input(
        "Tertiary enrollment (gross, %)",
        min_value=float(np.floor(x2_min)),
        max_value=float(np.ceil(x2_max)),
        value=float(x2_base) if x2_base is not None else float(np.clip((x2_min+x2_max)/2, x2_min, x2_max)),
        step=1.0,
        format="%.1f",
    )

    # Compute counterfactual change
    dx1 = (x1_new - (x1_base or x1_new))
    dx2 = (x2_new - (x2_base or x2_new))

    dy = betas[TREAT_LE60]["beta"] * dx1 + betas[TREAT_TER]["beta"] * dx2
    y_new = (y_base or 0.0) + dy

    st.divider()
    st.subheader("Result (point estimate)")
    st.metric(
        label="Predicted change in GDP per capita (USD)",
        value=f"{dy:,.0f} USD",
        delta=None
    )
    st.metric(
        label=f"Predicted GDP per capita in {y_sel} (USD)",
        value=f"{y_new:,.0f} USD",
        delta=f"{dy:,.0f} vs baseline" if y_base is not None else None
    )

    # Optional: crude uncertainty band (ignoring covariance)
    se1 = betas[TREAT_LE60].get("se")
    se2 = betas[TREAT_TER].get("se")
    if se1 is not None and se2 is not None:
        var = (dx1**2) * (se1**2) + (dx2**2) * (se2**2)  # no cov term available
        ci = 1.96 * np.sqrt(var)
        st.caption(f"≈ 95% interval (ignoring covariance): ±{ci:,.0f} USD")

with col_right:
    st.subheader(f"{c_sel}: GDP per capita over time")
    # Plot observed series
    plot_df = sub[["YEAR", TARGET]].rename(columns={TARGET: "GDP per capita (USD, const)"})
    st.line_chart(plot_df.set_index("YEAR"))

    if y_base is not None:
        st.caption(
            f"Baseline (nearest to {y_sel}): {y_base:,.0f} USD. "
            f"What-if: {y_new:,.0f} USD (Δ {dy:,.0f})."
        )
    else:
        st.caption("Baseline value unavailable for the selected year; used nearest available for the computation.")

st.divider()
with st.expander("Details & assumptions", expanded=False):
    st.markdown(
        f"""
- **Outcome**: `{TARGET}` (GDP per capita, constant US$).
- **Treatments**: 
  - `{TREAT_LE60}` (Life expectancy at 60, years)  
  - `{TREAT_TER}` (Tertiary enrollment, gross %)
- **Estimation**: OLS with **country** and **year** fixed effects; cluster-robust SE by country.
- **Interpretation**: Point estimates show **partial** effects, holding other included controls fixed.
- **Counterfactual math**:  
  ΔY = β₁·ΔLE60 + β₂·ΔTER.  
  New level = Baseline GDPpc + ΔY.  
  (FE & intercept cancel when taking differences around the same country-year.)
- **Caveats**: No general-equilibrium or long-run feedbacks; uncertainty band shown ignores covariance between coefficients.
        """
    )

