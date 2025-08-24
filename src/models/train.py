"""
train.py

Cross-country panel-aware ML training on World Bank panel dataset.

Pipeline:
1) Load processed panel data from preprocess.py.
2) Select features and target.
3) Split dataset per country: train on earlier years, validate on later years.
4) Handle imputation and scaling based on the training data ONLY.
5) Train multiple models (Linear, Ridge, RandomForest, GradientBoosting, optional LightGBM).
6) Evaluate on validation set using MAE, RMSE, R2.
7) Compare models and select the best.
8) Save the best model, feature importance, and scalers for future inference.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

# ML models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# Optional: LightGBM if installed
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# ---------- Paths & Config ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data/processed/panel_clean.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Helpers ----------
def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict:
    """Load YAML config."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_data(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Processed panel data not found: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"[Data] Loaded panel with shape: {df.shape}")
    return df


def select_features_targets(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return features X and target y."""
    exclude_cols = exclude_cols or ["REF_AREA", "REF_AREA_LABEL", "YEAR"]
    X = df.drop(columns=[c for c in exclude_cols if c in df.columns] + [target_col], errors='ignore')
    y = df[target_col]
    return X, y


def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
    """Compute MAE, RMSE, R2."""
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = root_mean_squared_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def save_model(model, name: str) -> Path:
    """Save trained model to disk."""
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[Save] Model saved → {path}")
    return path


def save_feature_importance(model, feature_names: List[str], name: str):
    """Save feature importance if supported."""
    fi_path = MODEL_DIR / f"{name}_feature_importance.csv"
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        fi.to_csv(fi_path, index=False)
        print(f"[Save] Feature importance saved → {fi_path}")
    else:
        print("[Feature Importance] Not available for this model type.")


# ---------- Cross-country panel-aware Split ----------
def panel_time_split(df: pd.DataFrame, val_years: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the panel dataset temporally **per country**.
    - df: panel dataframe with 'REF_AREA' (country) and 'YEAR'
    - val_years: number of latest years for validation per country
    Returns: train_df, val_df
    """
    train_list, val_list = [], []
    # The 'country' variable is defined here...
    for country, grp in df.groupby("REF_AREA"):
        max_year = grp["YEAR"].max()
        cutoff = max_year - val_years
        train_list.append(grp[grp["YEAR"] <= cutoff])
        val_list.append(grp[grp["YEAR"] > cutoff])
        
        # ... and it is correctly accessed and used right here.
        print(f"[Split] {country}: train <= {cutoff} ({len(grp[grp['YEAR'] <= cutoff])}), "
              f"val > {cutoff} ({len(grp[grp['YEAR'] > cutoff])})")

    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    print(f"[Split] Total train rows: {train_df.shape[0]}, validation rows: {val_df.shape[0]}")
    return train_df, val_df


# ---------- Model Training ----------
def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict[str, Dict]:
    """
    Train multiple models and evaluate.
    Returns a dictionary: {model_name: {model, metrics_train, metrics_val}}
    """
    results = {}

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results["LinearRegression"] = {
        "model": lr,
        "metrics_train": evaluate_model(lr, X_train, y_train),
        "metrics_val": evaluate_model(lr, X_val, y_val),
    }

    # --- Ridge Regression ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results["Ridge"] = {
        "model": ridge,
        "metrics_train": evaluate_model(ridge, X_train, y_train),
        "metrics_val": evaluate_model(ridge, X_val, y_val),
    }

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["RandomForest"] = {
        "model": rf,
        "metrics_train": evaluate_model(rf, X_train, y_train),
        "metrics_val": evaluate_model(rf, X_val, y_val),
    }

    # --- Gradient Boosting ---
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)
    results["GradientBoosting"] = {
        "model": gb,
        "metrics_train": evaluate_model(gb, X_train, y_train),
        "metrics_val": evaluate_model(gb, X_val, y_val),
    }

    # --- LightGBM ---
    if lgb:
        lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)
        lgb_model.fit(X_train, y_train)
        results["LightGBM"] = {
            "model": lgb_model,
            "metrics_train": evaluate_model(lgb_model, X_train, y_train),
            "metrics_val": evaluate_model(lgb_model, X_val, y_val),
        }

    # --- Print metrics ---
    print("\n--- Model Evaluation ---")
    for name, r in results.items():
        m_train = r["metrics_train"]
        m_val = r["metrics_val"]
        print(f"[Eval] {name:<20} → Train R2: {m_train['R2']:.4f} | Val R2: {m_val['R2']:.4f} | Val RMSE: {m_val['RMSE']:.2f}")

    return results


def select_best_model(results: Dict[str, Dict], criterion: str = "RMSE") -> str:
    """Return the model name with best performance (lowest RMSE by default on validation set)."""
    # NOTE: We select the best model based on VALIDATION metrics to avoid choosing an overfit model.
    best_name = min(results.keys(), key=lambda k: results[k]["metrics_val"][criterion])
    print(f"\n[Best Model] Selected: {best_name} based on validation {criterion}")
    return best_name


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Cross-country panel-aware ML training.")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    p.add_argument("--target", type=str, default=None, help="Override target column from config")
    p.add_argument("--val_years", type=int, default=5, help="Number of latest years for validation per country")
    p.add_argument("--no_scale", action="store_true", help="Disable within-country standardization")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))

    df = load_data()

    # Target column (normalize legacy aliases to WB code)
    target_col = args.target or cfg["targets"].get("economic_target", "WB_GS_NY_GDP_PCAP_KD")
    alias_map = {
        "GDP per capita (constant 2010 US$)": "WB_GS_NY_GDP_PCAP_KD",
        "GDP per capita": "WB_GS_NY_GDP_PCAP_KD",
        "NY.GDP.PCAP.KD": "WB_GS_NY_GDP_PCAP_KD",
    }
    target_col = alias_map.get(target_col, target_col)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. Available columns: {df.columns.tolist()}")

    # Cross-country panel-aware split
    train_df, val_df = panel_time_split(df, val_years=args.val_years)

    # --- ADD THIS SECTION TO REMOVE FEATURES FOR THE EXPLANATORY MODEL ---
    # Define all features that are too similar to the target to remove them
    features_to_remove = [
        # --- Target variable's own lags and changes ---
        'WB_GS_NY_GDP_PCAP_KD_lag1', 'WB_GS_NY_GDP_PCAP_KD_lag3', 'WB_GS_NY_GDP_PCAP_KD_lag5', 'WB_GS_NY_GDP_PCAP_KD_lag10',
        'WB_GS_NY_GDP_PCAP_KD_chg1', 'WB_GS_NY_GDP_PCAP_KD_chg3', 'WB_GS_NY_GDP_PCAP_KD_chg5', 'WB_GS_NY_GDP_PCAP_KD_chg10',

        # --- GDP Growth Rate (rate of change of target) ---
        'WB_GS_NY_GDP_MKTP_KD_ZG',
        'WB_GS_NY_GDP_MKTP_KD_ZG_lag1', 'WB_GS_NY_GDP_MKTP_KD_ZG_lag3', 'WB_GS_NY_GDP_MKTP_KD_ZG_lag5', 'WB_GS_NY_GDP_MKTP_KD_ZG_lag10',
        'WB_GS_NY_GDP_MKTP_KD_ZG_chg1', 'WB_GS_NY_GDP_MKTP_KD_ZG_chg3', 'WB_GS_NY_GDP_MKTP_KD_ZG_chg5', 'WB_GS_NY_GDP_MKTP_KD_ZG_chg10',

        # --- Total GDP (removes effect of sheer economic size) ---
        'WB_GS_NY_GDP_MKTP_CD',
        'WB_GS_NY_GDP_MKTP_CD_lag1', 'WB_GS_NY_GDP_MKTP_CD_lag3', 'WB_GS_NY_GDP_MKTP_CD_lag5', 'WB_GS_NY_GDP_MKTP_CD_lag10',
        'WB_GS_NY_GDP_MKTP_CD_chg1', 'WB_GS_NY_GDP_MKTP_CD_chg3', 'WB_GS_NY_GDP_MKTP_CD_chg5', 'WB_GS_NY_GDP_MKTP_CD_chg10',

        # --- GNI per capita (a direct sibling of the target) ---
        'WB_GS_NY_GNP_PCAP_PP_CD',
        'WB_GS_NY_GNP_PCAP_PP_CD_lag1', 'WB_GS_NY_GNP_PCAP_PP_CD_lag3', 'WB_GS_NY_GNP_PCAP_PP_CD_lag5', 'WB_GS_NY_GNP_PCAP_PP_CD_lag10',
        'WB_GS_NY_GNP_PCAP_PP_CD_chg1', 'WB_GS_NY_GNP_PCAP_PP_CD_chg3', 'WB_GS_NY_GNP_PCAP_PP_CD_chg5', 'WB_GS_NY_GNP_PCAP_PP_CD_chg10',

        # --- ADDED: Essential Economic & Social Controls ---
        'WB_GS_SP_URB_TOTL_IN_ZS',
        'WB_GS_SP_URB_TOTL_IN_ZS_lag1', 'WB_GS_SP_URB_TOTL_IN_ZS_lag3', 'WB_GS_SP_URB_TOTL_IN_ZS_lag5', 'WB_GS_SP_URB_TOTL_IN_ZS_lag10',
        'WB_GS_SP_URB_TOTL_IN_ZS_chg1', 'WB_GS_SP_URB_TOTL_IN_ZS_chg3', 'WB_GS_SP_URB_TOTL_IN_ZS_chg5', 'WB_GS_SP_URB_TOTL_IN_ZS_chg10',
        
        'WB_GS_FP_CPI_TOTL_ZG',
        'WB_GS_FP_CPI_TOTL_ZG_lag1', 'WB_GS_FP_CPI_TOTL_ZG_lag3', 'WB_GS_FP_CPI_TOTL_ZG_lag5', 'WB_GS_FP_CPI_TOTL_ZG_lag10',
        'WB_GS_FP_CPI_TOTL_ZG_chg1', 'WB_GS_FP_CPI_TOTL_ZG_chg3', 'WB_GS_FP_CPI_TOTL_ZG_chg5', 'WB_GS_FP_CPI_TOTL_ZG_chg10',
        
        'WB_GS_SL_UEM_ZS',
        'WB_GS_SL_UEM_ZS_lag1', 'WB_GS_SL_UEM_ZS_lag3', 'WB_GS_SL_UEM_ZS_lag5', 'WB_GS_SL_UEM_ZS_lag10',
        'WB_GS_SL_UEM_ZS_chg1', 'WB_GS_SL_UEM_ZS_chg3', 'WB_GS_SL_UEM_ZS_chg5', 'WB_GS_SL_UEM_ZS_chg10',

    ]
    train_df = train_df.drop(columns=features_to_remove, errors='ignore')
    val_df = val_df.drop(columns=features_to_remove, errors='ignore')
    print(f"[Feature Eng] Removed features too similar to the target to improve explanatory power.")
    # --- END OF NEW SECTION ---
    
    # Drop rows with NaN in the target column from both train and val sets
    initial_train_rows = len(train_df)
    initial_val_rows = len(val_df)
    train_df = train_df.dropna(subset=[target_col])
    val_df = val_df.dropna(subset=[target_col])
    print(f"[Target] Dropped {initial_train_rows - len(train_df)} training rows and "
          f"{initial_val_rows - len(val_df)} validation rows with missing '{target_col}' values.")

    # Features & target
    X_train, y_train = select_features_targets(train_df, target_col)
    X_val, y_val = select_features_targets(val_df, target_col)

    # --- Handle missing values and standardize features correctly ---
    # We will handle imputation and scaling here, AFTER the train/val split,
    # to ensure no data leakage occurs.
    
    # Identify value columns to transform (excluding non-numeric)
    value_cols = [c for c in X_train.columns if c not in ["REF_AREA", "REF_AREA_LABEL", "YEAR"]]
    
    # Impute missing values with a SimpleImputer fit on training data
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train[value_cols]), columns=value_cols, index=X_train.index)
    X_val_imputed = pd.DataFrame(imputer.transform(X_val[value_cols]), columns=value_cols, index=X_val.index)
    
    # Standardize features with a StandardScaler fit on training data
    # We only apply scaling if the --no_scale argument is not used
    if not args.no_scale:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=value_cols, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val_imputed), columns=value_cols, index=X_val.index)
        print("[Scale] StandardScaler fit on training data and applied to both sets.")
    else:
        X_train_scaled = X_train_imputed
        X_val_scaled = X_val_imputed
        print("[Scale] Skipped standardization as per --no_scale argument.")
        # Create a dummy scaler to save for inference, to avoid errors later
        scaler = None
    
    # Save scaler + imputer for inference
    pickle.dump(imputer, open(MODEL_DIR / "imputer.pkl", "wb"))
    if scaler:
        pickle.dump(scaler, open(MODEL_DIR / "scaler.pkl", "wb"))
    print("[Save] Imputer & Scaler saved.")

    # Train models
    results = train_models(X_train_scaled, y_train, X_val_scaled, y_val)

    # Select best
    best_name = select_best_model(results)
    best_model = results[best_name]["model"]
    save_model(best_model, best_name)
    save_feature_importance(best_model, X_train_scaled.columns.tolist(), best_name)

if __name__ == "__main__":
    main()



