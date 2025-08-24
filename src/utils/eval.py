# src/utils/eval.py

import argparse
import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "src" / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "panel_clean.parquet"

def load_model(model_name: str):
    with open(MODEL_DIR / f"{model_name}.pkl", "rb") as f:
        return pickle.load(f)

def main(model_name: str, target_col: str):
    # load model + scaler + imputer
    model = load_model(model_name)
    imputer = pickle.load(open(MODEL_DIR / "imputer.pkl", "rb"))
    scaler = pickle.load(open(MODEL_DIR / "scaler.pkl", "rb"))

    # load data
    df = pd.read_parquet(DATA_PATH)

    # make a strict test split (e.g. last 5 years globally)
    cutoff = df["YEAR"].max() - 5
    test_df = df[df["YEAR"] > cutoff].dropna(subset=[target_col])

    X = test_df.drop(columns=["REF_AREA", "REF_AREA_LABEL", "YEAR", target_col], errors="ignore")
    y = test_df[target_col]

    # preprocess with saved transformers
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    # predict
    preds = model.predict(X_scaled)

    # metrics
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    print(f"[Eval] {model_name} → MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (without .pkl)")
    parser.add_argument("--target", type=str, default="WB_GS_NY_GDP_PCAP_KD", help="Target column")
    args = parser.parse_args()

    main(args.model, args.target)

