from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
FEATURE_CSV = OUT_DIR / "1_dataset_features.csv"
MODEL_PATH = OUT_DIR / "2_ml_model_i_l_ki.joblib"
REPORT_PATH = OUT_DIR / "2_ml_test_evaluation_report.txt"
PREDICTION_CSV = OUT_DIR / "2_ml_test_predictions.csv"


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> list[str]:
    lines = ["RandomForest Final Test Evaluation"]
    for i, target in enumerate(targets):
        mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        r2 = float(r2_score(y_true[:, i], y_pred[:, i])) if len(y_true) >= 2 else float("nan")
        acc1 = float((np.abs(y_true[:, i] - y_pred[:, i]) <= 1.0).mean())
        lines.append(f"{target}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, +/-1_acc={acc1:.3f}")
    return lines


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURE_CSV)
    test_df = df[df["split"] == "test"].copy() if "split" in df.columns else pd.DataFrame()
    if test_df.empty:
        raise SystemExit("No test rows found. Put MOVELOG csv files in data_test and run src/1_build_dataset.py first.")

    targets = bundle["targets"]
    features = bundle["features"]
    x_test = test_df.reindex(columns=features, fill_value=0.0).fillna(0.0)
    y_test = test_df[targets].astype(float)
    pred = bundle["model"].predict(x_test)

    report = evaluate(y_test.to_numpy(), pred, targets)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    output = test_df[["file", "split", *targets]].copy()
    for i, target in enumerate(targets):
        output[f"pred_{target}"] = pred[:, i]
        output[f"err_{target}"] = output[f"pred_{target}"] - output[target]
    output.to_csv(PREDICTION_CSV, index=False, encoding="utf-8-sig")

    print(f"test rows: {len(test_df)}")
    print(f"saved: {REPORT_PATH}")
    print(f"saved: {PREDICTION_CSV}")


if __name__ == "__main__":
    main()
