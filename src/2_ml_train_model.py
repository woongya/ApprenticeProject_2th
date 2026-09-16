from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split
from sklearn.multioutput import MultiOutputRegressor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
FEATURE_CSV = OUT_DIR / "1_dataset_features.csv"
MODEL_PATH = OUT_DIR / "2_ml_model_i_l_ki.joblib"
REPORT_PATH = OUT_DIR / "2_ml_evaluation_report.txt"
IMPORTANCE_PATH = OUT_DIR / "2_ml_feature_importance.csv"

TARGET_COLUMNS = ["inertia", "load", "ki"]
DROP_COLUMNS = ["file", "split", "quality_issues", "resistance", *TARGET_COLUMNS]


def plus_minus_one_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    return [float((np.abs(y_true[:, i] - y_pred[:, i]) <= 1.0).mean()) for i in range(y_true.shape[1])]


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[str]:
    lines = [f"[{name}]"]
    for i, column in enumerate(TARGET_COLUMNS):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        r2 = r2_score(y_true[:, i], y_pred[:, i]) if len(y_true) >= 2 else np.nan
        acc1 = float((np.abs(y_true[:, i] - y_pred[:, i]) <= 1.0).mean())
        lines.append(f"{column}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, +/-1_acc={acc1:.3f}")
    return lines


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURE_CSV)
    df = df[df["quality_issues"].fillna("").str.contains("missing_label") == False].copy()
    if len(df) < 3:
        raise SystemExit("Need at least 3 labeled CSV files to train a useful model.")

    y = df[TARGET_COLUMNS].astype(float)
    x = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).select_dtypes(include=[np.number]).fillna(0.0)

    base = RandomForestRegressor(random_state=42, n_estimators=200)
    model = MultiOutputRegressor(base)
    param_grid = {
        "estimator__n_estimators": [100, 200],
        "estimator__max_depth": [None, 4, 8],
        "estimator__min_samples_leaf": [1, 2],
    }

    report: list[str] = []
    if "split" in df.columns and (df["split"] == "train").any() and (df["split"] == "validation").any():
        train_mask = df["split"] == "train"
        x_train, y_train = x[train_mask], y[train_mask]
        val_mask = df["split"] == "validation"
        x_val, y_val = x[val_mask], y[val_mask]
        cv = min(5, len(x_train))
        search = GridSearchCV(model, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        search.fit(x_train, y_train)
        best_model = search.best_estimator_
        val_pred = best_model.predict(x_val)
        report.append("split=directory_train_validation")
        report.append(f"rows={len(df)}, train={len(x_train)}, validation={len(x_val)}, test_excluded={int((df['split'] == 'test').sum())}")
        report.append(f"best_params={search.best_params_}")
        report.extend(evaluate("validation", y_val.to_numpy(), val_pred))
        best_model.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))
    elif len(df) >= 20:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        cv = min(5, len(x_train))
        search = GridSearchCV(model, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        search.fit(x_train, y_train)
        best_model = search.best_estimator_
        pred = best_model.predict(x_test)
        report.append("split=train/test")
        report.append(f"rows={len(df)}, train={len(x_train)}, test={len(x_test)}")
        report.append(f"best_params={search.best_params_}")
        report.extend(evaluate("test", y_test.to_numpy(), pred))
        best_model.fit(x, y)
    else:
        # Small assignment sample: use Leave-One-Out for an honest baseline and still save a final model trained on all data.
        loo = LeaveOneOut()
        preds = np.zeros_like(y.to_numpy(dtype=float))
        for train_idx, test_idx in loo.split(x):
            fold_model = MultiOutputRegressor(RandomForestRegressor(random_state=42, n_estimators=200, max_depth=None))
            fold_model.fit(x.iloc[train_idx], y.iloc[train_idx])
            preds[test_idx] = fold_model.predict(x.iloc[test_idx])
        best_model = MultiOutputRegressor(RandomForestRegressor(random_state=42, n_estimators=200, max_depth=None))
        best_model.fit(x, y)
        report.append("split=leave-one-out-cross-validation")
        report.append(f"rows={len(df)}")
        report.extend(evaluate("loo_cv", y.to_numpy(), preds))

    importances = np.mean([est.feature_importances_ for est in best_model.estimators_], axis=0)
    pd.DataFrame({"feature": x.columns, "importance": importances}).sort_values("importance", ascending=False).to_csv(
        IMPORTANCE_PATH, index=False, encoding="utf-8-sig"
    )

    joblib.dump({"model": best_model, "features": list(x.columns), "targets": TARGET_COLUMNS}, MODEL_PATH)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"saved: {MODEL_PATH}")
    print(f"saved: {REPORT_PATH}")
    print(f"saved: {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
