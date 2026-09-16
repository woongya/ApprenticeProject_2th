from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
FEATURE_CSV = OUT_DIR / "1_dataset_features.csv"
REPORT_PATH = OUT_DIR / "2_preprocessing_tuning_comparison.txt"
RESULT_CSV = OUT_DIR / "2_preprocessing_tuning_comparison.csv"

TARGET_COLUMNS = ["inertia", "load", "ki"]
DROP_COLUMNS = ["file", "split", "quality_issues", "resistance", *TARGET_COLUMNS]


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(FEATURE_CSV)
    df = df[df["quality_issues"].fillna("").str.contains("missing_label") == False].copy()
    if len(df) < 3:
        raise SystemExit("Need at least 3 labeled rows. Run build_dataset.py first.")

    y = df[TARGET_COLUMNS].astype(float)
    x = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).select_dtypes(include=[np.number]).fillna(0.0)
    return x, y


def metric_rows(experiment: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for i, target in enumerate(TARGET_COLUMNS):
        mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        r2 = float(r2_score(y_true[:, i], y_pred[:, i])) if len(y_true) >= 2 else float("nan")
        acc1 = float((np.abs(y_true[:, i] - y_pred[:, i]) <= 1.0).mean())
        rows.append({"experiment": experiment, "target": target, "mae": mae, "rmse": rmse, "r2": r2, "plus_minus_1_acc": acc1})
    return rows


def cv_predict(model, x: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
    return cross_val_predict(model, x, y, cv=LeaveOneOut(), n_jobs=-1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y = load_dataset()
    y_np = y.to_numpy()

    experiments = []
    base_rf = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=None, min_samples_leaf=1)

    experiments.append((
        "baseline_random_forest_no_scaling",
        MultiOutputRegressor(base_rf),
        "기본 RandomForest, scaling 없음",
    ))
    experiments.append((
        "standard_scaler_random_forest",
        Pipeline([("scaler", StandardScaler()), ("model", MultiOutputRegressor(base_rf))]),
        "StandardScaler 적용 후 RandomForest",
    ))
    experiments.append((
        "minmax_scaler_random_forest",
        Pipeline([("scaler", MinMaxScaler()), ("model", MultiOutputRegressor(base_rf))]),
        "MinMaxScaler 적용 후 RandomForest",
    ))

    rows: list[dict[str, float | str]] = []
    report_lines = [
        "Preprocessing and Hyperparameter Tuning Comparison",
        f"rows={len(x)}, features={len(x.columns)}, validation=LeaveOneOutCV",
        "",
        "Data Leakage 방지 방법:",
        "- scaling 실험은 sklearn Pipeline 안에 scaler를 넣었다.",
        "- cross_val_predict가 fold마다 train fold에만 scaler.fit을 수행한다.",
        "- test fold에는 transform만 적용되므로 validation data leakage가 없다.",
        "",
    ]

    for name, model, description in experiments:
        pred = cv_predict(model, x, y)
        rows.extend(metric_rows(name, y_np, pred))
        report_lines.append(f"[{name}] {description}")
        for row in rows[-len(TARGET_COLUMNS):]:
            report_lines.append(
                f"  {row['target']}: MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, "
                f"R2={row['r2']:.3f}, +/-1_acc={row['plus_minus_1_acc']:.3f}"
            )
        report_lines.append("")

    tuned = GridSearchCV(
        MultiOutputRegressor(RandomForestRegressor(random_state=42)),
        {
            "estimator__n_estimators": [100, 200],
            "estimator__max_depth": [None, 4, 8],
            "estimator__min_samples_leaf": [1, 2],
        },
        cv=min(5, len(x)),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    tuned.fit(x, y)
    tuned_pred = cross_val_predict(tuned.best_estimator_, x, y, cv=LeaveOneOut(), n_jobs=-1)
    rows.extend(metric_rows("tuned_random_forest_no_scaling", y_np, tuned_pred))
    report_lines.append("[tuned_random_forest_no_scaling] GridSearchCV로 선택한 RandomForest")
    report_lines.append(f"  best_params={tuned.best_params_}")
    for row in rows[-len(TARGET_COLUMNS):]:
        report_lines.append(
            f"  {row['target']}: MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, "
            f"R2={row['r2']:.3f}, +/-1_acc={row['plus_minus_1_acc']:.3f}"
        )

    report_lines.extend([
        "",
        "해석:",
        "- RandomForest는 tree 기반 모델이므로 feature scale에 민감하지 않다.",
        "- 따라서 StandardScaler/MinMaxScaler 적용 전후 차이가 작을 수 있다.",
        "- 하지만 scaling 실험은 preprocessing 필요성 검토와 data leakage 방지 설명을 위해 수행했다.",
        "- hyperparameter tuning도 수행했으며, 데이터가 작으면 성능 차이가 제한적일 수 있다.",
    ])

    pd.DataFrame(rows).to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"saved: {RESULT_CSV}")
    print(f"saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
