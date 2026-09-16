from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots" / "grid_search_ml"
FEATURE_CSV = OUT_DIR / "1_dataset_features.csv"
MODEL_PATH = OUT_DIR / "5_grid_search_ml_best_model.joblib"
CV_RESULTS_CSV = OUT_DIR / "5_grid_search_ml_cv_results.csv"
VALIDATION_CSV = OUT_DIR / "5_grid_search_ml_validation_results.csv"
TEST_METRICS_CSV = OUT_DIR / "5_grid_search_ml_test_metrics.csv"
TEST_PREDICTIONS_CSV = OUT_DIR / "5_grid_search_ml_test_predictions.csv"
REPORT_TXT = OUT_DIR / "5_grid_search_ml_report.txt"
REPORT_HTML = OUT_DIR / "5_grid_search_ml_report.html"

TARGET_COLUMNS = ["inertia", "load", "ki"]
DROP_COLUMNS = ["file", "split", "quality_issues", "resistance", *TARGET_COLUMNS]


def make_x(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns]).select_dtypes(include=[np.number]).fillna(0.0)


def make_searches() -> list[tuple[str, Pipeline, dict[str, list[object]]]]:
    return [
        ("random_forest", Pipeline([("model", MultiOutputRegressor(RandomForestRegressor(random_state=42)))]), {
            "model__estimator__n_estimators": [100, 200],
            "model__estimator__max_depth": [None, 6, 10],
            "model__estimator__min_samples_leaf": [1, 2],
        }),
        ("gradient_boosting", Pipeline([("model", MultiOutputRegressor(GradientBoostingRegressor(random_state=42)))]), {
            "model__estimator__n_estimators": [80, 150],
            "model__estimator__learning_rate": [0.05, 0.1],
            "model__estimator__max_depth": [2, 3],
        }),
        ("svr_standard", Pipeline([("scaler", StandardScaler()), ("model", MultiOutputRegressor(SVR()))]), {
            "model__estimator__C": [1.0, 10.0, 50.0],
            "model__estimator__epsilon": [0.05, 0.2],
            "model__estimator__gamma": ["scale", "auto"],
        }),
        ("knn_minmax", Pipeline([("scaler", MinMaxScaler()), ("model", KNeighborsRegressor())]), {
            "model__n_neighbors": [3, 5, 7],
            "model__weights": ["uniform", "distance"],
        }),
    ]


def metrics_rows(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float | str]]:
    rows = []
    for i, target in enumerate(TARGET_COLUMNS):
        error = y_pred[:, i] - y_true[:, i]
        rows.append({
            "model": model_name,
            "target": target,
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])) if len(y_true) >= 2 else np.nan,
            "plus_minus_1_acc": float((np.abs(error) <= 1.0).mean()),
        })
    rows.append({
        "model": model_name,
        "target": "mean",
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": np.nan,
        "plus_minus_1_acc": np.nan,
    })
    return rows


def plot_validation_results(results: pd.DataFrame) -> None:
    mean_rows = results[results["target"] == "mean"].sort_values("val_mae")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(mean_rows["model"], mean_rows["val_mae"], color="tab:blue")
    ax.set_title("Grid Search ML Validation Mean MAE")
    ax.set_xlabel("model")
    ax.set_ylabel("validation mean MAE")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "validation_model_comparison.png", dpi=140)
    plt.close(fig)


def plot_test_pred(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    for i, target in enumerate(TARGET_COLUMNS):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.75)
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set_xlabel(f"true {target}")
        ax.set_ylabel(f"predicted {target}")
        ax.set_title(f"Grid Search Best Model Test: {target}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"test_pred_vs_true_{target}.png", dpi=140)
        plt.close(fig)


def write_html(validation: pd.DataFrame, test_metrics: pd.DataFrame, best_name: str, best_params: dict[str, object]) -> None:
    images = [
        PLOT_DIR / "validation_model_comparison.png",
        PLOT_DIR / "test_pred_vs_true_inertia.png",
        PLOT_DIR / "test_pred_vs_true_load.png",
        PLOT_DIR / "test_pred_vs_true_ki.png",
    ]
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Grid Search ML Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827;} table{border-collapse:collapse;margin:12px 0 24px 0;} th,td{border:1px solid #cbd5e1;padding:6px 10px;text-align:right;} th:first-child,td:first-child{text-align:left;} img{max-width:100%;border:1px solid #cbd5e1;margin:8px 0 28px 0;} .desc{color:#334155;line-height:1.45;} code{background:#e2e8f0;padding:2px 4px;}</style>",
        "</head><body><h1>Grid Search ML Report</h1>",
        "<p class='desc'>이 report는 RandomForest, GradientBoosting, SVR, KNN 후보를 Grid Search로 비교한 결과입니다. resistance는 입력 feature에서 제외했습니다.</p>",
        f"<h2>Selected Model</h2><p>best_model: <code>{best_name}</code></p><p>best_params: <code>{best_params}</code></p>",
        "<h2>Validation Results</h2>",
        validation.fillna("-").to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Final Test Metrics</h2>",
        test_metrics.fillna("-").to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Plots</h2>",
    ]
    for image in images:
        if image.exists():
            rel = image.relative_to(OUT_DIR).as_posix()
            lines.append(f"<h3>{image.name}</h3><img src='{rel}'>")
    lines.append("</body></html>")
    REPORT_HTML.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURE_CSV)
    df = df[df["quality_issues"].fillna("").str.contains("missing_label") == False].copy()
    if not {"train", "validation", "test"}.issubset(set(df["split"].unique())):
        raise SystemExit("Need split=train, validation, test in outputs/1_dataset_features.csv")

    x = make_x(df)
    y = df[TARGET_COLUMNS].astype(float)
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "validation"
    test_mask = df["split"] == "test"
    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    cv_rows = []
    validation_rows = []
    best_model = None
    best_name = ""
    best_params: dict[str, object] = {}
    best_val_mae = float("inf")
    for name, pipeline, param_grid in make_searches():
        cv = min(5, len(x_train))
        search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        search.fit(x_train, y_train)
        cv_result = pd.DataFrame(search.cv_results_)
        cv_result["model"] = name
        cv_rows.append(cv_result)
        pred_val = search.best_estimator_.predict(x_val)
        rows = metrics_rows(name, y_val.to_numpy(), pred_val)
        for row in rows:
            row["val_mae"] = row.pop("mae")
            row["val_rmse"] = row.pop("rmse")
            row["val_r2"] = row.pop("r2")
            row["val_plus_minus_1_acc"] = row.pop("plus_minus_1_acc")
            row["best_params"] = str(search.best_params_)
        validation_rows.extend(rows)
        mean_mae = float(mean_absolute_error(y_val.to_numpy(), pred_val))
        print(f"{name}: validation mean MAE={mean_mae:.4f}, params={search.best_params_}")
        if mean_mae < best_val_mae:
            best_val_mae = mean_mae
            best_model = search.best_estimator_
            best_name = name
            best_params = search.best_params_

    assert best_model is not None
    cv_results = pd.concat(cv_rows, ignore_index=True)
    cv_results.to_csv(CV_RESULTS_CSV, index=False, encoding="utf-8-sig")
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(VALIDATION_CSV, index=False, encoding="utf-8-sig")
    plot_validation_results(validation)

    final_train_x = pd.concat([x_train, x_val])
    final_train_y = pd.concat([y_train, y_val])
    best_model.fit(final_train_x, final_train_y)
    test_pred = best_model.predict(x_test)
    test_metrics = pd.DataFrame(metrics_rows(best_name, y_test.to_numpy(), test_pred))
    test_metrics.to_csv(TEST_METRICS_CSV, index=False, encoding="utf-8-sig")
    pred_out = df[test_mask][["file", "split", *TARGET_COLUMNS]].copy()
    for i, target in enumerate(TARGET_COLUMNS):
        pred_out[f"pred_{target}"] = test_pred[:, i]
        pred_out[f"err_{target}"] = pred_out[f"pred_{target}"] - pred_out[target]
    pred_out.to_csv(TEST_PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
    plot_test_pred(y_test.to_numpy(), test_pred)

    joblib.dump({"model": best_model, "features": list(x.columns), "targets": TARGET_COLUMNS, "best_model": best_name, "best_params": best_params}, MODEL_PATH)
    lines = [
        "Grid Search ML Report",
        f"rows={len(df)}, train={len(x_train)}, validation={len(x_val)}, test={len(x_test)}",
        "input_exclusion=resistance",
        f"best_model={best_name}",
        f"best_params={best_params}",
        "selection_metric=validation mean MAE",
        "",
        test_metrics.to_string(index=False),
    ]
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_html(validation, test_metrics, best_name, best_params)
    print(f"saved: {MODEL_PATH}")
    print(f"saved: {REPORT_TXT}")
    print(f"saved: {REPORT_HTML}")
    print(f"saved: {CV_RESULTS_CSV}")
    print(f"saved: {VALIDATION_CSV}")
    print(f"saved: {TEST_METRICS_CSV}")
    print(f"saved: {TEST_PREDICTIONS_CSV}")
    print(f"saved plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
