from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots" / "svr_hyperparameter_scaling"
FEATURE_CSV = OUT_DIR / "1_dataset_features.csv"
MODEL_PATH = OUT_DIR / "5_2_svr_hyperparameter_scaling_model.joblib"
RESULT_CSV = OUT_DIR / "5_2_svr_hyperparameter_scaling_results.csv"
TEST_CSV = OUT_DIR / "5_2_svr_hyperparameter_scaling_test_metrics.csv"
REPORT_TXT = OUT_DIR / "5_2_svr_hyperparameter_scaling_report.txt"
REPORT_HTML = OUT_DIR / "5_2_svr_hyperparameter_scaling_report.html"

TARGET_COLUMNS = ["inertia", "load", "ki"]
DROP_COLUMNS = ["file", "split", "quality_issues", "resistance", *TARGET_COLUMNS]


def make_x(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).select_dtypes(include=[np.number]).fillna(0.0)


def metric_rows(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> list[dict[str, float | str]]:
    rows = []
    for i, target in enumerate(TARGET_COLUMNS):
        rows.append({
            "dataset": prefix,
            "target": target,
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])) if len(y_true) >= 2 else np.nan,
            "plus_minus_1_acc": float((np.abs(y_true[:, i] - y_pred[:, i]) <= 1.0).mean()),
        })
    rows.append({"dataset": prefix, "target": "mean", "mae": float(mean_absolute_error(y_true, y_pred)), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))), "r2": np.nan, "plus_minus_1_acc": np.nan})
    return rows


def make_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    return "passthrough"


def plot_scaling_results(results: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    mean_rows = results[results["target"] == "mean"].sort_values("mae")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(mean_rows["scaler"], mean_rows["mae"], color="tab:green")
    ax.set_title("SVR Hyperparameter + Scaling: Validation Mean MAE")
    ax.set_xlabel("scaler")
    ax.set_ylabel("validation mean MAE")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "validation_mean_mae_by_scaler.png", dpi=140)
    plt.close(fig)


def plot_test_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    for i, target in enumerate(TARGET_COLUMNS):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.75)
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set_title(f"SVR Scaling Best Model Test: {target}")
        ax.set_xlabel(f"true {target}")
        ax.set_ylabel(f"predicted {target}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"test_pred_vs_true_{target}.png", dpi=140)
        plt.close(fig)


def write_html_report(validation_results: pd.DataFrame, test_metrics: pd.DataFrame, best_scaler: str, best_params: dict[str, object]) -> None:
    images = [
        PLOT_DIR / "validation_mean_mae_by_scaler.png",
        PLOT_DIR / "test_pred_vs_true_inertia.png",
        PLOT_DIR / "test_pred_vs_true_load.png",
        PLOT_DIR / "test_pred_vs_true_ki.png",
    ]
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>SVR Hyperparameter and Scaling Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827;} table{border-collapse:collapse;margin:12px 0 24px 0;} th,td{border:1px solid #cbd5e1;padding:6px 10px;text-align:right;} th:first-child,td:first-child{text-align:left;} img{max-width:100%;border:1px solid #cbd5e1;margin:8px 0 28px 0;} .desc{color:#334155;line-height:1.45;} code{background:#e2e8f0;padding:2px 4px;}</style>",
        "</head><body><h1>SVR Hyperparameter and Data Scaling Report</h1>",
        "<p class='desc'>하나의 AI 모델(SVR)에 대해 하이퍼파라미터와 Data Scaling 종류를 함께 비교한 결과입니다. resistance는 입력 feature에서 제외했습니다.</p>",
        f"<h2>Selected Combination</h2><p>best_scaler: <code>{best_scaler}</code></p><p>best_params: <code>{best_params}</code></p>",
        "<h2>Validation Result Table</h2>",
        validation_results.fillna("-").to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Final Test Metrics</h2>",
        test_metrics.fillna("-").to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Graphs</h2>",
    ]
    for image in images:
        if image.exists():
            lines.append(f"<h3>{image.name}</h3><img src='{image.relative_to(OUT_DIR).as_posix()}'>")
    lines.append("</body></html>")
    REPORT_HTML.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURE_CSV)
    df = df[df["quality_issues"].fillna("").str.contains("missing_label") == False].copy()
    x = make_x(df)
    y = df[TARGET_COLUMNS].astype(float)
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "validation"
    test_mask = df["split"] == "test"
    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    results = []
    best_model = None
    best_scaler = ""
    best_params: dict[str, object] = {}
    best_val_mae = float("inf")
    for scaler_name in ["none", "standard", "minmax", "robust"]:
        pipeline = Pipeline([("scaler", make_scaler(scaler_name)), ("model", MultiOutputRegressor(SVR()))])
        param_grid = {
            "model__estimator__C": [1.0, 10.0, 50.0, 100.0],
            "model__estimator__epsilon": [0.05, 0.1, 0.2],
            "model__estimator__gamma": ["scale", "auto"],
        }
        search = GridSearchCV(pipeline, param_grid, cv=min(5, len(x_train)), scoring="neg_mean_absolute_error", n_jobs=-1)
        search.fit(x_train, y_train)
        val_pred = search.best_estimator_.predict(x_val)
        rows = metric_rows(y_val.to_numpy(), val_pred, "validation")
        for row in rows:
            row["scaler"] = scaler_name
            row["best_params"] = str(search.best_params_)
            results.append(row)
        val_mae = float(mean_absolute_error(y_val.to_numpy(), val_pred))
        print(f"scaler={scaler_name}: validation mean MAE={val_mae:.4f}, params={search.best_params_}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = search.best_estimator_
            best_scaler = scaler_name
            best_params = search.best_params_

    assert best_model is not None
    result_df = pd.DataFrame(results)
    result_df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    plot_scaling_results(result_df)
    best_model.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))
    test_pred = best_model.predict(x_test)
    test_metrics = pd.DataFrame(metric_rows(y_test.to_numpy(), test_pred, "test"))
    test_metrics["scaler"] = best_scaler
    test_metrics["best_params"] = str(best_params)
    test_metrics.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")
    plot_test_predictions(y_test.to_numpy(), test_pred)
    write_html_report(result_df, test_metrics, best_scaler, best_params)
    joblib.dump({"model": best_model, "features": list(x.columns), "targets": TARGET_COLUMNS, "scaler": best_scaler, "best_params": best_params}, MODEL_PATH)
    lines = ["SVR Hyperparameter and Scaling Report", "input_exclusion=resistance", f"train={len(x_train)}, validation={len(x_val)}, test={len(x_test)}", f"best_scaler={best_scaler}", f"best_params={best_params}", "", test_metrics.to_string(index=False)]
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {MODEL_PATH}")
    print(f"saved: {RESULT_CSV}")
    print(f"saved: {TEST_CSV}")
    print(f"saved: {REPORT_TXT}")
    print(f"saved: {REPORT_HTML}")
    print(f"saved plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
