from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots" / "svr_hyperparameter_only"
FEATURE_CSV = OUT_DIR / "1_dataset_features.csv"
MODEL_PATH = OUT_DIR / "5_1_svr_hyperparameter_only_model.joblib"
RESULT_CSV = OUT_DIR / "5_1_svr_hyperparameter_only_results.csv"
TEST_CSV = OUT_DIR / "5_1_svr_hyperparameter_only_test_metrics.csv"
REPORT_TXT = OUT_DIR / "5_1_svr_hyperparameter_only_report.txt"
REPORT_HTML = OUT_DIR / "5_1_svr_hyperparameter_only_report.html"
CV_RESULTS_CSV = OUT_DIR / "5_1_svr_hyperparameter_only_cv_results.csv"

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
    rows.append({
        "dataset": prefix,
        "target": "mean",
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": np.nan,
        "plus_minus_1_acc": np.nan,
    })
    return rows


def plot_test_metrics(metrics: pd.DataFrame) -> None:
    rows = metrics[metrics["target"] == "mean"].sort_values("mae")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rows["selection"], rows["mae"], color=["tab:green", "tab:orange", "tab:red"][:len(rows)])
    ax.set_title("SVR Hyperparameter Only: Best/Middle/Worst Test Mean MAE")
    ax.set_xlabel("validation selection")
    ax.set_ylabel("test mean MAE")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "test_mae_by_target.png", dpi=140)
    plt.close(fig)


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    for i, target in enumerate(TARGET_COLUMNS):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.75)
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set_title(f"SVR Hyperparameter Only: {target}")
        ax.set_xlabel(f"true {target}")
        ax.set_ylabel(f"predicted {target}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"test_pred_vs_true_{target}.png", dpi=140)
        plt.close(fig)


def write_html_report(validation_metrics: pd.DataFrame, test_metrics: pd.DataFrame) -> None:
    images = [
        PLOT_DIR / "test_mae_by_target.png",
        PLOT_DIR / "test_pred_vs_true_inertia.png",
        PLOT_DIR / "test_pred_vs_true_load.png",
        PLOT_DIR / "test_pred_vs_true_ki.png",
    ]
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>SVR Hyperparameter Only Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827;} table{border-collapse:collapse;margin:12px 0 24px 0;} th,td{border:1px solid #cbd5e1;padding:6px 10px;text-align:right;} th:first-child,td:first-child{text-align:left;} img{max-width:100%;border:1px solid #cbd5e1;margin:8px 0 28px 0;} .desc{color:#334155;line-height:1.45;} code{background:#e2e8f0;padding:2px 4px;}</style>",
        "</head><body><h1>SVR Hyperparameter Only Report</h1>",
        "<p class='desc'>Data Scaling을 적용하지 않고 하나의 AI 모델(SVR)에 대해 하이퍼파라미터만 Grid Search한 결과입니다. resistance는 입력 feature에서 제외했습니다.</p>",
        "<h2>Selected Hyperparameters</h2>",
        "<p class='desc'>24개 SVR 하이퍼파라미터 조합을 외부 validation mean MAE 기준으로 정렬한 뒤 최상/중간/최악 3개 조합을 선택했습니다.</p>",
        validation_metrics[validation_metrics["target"] == "mean"].fillna("-").to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Validation Metrics</h2>",
        validation_metrics.fillna("-").to_html(index=False, float_format=lambda x: f"{x:.4f}"),
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

    model = MultiOutputRegressor(SVR())
    param_grid = {
        "estimator__C": [1.0, 10.0, 50.0, 100.0],
        "estimator__epsilon": [0.05, 0.1, 0.2],
        "estimator__gamma": ["scale", "auto"],
    }
    search = GridSearchCV(model, param_grid, cv=min(5, len(x_train)), scoring="neg_mean_absolute_error", n_jobs=-1)
    search.fit(x_train, y_train)
    pd.DataFrame(search.cv_results_).to_csv(CV_RESULTS_CSV, index=False, encoding="utf-8-sig")

    validation_candidates = []
    trained_candidates = []
    for params in ParameterGrid(param_grid):
        svr_params = {key.replace("estimator__", ""): value for key, value in params.items()}
        candidate = MultiOutputRegressor(SVR(**svr_params))
        candidate.fit(x_train, y_train)
        val_pred = candidate.predict(x_val)
        mean_mae = float(mean_absolute_error(y_val.to_numpy(), val_pred))
        validation_candidates.append({"params": params, "mean_mae": mean_mae})
        trained_candidates.append(candidate)

    ranked = pd.DataFrame(validation_candidates).sort_values("mean_mae").reset_index()
    selected = {
        "best": int(ranked.iloc[0]["index"]),
        "middle": int(ranked.iloc[len(ranked) // 2]["index"]),
        "worst": int(ranked.iloc[-1]["index"]),
    }

    rows = []
    for selection, candidate_index in selected.items():
        candidate = trained_candidates[candidate_index]
        val_pred = candidate.predict(x_val)
        params = validation_candidates[candidate_index]["params"]
        for row in metric_rows(y_val.to_numpy(), val_pred, "validation"):
            row["selection"] = selection
            row["rank_by_validation_mae"] = int(ranked[ranked["index"] == candidate_index].index[0]) + 1
            row["scaler"] = "none"
            row["params"] = str(params)
            rows.append(row)
    validation_metrics = pd.DataFrame(rows)
    validation_metrics.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    test_rows = []
    best_test_pred = None
    final_model = None
    for selection, candidate_index in selected.items():
        params = validation_candidates[candidate_index]["params"]
        svr_params = {key.replace("estimator__", ""): value for key, value in params.items()}
        candidate = MultiOutputRegressor(SVR(**svr_params))
        candidate.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))
        test_pred = candidate.predict(x_test)
        if selection == "best":
            best_test_pred = test_pred
            final_model = candidate
        for row in metric_rows(y_test.to_numpy(), test_pred, "test"):
            row["selection"] = selection
            row["rank_by_validation_mae"] = int(ranked[ranked["index"] == candidate_index].index[0]) + 1
            row["scaler"] = "none"
            row["params"] = str(params)
            test_rows.append(row)
    assert best_test_pred is not None and final_model is not None
    test_metrics = pd.DataFrame(test_rows)
    test_metrics.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")
    plot_test_metrics(test_metrics)
    plot_predictions(y_test.to_numpy(), best_test_pred)
    write_html_report(validation_metrics, test_metrics)
    best_params = validation_candidates[selected["best"]]["params"]
    joblib.dump({"model": final_model, "features": list(x.columns), "targets": TARGET_COLUMNS, "scaler": "none", "best_params": best_params}, MODEL_PATH)

    lines = [
        "SVR Hyperparameter Only Report",
        "data_scaling=none",
        "input_exclusion=resistance",
        f"train={len(x_train)}, validation={len(x_val)}, test={len(x_test)}",
        f"selected_best_params={best_params}",
        f"selected_middle_params={validation_candidates[selected['middle']]['params']}",
        f"selected_worst_params={validation_candidates[selected['worst']]['params']}",
        "selection_metric=validation mean MAE",
        "",
        test_metrics.to_string(index=False),
    ]
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {MODEL_PATH}")
    print(f"saved: {RESULT_CSV}")
    print(f"saved: {TEST_CSV}")
    print(f"saved: {REPORT_TXT}")
    print(f"saved: {REPORT_HTML}")
    print(f"saved: {CV_RESULTS_CSV}")
    print(f"saved plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
