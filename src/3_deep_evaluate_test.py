from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots" / "deep_test"
FEATURE_CSV = OUT_DIR / "1_dataset_features.csv"
MODEL_PATH = OUT_DIR / "3_deep_mlp_model.joblib"
MODEL_PATHS = {
    "best": OUT_DIR / "3_deep_mlp_model_best.joblib",
    "middle": OUT_DIR / "3_deep_mlp_model_middle.joblib",
    "worst": OUT_DIR / "3_deep_mlp_model_worst.joblib",
}
REPORT_PATH = OUT_DIR / "3_deep_test_evaluation_report.txt"
PREDICTION_CSV = OUT_DIR / "3_deep_test_predictions.csv"
METRICS_CSV = OUT_DIR / "3_deep_test_metrics.csv"
ERROR_SUMMARY_CSV = OUT_DIR / "3_deep_test_error_summary.csv"
REPORT_HTML = OUT_DIR / "3_deep_test_report.html"
COMPARISON_CSV = OUT_DIR / "3_deep_test_model_comparison.csv"
SELECTED_MODELS_CSV = OUT_DIR / "3_deep_selected_models.csv"


def write_csv_with_fallback(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"warning: {path} is locked. saved fallback: {fallback}")
        return fallback


def write_text_with_fallback(path: Path, text: str) -> Path:
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        fallback.write_text(text, encoding="utf-8")
        print(f"warning: {path} is locked. saved fallback: {fallback}")
        return fallback


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> list[str]:
    lines = ["Deep MLP Final Test Evaluation"]
    for i, target in enumerate(targets):
        mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        r2 = float(r2_score(y_true[:, i], y_pred[:, i])) if len(y_true) >= 2 else float("nan")
        acc1 = float((np.abs(y_true[:, i] - y_pred[:, i]) <= 1.0).mean())
        lines.append(f"{target}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, +/-1_acc={acc1:.3f}")
    return lines


def metric_rows(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for i, target in enumerate(targets):
        error = y_pred[:, i] - y_true[:, i]
        abs_error = np.abs(error)
        rows.append({
            "target": target,
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])) if len(y_true) >= 2 else float("nan"),
            "plus_minus_1_acc": float((abs_error <= 1.0).mean()),
            "mean_error": float(error.mean()),
            "max_abs_error": float(abs_error.max()),
            "p90_abs_error": float(np.quantile(abs_error, 0.90)),
        })
    return rows


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for i, target in enumerate(targets):
        out = PLOT_DIR / f"test_pred_vs_true_{target}.png"
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true[:, i], y_pred[:, i], s=50, alpha=0.85, edgecolors="k", linewidths=0.3)
        low = min(float(y_true[:, i].min()), float(y_pred[:, i].min()))
        high = max(float(y_true[:, i].max()), float(y_pred[:, i].max()))
        ax.plot([low, high], [low, high], color="tab:red", linestyle="--", label="ideal")
        ax.set_title(f"Final Test Prediction vs True: {target}")
        ax.set_xlabel(f"true {target}")
        ax.set_ylabel(f"predicted {target}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    out = PLOT_DIR / "test_model_comparison_mean_mae.png"
    mean_rows = metrics[metrics["target"] == "mean"].copy()
    if mean_rows.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(mean_rows["model"], mean_rows["mae"], color=["tab:green", "tab:orange", "tab:red"][:len(mean_rows)])
    ax.set_title("Final Test Mean MAE: Best vs Middle vs Worst")
    ax.set_xlabel("model selected by validation performance")
    ax.set_ylabel("test mean MAE")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_error_report(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    abs_errors = [np.abs(y_pred[:, i] - y_true[:, i]) for i in range(len(targets))]

    out = PLOT_DIR / "test_abs_error_boxplot.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(abs_errors, labels=targets)
    ax.set_title("Final Test Absolute Error Distribution")
    ax.set_ylabel("absolute error")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)

    for i, target in enumerate(targets):
        out = PLOT_DIR / f"test_error_hist_{target}.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(y_pred[:, i] - y_true[:, i], bins=min(20, max(5, len(y_true) // 4)), color="tab:purple", alpha=0.8)
        ax.axvline(0.0, color="tab:red", linestyle="--", label="zero error")
        ax.set_title(f"Final Test Error Histogram: {target}")
        ax.set_xlabel("prediction error")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)


def write_html_report(comparison: pd.DataFrame, prediction_csv: Path) -> None:
    model_columns = ["model", "model_file", "rank_by_validation_mae", "scaler", "hidden_layers", "learning_rate", "alpha", "epochs"]
    available_model_columns = [column for column in model_columns if column in comparison.columns]
    model_summary = comparison[comparison["target"] == "mean"][available_model_columns].copy()
    metric_columns = ["model", "target", "mae", "rmse", "r2", "plus_minus_1_acc", "mean_error", "max_abs_error", "p90_abs_error"]
    available_metric_columns = [column for column in metric_columns if column in comparison.columns]
    test_metrics = comparison[available_metric_columns].copy()
    model_summary = model_summary.fillna("-")
    test_metrics = test_metrics.fillna("-")
    images = [
        PLOT_DIR / "test_pred_vs_true_inertia.png",
        PLOT_DIR / "test_pred_vs_true_load.png",
        PLOT_DIR / "test_pred_vs_true_ki.png",
        PLOT_DIR / "test_abs_error_boxplot.png",
        PLOT_DIR / "test_error_hist_inertia.png",
        PLOT_DIR / "test_error_hist_load.png",
        PLOT_DIR / "test_error_hist_ki.png",
        PLOT_DIR / "test_model_comparison_mean_mae.png",
    ]
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Deep MLP Final Test Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827;} table{border-collapse:collapse;margin:12px 0 24px 0;} th,td{border:1px solid #cbd5e1;padding:6px 10px;text-align:right;} th:first-child,td:first-child{text-align:left;} img{max-width:100%;border:1px solid #cbd5e1;margin:8px 0 28px 0;} .desc{color:#334155;line-height:1.45;} code{background:#e2e8f0;padding:2px 4px;}</style>",
        "</head><body>",
        "<h1>Deep MLP Final Test Report</h1>",
        "<p class='desc'>이 report는 validation 성능 기준으로 선택한 best/middle/worst MLP 모델 3개를 동일한 data_test로 비교 평가한 결과입니다. Test 데이터는 모델 선택에 사용하지 않았습니다.</p>",
        "<h2>Selected Models</h2>",
        "<p class='desc'>각 모델이 어떤 전처리와 하이퍼파라미터 조합으로 만들어졌는지 보여줍니다. rank_by_validation_mae는 36개 후보 중 validation mean MAE 기준 순위입니다.</p>",
        model_summary.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Test Metrics</h2>",
        "<p class='desc'>동일한 test 데이터에서 best/middle/worst 모델의 label별 성능을 비교합니다. mean 행은 inertia/load/ki 전체 평균 MAE/RMSE입니다.</p>",
        test_metrics.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        f"<p>Model comparison CSV: <code>{COMPARISON_CSV.name}</code></p>",
        f"<p>Prediction CSV: <code>{prediction_csv.name}</code></p>",
        "<h2>Plots</h2>",
    ]
    descriptions = {
        "test_pred_vs_true_inertia.png": "inertia 실제값과 예측값 비교입니다. 점이 빨간 대각선에 가까울수록 예측이 정확합니다.",
        "test_pred_vs_true_load.png": "load 실제값과 예측값 비교입니다. 부하 label에 대한 최종 일반화 성능을 확인합니다.",
        "test_pred_vs_true_ki.png": "ki 실제값과 예측값 비교입니다. KI label이 계단형으로 잘 분리되는지 확인합니다.",
        "test_abs_error_boxplot.png": "세 label의 절대 오차 분포입니다. 중앙값과 outlier를 비교하여 어떤 label이 어려운지 판단합니다.",
        "test_error_hist_inertia.png": "inertia 예측 오차 histogram입니다. 0을 중심으로 모이면 bias가 작습니다.",
        "test_error_hist_load.png": "load 예측 오차 histogram입니다. 양수/음수로 치우치면 과대/과소 예측 경향이 있습니다.",
        "test_error_hist_ki.png": "ki 예측 오차 histogram입니다. 오차가 ±1 안에 많이 모이는지 확인합니다.",
        "test_model_comparison_mean_mae.png": "Validation 성능 기준 best/middle/worst로 선택한 3개 모델의 최종 test 평균 MAE 비교입니다.",
    }
    for image in images:
        if image.exists():
            rel = image.relative_to(OUT_DIR).as_posix()
            lines.append(f"<h3>{image.name}</h3><p class='desc'>{descriptions.get(image.name, '')}</p><img src='{rel}'>")
    lines.append("</body></html>")
    write_text_with_fallback(REPORT_HTML, "\n".join(lines))


def main() -> None:
    df = pd.read_csv(FEATURE_CSV)
    test_df = df[df["split"] == "test"].copy() if "split" in df.columns else pd.DataFrame()
    if test_df.empty:
        raise SystemExit("No test rows found. Put MOVELOG csv files in data_test and run src/1_build_dataset.py first.")

    bundles = {name: joblib.load(path) for name, path in MODEL_PATHS.items() if path.exists()}
    if not bundles:
        bundles = {"best": joblib.load(MODEL_PATH)}
    first_bundle = next(iter(bundles.values()))
    targets = first_bundle["targets"]
    y_test = test_df[targets].astype(float)

    comparison_rows = []
    prediction_outputs = test_df[["file", "split", *targets]].copy()
    report = ["Deep MLP Final Test Model Comparison", f"test_rows={len(test_df)}", ""]
    best_pred = None
    for model_name, bundle in bundles.items():
        x_test = test_df.reindex(columns=bundle["features"], fill_value=0.0).fillna(0.0)
        pred = bundle["model"].predict(x_test)
        if model_name == "best":
            best_pred = pred
        report.extend(evaluate(y_test.to_numpy(), pred, targets))
        report.append("")
        rows = metric_rows(y_test.to_numpy(), pred, targets)
        mean_mae = float(mean_absolute_error(y_test.to_numpy(), pred))
        mean_rmse = float(np.sqrt(mean_squared_error(y_test.to_numpy(), pred)))
        rows.append({"target": "mean", "mae": mean_mae, "rmse": mean_rmse, "r2": np.nan, "plus_minus_1_acc": np.nan, "mean_error": np.nan, "max_abs_error": np.nan, "p90_abs_error": np.nan})
        for row in rows:
            row["model"] = model_name
            comparison_rows.append(row)
        for i, target in enumerate(targets):
            prediction_outputs[f"pred_{model_name}_{target}"] = pred[:, i]
            prediction_outputs[f"err_{model_name}_{target}"] = prediction_outputs[f"pred_{model_name}_{target}"] - prediction_outputs[target]

    if best_pred is None:
        best_pred = next(iter(bundles.values()))["model"].predict(test_df.reindex(columns=first_bundle["features"], fill_value=0.0).fillna(0.0))
    report_path = write_text_with_fallback(REPORT_PATH, "\n".join(report) + "\n")
    plot_pred_vs_true(y_test.to_numpy(), best_pred, targets)
    plot_error_report(y_test.to_numpy(), best_pred, targets)

    comparison = pd.DataFrame(comparison_rows)
    selected_columns = ["selection", "model_file", "rank_by_validation_mae", "scaler", "hidden_layers", "learning_rate", "alpha", "epochs"]
    if SELECTED_MODELS_CSV.exists():
        selected_info = pd.read_csv(SELECTED_MODELS_CSV)
        comparison = comparison.merge(selected_info[selected_columns], left_on="model", right_on="selection", how="left")
        comparison = comparison.drop(columns=["selection"])
    ordered_columns = [
        "model", "model_file", "rank_by_validation_mae", "scaler", "hidden_layers", "learning_rate", "alpha", "epochs",
        "target", "mae", "rmse", "r2", "plus_minus_1_acc", "mean_error", "max_abs_error", "p90_abs_error",
    ]
    comparison = comparison[[column for column in ordered_columns if column in comparison.columns]]
    comparison_csv = write_csv_with_fallback(comparison, COMPARISON_CSV)
    metrics = comparison[comparison["model"] == "best"].drop(columns=["model"])
    metrics_csv = write_csv_with_fallback(metrics, METRICS_CSV)
    plot_model_comparison(comparison)

    prediction_csv = write_csv_with_fallback(prediction_outputs, PREDICTION_CSV)
    error_columns = [column for column in prediction_outputs.columns if column.startswith("err_")]
    error_summary_csv = write_csv_with_fallback(prediction_outputs[error_columns].describe().reset_index(), ERROR_SUMMARY_CSV)
    write_html_report(comparison, PREDICTION_CSV)

    print(f"test rows: {len(test_df)}")
    print(f"saved: {report_path}")
    print(f"saved: {metrics_csv}")
    print(f"saved: {error_summary_csv}")
    print(f"saved: {prediction_csv}")
    print(f"saved: {comparison_csv}")
    print(f"saved: {REPORT_HTML}")
    print(f"saved plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
