from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots" / "svr_scaling_report"
HYPER_ONLY = OUT_DIR / "5_1_svr_hyperparameter_only_results.csv"
HYPER_ONLY_TEST = OUT_DIR / "5_1_svr_hyperparameter_only_test_metrics.csv"
SCALING_RESULTS = OUT_DIR / "5_2_svr_hyperparameter_scaling_results.csv"
SCALING_TEST = OUT_DIR / "5_2_svr_hyperparameter_scaling_test_metrics.csv"
SUMMARY_CSV = OUT_DIR / "5_3_svr_scaling_summary_table.csv"
IMPROVEMENT_CSV = OUT_DIR / "5_3_svr_scaling_improvement_table.csv"
REPORT_HTML = OUT_DIR / "5_3_svr_scaling_result_report.html"
REPORT_TXT = OUT_DIR / "5_3_svr_scaling_result_report.txt"


def plot_summary(summary: pd.DataFrame) -> None:
    mean_rows = summary[(summary["dataset"] == "validation") & (summary["target"] == "mean")].sort_values("mae")
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = mean_rows["experiment"] + "\n" + mean_rows["scaler"]
    ax.bar(labels, mean_rows["mae"], color="tab:blue")
    ax.set_title("SVR Validation Mean MAE by Scaling")
    ax.set_xlabel("experiment / scaler")
    ax.set_ylabel("validation mean MAE")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "svr_validation_mean_mae_by_scaling.png", dpi=140)
    plt.close(fig)


def plot_improvement_flow(improvement: pd.DataFrame) -> None:
    mean_rows = improvement[improvement["target"] == "mean"].sort_values("dataset")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(mean_rows))
    width = 0.35
    ax.bar(x - width / 2, mean_rows["baseline_mae"], width, label="No Scaling", color="tab:gray")
    ax.bar(x + width / 2, mean_rows["scaling_mae"], width, label="Best Scaling", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(mean_rows["dataset"])
    ax.set_title("SVR Performance Flow: No Scaling vs Best Scaling")
    ax.set_xlabel("dataset")
    ax.set_ylabel("mean MAE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "svr_no_scaling_to_best_scaling_flow.png", dpi=140)
    plt.close(fig)


def plot_target_improvement(improvement: pd.DataFrame) -> None:
    rows = improvement[(improvement["dataset"] == "test") & (improvement["target"] != "mean")]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rows["target"], rows["mae_improvement_percent"], color="tab:purple")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Test MAE Improvement by Target")
    ax.set_xlabel("target")
    ax.set_ylabel("MAE improvement (%)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "svr_test_target_improvement_percent.png", dpi=140)
    plt.close(fig)


def write_html(summary: pd.DataFrame, improvement: pd.DataFrame) -> None:
    validation_mean = summary[(summary["dataset"] == "validation") & (summary["target"] == "mean")]
    best = validation_mean.sort_values("mae").iloc[0]
    test_mean_improvement = improvement[(improvement["dataset"] == "test") & (improvement["target"] == "mean")].iloc[0]
    images = [
        PLOT_DIR / "svr_no_scaling_to_best_scaling_flow.png",
        PLOT_DIR / "svr_validation_mean_mae_by_scaling.png",
        PLOT_DIR / "svr_test_target_improvement_percent.png",
    ]
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>SVR Scaling Result Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827;} table{border-collapse:collapse;margin:12px 0 24px 0;} th,td{border:1px solid #cbd5e1;padding:6px 10px;text-align:right;} th:first-child,td:first-child{text-align:left;} img{max-width:100%;border:1px solid #cbd5e1;margin:8px 0 28px 0;} .desc{color:#334155;line-height:1.45;} code{background:#e2e8f0;padding:2px 4px;}</style>",
        "</head><body><h1>SVR Hyperparameter and Data Scaling Result Report</h1>",
        "<p class='desc'>하나의 AI 모델(SVR)을 사용하여, data scaling을 적용하지 않은 경우와 여러 scaling 방식을 적용한 경우를 비교했다. resistance는 입력 feature에서 제외했다. 목적은 모델 종류 변경이 아니라 dataset 전처리 방식이 성능에 미치는 영향을 확인하는 것이다.</p>",
        "<h2>Best Validation Result</h2>",
        f"<p>experiment: <code>{best['experiment']}</code>, scaler: <code>{best['scaler']}</code>, validation mean MAE: <code>{best['mae']:.4f}</code></p>",
        "<h2>Performance Improvement</h2>",
        f"<p class='desc'>최종 test mean MAE는 no scaling 대비 best scaling 적용 후 <code>{test_mean_improvement['mae_improvement_percent']:.2f}%</code> 개선되었다.</p>",
        improvement.fillna("-").to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Summary Table</h2>",
        summary.fillna("-").to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Graphs</h2>",
    ]
    descriptions = {
        "svr_no_scaling_to_best_scaling_flow.png": "Validation과 Test에서 No Scaling 결과가 Best Scaling 결과로 개선되는 흐름을 보여준다.",
        "svr_validation_mean_mae_by_scaling.png": "Scaling 방식별 validation mean MAE를 비교한다. 낮을수록 좋다.",
        "svr_test_target_improvement_percent.png": "최종 test 기준 target별 MAE 개선률을 보여준다. 양수이면 scaling 적용 후 성능이 개선된 것이다.",
    }
    for image in images:
        if image.exists():
            lines.append(f"<h3>{image.name}</h3><p class='desc'>{descriptions[image.name]}</p><img src='{image.relative_to(OUT_DIR).as_posix()}'>")
    lines.append("</body></html>")
    REPORT_HTML.write_text("\n".join(lines), encoding="utf-8")


def best_rows(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    if "selection" in df.columns:
        rows = df[df["selection"] == "best"].copy()
    else:
        mean_rows = df[df["target"] == "mean"].sort_values("mae")
        best_scaler = mean_rows.iloc[0]["scaler"]
        rows = df[df["scaler"] == best_scaler].copy()
    rows["experiment"] = experiment
    return rows


def make_improvement(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["experiment"] == "hyperparameter_only"]
    scaling = summary[summary["experiment"] == "hyperparameter_plus_scaling"]
    merged = baseline.merge(scaling, on=["dataset", "target"], suffixes=("_baseline", "_scaling"))
    rows = pd.DataFrame({
        "dataset": merged["dataset"],
        "target": merged["target"],
        "baseline_experiment": merged["experiment_baseline"],
        "baseline_scaler": merged["scaler_baseline"],
        "baseline_mae": merged["mae_baseline"],
        "scaling_experiment": merged["experiment_scaling"],
        "scaling_scaler": merged["scaler_scaling"],
        "scaling_mae": merged["mae_scaling"],
    })
    rows["mae_improvement"] = rows["baseline_mae"] - rows["scaling_mae"]
    rows["mae_improvement_percent"] = rows["mae_improvement"] / rows["baseline_mae"] * 100.0
    return rows


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    hyper = pd.read_csv(HYPER_ONLY)
    hyper = best_rows(hyper, "hyperparameter_only")
    hyper["experiment"] = "hyperparameter_only"
    if "params" in hyper.columns and "best_params" not in hyper.columns:
        hyper["best_params"] = hyper["params"]
    scaling = pd.read_csv(SCALING_RESULTS)
    scaling = best_rows(scaling, "hyperparameter_plus_scaling")
    scaling["experiment"] = "hyperparameter_plus_scaling"
    summary = pd.concat([hyper, scaling], ignore_index=True)
    summary["dataset"] = "validation"
    hyper_test = pd.read_csv(HYPER_ONLY_TEST)
    hyper_test = best_rows(hyper_test, "hyperparameter_only")
    if "params" in hyper_test.columns and "best_params" not in hyper_test.columns:
        hyper_test["best_params"] = hyper_test["params"]
    scaling_test = pd.read_csv(SCALING_TEST)
    scaling_test["experiment"] = "hyperparameter_plus_scaling"
    test_summary = pd.concat([hyper_test, scaling_test], ignore_index=True)
    test_summary["dataset"] = "test"
    summary = pd.concat([summary, test_summary], ignore_index=True)
    summary = summary[["dataset", "experiment", "scaler", "target", "mae", "rmse", "r2", "plus_minus_1_acc", "best_params"]]
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    improvement = make_improvement(summary)
    improvement.to_csv(IMPROVEMENT_CSV, index=False, encoding="utf-8-sig")
    plot_summary(summary)
    plot_improvement_flow(improvement)
    plot_target_improvement(improvement)
    write_html(summary, improvement)
    lines = ["SVR Scaling Result Report", "input_exclusion=resistance", "", "Improvement", improvement.to_string(index=False), "", "Summary", summary.to_string(index=False)]
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {SUMMARY_CSV}")
    print(f"saved: {IMPROVEMENT_CSV}")
    print(f"saved: {REPORT_HTML}")
    print(f"saved: {REPORT_TXT}")
    print(f"saved plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
