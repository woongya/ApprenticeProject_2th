from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots" / "heuristic"
FEATURE_CSV = OUT_DIR / "1_dataset_features.csv"
REPORT_TXT = OUT_DIR / "4_heuristic_analysis_report.txt"
REPORT_HTML = OUT_DIR / "4_heuristic_analysis_report.html"
CORRELATION_CSV = OUT_DIR / "4_heuristic_feature_label_correlation.csv"
RULE_CSV = OUT_DIR / "4_heuristic_rule_predictions.csv"
SUMMARY_CSV = OUT_DIR / "4_heuristic_rule_metrics.csv"

TARGET_COLUMNS = ["inertia", "load", "ki"]
DROP_COLUMNS = ["file", "split", "quality_issues", "resistance", *TARGET_COLUMNS]
CORE_FEATURES = [
    "duration_ms",
    "speed_peak",
    "speed_avg",
    "speed_rise_time_ms",
    "current_peak",
    "current_avg",
    "current_ripple",
    "pwm_avg",
    "speed_per_pwm_gain",
    "energy_proxy",
]


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["split"] == "train"].copy() if "split" in df.columns else df.copy()
    val_df = df[df["split"] == "validation"].copy() if "split" in df.columns else pd.DataFrame()
    test_df = df[df["split"] == "test"].copy() if "split" in df.columns else pd.DataFrame()
    return train_df, val_df, test_df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns]).select_dtypes(include=[np.number]).fillna(0.0)


def rank_to_label(train_feature: pd.Series, train_label: pd.Series, eval_feature: pd.Series) -> np.ndarray:
    order = np.argsort(train_feature.to_numpy())
    sorted_labels = train_label.to_numpy()[order]
    ranks = eval_feature.rank(method="average", pct=True).to_numpy()
    indexes = np.clip(np.round(ranks * (len(sorted_labels) - 1)).astype(int), 0, len(sorted_labels) - 1)
    return sorted_labels[indexes]


def heuristic_predict(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    train_x = build_feature_matrix(train_df)
    eval_x = build_feature_matrix(eval_df)
    candidates = {
        "inertia": ["speed_rise_time_ms", "duration_ms", "energy_proxy"],
        "load": ["current_avg", "current_peak", "current_ripple", "energy_proxy"],
        "ki": ["speed_per_pwm_gain", "speed_avg", "speed_peak"],
    }

    output = eval_df[["file", "split", *TARGET_COLUMNS]].copy()
    for target, features in candidates.items():
        available = [feature for feature in features if feature in train_x.columns and feature in eval_x.columns]
        if not available:
            output[f"pred_{target}"] = float(train_df[target].mean())
            continue
        corrs = train_x[available].corrwith(train_df[target].astype(float)).abs().fillna(0.0)
        feature = str(corrs.sort_values(ascending=False).index[0])
        pred = rank_to_label(train_x[feature], train_df[target].astype(float), eval_x[feature])
        output[f"pred_{target}"] = pred
        output[f"rule_feature_{target}"] = feature
        output[f"rule_corr_abs_{target}"] = float(corrs[feature])

    for target in TARGET_COLUMNS:
        output[f"err_{target}"] = output[f"pred_{target}"] - output[target]
    return output


def metric_rows(pred_df: pd.DataFrame) -> list[dict[str, float | str]]:
    rows = []
    for target in TARGET_COLUMNS:
        y_true = pred_df[target].astype(float).to_numpy()
        y_pred = pred_df[f"pred_{target}"].astype(float).to_numpy()
        rows.append({
            "target": target,
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
            "plus_minus_1_acc": float((np.abs(y_true - y_pred) <= 1.0).mean()),
        })
    return rows


def plot_correlation(corr: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    pivot = corr.pivot(index="target", columns="feature", values="corr").fillna(0.0)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=55, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Heuristic Feature-Label Correlation")
    fig.colorbar(im, ax=ax, label="correlation")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "feature_label_correlation_heatmap.png", dpi=140)
    plt.close(fig)


def plot_rule_predictions(pred_df: pd.DataFrame) -> None:
    for target in TARGET_COLUMNS:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(pred_df[target], pred_df[f"pred_{target}"], alpha=0.75)
        lo = min(pred_df[target].min(), pred_df[f"pred_{target}"].min())
        hi = max(pred_df[target].max(), pred_df[f"pred_{target}"].max())
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set_xlabel(f"true {target}")
        ax.set_ylabel(f"heuristic predicted {target}")
        ax.set_title(f"Heuristic Rule Prediction: {target}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"rule_pred_vs_true_{target}.png", dpi=140)
        plt.close(fig)


def write_html(corr: pd.DataFrame, metrics: pd.DataFrame) -> None:
    top_corr = corr.sort_values("abs_corr", ascending=False).groupby("target").head(5)
    images = [
        PLOT_DIR / "feature_label_correlation_heatmap.png",
        PLOT_DIR / "rule_pred_vs_true_inertia.png",
        PLOT_DIR / "rule_pred_vs_true_load.png",
        PLOT_DIR / "rule_pred_vs_true_ki.png",
    ]
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Heuristic Data Analysis</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827;} table{border-collapse:collapse;margin:12px 0 24px 0;} th,td{border:1px solid #cbd5e1;padding:6px 10px;text-align:right;} th:first-child,td:first-child{text-align:left;} img{max-width:100%;border:1px solid #cbd5e1;margin:8px 0 28px 0;} .desc{color:#334155;line-height:1.45;} code{background:#e2e8f0;padding:2px 4px;}</style>",
        "</head><body><h1>Heuristic Data Analysis</h1>",
        "<p class='desc'>이 report는 ML 모델 학습 전에 feature와 label 사이의 관계를 사람이 해석할 수 있도록 만든 휴리스틱 분석입니다. resistance는 입력 feature에서 제외했습니다.</p>",
        "<h2>Top Correlations</h2>",
        top_corr.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Heuristic Rule Metrics</h2>",
        metrics.fillna("-").to_html(index=False, float_format=lambda x: f"{x:.4f}"),
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
    train_df, val_df, test_df = split_data(df)
    eval_df = test_df if not test_df.empty else val_df
    if train_df.empty or eval_df.empty:
        raise SystemExit("Need train and test/validation rows in outputs/1_dataset_features.csv")

    x = build_feature_matrix(df)
    features = [feature for feature in CORE_FEATURES if feature in x.columns]
    if not features:
        features = list(x.columns)
    corr_rows = []
    for target in TARGET_COLUMNS:
        for feature in features:
            corr = float(df[[feature, target]].corr().iloc[0, 1])
            corr_rows.append({"target": target, "feature": feature, "corr": corr, "abs_corr": abs(corr)})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(CORRELATION_CSV, index=False, encoding="utf-8-sig")
    plot_correlation(corr_df)

    pred_df = heuristic_predict(train_df, eval_df)
    pred_df.to_csv(RULE_CSV, index=False, encoding="utf-8-sig")
    metrics = pd.DataFrame(metric_rows(pred_df))
    metrics.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    plot_rule_predictions(pred_df)
    write_html(corr_df, metrics)

    lines = [
        "Heuristic Data Analysis Report",
        f"rows={len(df)}, train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}",
        "input_exclusion=resistance",
        "rule: train feature rank를 label rank에 매핑하여 test/validation label을 추정",
        "",
        metrics.to_string(index=False),
    ]
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {REPORT_TXT}")
    print(f"saved: {REPORT_HTML}")
    print(f"saved: {CORRELATION_CSV}")
    print(f"saved: {RULE_CSV}")
    print(f"saved: {SUMMARY_CSV}")
    print(f"saved plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
