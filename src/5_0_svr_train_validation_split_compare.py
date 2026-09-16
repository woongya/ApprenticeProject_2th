from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots" / "svr_train_validation_compare"
FEATURE_CSV = OUT_DIR / "1_dataset_features.csv"
REPORT_TXT = OUT_DIR / "5_0_svr_train_validation_split_compare_report.txt"
REPORT_HTML = OUT_DIR / "5_0_svr_train_validation_split_compare_report.html"
SUMMARY_CSV = OUT_DIR / "5_0_svr_train_validation_split_compare_summary.csv"
PREDICTION_CSV = OUT_DIR / "5_0_svr_train_validation_split_compare_predictions.csv"
MODEL_SPLIT_PATH = OUT_DIR / "5_0_svr_split_selected_model.joblib"
MODEL_MERGED_PATH = OUT_DIR / "5_0_svr_train_validation_merged_model.joblib"

TARGET_COLUMNS = ["inertia", "load", "ki"]
DROP_COLUMNS = ["file", "split", "quality_issues", "resistance", *TARGET_COLUMNS]
FIXED_PARAMS = {
    "model__estimator__C": 10.0,
    "model__estimator__epsilon": 0.1,
    "model__estimator__gamma": "scale",
}


def make_x(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns]).select_dtypes(include=[np.number]).fillna(0.0)


def make_model() -> Pipeline:
    model = Pipeline([("scaler", RobustScaler()), ("model", MultiOutputRegressor(SVR()))])
    model.set_params(**FIXED_PARAMS)
    return model


def metric_rows(experiment: str, y_true: np.ndarray, y_pred: np.ndarray, note: str) -> list[dict[str, float | str]]:
    rows = []
    for i, target in enumerate(TARGET_COLUMNS):
        error = y_pred[:, i] - y_true[:, i]
        rows.append({
            "experiment": experiment,
            "note": note,
            "target": target,
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])) if len(y_true) >= 2 else np.nan,
            "plus_minus_1_acc": float((np.abs(error) <= 1.0).mean()),
        })
    rows.append({
        "experiment": experiment,
        "note": note,
        "target": "mean",
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": np.nan,
        "plus_minus_1_acc": np.nan,
    })
    return rows


def plot_comparison(summary: pd.DataFrame) -> None:
    mean_rows = summary[summary["target"] == "mean"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(mean_rows["experiment"], mean_rows["mae"], color=["tab:green", "tab:blue"][:len(mean_rows)])
    ax.set_title("Final Test Mean MAE: Split vs Merged Train/Validation")
    ax.set_xlabel("training strategy")
    ax.set_ylabel("test mean MAE")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "test_mean_mae_split_vs_merged.png", dpi=140)
    plt.close(fig)


def write_html(summary: pd.DataFrame, split_val_mae: float, merged_cv_mae: float) -> None:
    mean_rows = summary[summary["target"] == "mean"].set_index("experiment")
    diff = float(mean_rows.loc["merged_train_validation", "mae"] - mean_rows.loc["split_train_validation", "mae"])
    image = PLOT_DIR / "test_mean_mae_split_vs_merged.png"
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Train Validation Split Compare</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827;} table{border-collapse:collapse;margin:12px 0 24px 0;} th,td{border:1px solid #cbd5e1;padding:6px 10px;text-align:right;} th:first-child,td:first-child{text-align:left;} img{max-width:100%;border:1px solid #cbd5e1;margin:8px 0 28px 0;} .desc{color:#334155;line-height:1.45;} code{background:#e2e8f0;padding:2px 4px;}</style>",
        "</head><body><h1>Train/Validation Split Compare</h1>",
        "<p class='desc'>동일한 SVR + RobustScaler 모델과 고정 하이퍼파라미터를 사용하여 train과 validation을 분리한 경우와 train+validation을 하나로 뭉친 경우를 비교한다. resistance는 입력 feature에서 제외했다.</p>",
        "<h2>Compared Strategies</h2>",
        "<p><b>split_train_validation</b>: train으로 학습하고 validation으로 중간 성능을 확인한 뒤 train+validation으로 최종 재학습했다.</p>",
        "<p><b>merged_train_validation</b>: train+validation을 처음부터 하나의 학습 데이터로 합쳐 최종 모델을 학습했다. 별도 validation이 없어지므로 report용 성능 추정에는 cross-validation을 사용했다.</p>",
        "<h2>Fixed Parameters</h2>",
        f"<p>fixed_params: <code>{FIXED_PARAMS}</code></p>",
        f"<p>split external validation mean MAE: <code>{split_val_mae:.4f}</code></p>",
        f"<p>merged cross-validation mean MAE: <code>{merged_cv_mae:.4f}</code></p>",
        "<h2>Test Result Summary</h2>",
        f"<p class='desc'>test mean MAE difference(merged - split)는 <code>{diff:.4f}</code>이다. 양수이면 split 방식이 더 좋고, 음수이면 merged 방식이 더 좋다.</p>",
        summary.fillna("-").to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Graph</h2>",
    ]
    if image.exists():
        lines.append(f"<img src='{image.relative_to(OUT_DIR).as_posix()}'>")
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

    split_model = make_model()
    split_model.fit(x_train, y_train)
    split_val_pred = split_model.predict(x_val)
    split_val_mae = float(mean_absolute_error(y_val.to_numpy(), split_val_pred))
    split_model.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))
    split_pred = split_model.predict(x_test)

    merged_x = pd.concat([x_train, x_val])
    merged_y = pd.concat([y_train, y_val])
    merged_cv_pred = cross_val_predict(make_model(), merged_x, merged_y, cv=min(5, len(merged_x)), n_jobs=-1)
    merged_cv_mae = float(mean_absolute_error(merged_y.to_numpy(), merged_cv_pred))
    merged_model = make_model()
    merged_model.fit(merged_x, merged_y)
    merged_pred = merged_model.predict(x_test)

    rows = []
    rows.extend(metric_rows("split_train_validation", y_test.to_numpy(), split_pred, "fixed hyperparameters, train/validation kept separate before final refit"))
    rows.extend(metric_rows("merged_train_validation", y_test.to_numpy(), merged_pred, "fixed hyperparameters, train+validation merged for final training"))
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    pred_out = df[test_mask][["file", "split", *TARGET_COLUMNS]].copy()
    for i, target in enumerate(TARGET_COLUMNS):
        pred_out[f"pred_split_{target}"] = split_pred[:, i]
        pred_out[f"pred_merged_{target}"] = merged_pred[:, i]
        pred_out[f"err_split_{target}"] = pred_out[f"pred_split_{target}"] - pred_out[target]
        pred_out[f"err_merged_{target}"] = pred_out[f"pred_merged_{target}"] - pred_out[target]
    pred_out.to_csv(PREDICTION_CSV, index=False, encoding="utf-8-sig")

    joblib.dump({"model": split_model, "features": list(x.columns), "targets": TARGET_COLUMNS, "params": FIXED_PARAMS, "strategy": "split_train_validation"}, MODEL_SPLIT_PATH)
    joblib.dump({"model": merged_model, "features": list(x.columns), "targets": TARGET_COLUMNS, "params": FIXED_PARAMS, "strategy": "merged_train_validation"}, MODEL_MERGED_PATH)
    plot_comparison(summary)
    write_html(summary, split_val_mae, merged_cv_mae)

    lines = ["SVR Train/Validation Split Compare Report", "input_exclusion=resistance", "hyperparameter_tuning=not_used", f"fixed_params={FIXED_PARAMS}", f"train={len(x_train)}, validation={len(x_val)}, test={len(x_test)}", f"split_external_validation_mean_mae={split_val_mae:.4f}", f"merged_cross_validation_mean_mae={merged_cv_mae:.4f}", "", summary.to_string(index=False)]
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {SUMMARY_CSV}")
    print(f"saved: {PREDICTION_CSV}")
    print(f"saved: {REPORT_TXT}")
    print(f"saved: {REPORT_HTML}")
    print(f"saved: {MODEL_SPLIT_PATH}")
    print(f"saved: {MODEL_MERGED_PATH}")
    print(f"saved plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
