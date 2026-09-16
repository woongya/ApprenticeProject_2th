from __future__ import annotations

from pathlib import Path
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots" / "deep"
FEATURE_CSV = OUT_DIR / "1_dataset_features.csv"
MODEL_PATH = OUT_DIR / "3_deep_mlp_model.joblib"
MODEL_BEST_PATH = OUT_DIR / "3_deep_mlp_model_best.joblib"
MODEL_MIDDLE_PATH = OUT_DIR / "3_deep_mlp_model_middle.joblib"
MODEL_WORST_PATH = OUT_DIR / "3_deep_mlp_model_worst.joblib"
MODEL_SELECTED_CSV = OUT_DIR / "3_deep_selected_models.csv"
MODEL_SELECTED_REPORT = OUT_DIR / "3_deep_selected_models_report.txt"
REPORT_PATH = OUT_DIR / "3_deep_learning_report.txt"
RESULTS_CSV = OUT_DIR / "3_deep_hyperparameter_results.csv"
SELECTION_REPORT_PATH = OUT_DIR / "3_deep_model_selection_report.txt"
SELECTION_TOP_CSV = OUT_DIR / "3_deep_model_selection_top10.csv"

TARGET_COLUMNS = ["inertia", "load", "ki"]
DROP_COLUMNS = ["file", "split", "quality_issues", "resistance", *TARGET_COLUMNS]


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


def make_scaler(name: str):
    if name == "none":
        return "passthrough"
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError(name)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_targets(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for index, target in enumerate(TARGET_COLUMNS):
        metrics[f"{prefix}_{target}_mae"] = float(mean_absolute_error(y_true[:, index], y_pred[:, index]))
        metrics[f"{prefix}_{target}_rmse"] = rmse(y_true[:, index], y_pred[:, index])
        metrics[f"{prefix}_{target}_r2"] = float(r2_score(y_true[:, index], y_pred[:, index])) if len(y_true) >= 2 else np.nan
        metrics[f"{prefix}_{target}_acc_pm1"] = float((np.abs(y_true[:, index] - y_pred[:, index]) <= 1.0).mean())
    metrics[f"{prefix}_mean_mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics[f"{prefix}_mean_rmse"] = rmse(y_true, y_pred)
    return metrics


def plot_loss_curve(model: Pipeline, out: Path) -> None:
    mlp = model.named_steps["mlp"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mlp.loss_curve_, label="train loss")
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
        ax2 = ax.twinx()
        ax2.plot(mlp.validation_scores_, color="tab:orange", label="validation score")
        ax2.set_ylabel("validation score")
    ax.set_title("MLP Training Loss Curve")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    for index, target in enumerate(TARGET_COLUMNS):
        out = PLOT_DIR / f"pred_vs_true_{target}.png"
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true[:, index], y_pred[:, index], s=50, alpha=0.85, edgecolors="k", linewidths=0.3)
        low = min(float(y_true[:, index].min()), float(y_pred[:, index].min()))
        high = max(float(y_true[:, index].max()), float(y_pred[:, index].max()))
        ax.plot([low, high], [low, high], color="tab:red", linestyle="--", label="ideal")
        ax.set_title(f"Prediction vs True: {target}")
        ax.set_xlabel(f"true {target}")
        ax.set_ylabel(f"predicted {target}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)


def plot_model_selection(results: pd.DataFrame) -> None:
    top = results.sort_values("val_mean_mae").head(12).copy()
    labels = [f"{row.scaler}\n{row.hidden_layers}\nlr={row.learning_rate}\na={row.alpha}" for row in top.itertuples()]
    out = PLOT_DIR / "model_selection_top12_val_mae.png"
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(range(len(top)), top["val_mean_mae"], color="tab:blue", alpha=0.85)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("validation mean MAE")
    ax.set_title("Top 12 MLP Hyperparameter Candidates")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)

    out = PLOT_DIR / "model_selection_scaler_boxplot.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    scaler_groups = [group["val_mean_mae"].to_numpy() for _, group in results.groupby("scaler")]
    scaler_names = [name for name, _ in results.groupby("scaler")]
    ax.boxplot(scaler_groups, labels=scaler_names)
    ax.set_ylabel("validation mean MAE")
    ax.set_title("Validation MAE by Scaling Method")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_model_selection_report(results: pd.DataFrame, best_row: dict[str, float | str | int], feature_count: int) -> None:
    ranked = results.sort_values("val_mean_mae").reset_index(drop=True)
    write_csv_with_fallback(ranked.head(10), SELECTION_TOP_CSV)
    best = ranked.iloc[0]
    second = ranked.iloc[1] if len(ranked) > 1 else ranked.iloc[0]
    improvement = float(second["val_mean_mae"] - best["val_mean_mae"])

    scaler_summary = ranked.groupby("scaler")["val_mean_mae"].agg(["count", "mean", "min", "std"]).sort_values("min")
    hidden_summary = ranked.groupby("hidden_layers")["val_mean_mae"].agg(["count", "mean", "min", "std"]).sort_values("min")

    lines = [
        "Deep MLP Model Selection Report",
        "",
        "Purpose",
        "Validation 데이터를 사용하여 여러 MLP 하이퍼파라미터 후보 중 최종 모델을 선택한다.",
        "Test 데이터는 이 단계에서 사용하지 않고, 최종 평가 스크립트에서만 사용한다.",
        "",
        "Search Space",
        "scaler: none, standard, minmax",
        "hidden_layers: (32,), (64, 32), (128, 64, 32)",
        "learning_rate: 0.001, 0.0005",
        "alpha(L2 regularization): 0.0001, 0.001",
        f"total_candidates: {len(ranked)}",
        f"feature_count: {feature_count}",
        "input_exclusion: resistance",
        "selection_metric: validation mean MAE",
        "",
        "Best Model",
        f"scaler: {best['scaler']}",
        f"hidden_layers: {best['hidden_layers']}",
        f"learning_rate: {best['learning_rate']}",
        f"alpha: {best['alpha']}",
        f"epochs: {best['epochs']}",
        f"validation_mean_mae: {best['val_mean_mae']:.4f}",
        f"validation_mean_rmse: {best['val_mean_rmse']:.4f}",
        f"gap_to_second_best_mae: {improvement:.4f}",
        "",
        "Best Model Target Metrics",
    ]
    for target in TARGET_COLUMNS:
        lines.append(
            f"{target}: MAE={best[f'val_{target}_mae']:.4f}, RMSE={best[f'val_{target}_rmse']:.4f}, "
            f"R2={best[f'val_{target}_r2']:.4f}, +/-1_acc={best[f'val_{target}_acc_pm1']:.4f}"
        )

    lines.extend([
        "",
        "Top 10 Candidates",
    ])
    for rank, row in ranked.head(10).iterrows():
        lines.append(
            f"{rank + 1}. scaler={row['scaler']}, hidden={row['hidden_layers']}, lr={row['learning_rate']}, "
            f"alpha={row['alpha']}, val_mean_mae={row['val_mean_mae']:.4f}, epochs={row['epochs']}"
        )

    lines.extend([
        "",
        "Scaler Summary",
        scaler_summary.to_string(),
        "",
        "Hidden Layer Summary",
        hidden_summary.to_string(),
        "",
        "Selection Rationale",
        "최종 모델은 validation mean MAE가 가장 낮은 후보를 선택했다.",
        "Validation은 하이퍼파라미터 선택에 사용하고, Test는 최종 일반화 성능 평가에만 사용하여 data leakage를 방지한다.",
        "Scaling은 MLP의 gradient 학습 안정성에 영향을 주기 때문에 none/standard/minmax를 비교했다.",
    ])
    write_text_with_fallback(SELECTION_REPORT_PATH, "\n".join(lines) + "\n")


def write_selected_models_report(results: pd.DataFrame, selected: dict[str, int], selected_paths: dict[str, Path], feature_count: int) -> None:
    ranked = results.sort_values("val_mean_mae").reset_index()
    rank_by_index = {int(row["index"]): int(pos + 1) for pos, row in ranked.iterrows()}
    total = len(ranked)
    rows = []
    lines = [
        "Selected Deep MLP Models",
        "",
        "Selection Rule",
        "best: validation mean MAE가 가장 낮은 모델",
        "middle: validation mean MAE 순위상 중간 위치의 모델",
        "worst: validation mean MAE가 가장 높은 모델",
        "",
        "Common Settings",
        "model: sklearn.neural_network.MLPRegressor",
        "activation: relu",
        "solver: adam",
        "max_iter: 800",
        "early_stopping: True",
        "validation_fraction: 0.15 inside training split",
        "n_iter_no_change: 40",
        "input_exclusion: resistance",
        f"feature_count: {feature_count}",
        "",
        "Selected Models",
    ]

    for name in ["best", "middle", "worst"]:
        model_index = selected[name]
        row = results.loc[model_index]
        info = {
            "selection": name,
            "rank_by_validation_mae": rank_by_index[model_index],
            "total_candidates": total,
            "model_file": selected_paths[name].name,
            "scaler": row["scaler"],
            "hidden_layers": row["hidden_layers"],
            "learning_rate": row["learning_rate"],
            "alpha": row["alpha"],
            "epochs": row["epochs"],
            "val_mean_mae": row["val_mean_mae"],
            "val_mean_rmse": row["val_mean_rmse"],
            "feature_count": feature_count,
        }
        for target in TARGET_COLUMNS:
            info[f"val_{target}_mae"] = row[f"val_{target}_mae"]
            info[f"val_{target}_rmse"] = row[f"val_{target}_rmse"]
            info[f"val_{target}_r2"] = row[f"val_{target}_r2"]
            info[f"val_{target}_acc_pm1"] = row[f"val_{target}_acc_pm1"]
        rows.append(info)

        lines.extend([
            f"[{name}]",
            f"rank_by_validation_mae: {rank_by_index[model_index]}/{total}",
            f"model_file: {selected_paths[name].name}",
            f"scaler: {row['scaler']}",
            f"hidden_layers: {row['hidden_layers']}",
            f"learning_rate: {row['learning_rate']}",
            f"alpha: {row['alpha']}",
            f"epochs: {row['epochs']}",
            f"val_mean_mae: {row['val_mean_mae']:.4f}",
            f"val_mean_rmse: {row['val_mean_rmse']:.4f}",
            "",
        ])

    write_csv_with_fallback(pd.DataFrame(rows), MODEL_SELECTED_CSV)
    write_text_with_fallback(MODEL_SELECTED_REPORT, "\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURE_CSV)
    df = df[df["quality_issues"].fillna("").str.contains("missing_label") == False].copy()
    if len(df) < 20:
        raise SystemExit("Deep learning experiment needs at least 20 labeled rows. Collect more data or use train_model.py baseline.")

    y = df[TARGET_COLUMNS].astype(float)
    x = df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns]).select_dtypes(include=[np.number]).fillna(0.0)

    if "split" in df.columns and (df["split"] == "train").any() and (df["split"] == "validation").any():
        train_mask = df["split"] == "train"
        val_mask = df["split"] == "validation"
        x_train, y_train = x[train_mask], y[train_mask]
        x_val, y_val = x[val_mask], y[val_mask]
        x_test, y_test = pd.DataFrame(), pd.DataFrame()
        split_name = "directory_train_validation"
    else:
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.30, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=42)
        split_name = "random_train_validation_test"

    configs = []
    for scaler in ["none", "standard", "minmax"]:
        for hidden_layers in [(32,), (64, 32), (128, 64, 32)]:
            for learning_rate in [0.001, 0.0005]:
                for alpha in [0.0001, 0.001]:
                    configs.append((scaler, hidden_layers, learning_rate, alpha))

    rows = []
    trained_models: list[Pipeline] = []
    best_model: Pipeline | None = None
    best_row: dict[str, float | str | int] | None = None
    print(f"split={split_name}")
    print(f"rows={len(df)}, train={len(x_train)}, val={len(x_val)}, test_excluded={int((df['split'] == 'test').sum()) if 'split' in df.columns else len(x_test)}")
    print(f"features={len(x.columns)}; resistance is excluded from X")

    for index, (scaler, hidden_layers, learning_rate, alpha) in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] scaler={scaler}, hidden={hidden_layers}, lr={learning_rate}, alpha={alpha}")
        model = Pipeline([
            ("scaler", make_scaler(scaler)),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                solver="adam",
                alpha=alpha,
                learning_rate_init=learning_rate,
                max_iter=800,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=40,
                random_state=42,
            )),
        ])
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        row: dict[str, float | str | int] = {
            "scaler": scaler,
            "hidden_layers": str(hidden_layers),
            "learning_rate": learning_rate,
            "alpha": alpha,
            "epochs": int(model.named_steps["mlp"].n_iter_),
        }
        row.update(evaluate_targets(y_val.to_numpy(), val_pred, "val"))
        rows.append(row)
        trained_models.append(model)
        if best_row is None or float(row["val_mean_mae"]) < float(best_row["val_mean_mae"]):
            best_row = row
            best_model = model

    assert best_model is not None and best_row is not None
    results = pd.DataFrame(rows)
    ranked = results.sort_values("val_mean_mae").reset_index()
    selected = {
        "best": int(ranked.iloc[0]["index"]),
        "middle": int(ranked.iloc[len(ranked) // 2]["index"]),
        "worst": int(ranked.iloc[-1]["index"]),
    }
    results["selection"] = ""
    for name, model_index in selected.items():
        results.loc[model_index, "selection"] = name
    results = results.sort_values("val_mean_mae")
    results_csv = write_csv_with_fallback(results, RESULTS_CSV)
    plot_model_selection(results)
    write_model_selection_report(results, best_row, len(x.columns))

    plot_loss_curve(best_model, PLOT_DIR / "loss_curve.png")
    val_pred = best_model.predict(x_val)
    validation_metrics = evaluate_targets(y_val.to_numpy(), val_pred, "validation")
    plot_pred_vs_true(y_val.to_numpy(), val_pred)

    selected_paths = {
        "best": MODEL_BEST_PATH,
        "middle": MODEL_MIDDLE_PATH,
        "worst": MODEL_WORST_PATH,
    }
    for selection_name, model_index in selected.items():
        bundle = {
            "model": trained_models[model_index],
            "features": list(x.columns),
            "targets": TARGET_COLUMNS,
            "selection": selection_name,
            "config": rows[model_index],
        }
        joblib.dump(bundle, selected_paths[selection_name])
    write_selected_models_report(results, selected, selected_paths, len(x.columns))
    joblib.dump({"model": best_model, "features": list(x.columns), "targets": TARGET_COLUMNS, "best_config": best_row}, MODEL_PATH)

    lines = [
        "Deep Learning MLP Report",
        f"split={split_name}",
        f"rows={len(df)}, train={len(x_train)}, validation={len(x_val)}, test_excluded={int((df['split'] == 'test').sum()) if 'split' in df.columns else len(x_test)}",
        "input_exclusion=resistance",
        "scaling_experiment=none, standard, minmax",
        f"best_config={best_row}",
        "",
        "Validation Metrics",
    ]
    lines.extend(f"{key}={value:.4f}" for key, value in validation_metrics.items())
    lines.extend([
        "",
        "Scaling Explanation",
        "MLP는 gradient 기반 모델이므로 feature scale 차이에 민감하다.",
        "duration_ms, energy_proxy, current_avg, speed_peak 등 단위가 다르기 때문에 StandardScaler/MinMaxScaler를 비교했다.",
        "Scaler는 train split에만 fit되고 validation/test에는 transform만 적용되므로 data leakage를 방지한다.",
    ])
    report_path = write_text_with_fallback(REPORT_PATH, "\n".join(lines) + "\n")

    print(f"saved: {MODEL_PATH}")
    print(f"saved: {MODEL_BEST_PATH}")
    print(f"saved: {MODEL_MIDDLE_PATH}")
    print(f"saved: {MODEL_WORST_PATH}")
    print(f"saved: {MODEL_SELECTED_CSV}")
    print(f"saved: {MODEL_SELECTED_REPORT}")
    print(f"saved: {report_path}")
    print(f"saved: {results_csv}")
    print(f"saved: {SELECTION_REPORT_PATH}")
    print(f"saved: {SELECTION_TOP_CSV}")
    print(f"saved plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
