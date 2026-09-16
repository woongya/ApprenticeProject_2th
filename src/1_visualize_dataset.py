from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BUILD_DATASET_PATH = Path(__file__).resolve().parent / "1_build_dataset.py"
BUILD_DATASET_SPEC = importlib.util.spec_from_file_location("build_dataset_step1", BUILD_DATASET_PATH)
if BUILD_DATASET_SPEC is None or BUILD_DATASET_SPEC.loader is None:
    raise RuntimeError(f"failed to load {BUILD_DATASET_PATH}")
build_dataset = importlib.util.module_from_spec(BUILD_DATASET_SPEC)
BUILD_DATASET_SPEC.loader.exec_module(build_dataset)

DATA_DIR = build_dataset.DATA_DIR
DATA_DIRS = build_dataset.DATA_DIRS
FEATURE_CSV = build_dataset.FEATURE_CSV
OUT_DIR = build_dataset.OUT_DIR
parse_metadata = build_dataset.parse_metadata
read_movelog = build_dataset.read_movelog


PLOT_DIR = OUT_DIR / "plots"
RAW_DIR = PLOT_DIR / "raw"
COMPARE_DIR = PLOT_DIR / "compare"
FEATURE_DIR = PLOT_DIR / "features"
REPORT_HTML = OUT_DIR / "1_visual_report.html"
LABEL_TARGETS = ["inertia", "load", "ki"]
LABEL_CORRELATION_FEATURES = [
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


@dataclass
class ReportImage:
    path: Path
    description: str


def ensure_dirs() -> None:
    for path in [RAW_DIR, COMPARE_DIR, FEATURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)


def describe_image(path: Path) -> str:
    name = path.name
    if path.parent == RAW_DIR:
        return "개별 MOVELOG 파일의 전체 시계열입니다. 속도, PWM, 전류, 위치 변화가 시간에 따라 정상적으로 증가하는지 확인합니다."
    if name.startswith("overlay_resistance_control_speed"):
        return "동일 조건에서 Resistance만 바꾼 속도 비교입니다. 부하가 증가할 때 최대 속도 도달 시간과 속도 상승 기울기가 어떻게 변하는지 확인합니다."
    if name.startswith("overlay_resistance_current"):
        return "동일 조건에서 Resistance만 바꾼 전류 비교입니다. 부하 증가에 따라 전류 평균/피크가 증가하는지 확인합니다."
    if name.startswith("overlay_accel_control_speed"):
        return "동일 조건에서 Accel만 바꾼 속도 비교입니다. 가속 파라미터가 실제 속도 상승률에 영향을 주는지 확인합니다."
    if name.startswith("overlay_max_speed_control_speed"):
        return "동일 조건에서 Max Speed만 바꾼 속도 비교입니다. 명령 최대속도에 실제 속도가 잘 따라가는지 확인합니다."
    if name.startswith("scatter_"):
        return "feature와 label/조건의 관계를 보는 산점도입니다. 점들이 색상 기준으로 분리되면 해당 feature가 모델 입력으로 유용할 가능성이 큽니다."
    if name.startswith("hist_"):
        return "feature 또는 label의 분포입니다. 특정 값에 너무 몰려 있는지, outlier가 있는지, 데이터 균형이 적절한지 확인합니다."
    if name == "correlation_heatmap.png":
        return "feature 간 상관관계입니다. 강한 상관 feature는 중복일 수 있고, label과 관련된 feature는 모델에 중요할 가능성이 큽니다."
    if name == "target_label_correlation_heatmap.png":
        return "Y축은 inertia, load, ki이고 X축은 주요 후보 feature입니다. label별로 어떤 feature가 강하게 관련되는지 직접 확인하기 위한 핵심 그래프입니다."
    return "수집 데이터의 특성을 확인하기 위한 그래프입니다."


def plot_raw_waveform(path: Path) -> Path:
    df = read_movelog(path)
    meta = parse_metadata(path)
    out = RAW_DIR / f"{path.stem}_waveform.png"
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"{path.name}\nT={meta.get('target', '')} R={meta.get('resistance', '')} "
        f"A={meta.get('accel', '')} S={meta.get('max_speed', '')} I={meta.get('inertia', '')} "
        f"L={meta.get('load', '')} KI={meta.get('ki', '')}",
        fontsize=10,
    )
    x = df["time_ms"] if "time_ms" in df.columns else df.index
    for ax, column, ylabel, color in [
        (axes[0], "control_speed", "Speed", "tab:blue"),
        (axes[1], "pwm_out", "PWM", "tab:orange"),
        (axes[2], "current", "Current", "tab:green"),
        (axes[3], "pos_delta", "Pos Delta", "tab:red"),
    ]:
        if column in df.columns:
            ax.plot(x, df[column], color=color, linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time_ms")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def metadata_table(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        meta = parse_metadata(path)
        meta["file"] = path.name
        meta["path"] = str(path)
        rows.append(meta)
    return pd.DataFrame(rows)


def select_representative_raw_files(files: list[Path]) -> list[Path]:
    meta_df = metadata_table(files)
    group_columns = ["target", "resistance", "accel", "max_speed", "max_pwm", "current_limit", "log"]
    if not set(group_columns + ["path"]).issubset(meta_df.columns):
        return files[:20]

    representatives: list[Path] = []
    sort_columns = [column for column in ["repeat", "run_id", *group_columns] if column in meta_df.columns]
    sorted_meta = meta_df.sort_values(sort_columns) if sort_columns else meta_df
    for _, group in sorted_meta.groupby(group_columns, dropna=False, sort=True):
        representatives.append(Path(group.iloc[0]["path"]))
    return representatives


def plot_overlay(meta_df: pd.DataFrame, group_columns: list[str], varying: str, value_column: str, limit: int = 12) -> list[Path]:
    outputs: list[Path] = []
    if not set(group_columns + [varying, "path"]).issubset(meta_df.columns):
        return outputs
    for key, group in meta_df.dropna(subset=group_columns + [varying]).groupby(group_columns, dropna=False):
        if group[varying].nunique() < 2:
            continue
        title_key = key if isinstance(key, tuple) else (key,)
        label = "_".join(f"{col}{val}" for col, val in zip(group_columns, title_key))
        out = COMPARE_DIR / safe_name(f"overlay_{varying}_{value_column}_{label}.png")
        fig, ax = plt.subplots(figsize=(12, 5))
        for _, row in group.sort_values(varying).iterrows():
            df = read_movelog(Path(row["path"]))
            if value_column not in df.columns:
                continue
            x = df["time_ms"] if "time_ms" in df.columns else df.index
            ax.plot(x, df[value_column], linewidth=1.0, label=f"{varying}={row[varying]}")
        ax.set_title(f"{value_column} overlay by {varying}\n{label}")
        ax.set_xlabel("time_ms")
        ax.set_ylabel(value_column)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        outputs.append(out)
        if len(outputs) >= limit:
            break
    return outputs


def plot_feature_scatter(df: pd.DataFrame, x: str, y: str, color: str) -> Path | None:
    if not {x, y, color}.issubset(df.columns):
        return None
    out = FEATURE_DIR / safe_name(f"scatter_{x}_{y}_color_{color}.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df[x], df[y], c=df[color], cmap="viridis", s=48, alpha=0.85, edgecolors="k", linewidths=0.2)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}, color={color}")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax).set_label(color)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_histogram(df: pd.DataFrame, column: str) -> Path | None:
    if column not in df.columns:
        return None
    out = FEATURE_DIR / safe_name(f"hist_{column}.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[column].dropna(), bins=min(30, max(5, len(df) // 3)), color="tab:blue", alpha=0.8)
    ax.set_title(f"Distribution: {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_correlation(df: pd.DataFrame) -> Path:
    corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True).fillna(0.0)
    out = FEATURE_DIR / "correlation_heatmap.png"
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_target_label_correlation(df: pd.DataFrame) -> Path:
    available_targets = [column for column in LABEL_TARGETS if column in df.columns]
    available_features = [column for column in LABEL_CORRELATION_FEATURES if column in df.columns]
    corr_rows = []
    for target in available_targets:
        row = []
        for feature in available_features:
            row.append(df[target].corr(df[feature]))
        corr_rows.append(row)

    corr = pd.DataFrame(corr_rows, index=available_targets, columns=available_features).fillna(0.0)
    out = FEATURE_DIR / "target_label_correlation_heatmap.png"
    fig, ax = plt.subplots(figsize=(12, 3.8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=11)
    ax.set_title("Target Label Correlation Heatmap")
    for y in range(len(corr.index)):
        for x in range(len(corr.columns)):
            value = corr.iloc[y, x]
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def write_report(images: list[ReportImage]) -> None:
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>MOVELOG Visual Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827;} img{max-width:100%;border:1px solid #cbd5e1;margin:8px 0 28px 0;} h1,h2{color:#0f172a;} .path{font-size:12px;color:#475569;} .desc{font-size:14px;color:#334155;margin:4px 0 6px 0;line-height:1.45;}</style>",
        "</head><body><h1>MOVELOG Visual Report</h1>",
        "<p>수집된 MOVELOG 시계열과 feature dataset을 시각화한 report입니다. 사람의 눈으로 데이터 품질, feature 후보, 모델 방향을 판단하기 위한 목적입니다.</p>",
    ]
    current_section = ""
    for image in images:
        section = image.path.parent.name
        if section != current_section:
            current_section = section
            lines.append(f"<h2>{section}</h2>")
        rel = image.path.relative_to(OUT_DIR).as_posix()
        lines.append(f"<div class='path'>{image.path.name}</div><div class='desc'>{image.description}</div><img src='{rel}'>")
    lines.append("</body></html>")
    REPORT_HTML.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    files = []
    for data_dir in DATA_DIRS.values():
        if data_dir.exists():
            files.extend(sorted(data_dir.glob("MOVELOG_*.csv")))
    if not files:
        raise SystemExit(f"No MOVELOG_*.csv files in {', '.join(str(path) for path in DATA_DIRS.values())}")
    if not FEATURE_CSV.exists():
        raise SystemExit("Run src/build_dataset.py first.")

    images: list[ReportImage] = []
    raw_files = select_representative_raw_files(files)
    print(f"[1/6] raw waveform plot 생성 중: parameter 조합별 대표 {len(raw_files)}개")
    for index, path in enumerate(raw_files, start=1):
        print(f"  raw {index}/{len(raw_files)}: {path.name}")
        image = plot_raw_waveform(path)
        images.append(ReportImage(image, describe_image(image)))

    print("[2/6] metadata table 생성 중")
    meta_df = metadata_table(files)
    print("[3/6] parameter overlay plot 생성 중")
    overlay_paths = []
    overlay_paths.extend(plot_overlay(meta_df, ["target", "accel", "max_speed", "repeat"], "resistance", "control_speed"))
    overlay_paths.extend(plot_overlay(meta_df, ["target", "accel", "max_speed", "repeat"], "resistance", "current"))
    overlay_paths.extend(plot_overlay(meta_df, ["target", "resistance", "max_speed", "repeat"], "accel", "control_speed"))
    overlay_paths.extend(plot_overlay(meta_df, ["target", "resistance", "accel", "repeat"], "max_speed", "control_speed"))
    for image in overlay_paths:
        print(f"  overlay: {image.name}")
        images.append(ReportImage(image, describe_image(image)))

    print("[4/6] feature dataset 로딩 및 scatter plot 생성 중")
    feature_df = pd.read_csv(FEATURE_CSV)
    for spec in [
        ("resistance", "current_avg", "load"),
        ("resistance", "speed_peak", "inertia"),
        ("speed_rise_time_ms", "current_peak", "load"),
        ("pwm_avg", "speed_peak", "ki"),
        ("energy_proxy", "load", "resistance"),
        ("speed_per_pwm_gain", "inertia", "resistance"),
    ]:
        image = plot_feature_scatter(feature_df, *spec)
        if image is not None:
            print(f"  scatter: {image.name}")
            images.append(ReportImage(image, describe_image(image)))

    print("[5/6] histogram 및 correlation heatmap 생성 중")
    for column in ["sample_count", "duration_ms", "speed_peak", "current_peak", "current_avg", "pwm_avg", "energy_proxy", "inertia", "load", "ki"]:
        image = plot_histogram(feature_df, column)
        if image is not None:
            print(f"  histogram: {image.name}")
            images.append(ReportImage(image, describe_image(image)))
    corr_image = plot_correlation(feature_df)
    print(f"  correlation: {corr_image.name}")
    images.append(ReportImage(corr_image, describe_image(corr_image)))
    label_corr_image = plot_target_label_correlation(feature_df)
    print(f"  target label correlation: {label_corr_image.name}")
    images.append(ReportImage(label_corr_image, describe_image(label_corr_image)))

    print("[6/6] HTML report 생성 중")
    write_report(images)
    print(f"saved plots: {PLOT_DIR}")
    print(f"saved report: {REPORT_HTML}")


if __name__ == "__main__":
    main()
