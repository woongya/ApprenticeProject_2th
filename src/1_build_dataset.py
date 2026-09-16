from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_train"
DATA_DIRS = {
    "train": ROOT / "data_train",
    "validation": ROOT / "data_validation",
    "test": ROOT / "data_test",
}
OUT_DIR = ROOT / "outputs"
FEATURE_CSV = OUT_DIR / "1_dataset_features.csv"
QUALITY_REPORT = OUT_DIR / "1_dataset_quality_report.txt"
RAW_SUMMARY_CSV = OUT_DIR / "1_movelog_raw_summary.csv"


FILENAME_RE = re.compile(
    r"RUN(?P<run_id>\d+)_REP(?P<repeat>\d+)_T(?P<target>-?\d+)_R(?P<resistance>\d+)_"
    r"I(?P<inertia>\d+)_L(?P<load>\d+)_KI(?P<ki>\d+)_A(?P<accel>\d+)_D(?P<decel>\d+)_"
    r"S(?P<max_speed>\d+)_PWM(?P<max_pwm>\d+)_CL(?P<current_limit>\d+)",
    re.IGNORECASE,
)


def parse_metadata(path: Path) -> dict[str, int | float | str]:
    meta: dict[str, int | float | str] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if "=" not in body:
                continue
            key, value = body.split("=", 1)
            meta[key.strip().lower()] = coerce_value(value.strip())

    match = FILENAME_RE.search(path.name)
    if match:
        for key, value in match.groupdict().items():
            meta.setdefault(key, int(value))
    return meta


def coerce_value(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_movelog(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", on_bad_lines="skip")


def first_time_reaches(series: pd.Series, threshold: float, times: pd.Series) -> float:
    reached = np.flatnonzero(series.to_numpy() >= threshold)
    if len(reached) == 0:
        return np.nan
    return float(times.iloc[int(reached[0])])


def extract_features(path: Path, split: str = "") -> tuple[dict[str, int | float | str], list[str]]:
    meta = parse_metadata(path)
    df = read_movelog(path)
    issues: list[str] = []

    for column in ["time_ms", "pos_delta", "control_speed", "pwm_out", "current", "flags"]:
        if column not in df.columns:
            raise ValueError(f"missing column: {column}")

    flags = df["flags"].astype(str).str.replace("0x", "", case=False, regex=False).apply(lambda x: int(x, 16))
    speed = df["control_speed"].astype(float)
    current = df["current"].astype(float)
    pwm = df["pwm_out"].astype(float)
    pos_delta = df["pos_delta"].astype(float)
    time_ms = df["time_ms"].astype(float)

    duration_ms = float(time_ms.max() - time_ms.min()) if len(df) else 0.0
    pos_total = float(pos_delta.sum())
    abs_pos_total = float(pos_delta.abs().sum())
    speed_peak = float(speed.max()) if len(df) else 0.0
    current_peak = float(current.max()) if len(df) else 0.0
    pwm_peak = float(pwm.max()) if len(df) else 0.0
    moving = speed > max(1.0, speed_peak * 0.1)

    features: dict[str, int | float | str] = {
        "file": path.name,
        "split": split,
        "sample_count": int(len(df)),
        "duration_ms": duration_ms,
        "pos_total": pos_total,
        "abs_pos_total": abs_pos_total,
        "speed_peak": speed_peak,
        "speed_avg": float(speed.mean()) if len(df) else 0.0,
        "speed_std": float(speed.std(ddof=0)) if len(df) else 0.0,
        "speed_p50": float(speed.quantile(0.50)) if len(df) else 0.0,
        "speed_p90": float(speed.quantile(0.90)) if len(df) else 0.0,
        "speed_rise_time_ms": first_time_reaches(speed, speed_peak * 0.9, time_ms) if speed_peak > 0 else np.nan,
        "current_peak": current_peak,
        "current_avg": float(current.mean()) if len(df) else 0.0,
        "current_std": float(current.std(ddof=0)) if len(df) else 0.0,
        "current_ripple": float(current.max() - current.min()) if len(df) else 0.0,
        "current_p90": float(current.quantile(0.90)) if len(df) else 0.0,
        "pwm_peak": pwm_peak,
        "pwm_avg": float(pwm.mean()) if len(df) else 0.0,
        "pwm_std": float(pwm.std(ddof=0)) if len(df) else 0.0,
        "speed_per_pwm_gain": speed_peak / pwm_peak if pwm_peak > 0 else 0.0,
        "energy_proxy": float((current * pwm).sum()),
        "moving_ratio": float(moving.mean()) if len(df) else 0.0,
        "fault_ratio": float(((flags & 0x10) != 0).mean()) if len(df) else 0.0,
        "stopping_ratio": float(((flags & 0x08) != 0).mean()) if len(df) else 0.0,
    }

    for key in ["run_id", "repeat", "target", "resistance", "accel", "decel", "max_speed", "max_pwm", "current_limit", "log", "inertia", "load", "ki"]:
        features[key] = meta.get(key, np.nan)

    if len(df) < 100:
        issues.append("sample_count_lt_100")
    if current_peak <= 0:
        issues.append("current_peak_zero")
    if speed_peak <= 0:
        issues.append("speed_peak_zero")
    if abs_pos_total <= 0:
        issues.append("position_delta_zero")
    if features["inertia"] is np.nan or features["load"] is np.nan or features["ki"] is np.nan:
        issues.append("missing_label")

    features["quality_issues"] = ";".join(issues)
    return features, issues


def extract_raw_summary(path: Path, split: str = "") -> dict[str, int | float | str]:
    df = read_movelog(path)
    row: dict[str, int | float | str] = {"file": path.name, "split": split, "row_count": int(len(df))}

    numeric_columns = ["pos_delta", "control_speed", "pwm_out", "current"]
    for column in numeric_columns:
        if column not in df.columns:
            row[f"{column}_min"] = ""
            row[f"{column}_max"] = ""
            row[f"{column}_mean"] = ""
            row[f"{column}_std"] = ""
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        row[f"{column}_min"] = float(values.min()) if values.notna().any() else ""
        row[f"{column}_max"] = float(values.max()) if values.notna().any() else ""
        row[f"{column}_mean"] = float(values.mean()) if values.notna().any() else ""
        row[f"{column}_std"] = float(values.std(ddof=0)) if values.notna().any() else ""
    return row


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_summary_rows = []
    report_lines = []
    for split, data_dir in DATA_DIRS.items():
        if not data_dir.exists():
            continue
        for path in sorted(data_dir.glob("MOVELOG_*.csv")):
            try:
                features, issues = extract_features(path, split)
                rows.append(features)
                raw_summary_rows.append(extract_raw_summary(path, split))
                report_lines.append(f"OK {split} {path.name} issues={','.join(issues) if issues else 'none'}")
            except Exception as exc:
                report_lines.append(f"FAIL {split} {path.name} error={exc}")
    if not rows and (ROOT / "datas").exists():
        for path in sorted((ROOT / "datas").glob("MOVELOG_*.csv")):
            try:
                features, issues = extract_features(path, "train")
                rows.append(features)
                raw_summary_rows.append(extract_raw_summary(path, "train"))
                report_lines.append(f"OK train {path.name} issues={','.join(issues) if issues else 'none'}")
            except Exception as exc:
                report_lines.append(f"FAIL train {path.name} error={exc}")

    dataset = pd.DataFrame(rows)
    feature_csv = write_csv_with_fallback(dataset, FEATURE_CSV)
    raw_summary = pd.DataFrame(raw_summary_rows)
    raw_summary_csv = write_csv_with_fallback(raw_summary, RAW_SUMMARY_CSV)
    quality_report = write_text_with_fallback(QUALITY_REPORT, "\n".join(report_lines) + "\n")
    print(f"saved: {feature_csv}")
    print(f"saved: {raw_summary_csv}")
    print(f"saved: {quality_report}")
    print(f"rows: {len(dataset)}")


if __name__ == "__main__":
    main()
