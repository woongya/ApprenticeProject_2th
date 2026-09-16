from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util

import joblib
import pandas as pd

BUILD_DATASET_PATH = Path(__file__).resolve().parent / "1_build_dataset.py"
BUILD_DATASET_SPEC = importlib.util.spec_from_file_location("build_dataset_step1", BUILD_DATASET_PATH)
if BUILD_DATASET_SPEC is None or BUILD_DATASET_SPEC.loader is None:
    raise RuntimeError(f"failed to load {BUILD_DATASET_PATH}")
build_dataset = importlib.util.module_from_spec(BUILD_DATASET_SPEC)
BUILD_DATASET_SPEC.loader.exec_module(build_dataset)
extract_features = build_dataset.extract_features


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "2_ml_model_i_l_ki.joblib"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="MOVELOG csv path")
    args = parser.parse_args()

    bundle = joblib.load(MODEL_PATH)
    features, issues = extract_features(Path(args.file))
    x = pd.DataFrame([{name: features.get(name, 0.0) for name in bundle["features"]}]).fillna(0.0)
    pred = bundle["model"].predict(x)[0]

    print(f"file: {args.file}")
    print(f"quality_issues: {','.join(issues) if issues else 'none'}")
    for target, value in zip(bundle["targets"], pred):
        print(f"pred_{target}: {value:.2f}")


if __name__ == "__main__":
    main()
