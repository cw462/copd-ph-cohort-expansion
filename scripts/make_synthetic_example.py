#!/usr/bin/env python3
"""Write invented random tables for software testing; never read patient data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="New or empty directory outside the repository")
    parser.add_argument("--seed", type=int, default=731, help="Random synthetic fixture seed")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).expanduser().resolve()
    if output.is_relative_to(root):
        parser.error("Synthetic examples and outputs must stay outside the public repository")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory must be absent or empty")
    rng = np.random.default_rng(args.seed)
    # Arbitrary fixture dimensions, deliberately unrelated to any clinical cohort.
    ids = np.array([f"synthetic_{index:04d}" for index in range(1, 161)])
    labels = np.concatenate([rng.permutation(np.tile([0, 1], 16)) for _ in range(5)])
    main_n = 96
    protocols = np.repeat(["CT", "CTA", "CTPA"], 32)
    days = rng.integers(0, 25, main_n).astype(float)
    # Exercise temporal filtering while retaining both classes in all three protocols.
    excluded_window = np.flatnonzero((protocols == "CTA") & (labels[:main_n] == 0))[7:]
    days[excluded_window] = rng.integers(45, 150, len(excluded_window))

    def invented_measurements(count: int) -> dict:
        return {
            "lung_volume_ml": rng.uniform(2500, 7000, count),
            "LAA950_pct": rng.uniform(0, 40, count),
            "lung_mean_HU": rng.uniform(-920, -730, count),
            "vessel_volume_ml": rng.uniform(50, 250, count),
            "artery_BV5_ml": rng.uniform(10, 90, count),
        }

    features = pd.DataFrame({"participant_id": ids[:main_n], "label": labels[:main_n],
        "age": rng.integers(35, 86, main_n).astype(float),
        "sex_code": rng.choice(["category_a", "category_b"], main_n),
        "protocol": protocols, "gold_grade": rng.integers(1, 5, main_n).astype(float),
        "abs_ct_rhc_days": days, **invented_measurements(main_n)})
    features.loc[[3, 44, 71], "age"] = np.nan
    features.loc[[7, 48, 73], "artery_BV5_ml"] = np.nan
    features.loc[[2, 35, 66], "gold_grade"] = [-1, 0, np.nan]
    features.loc[80, "abs_ct_rhc_days"] = np.nan
    metadata = pd.DataFrame({"participant_id": ids, "label": labels,
        "source_group": rng.choice(["source_a", "source_b", "source_c"], len(ids)),
        "quant_source": rng.choice(["measurement_a", "measurement_b"], len(ids)),
        "shared_modality": rng.choice(["CT", "CTA", "CTPA"], len(ids)),
        "has_ct_quant": (np.arange(len(ids)) < 144).astype(int)})
    # A second measurement table may overlap the main table, with consistent labels.
    export_slice = slice(64, 144)
    ctpa = pd.DataFrame({"participant_id": ids[export_slice], "label": labels[export_slice],
        **invented_measurements(80)})
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [("features.csv", features), ("metadata.csv", metadata), ("ctpa_export.csv", ctpa)]:
        table.to_csv(output / name, index=False)
    note = {"synthetic": True, "clinical_data_read": False, "seed": args.seed,
        "purpose": "Software execution and leakage-guard testing only; labels and measurements are invented",
        "expected_cohorts_with_all_three_inputs": 11, "expected_model_combinations": 24,
        "not_evidence": "Synthetic metrics have no clinical interpretation"}
    (output / "SYNTHETIC_ONLY.json").write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"synthetic_example_created": True, "tables": 3, "clinical_data_read": False}))


if __name__ == "__main__":
    main()
