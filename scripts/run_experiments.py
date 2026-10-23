#!/usr/bin/env python3
"""Run exploratory nested-CV experiments on privately prepared participant tables.

This module performs no patient linkage, clinical diagnosis, label adjudication,
image processing, or source-specific exclusion. See docs/INPUT_SCHEMA.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
import xgboost
from xgboost import XGBClassifier


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [42, 2020, 2026]
CT_FEATURES = ["lung_volume_ml", "LAA950_pct", "lung_mean_HU", "vessel_volume_ml", "artery_BV5_ml"]
FEATURE_SETS = {
    "LR_clinical": (["age"], ["sex_code"]),
    "LR_CT5": (CT_FEATURES, []),
    "LR_clinical_CT5": (["age"] + CT_FEATURES, ["sex_code"]),
    "RF_clinical_CT5": (["age"] + CT_FEATURES, ["sex_code"]),
    "XGB_clinical_CT5": (["age"] + CT_FEATURES, ["sex_code"]),
    "LR_protocol_only": ([], ["protocol"]),
    "LR_protocol_clinical_CT5": (["age"] + CT_FEATURES, ["sex_code", "protocol"]),
    "LR_source_metadata_only": ([], ["source_group", "quant_source", "shared_modality"]),
}
BASE_COLUMNS = {"participant_id", "label"}
FEATURE_COLUMNS = BASE_COLUMNS | set(CT_FEATURES) | {
    "age", "sex_code", "protocol", "gold_grade", "abs_ct_rhc_days"
}
METADATA_COLUMNS = BASE_COLUMNS | {"source_group", "quant_source", "shared_modality", "has_ct_quant"}
EXPORT_COLUMNS = BASE_COLUMNS | set(CT_FEATURES)
METRIC_NAMES = ["roc_auc", "average_precision", "brier", "sensitivity", "specificity", "balanced_accuracy"]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def external_path(value: str, purpose: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_relative_to(REPOSITORY):
        raise ValueError(f"{purpose} must be outside this repository; row-level artifacts are private")
    return path


def numeric(frame: pd.DataFrame, name: str, required: bool = False) -> None:
    raw = frame[name]
    blank = raw.eq("")
    values = pd.to_numeric(raw.mask(blank), errors="coerce")
    if ((~blank) & (~np.isfinite(values))).any() or (required and blank.any()):
        raise ValueError(f"Invalid or missing numeric values in {name}; use empty cells for allowed missing values")
    frame[name] = values.astype(float)


def read_table(path: Path, required_columns: set[str], kind: str) -> pd.DataFrame:
    # Read strings first: do not lose leading zeroes or interpret a category as NA.
    import csv
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError(f"{kind}: duplicate column names")
    if set(header) != required_columns:
        missing = sorted(required_columns - set(header))
        extra = sorted(set(header) - required_columns)
        raise ValueError(f"{kind}: schema mismatch (missing={missing}, extra={extra}); select only documented columns")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for column in frame:
        frame[column] = frame[column].str.strip()
    if frame.empty or frame.participant_id.eq("").any() or not frame.participant_id.is_unique:
        raise ValueError(f"{kind}: expected one nonempty, unique participant_id per person; resolve repeated patients privately")
    numeric(frame, "label", required=True)
    if not frame.label.isin([0, 1]).all():
        raise ValueError(f"{kind}: label must be 0 or 1; no automatic clinical relabeling is performed")
    frame["label"] = frame.label.astype(int)
    return frame


def check_ct(frame: pd.DataFrame, kind: str) -> None:
    for column in CT_FEATURES:
        numeric(frame, column)
    if frame[CT_FEATURES].isna().all(axis=1).any():
        raise ValueError(f"{kind}: a whole missing CT block cannot be imputed into a measured participant")
    for column in ["lung_volume_ml", "vessel_volume_ml", "artery_BV5_ml"]:
        if not (frame[column].dropna() > 0).all():
            raise ValueError(f"{kind}: {column} must be positive when present")
    if not frame.LAA950_pct.dropna().between(0, 100).all():
        raise ValueError(f"{kind}: LAA950_pct must be in [0, 100]")


def load_inputs(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    features = read_table(paths["features"], FEATURE_COLUMNS, "features")
    check_ct(features, "features")
    for column in ["age", "gold_grade", "abs_ct_rhc_days"]:
        numeric(features, column)
    if not features.gold_grade.dropna().isin([-1, 0, 1, 2, 3, 4]).all():
        raise ValueError("gold_grade must be blank or one of -1, 0, 1, 2, 3, 4")
    if (features.abs_ct_rhc_days.dropna() < 0).any():
        raise ValueError("abs_ct_rhc_days must be nonnegative or blank")
    features["protocol"] = features.protocol.replace("", "unknown").map(
        lambda value: "unknown" if value.lower() == "unknown" else value.upper())
    if not features.protocol.isin(["CT", "CTA", "CTPA", "unknown"]).all():
        raise ValueError("protocol must be CT, CTA, CTPA, unknown, or blank")
    features["sex_code"] = features.sex_code.replace("", "missing")
    tables = {"features": features}
    if "metadata" in paths:
        frame = read_table(paths["metadata"], METADATA_COLUMNS, "metadata")
        numeric(frame, "has_ct_quant", required=True)
        if not frame.has_ct_quant.isin([0, 1]).all():
            raise ValueError("has_ct_quant must be 0 or 1")
        frame["has_ct_quant"] = frame.has_ct_quant.astype(bool)
        for column in ["source_group", "quant_source", "shared_modality"]:
            if frame[column].eq("").any():
                raise ValueError(f"metadata: {column} needs an explicit category, e.g. none or unknown")
        frame["protocol"] = "unknown"  # Not used by this diagnostic's predictors.
        tables["metadata"] = frame
    if "ctpa_export" in paths:
        frame = read_table(paths["ctpa_export"], EXPORT_COLUMNS, "ctpa_export")
        check_ct(frame, "ctpa_export")
        frame["protocol"] = "CTPA"
        tables["ctpa_export"] = frame
    labels: dict[str, int] = {}
    for kind, frame in tables.items():
        for participant_id, label in zip(frame.participant_id, frame.label):
            if participant_id in labels and labels[participant_id] != label:
                raise ValueError("Cross-table label conflict; adjudicate or exclude it before training")
            labels[participant_id] = label
    if "metadata" in tables:
        metadata = tables["metadata"].set_index("participant_id")
        for kind in ["features", "ctpa_export"]:
            if kind not in tables:
                continue
            ids = tables[kind].participant_id
            if not set(ids).issubset(metadata.index):
                raise ValueError(f"{kind}: every measured participant must exist in the supplied metadata cohort")
            if not metadata.loc[ids, "has_ct_quant"].all():
                raise ValueError(f"{kind}: metadata contradicts CT measurement availability")
    return tables


def make_cohorts(tables: dict[str, pd.DataFrame]) -> tuple[dict, list]:
    cohorts, skipped = {}, []

    def add(name: str, frame: pd.DataFrame, role: str, models: list[str], required: bool = False) -> None:
        frame = frame.reset_index(drop=True)
        counts = {str(label): int((frame.label == label).sum()) for label in [0, 1]}
        if min(counts.values()) < 3:
            if required:
                raise ValueError("The main cohort requires at least three participants in each class for nested CV")
            skipped.append({"cohort": name, "reason": "fewer than three participants in a class", "class_counts": counts})
            return
        cohorts[name] = (frame, role, models)

    frame = tables["features"]
    models = ["LR_clinical", "LR_CT5", "LR_clinical_CT5", "RF_clinical_CT5", "XGB_clinical_CT5", "LR_protocol_only", "LR_protocol_clinical_CT5"]
    add("main_mixed_protocol_exploratory", frame,
        "Exploratory pooled measurements; protocol confounding must be assessed", models, required=True)
    for protocol in ["CT", "CTA", "CTPA"]:
        selected = frame[frame.protocol == protocol]
        add(f"main_{protocol}_all_intervals", selected,
            "Exploratory within-protocol analysis; report minority class and temporal gaps",
            ["LR_clinical", "LR_clinical_CT5"])
        add(f"main_{protocol}_within30days", selected[selected.abs_ct_rhc_days <= 30],
            "Temporal sensitivity using a prespecified 30-day window, not a clinical guideline criterion",
            ["LR_clinical", "LR_clinical_CT5"])
    add("main_GOLD1to4_sensitivity", frame[frame.gold_grade.between(1, 4)],
        "Severity sensitivity only; absent, -1 and 0 are not severity grades",
        ["LR_clinical", "LR_clinical_CT5"])
    if "metadata" in tables:
        frame = tables["metadata"]
        add("parent_source_metadata_diagnostic", frame,
            "NONCLINICAL source-confounding diagnostic; not a diagnostic classifier",
            ["LR_source_metadata_only"])
        add("measured_source_metadata_diagnostic", frame[frame.has_ct_quant],
            "NONCLINICAL source-confounding diagnostic among participants with any CT quantification",
            ["LR_source_metadata_only"])
    else:
        skipped.append({"cohort": "source_metadata_diagnostics", "reason": "metadata input not supplied"})
    if "ctpa_export" in tables:
        add("homogeneous_CTPA_feasibility_only", tables["ctpa_export"],
            "Feasibility only; inspect negative-class support before interpreting performance",
            ["LR_CT5"])
    else:
        skipped.append({"cohort": "homogeneous_CTPA_feasibility_only", "reason": "CTPA export input not supplied"})
    return cohorts, skipped


def metrics(y, probabilities) -> dict:
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    tn, fp, fn, tp = confusion_matrix(y, probabilities >= 0.5, labels=[0, 1]).ravel()
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "brier": float(brier_score_loss(y, probabilities)),
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def make_pipeline(model: str, seed: int, fold: int, train_y: np.ndarray) -> Pipeline:
    numbers, categories = FEATURE_SETS[model]
    transformers = []
    if numbers:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numbers))
    if categories:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categories))
    preprocess = ColumnTransformer(transformers, remainder="drop")
    if model.startswith("RF"):
        estimator = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=seed + fold, n_jobs=2)
    elif model.startswith("XGB"):
        estimator = XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0,
            scale_pos_weight=float((train_y == 0).sum() / (train_y == 1).sum()),
            random_state=seed + fold, n_jobs=2, eval_metric="logloss")
    else:
        estimator = LogisticRegression(max_iter=4000, solver="lbfgs")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def fit_all(cohorts: dict, seeds: list[int], output: Path) -> dict:
    predictions, fit_rows, fold_rows = [], [], []
    with threadpool_limits(limits=2):
        for cohort, (frame, role, models) in cohorts.items():
            y = frame.label.to_numpy(dtype=int)
            outer_count = 5 if min(Counter(y).values()) >= 10 else 3
            for seed in seeds:
                splits = list(StratifiedKFold(outer_count, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
                for fold, (train, test) in enumerate(splits, 1):
                    if not set(frame.participant_id.iloc[train]).isdisjoint(frame.participant_id.iloc[test]):
                        raise RuntimeError("Patient overlap between outer train and test")
                    for index in test:
                        fold_rows.append({"cohort": cohort, "seed": seed, "fold": fold,
                            "participant_id": frame.participant_id.iloc[index], "label": int(y[index])})
                    for model in models:
                        numbers, categories = FEATURE_SETS[model]
                        fields = numbers + categories
                        X = frame[fields]
                        pipe = make_pipeline(model, seed, fold, y[train])
                        inner_count, best_c = None, None
                        if model.startswith("LR"):
                            inner_count = 3 if min(Counter(y[train]).values()) >= 6 else 2
                            inner = StratifiedKFold(inner_count, shuffle=True, random_state=seed + fold + 1000)
                            # The entire preprocessing pipeline is refit inside each inner split.
                            fit = GridSearchCV(pipe, {"model__C": [0.1, 1.0, 10.0]}, cv=inner,
                                scoring="roc_auc", n_jobs=1, refit=True, error_score="raise")
                            fit.fit(X.iloc[train], y[train])
                            best_c = float(fit.best_params_["model__C"])
                        else:
                            fit = pipe.fit(X.iloc[train], y[train])
                        probabilities = fit.predict_proba(X.iloc[test])[:, 1]
                        if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
                            raise RuntimeError("Invalid predicted probability")
                        fit_rows.append({"cohort": cohort, "model": model, "seed": seed, "fold": fold,
                            "train_n": len(train), "test_n": len(test),
                            "train_noPH": int((y[train] == 0).sum()), "train_PH": int(y[train].sum()),
                            "test_noPH": int((y[test] == 0).sum()), "test_PH": int(y[test].sum()),
                            "inner_splits": inner_count, "best_C": best_c,
                            "fields": "|".join(fields), "patient_overlap": 0})
                        for index, probability in zip(test, probabilities):
                            predictions.append({"cohort": cohort, "model": model, "seed": seed,
                                "fold": fold, "participant_id": frame.participant_id.iloc[index],
                                "label": int(y[index]), "probability_PH": float(probability),
                                "protocol": frame.protocol.iloc[index]})
                print(json.dumps({"cohort": cohort, "seed_complete": seed, "n": len(frame)}), flush=True)
    predicted = pd.DataFrame(predictions)
    repeat_rows = []
    for (cohort, model, seed), frame in predicted.groupby(["cohort", "model", "seed"], sort=False):
        expected = cohorts[cohort][0]
        if not frame.participant_id.is_unique or set(frame.participant_id) != set(expected.participant_id):
            raise RuntimeError("OOF membership is incomplete or duplicated")
        repeat_rows.append({"cohort": cohort, "model": model, "seed": int(seed), "n": len(frame),
            "PH": int(frame.label.sum()), "noPH": int((frame.label == 0).sum()),
            **metrics(frame.label, frame.probability_PH)})
    repeats = pd.DataFrame(repeat_rows)
    summaries = []
    for (cohort, model), frame in repeats.groupby(["cohort", "model"], sort=False):
        row = {"cohort": cohort, "model": model, "n": int(frame.n.iloc[0]),
            "PH": int(frame.PH.iloc[0]), "noPH": int(frame.noPH.iloc[0]), "role": cohorts[cohort][1]}
        for metric in METRIC_NAMES:
            row.update({metric + "_mean": float(frame[metric].mean()),
                metric + "_min": float(frame[metric].min()), metric + "_max": float(frame[metric].max())})
        summaries.append(row)
    within_protocol = []
    for (model, seed, protocol), frame in predicted[predicted.cohort == "main_mixed_protocol_exploratory"].groupby(["model", "seed", "protocol"]):
        if frame.label.nunique() == 2:
            within_protocol.append({"cohort": "main_mixed_protocol_exploratory", "model": model,
                "seed": int(seed), "protocol": protocol, "n": len(frame),
                **metrics(frame.label, frame.probability_PH)})
    artifacts = {
        "OOF_predictions.csv": predicted, "metrics_by_repeat.csv": repeats,
        "metrics_summary.csv": pd.DataFrame(summaries), "fold_membership.csv": pd.DataFrame(fold_rows),
        "fit_audit.csv": pd.DataFrame(fit_rows),
        "pooled_model_within_protocol_metrics.csv": pd.DataFrame(within_protocol),
    }
    for name, frame in artifacts.items():
        frame.to_csv(output / name, index=False, encoding="utf-8")
    return {"outer_fits": len(fit_rows), "OOF_rows": len(predicted),
        "cohort_count": len(cohorts), "model_combinations": len(summaries)}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True, help="Private, screened main participant feature CSV")
    parser.add_argument("--metadata", help="Optional eligible parent cohort metadata CSV (confounding diagnostics)")
    parser.add_argument("--ctpa-export", help="Optional privately linked homogeneous CTPA measurement CSV")
    parser.add_argument("--output-dir", required=True, help="New or empty directory OUTSIDE this repository")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Outer split seeds; default: 42 2020 2026. One seed is a smoke run, not the original repeated analysis")
    parser.add_argument("--prepare-only", action="store_true", help="Validate and freeze inputs/cohorts without fitting")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or any(seed < 0 or seed > 2**32 - 2000 for seed in args.seeds):
        raise ValueError("Seeds must be unique nonnegative integers smaller than 2**32 - 2000")
    paths = {kind: external_path(value, "Input") for kind, value in {
        "features": args.features, "metadata": args.metadata, "ctpa_export": args.ctpa_export}.items() if value}
    output = external_path(args.output_dir, "Output")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be absent or empty; existing results will not be overwritten")
    input_hashes = {kind: digest(path) for kind, path in paths.items()}
    tables = load_inputs(paths)
    cohorts, skipped = make_cohorts(tables)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output / "private_inputs"
    snapshots.mkdir()
    for kind, path in paths.items():
        destination = snapshots / f"{kind}.csv"
        shutil.copyfile(path, destination)
        if digest(destination) != input_hashes[kind] or digest(path) != input_hashes[kind]:
            raise RuntimeError("An input changed while it was being read/frozen")
    descriptions, membership = {}, []
    for name, (frame, role, models) in cohorts.items():
        has_time = "abs_ct_rhc_days" in frame
        descriptions[name] = {"n": len(frame), "PH": int(frame.label.sum()),
            "noPH": int((frame.label == 0).sum()), "models": models, "role": role,
            "CT_RHC_days_available": int(frame.abs_ct_rhc_days.notna().sum()) if has_time else None,
            "CT_RHC_over365days": int((frame.abs_ct_rhc_days > 365).sum()) if has_time else None}
        for participant_id, label, protocol in zip(frame.participant_id, frame.label, frame.protocol):
            membership.append({"cohort": name, "participant_id": participant_id, "label": int(label), "protocol": protocol})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "method_version": "1.0",
        "status": "Exploratory source-label methods; no independent clinical validation",
        "input_sha256": input_hashes, "input_rows": {kind: len(frame) for kind, frame in tables.items()},
        "cohorts": descriptions, "skipped_cohorts": skipped, "feature_sets": FEATURE_SETS,
        "CV": {"seeds": args.seeds, "original_default_seeds_used": args.seeds == DEFAULT_SEEDS,
            "outer": "5 folds if minority >= 10, otherwise 3",
            "inner_LR": "3 folds if outer-training minority >= 6, otherwise 2",
            "LR_C_grid": [0.1, 1.0, 10.0], "selection_metric": "ROC AUC", "threshold": 0.5,
            "preprocessing": "All imputation/scaling/encoding fitted within each training split",
            "patient_isolation": "Unique supplied participant IDs; repeated IDs are rejected. Private identity resolution is required",
            "RF": "200 trees, depth 3, leaf minimum 5, balanced_subsample",
            "XGB": "150 trees, depth 2, rate 0.03, subsample 0.8, colsample 0.8, lambda 10, outer-training class weight"},
        "uncertainty": "Seed min/max describe partition variability, not confidence intervals; no external test set",
        "label_rule": "Use privately audited source labels unchanged; no pressure-threshold relabeling",
        "predictor_exclusions": ["participant_id", "label", "gold_grade", "abs_ct_rhc_days", "RHC measures", "clinical label derivatives"],
        "source_protocol_rule": "Used only in explicitly named diagnostic/adjusted models",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "runner_sha256": digest(Path(__file__)),
    }
    write_json(output / "frozen_manifest.json", manifest)
    pd.DataFrame(membership).to_csv(output / "cohort_membership.csv", index=False)
    pd.DataFrame([dict(cohort=name, **{k: v for k, v in info.items() if k != "models"})
        for name, info in descriptions.items()]).to_csv(output / "cohort_counts.csv", index=False)
    shutil.copyfile(__file__, output / "run_experiments.py")
    if args.prepare_only:
        write_json(output / "preflight.json", {"preflight_passed": True, "training_completed": False,
            "cohort_count": len(cohorts), "model_combinations": sum(len(value[2]) for value in cohorts.values())})
        print(json.dumps({"preflight_passed": True, "cohort_count": len(cohorts)}))
        return 0
    counts = fit_all(cohorts, args.seeds, output)
    if any(digest(path) != input_hashes[kind] for kind, path in paths.items()):
        raise RuntimeError("An original input changed during training")
    completion = {"completed_utc": datetime.now(timezone.utc).isoformat(), "completed": True,
        "clinical_validation_completed": False, "original_inputs_unchanged": True, **counts,
        "OOF_rows_are_independent_patients": False,
        "independent_check": "Run validate_outputs.py before treating these artifacts as verified"}
    write_json(output / "completion.json", completion)
    print(json.dumps(completion), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
