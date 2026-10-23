#!/usr/bin/env python3
"""Independently recompute OOF metrics and verify private run artifacts.

This checks numerical/artifact consistency, not clinical validity or identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


METRICS = ["roc_auc", "average_precision", "brier", "sensitivity", "specificity", "balanced_accuracy"]


def independent_metrics(y, probabilities):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    positives, negatives = p[y == 1], p[y == 0]
    # Pairwise concordance, including half credit for ties, without sklearn metrics.
    auc = float(np.mean([np.mean(value > negatives) + 0.5 * np.mean(value == negatives) for value in positives]))
    order = np.argsort(-p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_p) != 0), len(y) - 1]
    cumulative_positive = np.cumsum(sorted_y)[endpoints]
    recall = cumulative_positive / len(positives)
    precision = cumulative_positive / (endpoints + 1)
    ap = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    binary = p >= 0.5
    tn = int(((y == 0) & (~binary)).sum())
    fp = int(((y == 0) & binary).sum())
    fn = int(((y == 1) & (~binary)).sum())
    tp = int(((y == 1) & binary).sum())
    se, sp = tp / (tp + fn), tn / (tn + fp)
    return {"roc_auc": auc, "average_precision": ap, "brier": float(np.mean((p - y) ** 2)),
        "sensitivity": se, "specificity": sp, "balanced_accuracy": (se + sp) / 2,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate(run: Path) -> dict:
    errors = []

    def check(condition, message):
        if not bool(condition):
            errors.append(message)

    manifest = json.loads((run / "frozen_manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((run / "completion.json").read_text(encoding="utf-8"))
    check(completion.get("completed") is True, "Run is not marked complete")
    check(completion.get("clinical_validation_completed") is False, "Unexpected clinical validation claim")
    for kind, expected in manifest["input_sha256"].items():
        check(sha(run / "private_inputs" / f"{kind}.csv") == expected, f"Frozen {kind} input hash mismatch")
    check(sha(run / "run_experiments.py") == manifest["runner_sha256"], "Frozen runner hash mismatch")
    read = lambda name: pd.read_csv(run / name, dtype={"participant_id": str}, keep_default_na=False)
    predictions = read("OOF_predictions.csv")
    folds = read("fold_membership.csv")
    membership = read("cohort_membership.csv")
    fits = read("fit_audit.csv")
    reported = read("metrics_by_repeat.csv")
    summary = read("metrics_summary.csv")
    seeds = manifest["CV"]["seeds"]
    expected_groups = {(cohort, model, seed) for cohort, entry in manifest["cohorts"].items()
        for model in entry["models"] for seed in seeds}
    actual_groups = set(predictions.groupby(["cohort", "model", "seed"]).groups)
    check(actual_groups == expected_groups, "Missing or unexpected OOF model/seed groups")
    check(set(membership.cohort) == set(manifest["cohorts"]), "Cohort membership set differs from manifest")
    check(not membership.duplicated(["cohort", "participant_id"]).any(), "Duplicate participant within a cohort")
    check(not folds.duplicated(["cohort", "seed", "participant_id"]).any(), "Duplicate outer-test membership")
    check(not fits.duplicated(["cohort", "model", "seed", "fold"]).any(), "Duplicate fit audit row")
    check(not reported.duplicated(["cohort", "model", "seed"]).any(), "Duplicate reported metric group")
    check(not summary.duplicated(["cohort", "model"]).any(), "Duplicate summary group")
    check(set(reported.groupby(["cohort", "model", "seed"]).groups) == expected_groups, "Reported metric group set differs")
    check(set(summary.groupby(["cohort", "model"]).groups) == {(c, m) for c, m, _ in expected_groups}, "Summary group set differs")
    check(predictions.label.isin([0, 1]).all(), "Nonbinary OOF labels")
    check(np.isfinite(predictions.probability_PH).all() and predictions.probability_PH.between(0, 1).all(), "Invalid OOF probability")
    expected_fold_keys, expected_fit_keys = set(), set()
    max_difference, recomputed = 0.0, []
    for cohort, entry in manifest["cohorts"].items():
        members = membership[membership.cohort == cohort].reset_index(drop=True)
        check(len(members) == entry["n"], f"{cohort}: membership count differs")
        check(int(members.label.sum()) == entry["PH"], f"{cohort}: PH count differs")
        check(int((members.label == 0).sum()) == entry["noPH"], f"{cohort}: noPH count differs")
        lookup = members.set_index("participant_id")["label"].to_dict()
        outer_count = 5 if min(entry["PH"], entry["noPH"]) >= 10 else 3
        for seed in seeds:
            split = StratifiedKFold(outer_count, shuffle=True, random_state=seed)
            for fold, (train, test) in enumerate(split.split(np.zeros(len(members)), members.label), 1):
                test_ids, train_ids = set(members.participant_id.iloc[test]), set(members.participant_id.iloc[train])
                check(test_ids.isdisjoint(train_ids), f"{cohort}: patient overlap")
                expected_fold_keys.add((cohort, seed, fold))
                fold_table = folds[(folds.cohort == cohort) & (folds.seed == seed) & (folds.fold == fold)]
                check(set(fold_table.participant_id) == test_ids and len(fold_table) == len(test_ids), f"{cohort}: outer split mismatch")
                check(all(lookup.get(row.participant_id) == row.label for row in fold_table.itertuples()), f"{cohort}: fold label mismatch")
                for model in entry["models"]:
                    expected_fit_keys.add((cohort, model, seed, fold))
                    audit = fits[(fits.cohort == cohort) & (fits.model == model) & (fits.seed == seed) & (fits.fold == fold)]
                    check(len(audit) == 1, f"{cohort}/{model}: missing fit record")
                    if len(audit) != 1:
                        continue
                    row = audit.iloc[0]
                    numbers, categories = manifest["feature_sets"][model]
                    check(row.fields == "|".join(numbers + categories), f"{cohort}/{model}: unexpected predictor fields")
                    check(int(row.patient_overlap) == 0, f"{cohort}/{model}: nonzero overlap")
                    check(int(row.train_n) == len(train) and int(row.test_n) == len(test), f"{cohort}/{model}: split count mismatch")
                    train_ph, test_ph = int(members.label.iloc[train].sum()), int(members.label.iloc[test].sum())
                    check(int(row.train_PH) == train_ph and int(row.test_PH) == test_ph and
                        int(row.train_noPH) == len(train) - train_ph and int(row.test_noPH) == len(test) - test_ph,
                        f"{cohort}/{model}: class counts differ")
                    if model.startswith("LR"):
                        expected_inner = 3 if min(train_ph, len(train) - train_ph) >= 6 else 2
                        check(int(float(row.inner_splits)) == expected_inner and float(row.best_C) in [0.1, 1.0, 10.0],
                            f"{cohort}/{model}: LR inner-CV audit mismatch")
                    predicted = predictions[(predictions.cohort == cohort) & (predictions.model == model) &
                        (predictions.seed == seed) & (predictions.fold == fold)]
                    check(set(predicted.participant_id) == test_ids and len(predicted) == len(test_ids), f"{cohort}/{model}: OOF fold mismatch")
            for model in entry["models"]:
                predicted = predictions[(predictions.cohort == cohort) & (predictions.model == model) & (predictions.seed == seed)]
                check(predicted.participant_id.is_unique and set(predicted.participant_id) == set(lookup), f"{cohort}/{model}: OOF membership differs")
                check(all(lookup.get(row.participant_id) == row.label for row in predicted.itertuples()), f"{cohort}/{model}: OOF label mismatch")
                measured = independent_metrics(predicted.label, predicted.probability_PH)
                target = reported[(reported.cohort == cohort) & (reported.model == model) & (reported.seed == seed)]
                check(len(target) == 1, f"{cohort}/{model}: missing metric row")
                if len(target) == 1:
                    for metric, value in measured.items():
                        difference = abs(float(target.iloc[0][metric]) - value)
                        max_difference = max(max_difference, difference)
                        check(difference <= 1e-10, f"{cohort}/{model}: {metric} mismatch")
                    check(int(target.iloc[0].n) == entry["n"] and int(target.iloc[0].PH) == entry["PH"] and int(target.iloc[0].noPH) == entry["noPH"],
                        f"{cohort}/{model}: metric count mismatch")
                recomputed.append({"cohort": cohort, "model": model, "seed": seed, **measured})
    check(set(folds.groupby(["cohort", "seed", "fold"]).groups) == expected_fold_keys, "Unexpected fold records")
    check(set(fits.groupby(["cohort", "model", "seed", "fold"]).groups) == expected_fit_keys, "Unexpected fit records")
    recomputed = pd.DataFrame(recomputed)
    for (cohort, model), repeated in recomputed.groupby(["cohort", "model"]):
        selected = summary[(summary.cohort == cohort) & (summary.model == model)]
        if len(selected) != 1:
            continue
        target = selected.iloc[0]
        for metric in METRICS:
            for suffix, value in [("mean", repeated[metric].mean()), ("min", repeated[metric].min()), ("max", repeated[metric].max())]:
                difference = abs(float(target[metric + "_" + suffix]) - float(value))
                max_difference = max(max_difference, difference)
                check(difference <= 1e-10, f"{cohort}/{model}: {metric}_{suffix} mismatch")
    check(completion["OOF_rows"] == len(predictions), "Completion OOF count mismatch")
    check(completion["outer_fits"] == len(fits), "Completion fit count mismatch")
    check(completion["cohort_count"] == len(manifest["cohorts"]), "Completion cohort count mismatch")
    check(completion["model_combinations"] == len(summary), "Completion model count mismatch")
    return {"passed": not errors, "errors": errors, "cohort_count": len(manifest["cohorts"]),
        "model_combinations": len(summary), "model_seed_groups": len(recomputed),
        "outer_fits": len(fits), "OOF_rows": len(predictions), "max_absolute_metric_difference": max_difference,
        "scope": "Artifact membership, split isolation and independent OOF/summary arithmetic; not clinical validity or source-identity adjudication"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Private completed run directory outside this repository")
    args = parser.parse_args()
    run = Path(args.run_dir).expanduser().resolve()
    if run.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("Run directory must be outside the repository")
    try:
        result = validate(run)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result = {"passed": False, "errors": [f"Validation could not complete: {type(error).__name__}: {error}"]}
    if run.is_dir():
        (run / "independent_validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
