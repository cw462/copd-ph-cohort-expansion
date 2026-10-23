# Private input and experiment contract

The public runner implements the experiment methods. It does not contain the
patient data or reproduce private source-table linkage, diagnosis adjudication,
exclusion decisions, CT segmentation, or measurement extraction. Reproducing the
reported clinical-data results requires the same privately audited, screened
inputs, row order, measurement definitions and numerical stack. Synthetic runs
test software execution only; their metrics are not clinical results.

Use Python 3.11 or later and install `requirements.txt`. The pinned stack was
tested with Python 3.14.3. No GPU is required. All CSV files use UTF-8 (an optional
BOM is accepted), a comma separator and a header. Blank numeric cells mean
missing. Other placeholder strings and nonfinite numbers are rejected.

## Identity and clinical preparation

Before calling the runner, resolve identity using trusted private identifiers,
choose one measurement row per participant per input, and exclude or adjudicate
unresolved label conflicts and weak identity links. Do not use names or image
filename guesses to transfer clinical labels. Use the same private pseudonym
across all three tables. Each table must have unique nonempty `participant_id`
values; the runner rejects repeated participants instead of silently treating
multiple scans as independent patients. Cross-table label conflicts are rejected.
Different pseudonyms for the same person cannot be detected by this software.

`label` is the already reviewed source label, `1` for PH and `0` for noPH. The
runner does not derive labels from pressure thresholds, change source labels,
or confirm a clinical diagnosis. COPD attribution and any RHC/date/unit issues
must be resolved or retained as review flags in a separate private audit. GOLD
is a severity sensitivity field, not the sole eligibility criterion for COPD.
Unknown GOLD codes `-1` and `0` are retained in the main cohort but never treated
as valid grades.

The prepared feature table is the analogue of the previously screened main
measurement cohort; the metadata table contains the eligible parent cohort,
not the full unscreened audit population. Historical exclusions are deliberately
not hardcoded. The complete audit population remains a separate private artifact.

## Required main feature table: `--features`

Exactly these columns are required. Their order is arbitrary; **row order must
remain fixed** to reproduce the same stratified folds.

| Column | Meaning and rule | Predictive use |
| --- | --- | --- |
| `participant_id` | Private pseudonym, unique within this table | Never |
| `label` | Audited source PH label, exactly 0 or 1 | Outcome only |
| `age` | Age in years; blank allowed | Clinical models |
| `sex_code` | Consistent categorical coding; blank becomes `missing` | Clinical models |
| `protocol` | `CT`, `CTA`, `CTPA`, or `unknown`; blank becomes `unknown` | Stratification; explicitly named protocol models only |
| `gold_grade` | 1–4, or -1/0/blank for unknown severity | Sensitivity selection only |
| `abs_ct_rhc_days` | Audited absolute CT–RHC interval in days, nonnegative; blank if unknown | Temporal sensitivity selection/reporting only |
| `lung_volume_ml` | Measured whole-lung volume in ml; positive when present | CT models |
| `LAA950_pct` | Measured whole-lung low-attenuation percentage at -950 HU, 0–100 | CT models |
| `lung_mean_HU` | Measured whole-lung mean attenuation in HU | CT models |
| `vessel_volume_ml` | Measured pulmonary vessel volume in ml; positive when present | CT models |
| `artery_BV5_ml` | Measured arterial BV5 volume in ml under the original extraction definition; positive when present | CT models |

The five measurements must be harmonized **before** input preparation. The runner
does not establish equivalence of alternative vessel compartments, contrast
protocols, image series, segmentation versions, or vendor field names. Do not
substitute lobar vascular volume for whole-lobe tissue volume. Partial missing
measurements are imputed within training folds; a participant with all five CT
measurements missing is rejected. An entirely absent field across a source is
not made scientifically comparable by imputation.

Derive `abs_ct_rhc_days` from audited original dates, not an unreviewed merged
date field. Preserve original date strings, normalization and unresolved issues
in the private audit. Missing intervals are excluded only from the 30-day
sensitivity analysis. The runner needs no actual calendar dates.

## Optional parent metadata table: `--metadata`

Exactly these columns are required:

| Column | Meaning |
| --- | --- |
| `participant_id`, `label` | Same definitions as above |
| `source_group` | Categorical provenance group, without patient information |
| `quant_source` | Categorical measurement provenance; use `none` if absent |
| `shared_modality` | Audited modality category; use `none` or `unknown` if needed |
| `has_ct_quant` | Exactly 0 or 1, indicating any actual CT quantification |

When supplied, this table must contain every participant in either feature
table, with consistent labels and `has_ct_quant=1` for those measured rows.
The runner fits source-metadata-only models in the full eligible parent cohort
and in its measured subset. These are **nonclinical confounding diagnostics**:
high discrimination can indicate that provenance predicts the source label.
They are not CT models or clinical diagnostic validation.

## Optional homogeneous measurement table: `--ctpa-export`

Exactly `participant_id`, `label` and the five CT measurement columns are required.
Rows must be privately linked to their source measurements and represent a
consistent CTPA extraction layer. The same person may appear in both input
feature tables, with consistent labels, because these are separate analyses;
the tables are never concatenated into extra independent observations. No age,
sex, GOLD, dates, or source metadata are imputed into this table. Only `LR_CT5`
is fitted here. This remains a feasibility analysis when the negative class is
very small.

## Cohorts, models and correspondence to released aggregates

The original prepared inputs support 11 cohorts and 24 cohort/model combinations.
This is not a universal fixed count: the public runner explicitly skips an
optional cohort when either class has fewer than three participants. The main
cohort instead fails preflight under that condition. Supplying only `--features`
omits the two metadata diagnostics and the homogeneous CTPA analysis.

| Public runner cohort | Corresponding aggregate-result cohort |
| --- | --- |
| `main_mixed_protocol_exploratory` | `yu194_mixed_protocol_exploratory` |
| `main_CT_all_intervals` / `main_CT_within30days` | `yu_CT_all_intervals` / `yu_CT_within30days` |
| `main_CTA_all_intervals` / `main_CTA_within30days` | `yu_CTA_all_intervals` / `yu_CTA_within30days` when eligible |
| `main_CTPA_all_intervals` / `main_CTPA_within30days` | `yu_CTPA_all_intervals` / `yu_CTPA_within30days` |
| `main_GOLD1to4_sensitivity` | `yu_GOLD1to4_sensitivity` |
| `parent_source_metadata_diagnostic` | `parent590_source_metadata_diagnostic` |
| `measured_source_metadata_diagnostic` | `CT443_source_metadata_diagnostic` |
| `homogeneous_CTPA_feasibility_only` | `original_CTPA374_feasibility_only` |

The main cohort runs clinical LR, CT5 LR, clinical+CT5 LR, clinical+CT5 random
forest, clinical+CT5 XGBoost, protocol-only LR, and protocol-adjusted clinical+CT5
LR. The protocol and GOLD sensitivities each run clinical LR and clinical+CT5
LR. The two metadata cohorts each run source-metadata-only LR; the homogeneous
CTPA cohort runs CT5 LR. No outcome-based feature selection is performed.

## Preserved validation method

- Default split seeds: 42, 2020, 2026. All models in a cohort use the same outer
  folds for a seed. Five outer folds are used when the minority class has at
  least 10 participants, otherwise three.
- LR tunes `C` over 0.1, 1 and 10 by ROC AUC in inner stratified CV. Inner folds
  number three when the outer-training minority has at least six participants,
  otherwise two. The inner seed is outer seed + fold number + 1000.
- Median numeric imputation, scaling, categorical imputation and one-hot encoding
  are inside the fitted pipeline, including each inner-CV fit. LR uses lbfgs and
  at most 4000 iterations. There is no oversampling or threshold tuning.
- Random forest uses 200 trees, depth 3, minimum leaf size 5, and
  `balanced_subsample`. XGBoost uses 150 trees, depth 2, learning rate 0.03,
  row/column subsampling 0.8, regularization lambda 10, and class weighting
  calculated from the outer training fold. Both use seed + fold number.
- OOF metrics are ROC AUC, average precision, Brier score, and sensitivity,
  specificity and balanced accuracy at probability 0.5. Mean/min/max across
  seeds describe partition variability, **not confidence intervals**. There is
  no independent external test cohort. A good AUC with only a few noPH cases
  cannot establish negative-class clinical performance.

IDs, labels, GOLD, temporal intervals and RHC measurements are never predictors.
Protocol/source fields are used only in explicitly named models. Row-level OOF
records repeat participants across models and seeds and are not extra patients.

## Commands and private outputs

Run these commands from the repository root. The sibling directories keep all
inputs and generated artifacts outside the public repository.

```console
python -m pip install -r requirements.txt
python scripts/make_synthetic_example.py --output-dir ../synthetic-example
python scripts/run_experiments.py --features ../synthetic-example/features.csv --metadata ../synthetic-example/metadata.csv --ctpa-export ../synthetic-example/ctpa_export.csv --output-dir ../synthetic-run
python scripts/validate_outputs.py --run-dir ../synthetic-run
```

For a quicker software smoke run, add `--seeds 42` and choose a fresh output
directory. This changes the number of repeats; it is not the default repeated
experiment. `--prepare-only` validates and freezes inputs without fitting and
does not write a training completion marker. The synthetic generator reads no
clinical files and exercises all 11 cohorts and 24 model comparisons.

Existing nonempty output directories are rejected. Private outputs include
frozen input CSVs, their runtime hashes, cohort membership, OOF predictions,
fold membership, model fit audit, metrics, environment versions and the runner
snapshot. They may contain linkable private pseudonyms and must not be committed
or published. The runner rejects inputs and outputs inside this repository.

`validate_outputs.py` independently computes pairwise AUC, tie-aware average
precision, Brier and confusion metrics, checks the summary arithmetic, recreates
outer folds, and checks participant coverage, predictor lists, fitted-fold
counts and snapshot hashes. It writes `independent_validation.json` in the
private run directory and returns exit code 2 on a failed or incomplete check.
This is a numerical/artifact audit, not a clinical or source-identity validation.

When a cohort has no timing field, timing summaries are null (not assessed),
not zero. For timed cohorts, missing intervals and measured intervals must be
distinguished when interpreting the count above one year.
