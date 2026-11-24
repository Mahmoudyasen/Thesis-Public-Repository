# WP4 · NLST per-exam evaluation and temporal false-positive reduction: pre-specified plan (DRAFT, 9 Nov 2026)

Status: **DRAFT; the user accepted all recommendations on 9 Nov 2026** (Section 10). It is final once the code's
known-answer checks are in place. Until the ablation queue ends, the WP4 modules live in `wp4_dev/`, outside
`lungcad/`, so the running ablations' recorded source hash stays the same. They move to `lungcad/wp4/` at the commit. Once approved, this file is committed **before** any WP4 inference is run on NLST and before any NLST
outcome is joined to a model output. Its commit hash is recorded in every WP4 result file. After that the plan is
not edited, and every deviation goes into `wp4_nlst_prespec_NOTES.md` with a date.

This plan covers RQ5 (C4, per-exam comparison with the NLST radiologists and Lung-RADS) and RQ4 (C3, temporal
false-positive reduction). WP1 is merged into WP4 (scope decision of 9 Nov 2026).

## 0. What has been seen before this plan (disclosure)

- **Labels only (no model output):** cancer counts per round in the analysed cohort, and the reproduction of the
  Pinsky et al. reference numbers from the NLST records (Section 2.3). These define the endpoint and the power.
  They cannot favour any system, because no system score was involved.
- **Detector outputs without outcomes (P2):** the fold-0 S3 and S4 detectors on the 1,955-series NLST sample
  (candidate counts and score distributions) and on 398 soft/sharp kernel pairs. None of these outputs was ever
  joined to a cancer label (the P2 scripts `p2_nlst_analysis.py`, `18`, `20` and `30` read no outcome field).
  WP4 does not reuse them. They are only the reason for the archive check in Section 3.
- **Not seen:** any system score joined to a cancer outcome, any stage-2 output on NLST, any registration result.

## 1. Questions

| | Question | Endpoint |
|---|---|---|
| RQ5 (C4) | Per screening exam, does the system reach a lower false-positive rate than the NLST radiologists **at the radiologists' sensitivity**, separately at the baseline screen (first screen received) and the screens after baseline? | ΔFPR at matched sensitivity, system − radiologists, on the same exams |
| RQ4 (C3) | At the later screens, do temporal features (the same nodule on the prior screen, change in size and score) lower the false-positive rate further at matched sensitivity? | ΔFPR at matched sensitivity, temporal − single-scan, after-baseline exams |

Published reference points (Pinsky et al., Ann Intern Med 2015;162:485–491, doi 10.7326/M14-2086). The full text
(online-first version, `~/Downloads/2015AnnIntMedPinskyLungRADSinNLSTretrospectiveassessment.pdf`) was read on
9 Nov 2026. Table 4 and Appendix Table 3:

| Screens | NLST radiologists: sensitivity / FPR | Lung-RADS: sensitivity / FPR |
|---|---|---|
| Baseline (first screen received), 26,455 | 93.5 % (90.7–96.3), 273/292 / 26.6 % (26.1–27.1), 6,939/26,090 | 84.9 % (80.8–89.0) / 12.8 % (12.4–13.2) |
| After baseline, 48,671 | 93.8 % (91.4–96.1), 376/401 / 21.8 % (21.4–22.2), 10,512/48,197 | 78.6 % (74.6–82.6) / 5.3 % (5.1–5.5) |
| · first after baseline | 94.5 %, 172/182 / 27.3 %, 6,794/24,877 | 78.0 % / 5.4 % |
| · second after baseline | 93.2 %, 204/219 / 15.9 %, 3,718/23,320 | 79.0 % / 5.2 % |

## 2. Cohort, unit of analysis and outcome

### 2.1 Cohort
- NLST CT arm, IDC 2021-05-27 delivery (`nlst_780`): 26,453 participants in the screen table.
- Image set: the converted archive minus `nlst_exclusions.csv` (`lungcad.data.nlst.usable_series()`): 72,545 series,
  26,237 participants, one primary series per participant and round.
- **Unit = one screening exam** (participant × round) with a usable series **and** an official screen result
  `scr_res{r}` in 1–6. Exams with other codes (inadequate image, not compliant, not expected) are excluded.
- **Baseline vs after baseline, as in Pinsky.** The baseline exam is the participant's **first screen received**
  (first round with `scr_res` in 1–6), even if it fell at T1 or T2; every later screen is "after baseline". This is
  defined by the screening history, not by image availability. 145 baseline exams in the image cohort are at T1 or T2.
- Analysed: **25,845 baseline exams** and **46,693 after-baseline exams** (23,696 first and 22,997 second after
  baseline).

### 2.2 Outcome (reference standard): Pinsky et al.'s definition
Pinsky, Methods: "Lung cancer was deemed to be present at a screening if it was diagnosed within 1 year or before
the next screening (whichever came first) or, for positive results, if it was diagnosed after a longer period but
with no time gap between diagnostic procedures of more than 1 year."

Implemented as follows. An exam is **cancer-present** if either:
1. `candx_days − scr_days{r}` is in [0, 365] and the diagnosis came before the next screen; or
2. the screen was positive and the NLST attributed the cancer to that screen's year (`cancyr == r`). The data
   dictionary says "a cancer associated with a positive screen is assigned the study year of that screen", the
   NLST's coding of the diagnostic-procedure chain.

Every other exam is **cancer-absent**.

- **Deviation from Pinsky, and why.** Pinsky additionally excluded exams with "indeterminate cancer status" (less
  than 1 year of follow-up, unless the participant died within 1 year): 73 at baseline and 73 after. The IDC delivery
  has no death dates, so this rule cannot be applied exactly. Treating all not-present exams as absent reproduces
  Pinsky's FPR best (K1 below). **Sensitivity analysis (pre-specified):** exclude not-present exams with
  `canc_free_days − scr_days{r}` < 365.
- **Analysed cohort: 284 cancer-present baseline exams, 375 after baseline** (164 first, 211 second after baseline).
- Cancers diagnosed after the screening years count only through rule 1 or 2. Interval and post-screening cancers
  are reported descriptively (`can_scr`).

### 2.3 Radiologists' operating point (computed, not copied)
- Radiologist-positive = `scr_res{r}` in {4, 5, 6}, the official result after comparison with prior images.
  Pinsky: "NCN of 4 mm or greater … Other abnormalities, including adenopathy or effusion, could also trigger
  positive screening results."
- **Known-answer check K1 (done 9 Nov, labels only).** On all 26,453 CT-arm participants of the IDC delivery:

  | | This implementation | Pinsky Table 4 |
  |---|---|---|
  | Baseline exams | 26,451 | 26,455 |
  | Cancer-present (baseline / after) | **292 / 401** | 292 / 401 |
  | Sensitivity (baseline / after) | **273/292 / 376/401**, exact | 273/292 / 376/401 |
  | Cancer-absent (baseline / after) | 26,159 / 48,267 | 26,090 / 48,197 (+73 / +73 indeterminate excluded) |
  | FPR (baseline / after) | **26.62 % / 21.81 %** | 26.60 % / 21.81 % |

  The cancer-present counts and the sensitivity numerators match Pinsky **exactly**. The cancer-absent counts differ
  by 69 and 70 exams, matching Pinsky's 73 indeterminate exams. The tested alternatives are worse: `cancyr == r`
  alone gives 288 / 405 cancers, and a pure 365-day window gives sensitivity 92.1 % / 92.4 %.
- **Same comparison on the image cohort** (context, labels only): the radiologists score 265/284 (93.3 %) with an
  FPR of 26.7 % at baseline, and 352/375 (93.9 %) with an FPR of 22.0 % after baseline.
- In every comparison the radiologists' sensitivity and FPR are recomputed **on exactly the analysed exams** (and
  inside every bootstrap replicate), so the system and the radiologists are measured on the same exams with the same
  outcome. The published values are quoted as context only.

## 3. Image input: 1.5 mm archive or native resolution (decided first, label-free)

The archive stores NLST at 1.5 mm isotropic spacing, coarser in-plane than the detector's 0.703 mm. P2 therefore
used native-resolution conversion (`convert_native`, 1,955-series sample, already on disk).

- **Check.** Run the full WP4 system (Section 4) on the 1,955 sample from both inputs: the archive resampled to
  detector spacing, and native.
- **Label-free criterion.** The archive is used only if it changes the system **less than a real reconstruction
  kernel change does**, the reference measured in P2 (F7: agreement of confident detections between soft and sharp
  reconstructions of the same acquisition, 0.891 for the S4 detector). Concretely, the archive is accepted if both
  of these hold:
  1. Confident-detection agreement between archive and native (score ≥ 0.5, P2 F7 matching rule, mean of both
     directions) ≥ 0.891.
  2. The Spearman correlation of the per-exam maximum fused score is ≥ 0.90.

  Otherwise all of WP4 uses native-resolution input.
- No outcome is joined in this check.
- **Cost if native:** ~58 MB per series → ~4.2 TB for 72,545 series, which does not fit on thesis-data (200 GB
  free). It goes on the 24 TB HDD (12 TB free, `/mnt/nlst-data/nlst_native_detector/`), with a CPU conversion of a few days
  (decided 9 Nov).

## 4. The system applied to NLST (frozen, no NLST training)

- **Detector:** the frozen stage-1 model of **fold 0** (`stage1/fold0/seed0`, tag `gbfd8744f0b-c1473ecc8d6`),
  the S4-augmented model that P2 tested on NLST.
  - Reason: one detector pass over 72,545 scans takes ~3 GPU-days (estimated from the P2 NLST run's file timestamps, to be measured in step 3: ~3.6 s per series per
    model). A 10-fold ensemble would take ~30 GPU-days.
  - Fold 0 is fixed now, by this rule, not by any NLST result.
  - Decided 9 Nov: fold 0 only, no ensemble.
- **Stage 2 (primary):** model **B**, own regime, fold 0, **seed 0**, fused with the detector score using that run's
  validation-chosen weight w (`fusion.json`), exactly as in S6. Candidate threshold 0.02, as in S6.
- **Secondary systems (cheap, same candidates):**
  - Detector alone.
  - B seeds 1 and 2, and the seed ensemble of B (S6 secondary system).
  - A and C fold 0 seed 0 (fused and stage 2 alone).

  These show how the LUNA16 ranking carries over (ch3 §3.9).
- **Per candidate, saved:** pid, round, world coordinates, detector score, stage-2 score, fused score, size
  (d_z, d_y, d_x from the box), z-fraction of the lung crop, and B's 3 × c embedding (`model.embed`) for MIL.
- **Blinding by construction.** The inference script reads only the image and the series table, never the
  clinical tables, and writes `nlst_candidates_*.parquet`. Outcomes are joined only by the analysis script, after
  this plan is committed.
- **Known-answer check K2.** Before NLST inference starts, the WP4 inference script (`scripts/41_wp4_infer.py --k2`),
  run on the LUNA16 fold-0 test scans, must reproduce the frozen fold-0 candidates and S6
  `own/B/fold0/seed0/test_candidates_fused.csv` within the tolerances of Section 8, and the fused CPM within 1e-4.

## 5. Per-exam aggregation

- **Primary for RQ5: max pooling.** The exam score is the maximum fused candidate score, or 0 if there is no
  candidate. It has no trainable part and no tuning, so it needs no split and leaves the fewest researcher degrees
  of freedom.
- **Secondary aggregators:**
  - FeatureGBM on single-scan features (`lungcad/wp4/aggregators.py`: candidate count, max / mean-top-3 score,
    largest diameter, diameter of the top candidate, z-fraction, score × diameter).
  - AttentionMIL over the top-k = 10 candidate embeddings.

  Both use the hyperparameters as committed in `aggregators.py`. No tuning on test folds.
- **Splits for the trained aggregators:**
  - 5 folds at patient level, stratified by "any cancer-present exam (Section 2.2)".
  - Fixed by `scripts/40_wp4_splits.py` (seed 0) and saved as `nlst/meta/wp4_splits.csv`.
  - Its SHA-256 is recorded in the NOTES file before any aggregator is trained.
  - All exams of one participant stay in one fold.
  - Every exam score used in the analysis is out-of-fold.
- **RQ4 aggregator:** FeatureGBM with the single-scan features plus the temporal features of Section 6, on the same
  splits. The single-scan FeatureGBM on the same exams is its paired comparator.

## 6. Temporal features (RQ4)

- **Prior exam** = the most recent earlier round with a usable series. The gap in days is a feature. A T2 exam
  whose T1 is missing uses T0. An after-baseline exam whose earlier screens have no usable series gets the missing-prior
  indicator.
- **Registration:** prior → current, rigid then affine, driven by the lung masks (no deformable registration).
  The conversion kept only the lung box, so masks are recomputed on the 1.5 mm archive (`lung_mask_classical`, with a
  no-border-rule fallback when that mask is < 1 L). The method is mean squares on the masks smoothed by 2 mm, a
  3-level pyramid, Euler3D then affine (`wp4_dev/register.py`).
- **Registration QC:**
  - Lung-mask Dice after registration ≥ 0.85, or the pair counts as a registration failure. A failure gets the
    missing-prior indicator and is reported, never dropped.
  - Visual check of 30 random pairs before scaling up.
  - The success rate is a methods number.
- **Matching:**
  - A current candidate matches the prior candidate nearest in registered world coordinates if the distance is
    ≤ max(10 mm, 1.5 × the larger diameter).
  - Only prior candidates with fused score ≥ 0.02 are considered (the same candidate threshold).
- **Features of the exam's top candidate and of the exam:**
  - present_prior (0/1), diameter change (mm), score change, prior exam's max score, gap (days).
  - Indicator "no usable prior" (baseline exams never enter RQ4).
- **Validation of the matching against the radiologists' records (descriptive, no threshold):**
  - `ctab`/`ctabc` give slice, lobe and diameter, plus "pre-existing" and "growth" flags.
  - We report how often an abnormality the radiologist recorded as pre-existing has a matched candidate pair, and
    how often recorded growth goes with a positive diameter change.
  - The slice from `ctab` is matched within ±5 slices, and the lobe by the lung-mask lobe proxy (z-fraction and side).
- **Known-answer check K3:** apply a known rigid + affine transform to 5 NLST scans and recover it to ≤ 1 mm
  (mean lung-surface distance).

## 7. Analysis (fixed before any NLST outcome is joined to a score)

**Matched-sensitivity FPR.**
- On a set of exams, s* = the radiologists' sensitivity on those exams.
- The system threshold is the highest threshold at which the system's sensitivity is ≥ s*.
- FPR_sys = positives among cancer-negative exams at that threshold, and Δ = FPR_sys − FPR_rad.
- **Bootstrap:** 10,000 resamples **over participants**, so the T1 and T2 exams of one person move together.
  s*, the threshold, FPR_sys and FPR_rad are recomputed in every replicate. 95 % percentile CI, two-sided
  P = 2·min(#Δ ≤ 0 + 1, #Δ ≥ 0 + 1)/(B + 1).

**Primary hypotheses.**

| Family | Hypothesis | Exams | System | Correction |
|---|---|---|---|---|
| F1 (RQ5) | H1a: ΔFPR < 0 at baseline | baseline | B fused, max pooling | Holm, m = 2 |
| F1 (RQ5) | H1b: ΔFPR < 0 after baseline | after baseline | B fused, max pooling | (same family) |
| F2 (RQ4) | H2: ΔFPR (temporal GBM − single-scan GBM) < 0 | after baseline | FeatureGBM ± temporal | m = 1 |

Decision wording, fixed now:
- **"Lower FPR than the radiologists at their sensitivity"** only if P_Holm < 0.05 **and** the CI lies below 0.
- **"Higher"** if the CI lies above 0.
- Otherwise **"not distinguishable"**.
- The same rule applies to H2 ("temporal features reduce false positives").

**Secondary, descriptive (CIs, no P values):**
- Per group (baseline, first and second after baseline, as in Pinsky Appendix Table 3): AUC, sensitivity and FPR at the transferred threshold (chosen on the
  training folds at s*, then applied to the held-out fold; the deployable operating point), and PPV.
- Each system and aggregator against the primary: detector alone, B seeds and seed ensemble, A, C, FeatureGBM,
  AttentionMIL (ΔAUC, ΔFPR at s*).
- Lung-RADS: the published points are plotted on the ROC as reference. Lung-RADS is **not** recomputed.
- Subgroups: manufacturer, kernel class, slice thickness, cancer stage and histology, lesion size. Interval cancers
  (`can_scr` = 2, negative screen) and missed-screen cancers are reported separately.
- **Farina et al. 2025 nodule locations (included, decided 9 Nov):**
  - 1,996 nodules in 450 participants, with locations in native voxel indices.
  - Measure: the detection sensitivity of the detector and of B fused at their operating points for these
    nodules, overall and for malignant ones.
  - Sensitivity only: the file marks a subset of nodules, so FPs per scan cannot be counted and no FROC is computed.
- Registration success rate; agreement of the matching with `ctabc`.

**Interpretation limits, fixed now:**
- The radiologists read with prior images, clinical context and follow-up. At baseline the system sees one scan, and
  after baseline it sees a prior only in H2.
- The reference standard depends partly on the radiologists' reads: a positive read triggered work-up, so cancers
  behind radiologist-negative screens surface only later, as interval or next-round cancers (verification bias).
  This favours the radiologists' sensitivity, and the thesis must say so.
- **If the system lands above the radiologists' FPR,** the framing is "what automated detection achieves against a
  trial benchmark" (roadmap WP4 caveat), reported in the same detail as a positive result.

**Power (approximate, from the label counts only).** With 284 baseline and 375 after-baseline cancers, s* is estimated to
about ±3 percentage points, and FPR to about ±0.4–0.6 percentage points. A system FPR difference of 2–3 percentage
points at matched sensitivity is detectable. Smaller differences will most likely come out "not distinguishable".

## 8. Known-answer checks (all must pass before the step they guard)

| Check | Guards | Status |
|---|---|---|
| K1 radiologist point reproduces Pinsky from the NLST records | outcome definition | PASS (9 Nov, Section 2.3) |
| K2 WP4 inference on LUNA16 fold-0 test reproduces the frozen fold-0 candidates and S6 B fused: same candidate set (coordinates to 1e-3 mm), max \|Δp\| ≤ 1e-3 for the detector and fused scores (AMP), fused CPM within 1e-4 | NLST inference | to do (needs the GPU) |
| K3 registration recovers a known transform: mean lung-surface error ≤ 1 mm | RQ4 features | PASS (9 Nov, 10 scans not used for tuning: 0.01–0.33 mm; NOTES 2) |
| K4 matched-sensitivity code reproduces a hand-computed toy example (10 exams), and on a table where system = radiologist gives Δ = 0 exactly | analysis script | PASS (9 Nov, `wp4_dev/metrics.py`) |
| K5 splits: every participant in exactly one fold, stratification within ±1 cancer per fold, SHA-256 recorded | aggregators | PASS (9 Nov, NOTES 1) |

## 9. Procedure and timeline (estimates)

1. Plan approved in principle (Section 10). Commit it after the C-ablation queue
   ends (~18 Nov), with no commits during that queue.
2. Write the code (CPU only, during the ablations): the inference script (K2 waits for a free GPU), splits (K5),
   analysis (K4), registration (K3).
3. ~18–19 Nov: K2 on the GPU (minutes). Then the archive-vs-native check on the 1,955 sample (Section 3; ~4 GPU-h
   with both inputs).
4. If native: convert 72,545 series to the HDD (~2–4 days CPU). Then full NLST inference (~3–4 GPU-days at the 300 W
   cap, one detector, plus stage 2 for A/B/C × seeds).
5. Aggregators, registration, temporal features (CPU, ~1 week), then analysis.
6. Chapter 5 §5.2–5.4 generated from the result files (as for Chapter 4).

Expected end: late January 2027, matching the scope decision of 9 Nov.

## 10. Decisions (9 Nov 2026, user: "go with your recommendations")

1. **Pinsky et al. read in full** (the user supplied the PDF). The definitions are in Section 2. They change the
   draft in two ways: (a) the outcome is Pinsky's (within 1 year or before the next screen; for positive screens the
   diagnostic chain via `cancyr`), not `cancyr` alone; (b) "baseline" is the first screen received, not round T0.
   K1 now reproduces Pinsky's cancer counts and sensitivity numerators exactly.
2. Detector: fold 0 only.
3. Native input, if needed: on the 24 TB HDD.
4. Farina 2025 nodule-level sensitivity: included (secondary, descriptive).
5. **Still open, with the supervisor:** the definition of "hybrid" in the title, since this plan's primary system
   is B fused.
6. **Sensitivity analyses on the outcome (pre-specified):** (a) exclude not-present exams with less than 365 days of
   cancer-free follow-up (approximating Pinsky's indeterminate exclusion); (b) round-based groups (T0 vs T1 + T2)
   instead of first-screen-received.
