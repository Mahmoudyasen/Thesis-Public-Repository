# Thesis code: hybrid deep-learning lung-nodule detection with false-positive reduction

Code for the master's thesis *A Hybrid Deep Learning Approach for the Early Detection of Lung Cancer: A Comparative
Study on the NLST Dataset with a Focus on Reducing False Positives* (Mahmoud S. Yaseen, Istanbul Aydin University,
Institute of Graduate Studies, Department of Software Engineering).

The system has two stages:

1. **Stage 1, nodule detection:** a 3D RetinaNet (MONAI) trained on LUNA16 in ten-fold cross-validation, with
   dose and reconstruction-kernel augmentation.
2. **Stage 2, false-positive reduction:** three classifiers on multi-scale candidate patches (24 / 48 / 96 mm),
   compared on identical inputs: **A** a single-scale 3D ResNet-18, **B** a multi-scale 3D ResNet-18 and **C** a
   multi-scale model with cross-scale attention. In the end-to-end ("own") regime the stage-2 score is fused with the
   detector score; the fusion weight is chosen on validation candidates only.

The complete system is then applied to NLST screening exams (per-exam classification against the radiologists, and
temporal false-positive reduction with prior-exam registration; WP4), and checked at nodule level on the TCIA
LDCT-and-Projection-data screening scans at full and 10 % dose (WP5).

The code of the separate robustness study (paired evaluation of dose and blur augmentation) is published on its own at
<https://github.com/Mahmoudyasen/lung-nodule-detector-robustness>. This repository contains only the few modules of that
study that the thesis steps import or read from (marked below).

## Contents by thesis chapter

| Thesis section | Code |
|---|---|
| III.A–B Data, preprocessing | `lungcad/preprocess.py`, `lungcad/data/luna16_index.py`, `lungcad/data/luna16_preprocess.py`; scripts `01`, `02` |
| III.C, IV.A Stage 1 detector | `lungcad/stage1/`, `lungcad/augment.py`, `configs/stage1_retinanet*.yaml`; scripts `03`, `04`, `05`, `build_own_candidates.py` |
| IV.B Missed nodules (E5) | `scripts/e5_miss_review.py`, `e5b_miss_low_threshold.py`, `e5_figures.py` |
| IV.C Dose and kernel robustness (E1) | `scripts/e1_dose_robustness.py`, `e1_figures.py` |
| III.D–E, IV.D Stage 2 and score fusion (S6) | `lungcad/stage2/`, `lungcad/eval/fusion.py`, `lungcad/eval/operating_points.py`, `configs/stage2.yaml`; scripts `06`, `08`, `09`, `13`, `s6_fusion_fold0.py`, `s6_fusion_figure.py`; `report/s6_figures.py`, `report/s6_seeds_figure.py` |
| III.F Evaluation and statistics | `lungcad/eval/froc.py` (equivalent to the official LUNA16 scorer, checked by `tests/test_froc_official.py`), `lungcad/eval/bootstrap.py`, `lungcad/eval/official/` (official LUNA16 evaluation script) |
| IV.E Seed ensemble | `scripts/12_seed_ensemble.py` |
| IV.F Ablations of model C | `scripts/07_ablations_C.sh`, `scripts/35_c_ablation_stats.py` |
| III.I, V.B–V.D NLST (WP4) | `lungcad/data/nlst*.py`, `lungcad/wp4/`; scripts `10`, `11`, `40`–`47`; `report/figs_nlst.py` |
| III.J, V.E TCIA nodule-level check (WP5) | `scripts/48_wp5_tcia.py` |
| Figures | `thesis/make_figures.py` (pooled detector FROC, per-fold CPM and recall by size) and the figure scripts above |

Modules shared with the robustness study, included because thesis steps depend on them:
`scripts/e1_multifold.py`, `p2_stats.py`, `22_tcia_e1.py` and `32_review_stats.py` (imported by `45` and `48`);
`15_tcia_ldct_download.py`, `16_tcia_ldct_preprocess.py` and `19_tcia_annotations.py` (TCIA volumes and reference
lesions used by WP5); `18_nlst_descriptive.py` and `20_nlst_native_sample.py` (the 1,955-series NLST sample used by
the WP4 input check, script `45`).

## Environment

Python 3.12, `pip install -r requirements.txt` (PyTorch with CUDA from pytorch.org). All experiments ran on one NVIDIA
RTX 3090 (24 GB). Training the detector takes about 19 h per fold; the full stage-2 programme takes about 18 GPU-days.

Data roots are set by environment variables (defaults in `lungcad/config.py`):

```bash
export LUNGCAD_DATA=/path/to/working-data      # LUNA16, preprocessed volumes, runs (work/wp0, work/wp4, work/wp5), results
export LUNGCAD_NLST_HDD=/path/to/raw-data      # raw downloads (NLST DICOM, tcia_ldct/)
export LUNGCAD_NLST_NATIVE=$LUNGCAD_NLST_HDD/nlst_native_detector   # WP4 full native-resolution NLST store (~4.2 TB)
export PY=/path/to/env/bin/python              # used by the shell scripts
```

## Data

No images, labels, checkpoints or result files are stored here. The data are public from their providers:

| Data set | Source | Used for |
|---|---|---|
| LUNA16 (from LIDC-IDRI) | Zenodo [10.5281/zenodo.3723295](https://doi.org/10.5281/zenodo.3723295), [10.5281/zenodo.4121926](https://doi.org/10.5281/zenodo.4121926) | training and evaluation of both stages |
| NLST CT and public clinical subset | TCIA [10.7937/TCIA.HMQ8-J677](https://doi.org/10.7937/TCIA.HMQ8-J677) | WP4 |
| LDCT-and-Projection-data | TCIA [10.7937/9NPB-2637](https://doi.org/10.7937/9NPB-2637) | WP5 |
| Farina et al. NLST nodule positions | the authors' public data record | WP4 detection-sensitivity check (`46`) |

## Run order

| Step | Command |
|---|---|
| Checks | `python tests/test_froc_official.py` (scorer equals the official script); `python tests/smoke_models.py` (GPU harness); `python tests/test_preprocess_roundtrip.py` |
| S1 LUNA16 index and folds | `bash scripts/01_s1_index.sh` |
| S2 preprocessing | `bash scripts/02_s2_preprocess_luna16.sh` |
| S3 fold-0 detector without augmentation | `bash scripts/03_s3_fold0_gate.sh`; `bash scripts/04_s3_score_fold0.sh` |
| S4/S5 ten-fold detector with augmentation | `CFG=configs/stage1_retinanet_aug.yaml bash scripts/05_s5_all_folds.sh`; `python scripts/build_own_candidates.py` |
| E1, E5 | `python scripts/e1_dose_robustness.py`; `python scripts/e5_miss_review.py`; `python scripts/e5b_miss_low_threshold.py` |
| S6 stage 2 (3 models × 10 folds × 3 seeds × 2 regimes) | `bash scripts/06_s6_stage2_all.sh`; `python scripts/08_results_tables.py`; `python scripts/12_seed_ensemble.py`; `python scripts/13_fp_between_models.py` |
| Ablations of model C | `bash scripts/07_ablations_C.sh`; `python scripts/35_c_ablation_stats.py --check-only`, then without the flag |
| NLST inventory and conversion | `bash scripts/10_nlst_inventory.sh`; `bash scripts/11_nlst_convert.sh pilot`, then `full` |
| WP4 input check | `python scripts/18_nlst_descriptive.py` (sample); `python scripts/20_nlst_native_sample.py`; `python scripts/41_wp4_infer.py --k2`; `41` with `--cohort sample1955 --input archive` and `--input native`; `python scripts/45_wp4_input_check.py` |
| WP4 full run | `python scripts/40_wp4_splits.py`; `python scripts/47_wp4_native_convert.py`; `python scripts/41_wp4_infer.py --cohort all --input native`; `python scripts/42_wp4_register.py`; `python scripts/43_wp4_aggregate.py`; `python scripts/44_wp4_analysis.py`; `python scripts/46_wp4_farina.py map`, then `sensitivity` |
| WP5 | `python scripts/15_tcia_ldct_download.py`; `python scripts/16_tcia_ldct_preprocess.py`; `python scripts/19_tcia_annotations.py`; `python scripts/48_wp5_tcia.py --selftest`, then without the flag |
| Figures | `python thesis/make_figures.py`; `python report/s6_figures.py`; `python report/figs_nlst.py` |

Each script's header lists its options. `python -m lungcad.wp4.selftest` tests scripts `43` and `44` end to end on a
synthetic cohort (no NLST data).

**WP4 analysis plan.** The WP4 scripts that join NLST outcomes to model scores (`43`, `44`, `46 sensitivity`) refuse to
run unless the WP4 analysis plan is committed in the repository and the participant splits match the recorded SHA-256
(`lungcad/wp4/guard.py`). The plan is therefore included, byte-for-byte as committed before any NLST system output
existed, as `report/wp4_nlst_prespec.md`; `sha256sum report/wp4_nlst_prespec.md` gives the value listed in the thesis
(Appendix C, Table 40). The other analysis plans are not read by any code and are not included.

## Traceability

Every checkpoint and result file carries a version tag made of the Git commit of the code and a hash of the
configuration (`lungcad/version.py`), and every run writes `run_manifest.json` with the full configuration and software
versions. Uncommitted code changes mark the tag as modified.

## Licence

MIT (see `LICENSE`). `lungcad/eval/official/` is the LUNA16 challenge evaluation script, ported to Python 3 with no
logic changes and used for the scorer check; see `lungcad/eval/official/README.md`.
