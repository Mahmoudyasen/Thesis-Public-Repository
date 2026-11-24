#!/bin/bash
# Option B: 40-vs-60 epoch study on fold 0, seed 0 (analysis runs, outside the S6 tree). A and C in the official regime,
# C in the own regime, each with a 40- and a 60-epoch cosine schedule. Stops at the first failure; relaunch resumes.
# Decision rule (amended 4 Oct, before any C result): validation only. Choose 60 only if model C's mean of the last
# 3 validation CPM evaluations gains >= 0.01 with 60; otherwise 40. The paired test ΔCPM (60 - 40) is confirmation
# only. Result (5 Oct): 40 epochs.
set -e
cd "$(dirname "$0")/.."
export OUT_ROOT=${LUNGCAD_DATA:-data}/work/wp0/stage2_dev/epoch_study FOLDS=0 SEEDS=0
EPOCHS=40 REGIMES=official MODELS="A C" bash scripts/06_s6_stage2_all.sh
EPOCHS=60 REGIMES=official MODELS="A C" bash scripts/06_s6_stage2_all.sh
EPOCHS=40 REGIMES=own MODELS=C bash scripts/06_s6_stage2_all.sh
EPOCHS=60 REGIMES=own MODELS=C bash scripts/06_s6_stage2_all.sh
echo "$(date '+%F %T') epoch study complete"
