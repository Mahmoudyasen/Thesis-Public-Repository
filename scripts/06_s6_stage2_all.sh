#!/bin/bash
# S6 — 3 models x 10 folds x 3 seeds x 2 regimes, back to back. Develop on fold 0 first (see README), then run this.
#   REGIMES="official" FOLDS="0" SEEDS="0" MODELS="C" bash scripts/06_s6_stage2_all.sh    # a single dev run
#   EPOCHS=60 OUT_ROOT=$LUNGCAD_DATA/work/wp0/stage2_dev/epoch_study ...   # analysis runs outside the S6 tree
# Order: seed -> regime -> model -> fold, so the complete 10-fold A/B/C comparison for seed 0 (both regimes) exists
# after ~6 days; seeds 1-2 then add the spread.
# Own regime only: after test scoring, the run's validation candidates are scored and the detector x stage-2 fusion
# weight is chosen on them, then applied once to the test candidates (lungcad.stage2.fuse).
# A run counts as done only when done.json AND froc_test.json exist (own regime: also fusion.json). Safety (same as the
# S5 queue): the queue STOPS at the first failure, so a half-trained run is never scored; relaunching the same command
# resumes it from last.pt. If training finished but a later step did not, only the missing steps are redone.
source "$(dirname "$0")/00_env.sh"
ROOT=${OUT_ROOT:-${LUNGCAD_DATA:-data}/work/wp0/stage2}
stop() { echo "$(date '+%F %T') $1 - queue stopped. Relaunch the same command to resume." | tee -a "$OUT/stdout.log"; exit 1; }
for s in ${SEEDS:-0 1 2}; do for regime in ${REGIMES:-official own}; do for model in ${MODELS:-A B C}; do for k in ${FOLDS:-0 1 2 3 4 5 6 7 8 9}; do
  NAME=$model${EPOCHS:+_e$EPOCHS}; RUN="$regime $NAME fold$k seed$s"
  OUT=$ROOT/$regime/$NAME/fold$k/seed$s; mkdir -p "$OUT"
  if [ -f "$OUT/done.json" ] && [ -f "$OUT/froc_test.json" ] && { [ "$regime" != own ] || [ -f "$OUT/fusion.json" ]; }; then echo "$RUN done"; continue; fi
  if [ ! -f "$OUT/done.json" ]; then
    echo "$(date '+%F %T') $RUN: training" | tee -a "$OUT/stdout.log"
    if ! $PY -m lungcad.stage2.train --config configs/stage2.yaml --model $model --fold $k --seed $s --regime $regime \
         ${EPOCHS:+--epochs $EPOCHS} --out "$OUT" --resume >> "$OUT/stdout.log" 2>&1 || [ ! -f "$OUT/done.json" ]; then
      stop "$RUN: TRAINING FAILED"
    fi
  fi
  if [ ! -f "$OUT/froc_test.json" ]; then
    echo "$(date '+%F %T') $RUN: scoring the test split" | tee -a "$OUT/stdout.log"
    if ! $PY -m lungcad.stage2.infer --checkpoint "$OUT/best.pt" >> "$OUT/stdout.log" 2>&1 || [ ! -f "$OUT/froc_test.json" ]; then
      stop "$RUN: TEST SCORING FAILED"
    fi
  fi
  [ "$regime" = own ] || continue
  if [ ! -f "$OUT/froc_val.json" ]; then
    echo "$(date '+%F %T') $RUN: scoring the validation split (for fusion)" | tee -a "$OUT/stdout.log"
    if ! $PY -m lungcad.stage2.infer --checkpoint "$OUT/best.pt" --split val --bootstrap 0 >> "$OUT/stdout.log" 2>&1 || [ ! -f "$OUT/froc_val.json" ]; then
      stop "$RUN: VALIDATION SCORING FAILED"
    fi
  fi
  echo "$(date '+%F %T') $RUN: fusion (w chosen on validation, applied once to test)" | tee -a "$OUT/stdout.log"
  if ! $PY -m lungcad.stage2.fuse --run-dir "$OUT" >> "$OUT/stdout.log" 2>&1 || [ ! -f "$OUT/fusion.json" ]; then
    stop "$RUN: FUSION FAILED"
  fi
done; done; done; done
