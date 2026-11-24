#!/bin/bash
# C ablations under the committed protocol (report/c_ablation_prespec.md): attention vs concat fusion, no scale
# embedding, depth 6; folds 0-9, seed 0. Arm A = official regime (primary), arm B = own regime + score fusion
# (secondary; same val-scoring + fusion steps as 06_s6_stage2_all.sh). Order: regime -> variant -> fold.
#   REGIMES="official" FOLDS="0" VARIANTS="depth=6" EPOCHS=2 OUT_ROOT=.../stage2_dev/ablation_smoke bash scripts/07_ablations_C.sh
# Same safety rules as the S6 queue: stop at the first failure; relaunching the same command resumes.
source "$(dirname "$0")/00_env.sh"
ROOT=${OUT_ROOT:-${LUNGCAD_DATA:-data}/work/wp0/stage2}
stop() { echo "$(date '+%F %T') $1 - queue stopped. Relaunch the same command to resume." | tee -a "$OUT/stdout.log"; exit 1; }
for regime in ${REGIMES:-official own}; do for v in ${VARIANTS:-fusion=concat use_scale_embedding=false depth=6}; do for k in ${FOLDS:-0 1 2 3 4 5 6 7 8 9}; do
  NAME=C_$(echo $v | tr -d '=' | tr ',' '_')${EPOCHS:+_e$EPOCHS}; RUN="$regime $NAME fold$k seed0"
  OUT=$ROOT/$regime/$NAME/fold$k/seed0; mkdir -p "$OUT"
  if [ -f "$OUT/done.json" ] && [ -f "$OUT/froc_test.json" ] && { [ "$regime" != own ] || [ -f "$OUT/fusion.json" ]; }; then echo "$RUN done"; continue; fi
  if [ ! -f "$OUT/done.json" ]; then
    echo "$(date '+%F %T') $RUN: training" | tee -a "$OUT/stdout.log"
    if ! $PY -m lungcad.stage2.train --config configs/stage2.yaml --model C --variant "$v" --fold $k --seed 0 --regime $regime \
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
done; done; done
