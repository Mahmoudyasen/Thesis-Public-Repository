#!/bin/bash
# S3 — the fold-0 gate. Resumable background job; TensorBoard in work/wp0/stage1/fold0/seed0/tb.
#   bash scripts/03_s3_fold0_gate.sh            # start
#   RESUME=1 bash scripts/03_s3_fold0_gate.sh   # continue after an interruption
source "$(dirname "$0")/00_env.sh"
# The thesis run was written here and moved to fold0/seed0_s3_gate_noaug before S4 was trained in fold0/seed0
# (the later scripts read the S3 fold-0 model from seed0_s3_gate_noaug).
OUT=${LUNGCAD_DATA:-data}/work/wp0/stage1/fold0/seed0; mkdir -p "$OUT"
nohup $PY -m lungcad.stage1.train --config configs/stage1_retinanet.yaml --fold 0 --seed 0 ${RESUME:+--resume} >> "$OUT/stdout.log" 2>&1 &
echo "started pid $! -> $OUT/stdout.log   (tensorboard --logdir $OUT/tb)"
