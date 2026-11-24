#!/bin/bash
# S3 — score the fold-0 TEST subset with the best checkpoint (official-equivalent FROC + 1000-resample bootstrap CI).
source "$(dirname "$0")/00_env.sh"
CK=${LUNGCAD_DATA:-data}/work/wp0/stage1/fold0/seed0/best.pt
$PY -m lungcad.stage1.infer --checkpoint "$CK" --fold 0 --split test --score
