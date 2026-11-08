#!/bin/bash
# Evaluates every finished job (outputs/<job>/DONE.json) that has no eval/metrics.json yet. CPU, low priority.
cd /home/administrator/imgsync
source /home/administrator/miniconda3/etc/profile.d/conda.sh
conda activate /home/administrator/miniconda3/envs/avatar-eval
while true; do
  for d in outputs/*/; do
    j=$(basename $d)
    [ -f $d/DONE.json ] || continue
    [ -f $d/eval/metrics.json ] && continue
    [ -f $d/eval/.failed ] && continue
    img=$(python -c "import json;print(json.load(open('$d/job_config.json'))['image'])")
    echo "[$(date '+%F %T')] evaluating $j" | tee -a logs/eval.log
    mkdir -p $d/eval
    if nice -n 15 python -u eval/eval_avatar.py --video $d/$j.mp4 --image $img --out $d/eval >> logs/eval.log 2>&1; then
      echo "[$(date '+%F %T')] eval done $j" | tee -a logs/eval.log
    else
      echo "[$(date '+%F %T')] eval FAILED $j" | tee -a logs/eval.log; touch $d/eval/.failed
    fi
  done
  sleep 60
done
