#!/bin/bash
# Re-run the best trial (cp_full_gazaman3_gaza_lora0.30_minus0.01x3_min0.20) through the persistent worker queue.
# This is EXACTLY how it was run on server 22. Change IMAGE / AUDIO / NAME / CROP_BOX for another person.
set -e
cd /home/administrator/imgsync
source /home/administrator/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate /home/administrator/miniconda3/envs/longcat-avatar15 \
  || export PATH=/home/administrator/miniconda3/envs/longcat-avatar15/bin:$PATH

IMAGE=${IMAGE:-inputs/gazaman3.png}
AUDIO=${AUDIO:-inputs/gaza_all_01_08.wav}
CROP_BOX=${CROP_BOX:-0,0,879,940}          # x0,y0,x1,y1 in input-image pixels (full height around the person)
NAME=${NAME:-cp_full_gazaman3_gaza_lora0.30_minus0.01x3_min0.20_rerun}

python submit_job.py --audio "$AUDIO" --image "$IMAGE" \
  --quant int8 --distill --resolution 720p --seed 42 \
  --lora_multiplier 0.30 --lora_schedule 0.30,0.01,3,0.20 \
  --crop_box "$CROP_BOX" --paste_mode rect \
  --prompt_file prompt_best.txt \
  --name "$NAME"

./start_worker.sh                          # does nothing if the worker is already running
echo "queued. follow it with:  tail -f logs/worker.log    or   screen -r longcat_worker   (detach Ctrl-a d)"
echo "results: outputs/$NAME/"
