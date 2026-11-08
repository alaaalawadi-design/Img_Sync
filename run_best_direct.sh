#!/bin/bash
# Same run with ONE command, without the queue/worker (run it inside screen: screen -S best ./run_best_direct.sh).
# Do not run it while the worker (screen longcat_worker) is generating - one job per GPU.
set -e
cd /home/administrator/imgsync
source /home/administrator/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate /home/administrator/miniconda3/envs/longcat-avatar15 \
  || export PATH=/home/administrator/miniconda3/envs/longcat-avatar15/bin:$PATH

python run_cp_avatar.py \
  --image ${IMAGE:-inputs/gazaman3.png} --audio ${AUDIO:-inputs/gaza_all_01_08.wav} \
  --quant int8 --lora_multiplier 0.30 --lora_schedule 0.30,0.01,3,0.20 \
  --reencode_continuation \
  --crop_box ${CROP_BOX:-0,0,879,940} --paste_mode rect \
  --prompt_file prompt_best.txt --seed 42 --resolution 720p \
  --name ${NAME:-cp_full_gazaman3_best_direct}
