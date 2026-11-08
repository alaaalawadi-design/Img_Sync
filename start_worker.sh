#!/bin/bash
# Starts the persistent worker inside a detached screen session "longcat_worker".
# Attach with:  screen -r longcat_worker      (detach: Ctrl-a d)
# Exit code 75 = worker asked for a restart (after a CUDA OOM, to fully release VRAM); jobs resume automatically.
cd /home/administrator/imgsync
mkdir -p logs jobs
# never run two workers on one GPU
screen -wipe >/dev/null 2>&1  # drop stale sockets of dead sessions
if pgrep -f "python -u avatar_worker.py --serve" >/dev/null; then echo "a worker is already running - not starting another"; exit 0; fi
PRELOAD=${PRELOAD:-int8}
cat > /home/administrator/imgsync/.worker_loop.sh <<EOF
if [ -f /home/administrator/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/administrator/miniconda3/etc/profile.d/conda.sh
  conda activate /home/administrator/miniconda3/envs/longcat-avatar15
else  # server without miniconda (env copied as a self-contained prefix)
  export PATH=/home/administrator/miniconda3/envs/longcat-avatar15/bin:\$PATH CONDA_PREFIX=/home/administrator/miniconda3/envs/longcat-avatar15
fi
cd /home/administrator/imgsync
while true; do
  python -u avatar_worker.py --serve --preload $PRELOAD 2>&1 | tee -a logs/worker.log
  rc=\${PIPESTATUS[0]}
  [ "\$rc" = "75" ] || { echo "worker exited with code \$rc" | tee -a logs/worker.log; break; }
  echo "=== worker restart (rc=75) \$(date)" | tee -a logs/worker.log
done
EOF
screen -dmS longcat_worker bash /home/administrator/imgsync/.worker_loop.sh
sleep 1; screen -ls | grep longcat_worker
