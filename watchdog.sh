#!/bin/bash
# Keeps the LongCat worker healthy until the queue is empty.
#  - worker gone (crash / OOM-kill / error) while jobs are pending/running -> restart (job resumes from last segment)
#  - worker alive but logs.worker.log silent for > STALE_MIN minutes -> kill it (then restarted next check)
#  - same job restarted MAX_RESTARTS times without a new finished segment -> give up and log
cd /home/administrator/imgsync
STALE_MIN=${STALE_MIN:-15}; MAX_RESTARTS=${MAX_RESTARTS:-4}
L=logs/watchdog.log; restarts=0; last_done=""
log(){ echo "[$(date '+%F %T')] $*" | tee -a $L; }
log "watchdog started (stale=${STALE_MIN}min, max_restarts=${MAX_RESTARTS})"
while true; do
  njobs=$(ls jobs/pending/*.json jobs/running/*.json 2>/dev/null | wc -l)
  alive=$(pgrep -f "python -u avatar_worker.py --serve" | wc -l)
  done_line=$(tr '\r' '\n' < logs/worker.log | grep -E 'segment [0-9]+/[0-9]+ done|JOB .* DONE' | tail -1)
  if [ "$done_line" != "$last_done" ]; then restarts=0; last_done="$done_line"; fi
  if [ "$njobs" = "0" ] && [ "$alive" != "0" ]; then :; fi
  if [ "$njobs" != "0" ] && [ "$alive" = "0" ]; then
    if [ $restarts -ge $MAX_RESTARTS ]; then log "GIVING UP: worker died $restarts times without progress; needs a human"; sleep 600; continue; fi
    restarts=$((restarts+1)); log "worker not running with $njobs job(s) queued -> restart #$restarts"
    screen -S longcat_worker -X quit >/dev/null 2>&1; sleep 2
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r -n1 ps -o pid=,cmd= -p | grep avatar_worker && log "stale GPU process still present"
    echo "=== watchdog restart #$restarts $(date)" >> logs/worker.log; ./start_worker.sh >> $L 2>&1
  elif [ "$alive" != "0" ] && [ "$njobs" != "0" ]; then  # a quiet log only means "hung" while a job is queued/running
    age=$(( $(date +%s) - $(stat -c %Y logs/worker.log) ))
    if [ $age -gt $((STALE_MIN*60)) ]; then
      log "worker log silent for ${age}s -> killing hung worker"; pkill -9 -f "python -u avatar_worker.py --serve"
    fi
  fi
  sleep 60
done
