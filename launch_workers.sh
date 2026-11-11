#!/usr/bin/env bash
# Start one queue worker per GPU, each in its own tmux window.
# Usage:  bash launch_workers.sh                      -> all GPUs, EVERYTHING: Steps (a), (b) and (c)
#         bash launch_workers.sh "small2d,3d,mid2d,big2d,chest"   -> choose the plan
#         bash launch_workers.sh all "--joint_rounds 60"           -> shorter Step (c) training (cheaper)
# Watch:  tmux attach -t workers   (Ctrl-b n = next window)   |   tail -f logs/worker_gpu*.log
# Re-running this after a crash is safe: finished jobs are skipped, unfinished ones resume.
set -euo pipefail
cd "$(dirname "$0")"
PLAN="${1:-all}"
EXTRA="${2:-}"
NGPU=$(nvidia-smi -L | wc -l)
mkdir -p logs
[ -d venv ] && source venv/bin/activate
bash setup_official.sh                       # authors' code for Steps (a)+(b) (safe to repeat)
export NGPU
echo "Downloading all datasets first (avoids parallel downloads of the same file)..."
python run_queue.py --plan "$PLAN" --download_only
tmux kill-session -t workers 2>/dev/null || true
tmux new-session -d -s workers -n gpu0
for ((i=0; i<NGPU; i++)); do
  [ "$i" -gt 0 ] && tmux new-window -t workers -n "gpu$i"
  tmux send-keys -t "workers:gpu$i" \
    "[ -d venv ] && source venv/bin/activate; NGPU=$NGPU CUDA_VISIBLE_DEVICES=$i python run_queue.py --worker --plan $PLAN --log_test $EXTRA 2>&1 | tee -a logs/worker_gpu$i.log" C-m
  sleep 20   # stagger starts so workers claim different jobs cleanly
done
echo "Started $NGPU workers in tmux session 'workers'.  tmux attach -t workers"
