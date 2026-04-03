#!/usr/bin/env bash
# =================================================================
# V2 Deploy: SP4096 + Depth Recurrence + Polar Express + SLOT
# One-command deployment on RunPod 8xH100
# =================================================================
set -euo pipefail

FORK="https://github.com/Omrigotlieb/parameter-golf.git"
RESULTS="/workspace/results_v2"
mkdir -p "$RESULTS"

echo "=== V2 DEPLOY $(date) ===" | tee "$RESULTS/status.txt"

cd /workspace/parameter-golf
git remote add fork "$FORK" 2>/dev/null || true
git fetch fork main --quiet
git checkout fork/main -- records/track_10min_16mb/2026-04-03_V2_SP4096_DepthRecur/

NGPU=$(nvidia-smi -L | wc -l)
echo "GPUs: $NGPU" | tee -a "$RESULTS/status.txt"

# Download SP4096 data if needed
SP4096_COUNT=$(ls data/datasets/fineweb10B_sp4096/fineweb_train_*.bin 2>/dev/null | wc -l)
if [ "$SP4096_COUNT" -lt 10 ]; then
    echo "Downloading SP4096 data..." | tee -a "$RESULTS/status.txt"
    MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 \
        data/cached_challenge_fineweb.py --variant sp4096 --train-shards 143
    echo "SP4096 data ready: $(ls data/datasets/fineweb10B_sp4096/fineweb_train_*.bin | wc -l) shards" | tee -a "$RESULTS/status.txt"
fi

SCRIPT="records/track_10min_16mb/2026-04-03_V2_SP4096_DepthRecur/train_gpt.py"

# === RUN 1: V2 baseline (no SLOT, no recurrence) ===
echo "" | tee -a "$RESULTS/status.txt"
echo "=== V2 RUN 1: Baseline SP4096 (seed=1337) ===" | tee -a "$RESULTS/status.txt"
echo "Started: $(date)" | tee -a "$RESULTS/status.txt"

MUON_BACKEND_STEPS=4 SLOT_ENABLED=0 RECUR_LAYERS="" \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 SEED=1337 \
torchrun --standalone --nproc_per_node=$NGPU "$SCRIPT" 2>&1 | tee "$RESULTS/v2_run1_baseline.log"
echo "V2 Run 1 done: $(date)" | tee -a "$RESULTS/status.txt"

# === RUN 2: V2 + depth recurrence ===
echo "" | tee -a "$RESULTS/status.txt"
echo "=== V2 RUN 2: + Depth Recurrence (seed=1337) ===" | tee -a "$RESULTS/status.txt"

MUON_BACKEND_STEPS=4 SLOT_ENABLED=0 RECUR_LAYERS=4,5 RECUR_START_STEP=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 SEED=1337 \
torchrun --standalone --nproc_per_node=$NGPU "$SCRIPT" 2>&1 | tee "$RESULTS/v2_run2_recur.log"
echo "V2 Run 2 done: $(date)" | tee -a "$RESULTS/status.txt"

# === RUN 3: V2 + depth recurrence + SLOT ===
echo "" | tee -a "$RESULTS/status.txt"
echo "=== V2 RUN 3: + DR + SLOT (seed=1337) ===" | tee -a "$RESULTS/status.txt"

MUON_BACKEND_STEPS=4 SLOT_ENABLED=1 SLOT_STEPS=8 SLOT_LR=0.005 \
RECUR_LAYERS=4,5 RECUR_START_STEP=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 SEED=1337 \
torchrun --standalone --nproc_per_node=$NGPU "$SCRIPT" 2>&1 | tee "$RESULTS/v2_run3_recur_slot.log"
echo "V2 Run 3 done: $(date)" | tee -a "$RESULTS/status.txt"

# === RUN 4-5: 3-seed validation of best config ===
echo "" | tee -a "$RESULTS/status.txt"
echo "=== V2 3-SEED VALIDATION ===" | tee -a "$RESULTS/status.txt"
for SEED in 42 2025; do
    echo "Seed $SEED started: $(date)" | tee -a "$RESULTS/status.txt"
    MUON_BACKEND_STEPS=4 SLOT_ENABLED=1 SLOT_STEPS=8 SLOT_LR=0.005 \
    RECUR_LAYERS=4,5 RECUR_START_STEP=3000 \
    ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 SEED=$SEED \
    torchrun --standalone --nproc_per_node=$NGPU "$SCRIPT" 2>&1 | tee "$RESULTS/v2_seed${SEED}.log"
    echo "Seed $SEED done: $(date)" | tee -a "$RESULTS/status.txt"
done

echo "" | tee -a "$RESULTS/status.txt"
echo "=== ALL V2 RUNS COMPLETE $(date) ===" | tee -a "$RESULTS/status.txt"
cat "$RESULTS/status.txt"
