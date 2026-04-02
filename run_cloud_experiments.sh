#!/usr/bin/env bash
# =================================================================
# Cloud experiment runner — executes on RunPod 8xH100
# Runs: 1) Smoke test 2) SOTA reproduction 3) Our improvements
# =================================================================
set -euo pipefail

cd /workspace/parameter-golf
LOG_DIR=/workspace/results
mkdir -p "$LOG_DIR"

echo "=== EXPERIMENT RUNNER STARTED $(date) ===" | tee "$LOG_DIR/status.txt"

# Use our submission script (based on SOTA record with Polar Express + SLOT)
SCRIPT="records/track_10min_16mb/2026-04-02_PolarExpress_SLOT_XSAall/train_gpt.py"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Submission script not found at $SCRIPT"
    echo "Falling back to base train_gpt.py"
    SCRIPT="train_gpt.py"
fi

echo "Using script: $SCRIPT" | tee -a "$LOG_DIR/status.txt"

# =================================================================
# RUN 1: SOTA reproduction (no SLOT, no PE, matches PR #1019 config)
# Expected: ~1.1147 BPB
# =================================================================
echo "" | tee -a "$LOG_DIR/status.txt"
echo "=== RUN 1: SOTA REPRODUCTION ===" | tee -a "$LOG_DIR/status.txt"
echo "Started: $(date)" | tee -a "$LOG_DIR/status.txt"

BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
XSA_LAST_N=11 \
WARMDOWN_ITERS=4000 \
MUON_BACKEND_STEPS=5 \
MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.05 \
SLOT_ENABLED=0 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 "$SCRIPT" 2>&1 | tee "$LOG_DIR/run1_sota_repro.log"

# Extract results
R1_BPB=$(grep "final_int6_sliding_window_s64 " "$LOG_DIR/run1_sota_repro.log" | tail -1 | sed 's/.*val_bpb://' | awk '{print $1}')
echo "RUN 1 RESULT: val_bpb=$R1_BPB" | tee -a "$LOG_DIR/status.txt"

# =================================================================
# RUN 2: Our improvements (Polar Express NS + SLOT)
# Expected: ~1.10 BPB
# =================================================================
echo "" | tee -a "$LOG_DIR/status.txt"
echo "=== RUN 2: POLAR EXPRESS + SLOT ===" | tee -a "$LOG_DIR/status.txt"
echo "Started: $(date)" | tee -a "$LOG_DIR/status.txt"

BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
XSA_LAST_N=11 \
WARMDOWN_ITERS=4000 \
MUON_BACKEND_STEPS=4 \
MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.05 \
SLOT_ENABLED=1 SLOT_STEPS=8 SLOT_LR=0.005 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 "$SCRIPT" 2>&1 | tee "$LOG_DIR/run2_pe_slot.log"

R2_BPB=$(grep "slot_eval " "$LOG_DIR/run2_pe_slot.log" | tail -1 | sed 's/.*val_bpb://' | awk '{print $1}')
R2_PRE=$(grep "final_int6_sliding_window_s64 " "$LOG_DIR/run2_pe_slot.log" | tail -1 | sed 's/.*val_bpb://' | awk '{print $1}')
echo "RUN 2 RESULT: pre_slot=$R2_PRE post_slot=$R2_BPB" | tee -a "$LOG_DIR/status.txt"

# =================================================================
# RUN 3: 3-seed validation (if run 2 looks good)
# =================================================================
echo "" | tee -a "$LOG_DIR/status.txt"
echo "=== RUN 3: 3-SEED VALIDATION ===" | tee -a "$LOG_DIR/status.txt"

for SEED in 42 2025; do
    echo "Seed $SEED started: $(date)" | tee -a "$LOG_DIR/status.txt"
    BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
    XSA_LAST_N=11 \
    WARMDOWN_ITERS=4000 \
    MUON_BACKEND_STEPS=4 \
    SLOT_ENABLED=1 SLOT_STEPS=8 SLOT_LR=0.005 \
    ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
    EVAL_STRIDE=64 \
    SEED=$SEED \
    torchrun --standalone --nproc_per_node=8 "$SCRIPT" 2>&1 | tee "$LOG_DIR/run3_seed${SEED}.log"

    SBPB=$(grep "slot_eval " "$LOG_DIR/run3_seed${SEED}.log" | tail -1 | sed 's/.*val_bpb://' | awk '{print $1}')
    echo "Seed $SEED RESULT: val_bpb=$SBPB" | tee -a "$LOG_DIR/status.txt"
done

echo "" | tee -a "$LOG_DIR/status.txt"
echo "=== ALL RUNS COMPLETE $(date) ===" | tee -a "$LOG_DIR/status.txt"
cat "$LOG_DIR/status.txt"
