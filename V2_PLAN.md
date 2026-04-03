# V2 Submission Plan

## Target: sub-1.09 BPB clean (no SLOT risk)

## Current Position
- PR #1298: 1.1043 BPP (with SLOT, legality disputed)
- Non-SLOT baseline: ~1.115 BPB

## V2 Stack (combines all top-performing clean techniques)

### Training-time changes
1. SP4096 tokenizer (need to build from raw docs or get from clarkkev)
2. MLP 4x expansion (from 3x)
3. Weight decay 0.090 (from 0.04)
4. Depth recurrence layers 4,5 (shared MLP, activate step 3000)
5. Polar Express NS (4 steps) — already have
6. MuonEq-R — already have
7. XSA all 11 layers — already have

### Quantization
8. Full Hessian GPTQ int6 (all 66 layers) with AR self-gen calibration
9. Brotli-11 + byte-shuffle compression

### Eval-time (legal)
10. LoRA TTT (rank-8 on Q,V,lm_head, score-first)
11. N-gram best_agree rescoring (single-pass, causal)

## Implementation Order
1. Quick win: depth recurrence on current SP1024 script → cloud test
2. Build SP4096 tokenizer + retokenize data
3. MLP 4x + WD 0.09 config change
4. Full GPTQ port
5. LoRA TTT port
6. N-gram port
7. 3-seed validation → submit V2 PR

## Estimated BPB: 1.03-1.07 (aggressive) / 1.08-1.09 (conservative)
