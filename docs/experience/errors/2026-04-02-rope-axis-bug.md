# 2026-04-02 · fast.rope axis bug in rust_transformer_layer

## Context

Investigating the ~13% decode throughput gap between agent-infer (174 tok/s) and mlx_lm (200 tok/s)
on Qwen3-0.6B-4bit with prompt=20/gen=512 tokens. Compared the MLX ops sequence in mlx_lm's
`Attention.__call__` with ours in `rust_transformer_layer`.

## Root Cause

`mlx::core::fast.rope` expects input shape `(B, *, T, D)` where **T is the second-to-last axis**
(sequence length) and D is the last axis (feature dimension).

Old code in `rust_transformer_layer` called rope on `[1, seq, n_heads, head_dim]`:
- second-to-last = `n_heads` = 16 → rope applied 16 **different** positions per step
- correct: second-to-last should be `seq` → rope applies the same position to all heads

mlx_lm correctly uses `[B, n_heads, seq, head_dim]` (post-transpose) for rope, so T = seq.

Verified with a Python test: for L=1, n=16, the two formats give `max diff ≈ 5.2` (not zero),
confirming the positional encodings were wrong in the old code.

## Fix

Changed the ops order in `rust_transformer_layer` from:
```
apply_head_norm (reshape→norm→reshape back) → reshape to 4D → rope → transpose
```
to (matching mlx_lm):
```
reshape to 4D → per-head norm → transpose → rope
```

This also removes the double reshape in the old `apply_head_norm` helper, saving 2 lazy
graph nodes per `q` and `k` projection = **4 nodes per layer × 28 layers = 112 fewer
lazy nodes per decode step**.

`apply_head_norm` helper removed (dead code).

## Rule

Before calling `fast.rope`, always transpose to `[B, n_heads, seq, head_dim]` so the
second-to-last axis is `seq`. Never apply rope to `[B, seq, n_heads, head_dim]` — rope
will treat `n_heads` as the sequence length, giving wrong positional encodings.
