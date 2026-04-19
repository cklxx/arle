# SGLang parity at fixed concurrency — Qwen3-4B on L4 (c=16 blocked by admission bug; c=8 is the clean comparison)

## Goal

Measure and compare infer vs sglang 0.5.10 **at fixed concurrency**, not
sweep. The user's question was "16 并发下的对比". Answer in two parts:

1. Whether we hit parity at c=16 (goal: optimization).
2. If c=16 is not viable on L4 today, what is the closest clean
   comparison (goal: baseline).

## Hypothesis

Based on `project_sglang_parity_2026-04-17.md`: ITL at parity,
TTFT +50% infer. Fixed-concurrency should show the same shape, with
aggregate throughput scaling N× vs sweep.

## Parameters

```
# infer (c=16 attempt)
./target/release/infer --model-path models/Qwen3-4B --port 8000 \
  --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --cuda-graph=false

# infer (c=8 clean)
./target/release/infer --model-path models/Qwen3-4B --port 8000 \
  --num-slots 8 --max-seq-len 4608 --mem-fraction-static 0.88

# sglang (c=8 clean)
python3 -m sglang.launch_server --model-path models/Qwen3-4B --host 0.0.0.0 \
  --port 8000 --max-running-requests 8 --mem-fraction-static 0.88

# guidellm (both)
scripts/bench_guidellm_concurrent.sh <label> --concurrency <N> --model <name>
# profile=concurrent, rate=N, data=prompt_tokens=4096,output_tokens=256,
# max-seconds=60, random-seed=20260416
```

Invoked via the new sibling wrapper
`scripts/bench_guidellm_concurrent.sh` (sweep canonical unchanged).

## Environment

- **Hardware:** NVIDIA L4 24GB, CUDA 13.0, driver 580.82.07, SM 8.9
- **Commit:** d91908f (+ local flashinfer-include build.rs fix)
- **infer feature set:** `cargo build --release -p infer --features cuda`
- **sglang:** 0.5.10.post1 (`pip install sglang`), default backend (`triton`)
- **Weights:** `Qwen/Qwen3-4B` BF16 from HF, shared local path via symlink
- **Serial bench order:** infer c=8 first, server killed, sglang c=8 second.
  Per `feedback_serial_bench.md` — never run both on the GPU concurrently.
- **Canonical tool:** guidellm 0.6.0 `openai_http` backend

## Results

### c=16 — infer fails, sglang untested

`cuda-l4-infer-c16-run3/` — **contaminated, do not cite**. During the
60-s window the server logged **220 pool alloc failures** out of 241
requests admitted. Failed requests emit empty 200-OK streams;
guidellm's p50 TTFT/ITL = 0 ms because the majority of completions are
zero-length.

Root cause: scheduler admits all 16 requests to slots without gating on
paged-pool token capacity. Fixed by design in sglang (gates admission
on `max-running-requests` **and** total-token budget). Full diagnosis:
[`errors/2026-04-18-c16-paged-pool-admission-overcommit.md`](../errors/2026-04-18-c16-paged-pool-admission-overcommit.md).

Decision: c=16 infer numbers are **unpublishable** on the current
admission policy. sglang c=16 was not run because without an infer
counterpart the data is one-sided. Unblocks after the admission-time
pool-capacity gate lands.

### c=8 — matched, clean

| metric | infer | sglang | Δ (infer − sglang) |
|---|---:|---:|---:|
| TTFT p50 (ms) | 3 782.5 | 2 997.3 | **+26.2 %** |
| TTFT p99 (ms) | 7 524.4 | 5 410.3 | **+39.1 %** |
| ITL p50 (ms)  | 70.47   | 59.85   | +17.7 % |
| ITL p99 (ms)  | 76.85   | 70.51   | **+9.0 %** |
| out tok/s     | 71.15   | 113.14  | **−37.1 %** |
| req/s actual  | 0.267   | 0.4     | −33.3 % |

### Comparison to the 04-17 sweep snapshot

From `2026-04-17-sglang-p99-parity-qwen3-4b.md` (same workload, same
hardware, sweep profile @ ~0.33 r/s — roughly matched concurrency ≈ 1
steady-state):

| metric | sweep 0.33 r/s (infer − sglang) | c=8 fixed (infer − sglang) |
|---|---|---|
| TTFT p99 | +54 % | **+39 %** ✓ better |
| ITL p99 | +3 % | +9 % (small regress) |
| saturation tok/s | −18 % (97.9 vs 115.8) | **−37 %** worse |

ITL p99 widens a bit vs sweep's parity (+3 % → +9 %), consistent with
c=8 driving higher per-step batch sizes where our per-step CPU overhead
(`scheduler-gpu-cpu-overlap.md`) becomes a larger slice of ITL.

The throughput gap **doubles** at c=8 (−18 % → −37 %). At c=8 the
workload is 94 % prefill-dominated; infer's TTFT deficit compounds
across the 8 in-flight requests. This is exactly what the P1 target
(Qwen3.5 full-forward CUDA Graph prefill / `qwen35-single-graph-prefill.md`)
and the P0 target (`scheduler-gpu-cpu-overlap.md`) are sized to close.

## Problems observed

1. **c=16 contaminated run** — see §Results and the linked errors
   entry. 220 / 241 pool-alloc failures; bench deleted from
   `bench-output/` index and kept at
   `2026-04-18-cuda-l4-infer-c16-run3/` for forensics only.

2. **sglang first run @ c=8 also returned zeros** — guidellm did not
   dispatch any requests until 104 s after bench start (~44 s after
   the 60-s window closed). First sglang prefill at 16:42:23 for a
   bench nominally running 16:40:39–16:41:39. Cause is guidellm-side
   synthetic-data generation lag on first invocation with a cold
   tokenizer cache (`transformers` first-load). Second run (tokenizer
   warm) immediately dispatched on schedule. The tactical workaround
   is: always do a throwaway warmup guidellm run for the first backend
   of a session — or, structurally, extend `bench_guidellm_concurrent.sh`
   to pre-warm via a 5-sec `--profile throughput --max-requests 2` probe
   before the real bench. Filed as a follow-up; not blocking the
   comparison here.

3. **Pool budget is very tight on L4** for c ≥ 8 at 4096-token prompts.
   After weights (8 GB) and CUDA Graph workspace (~1–2 GB) and
   FlashInfer workspace (~1–2 GB), the paged pool is 7–9 GB
   (53 000–59 000 tokens). 16 concurrent 4096-token prefills need
   65 500 tokens just for prefill KV — doesn't fit even before the
   admission bug. Raising `mem-fraction-static` from 0.88 → 0.94 and
   `cuda-graph=false` bumps pool from 7.4 GB → 8.8 GB but still
   under-capacity for c=16.

## Learnings

1. **`slot availability != pool availability`.** See
   `errors/2026-04-18-c16-paged-pool-admission-overcommit.md` — this
   wins entry is a symptom observation; that error entry is the root
   cause writeup. Every future scheduler change that touches
   `assign_slots()` must also consult paged-pool capacity; a test
   admitting N requests at a pool sized for N-k should observe exactly
   N-k `Phase::Prefilling`.

2. **Cold guidellm tokenizer is a silent 60-s zero-result hazard.**
   A pre-warm probe is cheap; absence of one is a serial-bench
   land-mine on first invocation.

3. **TTFT dominates fixed-concurrency throughput** at c=8 the same
   way it did at sweep saturation, only more so (−18 % → −37 %
   throughput gap). Closing the P0 + P1 prefill targets in
   `project_sglang_parity_2026-04-17.md` is directly on the critical
   path for the c=8 number — and, post-admission-fix, for c=16.

4. **infer ITL p99 at c=8 is within 9 %** of sglang's, which is the
   tightest non-sweep ITL number we've captured on L4.
   `feedback_architecture_ideal.md`: don't chase more ITL for its
   own sake; any decode-path kernel work needs to move the TTFT/admission
   numbers to be worth committing.

## Rule

For `c=16 × 4096-token-prompt` comparisons on L4 24 GB **do not run
infer today** — the scheduler over-admits and produces contaminated
numbers that look like a regression. When the admission-gate lands
(per the errors entry), re-run this snapshot and supersede.

For `c=8` workloads on L4 24 GB, **matched at num-slots=8 and
mem-fraction-static=0.88**, both engines run clean. The published
numbers above are the reference point; future optimization entries
should diff against them.

Do a **pre-warm guidellm probe** before the first bench of a session —
otherwise the first 60-s window is wasted on tokenizer load and all
percentiles report 0.

## Raw artefacts

- infer c=16 (contaminated): `bench-output/2026-04-18-cuda-l4-infer-c16-run3/`
- infer c=8 (clean):        `bench-output/2026-04-18-cuda-l4-infer-c8/`
- sglang c=8 cold (empty):  removed (0 successful completions — see §Problems #2)
- sglang c=8 (clean, v2):   `bench-output/2026-04-18-cuda-l4-sglang-c8-run2/`
- guidellm 0.6.0, sglang 0.5.10.post1, commit d91908f
