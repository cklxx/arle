# CUDA L4 C16 ROI2 C2 Fast R4 4608 No-Graph

## Context

- Mode: `scripts/bench_guidellm.sh cuda-l4-c16-roi2-c2-fast-r4-4608-nograph --fast`
- Server flags:
  - `--num-slots 16`
  - `--max-seq-len 4608`
  - `--mem-fraction-static 0.94`
  - `--cuda-graph=false`
- Purpose: keep a concrete fast-mode c=16 reference point for the ROI#2 C2 branch line.

## Results

- `TTFT p99`: `4279.8 ms`
- `ITL p99`: `83.7 ms`
- `out tok/s`: `92.6`
- `server output tok/s`: `108.7`
- `server total tok/s`: `3611.1`

## Artefacts

- `bench-output/2026-04-20-cuda-l4-c16-roi2-c2-fast-r4-4608-nograph/`

## Rule

- Keep this run as the branch-local c=16 fast reference unless a newer fast run clearly dominates it on both latency and completed-request throughput.
