# Remove legacy model family — guidellm benchmark (pending remote), 2026-04-23

## Goal
- Validate that removing the legacy model family does not regress Qwen3/Qwen3.5 serving performance.

## Hypothesis
- No throughput or latency regression is expected because deleted legacy-model paths were isolated from the active Qwen hot kernels.

## Params
- Planned command: `scripts/bench_guidellm.sh remove-legacy-model-family`
- Planned model: `Qwen/Qwen3-4B` and `Qwen/Qwen3.5-4B`

## Env
- Status: `pending-remote`
- Local blocker: this workspace lacks required CUDA/NVCC + benchmark runtime prerequisites.
- Remote execution plan: run on the standard CUDA benchmark host used by the team benchmark pipeline.

## Results
- Pending remote execution.

## Problems
- None in code path validation; benchmark environment unavailable locally.

## Learnings
- Structural model-family deletions should still be benchmarked to preserve regression discipline.
