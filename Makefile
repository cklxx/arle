# agent-infer — local development shortcuts
#
# Usage:
#   make build-metal               # macOS / Apple Silicon
#   make bench-metal MODEL=models/Qwen3-4B
#   make build-cuda                # Linux / NVIDIA GPU
#   make bench-cuda
#   make test                      # any platform (CPU-only)
#   make test-py

MODEL ?= models/Qwen3-4B

.PHONY: build-metal bench-metal bench-metal-compare build-cuda bench-cuda test test-py

# ── Metal (macOS / Apple Silicon) ────────────────────────────────────────────
build-metal:
	cargo build --release --no-default-features --features metal,no-cuda -p infer

bench-metal:
	cargo run --release --no-default-features --features metal,no-cuda \
		--bin metal_bench -- --model $(MODEL)

bench-metal-compare:
	cargo run --release --no-default-features --features metal,no-cuda \
		--bin metal_bench -- --model $(MODEL) --compare-baseline benchmarks/metal_baseline.json

# ── CUDA (Linux / NVIDIA GPU) ─────────────────────────────────────────────────
build-cuda:
	cargo build --release --features cuda -p infer

bench-cuda:
	cargo run --release --features cuda --bin bench_serving

# ── Platform-agnostic ─────────────────────────────────────────────────────────
test:
	cargo test --no-default-features --features no-cuda -p infer --lib

test-py:
	pytest tests/ -x
