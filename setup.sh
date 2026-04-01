#!/usr/bin/env bash
# ============================================================================
# agent-infer — one-shot dev environment setup
#
# Usage:
#   ./setup.sh              # Full setup: toolchain + deps + build + model
#   ./setup.sh --deps-only  # Install deps only, no build/model
#   ./setup.sh --build-only # Build only (assumes deps already installed)
#   ./setup.sh --model-only # Download model only
#   ./setup.sh --check      # Verify environment is ready
#
# Environment variables:
#   MODEL_ID      — HuggingFace model ID  (default: Qwen/Qwen3-8B)
#   MODEL_DIR     — Local path for model  (default: models/Qwen3-8B)
#   CUDA_HOME     — CUDA toolkit path     (default: /usr/local/cuda)
#   SKIP_MODEL    — Set to 1 to skip model download
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Colors & helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
fail()  { echo -e "${RED}[fail]${NC}  $*"; }
step()  { echo -e "\n${GREEN}▸ $*${NC}"; }

check_cmd() {
    if command -v "$1" &>/dev/null; then
        ok "$1 found: $(command -v "$1")"
        return 0
    else
        fail "$1 not found"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-8B}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
SKIP_MODEL="${SKIP_MODEL:-0}"

# ---------------------------------------------------------------------------
# Mode parsing
# ---------------------------------------------------------------------------
MODE="full"
case "${1:-}" in
    --deps-only)  MODE="deps" ;;
    --build-only) MODE="build" ;;
    --model-only) MODE="model" ;;
    --check)      MODE="check" ;;
    --help|-h)
        head -14 "$0" | tail -12
        exit 0
        ;;
esac

# ============================================================================
# CHECK mode — verify everything is ready
# ============================================================================
do_check() {
    step "Checking environment"
    local errors=0

    # Rust
    if check_cmd rustc; then
        info "  rustc $(rustc --version 2>/dev/null | awk '{print $2}')"
    else
        ((errors++))
    fi
    check_cmd cargo || ((errors++))

    # CUDA
    if [ -x "$CUDA_HOME/bin/nvcc" ]; then
        ok "nvcc found: $CUDA_HOME/bin/nvcc"
        info "  $("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep release)"
    else
        fail "nvcc not found at $CUDA_HOME/bin/nvcc"
        ((errors++))
    fi

    # GPU
    if check_cmd nvidia-smi; then
        local gpu
        gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        info "  GPU: $gpu"
    else
        ((errors++))
    fi

    # Python
    if check_cmd python3; then
        info "  python $(python3 --version 2>/dev/null | awk '{print $2}')"
    else
        ((errors++))
    fi

    # nsjail
    if check_cmd nsjail; then
        info "  sandbox isolation active"
    else
        warn "nsjail not found — tool execution will run without sandbox"
    fi

    # Python packages
    for pkg in triton flashinfer huggingface_hub; do
        if python3 -c "import $pkg" 2>/dev/null; then
            ok "python: $pkg available"
        else
            fail "python: $pkg missing"
            ((errors++))
        fi
    done

    # Binary
    if [ -x target/release/agent-infer ]; then
        ok "target/release/agent-infer built"
    else
        fail "target/release/agent-infer not found — run ./setup.sh --build-only"
        ((errors++))
    fi

    # Model
    if [ -f "$MODEL_DIR/config.json" ]; then
        ok "Model found: $MODEL_DIR"
    else
        warn "Model not found at $MODEL_DIR — run ./setup.sh --model-only"
    fi

    echo ""
    if [ "$errors" -eq 0 ]; then
        ok "Environment is ready!"
    else
        fail "$errors issue(s) found"
        return 1
    fi
}

# ============================================================================
# DEPS — install toolchain + Python packages
# ============================================================================
do_deps() {
    step "Installing Rust toolchain"
    if command -v rustc &>/dev/null; then
        ok "Rust already installed: $(rustc --version)"
    else
        info "Installing via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
        ok "Rust installed: $(rustc --version)"
    fi

    step "Checking CUDA toolkit"
    if [ -x "$CUDA_HOME/bin/nvcc" ]; then
        ok "nvcc: $("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep release)"
    else
        fail "CUDA toolkit not found at $CUDA_HOME"
        info "Install CUDA toolkit or set CUDA_HOME=/path/to/cuda"
        exit 1
    fi

    step "Installing nsjail (sandbox)"
    if command -v nsjail &>/dev/null; then
        ok "nsjail already installed"
    else
        info "Building nsjail from source..."
        apt-get install -y -qq autoconf bison flex gcc g++ git \
            libprotobuf-dev libnl-route-3-dev libtool make pkg-config protobuf-compiler \
            >/dev/null 2>&1
        local nsjail_tmp
        nsjail_tmp="$(mktemp -d)"
        git clone --depth 1 https://github.com/google/nsjail.git "$nsjail_tmp/nsjail" 2>/dev/null
        make -C "$nsjail_tmp/nsjail" -j"$(nproc)" >/dev/null 2>&1
        cp "$nsjail_tmp/nsjail/nsjail" /usr/local/bin/
        rm -rf "$nsjail_tmp"
        ok "nsjail built and installed"
    fi

    step "Installing Python packages (pinned versions)"

    # ---------- Pinned dependency versions ----------
    # FlashInfer headers — MUST match the version our csrc/cuda/flashinfer_decode.cu is compiled against.
    # 0.2.x: no enable_pdl param.  0.6.x: has enable_pdl param.
    FLASHINFER_VERSION="0.6.3"
    TRITON_VERSION="3.5.1"
    HUGGINGFACE_HUB_VERSION="0.36.2"
    HTTPX_VERSION="0.28.1"
    # ------------------------------------------------

    # FlashInfer headers (--no-deps: we only need the C++ headers, not the torch runtime)
    local fi_ver
    fi_ver=$(python3 -c "import flashinfer; print(flashinfer.__version__)" 2>/dev/null || echo "")
    if [ "$fi_ver" = "$FLASHINFER_VERSION" ]; then
        ok "flashinfer $FLASHINFER_VERSION already installed"
    else
        info "Installing flashinfer-python==$FLASHINFER_VERSION (headers only, --no-deps)..."
        pip3 install "flashinfer-python==$FLASHINFER_VERSION" --no-deps -q
        ok "flashinfer $FLASHINFER_VERSION installed"
    fi

    # Triton (needed for AOT kernel compilation at build time)
    local tri_ver
    tri_ver=$(python3 -c "import triton; print(triton.__version__)" 2>/dev/null || echo "")
    if [ "$tri_ver" = "$TRITON_VERSION" ]; then
        ok "triton $TRITON_VERSION already installed"
    else
        info "Installing triton==$TRITON_VERSION..."
        pip3 install "triton==$TRITON_VERSION" -q
        ok "triton $TRITON_VERSION installed"
    fi

    # HuggingFace Hub (for model downloads)
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        ok "huggingface_hub already installed"
    else
        info "Installing huggingface_hub==$HUGGINGFACE_HUB_VERSION..."
        pip3 install "huggingface_hub==$HUGGINGFACE_HUB_VERSION" -q
        ok "huggingface_hub installed"
    fi

    # httpx (for benchmarks)
    pip3 install "httpx>=$HTTPX_VERSION" -q 2>/dev/null || true

    # Project Python deps
    if [ -f pyproject.toml ]; then
        info "Installing Python project deps..."
        pip3 install -e ".[dev]" -q 2>/dev/null || true
    fi

    ok "All dependencies installed"
}

# ============================================================================
# BUILD — compile Rust + CUDA kernels
# ============================================================================
do_build() {
    step "Building agent-infer (release, CUDA)"

    # Ensure cargo is on PATH
    # shellcheck disable=SC1091
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    # CUDA driver stubs needed for linking when driver isn't in default lib path
    export LIBRARY_PATH="$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}"

    info "CUDA_HOME=$CUDA_HOME"
    info "SM targets: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr '\n' ' ')"

    local start
    start=$(date +%s)

    cargo build --release 2>&1 | while IFS= read -r line; do
        # Show warnings/errors, suppress noisy compile lines
        case "$line" in
            *warning:*|*error:*|*Compiling*infer*|*Compiling*agent*)
                echo "  $line" ;;
        esac
    done

    local elapsed=$(( $(date +%s) - start ))
    ok "Build complete in ${elapsed}s"

    # Show binary size
    if [ -x target/release/agent-infer ]; then
        local size
        size=$(du -h target/release/agent-infer | awk '{print $1}')
        info "Binary: target/release/agent-infer ($size)"
    fi
    if [ -x target/release/infer ]; then
        local size
        size=$(du -h target/release/infer | awk '{print $1}')
        info "Binary: target/release/infer ($size)"
    fi
}

# ============================================================================
# MODEL — download model weights
# ============================================================================
do_model() {
    step "Downloading model: $MODEL_ID → $MODEL_DIR"

    if [ -f "$MODEL_DIR/config.json" ]; then
        ok "Model already exists at $MODEL_DIR"
        info "Delete $MODEL_DIR to re-download"
        return 0
    fi

    mkdir -p "$MODEL_DIR"

    python3 -c "
from huggingface_hub import snapshot_download
import sys

print(f'Downloading {\"$MODEL_ID\"} ...')
path = snapshot_download(
    '$MODEL_ID',
    local_dir='$MODEL_DIR',
    ignore_patterns=['*.bin', '*.pt', 'original/*'],
)
print(f'Downloaded to: {path}')
"
    ok "Model downloaded to $MODEL_DIR"

    # Quick sanity check
    if [ -f "$MODEL_DIR/config.json" ]; then
        local params
        params=$(python3 -c "
import json
c = json.load(open('$MODEL_DIR/config.json'))
h = c.get('hidden_size', '?')
l = c.get('num_hidden_layers', '?')
print(f'hidden={h}, layers={l}')
" 2>/dev/null || echo "?")
        info "Model config: $params"
    fi
}

# ============================================================================
# FULL — run everything
# ============================================================================
do_full() {
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║       agent-infer — environment setup        ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""

    do_deps
    do_build
    if [ "$SKIP_MODEL" != "1" ]; then
        do_model
    else
        warn "Skipping model download (SKIP_MODEL=1)"
    fi

    echo ""
    step "Setup complete!"
    echo ""
    info "Quick start:"
    echo "  # Set runtime library paths"
    echo "  export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo ""
    echo "  # Run agent REPL (interactive, local inference)"
    echo "  ./target/release/agent-infer --model-path $MODEL_DIR"
    echo ""
    echo "  # Run HTTP server"
    echo "  cd infer && cargo run --release -- --model-path /root/models/Qwen3-8B --port 8000"
    echo ""
    echo "  # Run tests"
    echo "  cargo test --release"
    echo ""
    echo "  # Check environment"
    echo "  ./setup.sh --check"
    echo ""
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$MODE" in
    check) do_check ;;
    deps)  do_deps ;;
    build) do_build ;;
    model) do_model ;;
    full)  do_full ;;
esac
