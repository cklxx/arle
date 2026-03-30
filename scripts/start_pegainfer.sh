#!/bin/bash
# Start pegainfer inference server
#
# Usage:
#   ./scripts/start_pegainfer.sh [model_path] [port]
#
# Examples:
#   ./scripts/start_pegainfer.sh                                    # defaults
#   ./scripts/start_pegainfer.sh pegainfer/models/Qwen3-4B 8000

set -euo pipefail

MODEL_PATH="${1:-pegainfer/models/Qwen3-4B}"
PORT="${2:-8000}"

echo "=== Starting Pegainfer ==="
echo "  Model: ${MODEL_PATH}"
echo "  Port:  ${PORT}"
echo ""

cd "$(dirname "$0")/../pegainfer"

cargo run --release -- \
    --model-path "../${MODEL_PATH}" \
    --port "${PORT}"
