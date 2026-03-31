#!/bin/bash
# Start Dynamo frontend + infer backend worker
#
# Prerequisites:
#   - infer server running (./scripts/start_infer.sh)
#   - etcd running (or use --discovery-backend file)
#   - dynamo Python package installed
#
# Usage:
#   ./scripts/start_dynamo.sh [infer_url] [model_name] [model_path]
#
# Examples:
#   ./scripts/start_dynamo.sh
#   ./scripts/start_dynamo.sh http://localhost:8000 Qwen3-4B Qwen/Qwen3-4B

set -euo pipefail

PEGAINFER_URL="${1:-http://localhost:8000}"
MODEL_NAME="${2:-Qwen3-4B}"
MODEL_PATH="${3:-Qwen/Qwen3-4B}"
FRONTEND_PORT="${FRONTEND_PORT:-8080}"

echo "=== Starting Dynamo with Pegainfer Backend ==="
echo "  Pegainfer URL: ${PEGAINFER_URL}"
echo "  Model:         ${MODEL_NAME}"
echo "  Frontend Port: ${FRONTEND_PORT}"
echo ""

# Start infer backend worker
echo "[1/2] Starting infer backend worker..."
python -m dynamo.infer \
    --infer-url "${PEGAINFER_URL}" \
    --model-name "${MODEL_NAME}" \
    --model-path "${MODEL_PATH}" \
    --namespace dynamo \
    --component backend \
    --discovery-backend file \
    --request-plane tcp \
    --event-plane tcp &

BACKEND_PID=$!
echo "  Backend PID: ${BACKEND_PID}"

sleep 2

# Start Dynamo frontend
echo "[2/2] Starting Dynamo frontend..."
python -m dynamo.frontend \
    --http-port "${FRONTEND_PORT}" \
    --namespace dynamo \
    --discovery-backend file \
    --request-plane tcp \
    --event-plane tcp &

FRONTEND_PID=$!
echo "  Frontend PID: ${FRONTEND_PID}"

echo ""
echo "=== Dynamo ready ==="
echo "  Frontend: http://localhost:${FRONTEND_PORT}"
echo "  API:      http://localhost:${FRONTEND_PORT}/v1/chat/completions"
echo ""
echo "Press Ctrl+C to stop all services"

trap "kill ${BACKEND_PID} ${FRONTEND_PID} 2>/dev/null; exit 0" INT TERM
wait
