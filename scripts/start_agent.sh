#!/bin/bash
# Start the agent-infer agent
#
# Usage:
#   ./scripts/start_agent.sh [url] [extra-agent-args...]
#
# Examples:
#   ./scripts/start_agent.sh
#   ./scripts/start_agent.sh http://localhost:8000 --max-steps 8

set -euo pipefail

URL="${1:-http://localhost:8000}"
if [[ $# -gt 0 ]]; then
    shift
fi

CLIENT_MODE="completions"
echo "=== Agent-Infer (direct to infer) ==="

echo "  URL:  ${URL}"
echo "  Mode: ${CLIENT_MODE}"
echo ""

python -m agent_infer \
    --url "${URL}" \
    --mode "${CLIENT_MODE}" \
    "$@"
