#!/bin/bash
# Start the agent-infer agent
#
# Usage:
#   ./scripts/start_agent.sh [mode] [url]
#
# Modes:
#   direct  - Connect directly to infer (default)
#   dynamo  - Connect to Dynamo frontend
#
# Examples:
#   ./scripts/start_agent.sh                                    # direct to infer
#   ./scripts/start_agent.sh dynamo http://localhost:8080        # via dynamo

set -euo pipefail

MODE="${1:-direct}"
URL="${2:-}"

case "${MODE}" in
    direct)
        URL="${URL:-http://localhost:8000}"
        CLIENT_MODE="completions"
        echo "=== Agent-Infer (Direct Pegainfer) ==="
        ;;
    dynamo)
        URL="${URL:-http://localhost:8080}"
        CLIENT_MODE="chat"
        echo "=== Agent-Infer (via Dynamo) ==="
        ;;
    *)
        echo "Unknown mode: ${MODE}. Use 'direct' or 'dynamo'."
        exit 1
        ;;
esac

echo "  URL:  ${URL}"
echo "  Mode: ${CLIENT_MODE}"
echo ""

python -m agent_infer \
    --url "${URL}" \
    --mode "${CLIENT_MODE}" \
    "$@"
