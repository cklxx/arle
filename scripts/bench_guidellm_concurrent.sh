#!/usr/bin/env bash
# Canonical c=N concurrent-profile bench wrapper around vllm-project/guidellm.
#
# Sibling to scripts/bench_guidellm.sh. Same locked DATA/MAX_SECONDS/SEED;
# replaces `--profile sweep` with `--profile concurrent --rate N` so every
# request is one of N constant in-flight slots. Use this when the question
# is "how do we compare at fixed concurrency", not "where does saturation
# land". The sweep script remains the canonical regression-check tool.
#
# Usage:
#   ./scripts/bench_guidellm_concurrent.sh <backend-label> --concurrency N \
#       [--target URL] [--model NAME] [--processor PATH]
#
# Preconditions: guidellm, curl, jq on PATH; HTTP server running at --target
# with --num-slots >= N (else the last N - num_slots slots queue in
# admission, not kernels, and the numbers lie).

set -euo pipefail

# ---- Canonical params (locked; same as sweep wrapper except PROFILE+RATE) ----
DATA="prompt_tokens=4096,output_tokens=256"
MAX_SECONDS=60
RANDOM_SEED=20260416
OUTPUTS="json,csv,html"
BACKEND="openai_http"
BACKEND_KWARGS='{"validate_backend": "/v1/models"}'
PROFILE="concurrent"
# -----------------------------------------------------------------------------

TARGET="http://localhost:8000"
MODEL="Qwen/Qwen3-4B"
PROCESSOR_DEFAULT="models/Qwen3-4B"
PROCESSOR=""
LABEL=""
CONCURRENCY=""

usage() {
    cat <<EOF
usage: $(basename "$0") <backend-label> --concurrency N [--target URL] [--model NAME] [--processor PATH]

  <backend-label>   required, e.g. cuda-l4-infer, cuda-l4-sglang
  --concurrency N   required, number of constant in-flight requests
  --target URL      default: $TARGET
  --model  NAME     default: $MODEL
  --processor PATH  tokenizer source (default: $PROCESSOR_DEFAULT if present, else \$MODEL)

Server must be launched with --num-slots >= N.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)      [[ $# -ge 2 ]] || { echo "error: --target requires a value" >&2; exit 2; }; TARGET="$2"; shift 2 ;;
        --model)       [[ $# -ge 2 ]] || { echo "error: --model requires a value" >&2; exit 2; }; MODEL="$2"; shift 2 ;;
        --processor)   [[ $# -ge 2 ]] || { echo "error: --processor requires a value" >&2; exit 2; }; PROCESSOR="$2"; shift 2 ;;
        --concurrency) [[ $# -ge 2 ]] || { echo "error: --concurrency requires a value" >&2; exit 2; }; CONCURRENCY="$2"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        --*)           echo "error: unknown flag: $1" >&2; usage >&2; exit 2 ;;
        *)             if [[ -z "$LABEL" ]]; then LABEL="$1"; shift; else echo "error: unexpected positional: $1" >&2; usage >&2; exit 2; fi ;;
    esac
done

[[ -n "$LABEL" ]]       || { echo "error: <backend-label> required" >&2; usage >&2; exit 2; }
[[ -n "$CONCURRENCY" ]] || { echo "error: --concurrency N required" >&2; usage >&2; exit 2; }
[[ "$CONCURRENCY" =~ ^[0-9]+$ ]] || { echo "error: --concurrency must be an integer" >&2; exit 2; }

for tool in guidellm jq curl; do
    command -v "$tool" >/dev/null 2>&1 || { echo "error: $tool not on PATH" >&2; exit 2; }
done

if ! curl -sS -f "$TARGET/v1/models" >/dev/null 2>&1; then
    echo "error: server not running at $TARGET" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE="$(date +%Y-%m-%d)"

# dir suffix encodes concurrency so repeat runs at a different c don't collide
base_dir="$REPO_ROOT/bench-output/${DATE}-${LABEL}-c${CONCURRENCY}"
OUTPUT_DIR="$base_dir"
run=1
while [[ -e "$OUTPUT_DIR" ]]; do
    run=$((run + 1))
    OUTPUT_DIR="${base_dir}-run${run}"
done
mkdir -p "$OUTPUT_DIR"

COMMIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

if [[ -z "$PROCESSOR" ]]; then
    if [[ -d "$REPO_ROOT/$PROCESSOR_DEFAULT" ]]; then
        PROCESSOR="$REPO_ROOT/$PROCESSOR_DEFAULT"
    else
        PROCESSOR="$MODEL"
    fi
fi

echo ">>> guidellm concurrent c=$CONCURRENCY"
echo "    target : $TARGET"
echo "    model  : $MODEL"
echo "    label  : $LABEL"
echo "    data   : $DATA"
echo "    seconds: $MAX_SECONDS"
echo "    seed   : $RANDOM_SEED"
echo "    output : $OUTPUT_DIR"
echo

set +e
guidellm benchmark run \
    --target "$TARGET" \
    --model "$MODEL" \
    --processor "$PROCESSOR" \
    --profile "$PROFILE" \
    --rate "$CONCURRENCY" \
    --data "$DATA" \
    --max-seconds "$MAX_SECONDS" \
    --random-seed "$RANDOM_SEED" \
    --output-dir "$OUTPUT_DIR" \
    --outputs "$OUTPUTS" \
    --backend "$BACKEND" \
    --backend-kwargs "$BACKEND_KWARGS"
gdl_rc=$?
set -e

if [[ $gdl_rc -ne 0 ]]; then
    echo "error: guidellm exited with status $gdl_rc" >&2
    echo "       raw artefacts (if any): $OUTPUT_DIR" >&2
    exit 3
fi

JSON_FILE="$OUTPUT_DIR/benchmarks.json"
TABLE_FILE="$OUTPUT_DIR/headline_table.md"

{
    printf '| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |\n'
    printf '|---|---|---|---|---|---|---|\n'
    jq -r '
        def pctl($m): (.metrics[$m].successful.percentiles // {});
        def avg($m):  (.metrics[$m].successful.mean        // null);
        def rnd(d): if . == null then "n/a" else (. * pow(10;d) | round / pow(10;d)) end;
        .benchmarks
        | map(
            { rate: (.config.strategy | if .type_ == "concurrent" then "conc\(.max_concurrency // "?")" elif .type_ == "synchronous" then "sync" elif .type_ == "throughput" then "throughput" elif (.rate // null) != null then "\(.rate)r/s" else .type_ end),
              ttft_p50: (pctl("time_to_first_token_ms").p50 | rnd(1)),
              ttft_p99: (pctl("time_to_first_token_ms").p99 | rnd(1)),
              itl_p50:  (pctl("inter_token_latency_ms").p50 | rnd(2)),
              itl_p99:  (pctl("inter_token_latency_ms").p99 | rnd(2)),
              tok_s:    (avg("output_tokens_per_second") | rnd(2)),
              req_s:    (avg("requests_per_second")      | rnd(3)) })
        | .[]
        | "| \(.rate) | \(.ttft_p50) | \(.ttft_p99) | \(.itl_p50) | \(.itl_p99) | \(.tok_s) | \(.req_s) |"
    ' "$JSON_FILE" 2>/dev/null || printf '| _extraction failed_ | see | `benchmarks.html` | for | full | results | . |\n'
} > "$TABLE_FILE"

echo
echo ">>> headline table"
cat "$TABLE_FILE"
echo
echo ">>> artefacts"
echo "    raw  : $OUTPUT_DIR"
echo "    html : $OUTPUT_DIR/benchmarks.html"
echo "    commit: $COMMIT_SHA"
