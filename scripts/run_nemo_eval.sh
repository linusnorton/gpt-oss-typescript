#!/usr/bin/env bash
# run_nemo_eval.sh - Run NeMo Evaluator against a served model
# Uses BigCode harness tasks for code evaluation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_BASE_URL="http://localhost:${VLLM_PORT:-8000}/v1"
DEFAULT_MODEL_NAME="gpt-oss"
DEFAULT_TASKS="humaneval-ts,mbpp-ts"
DEFAULT_API_KEY="${NEMO_EVAL_API_KEY:-EMPTY}"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/outputs/eval"

# Parse arguments
MODEL_NAME=""
BASE_URL=""
API_KEY=""
TASKS=""
OUTPUT_DIR=""
CONFIG_FILE=""
NUM_SAMPLES=""
TEMPERATURE=""
MAX_TOKENS=""
DRY_RUN=false
USE_DOCKER=false

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run NeMo Evaluator or BigCode evaluation harness against a served model."
    echo ""
    echo "Options:"
    echo "  --model-name NAME     Model name in vLLM (default: $DEFAULT_MODEL_NAME)"
    echo "  --base-url URL        OpenAI-compatible API base URL (default: $DEFAULT_BASE_URL)"
    echo "  --api-key KEY         API key for authentication (default: EMPTY)"
    echo "  --tasks TASKS         Comma-separated task list (default: $DEFAULT_TASKS)"
    echo "                        Available: humaneval-ts, mbpp-ts"
    echo "  --out DIR             Output directory (default: $DEFAULT_OUTPUT_DIR)"
    echo "  --config FILE         Config file override (default: configs/nemo_eval.yaml)"
    echo "  --num-samples N       Number of samples per task"
    echo "  --temperature T       Sampling temperature"
    echo "  --max-tokens N        Maximum tokens to generate"
    echo "  --docker              Run evaluation in Docker container"
    echo "  --dry-run             Print command without executing"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Available Tasks:"
    echo "  humaneval-ts          HumanEval TypeScript (HumanEval-X)"
    echo "  mbpp-ts               MBPP TypeScript (MultiPL-E)"
    echo ""
    echo "Examples:"
    echo "  # Basic evaluation"
    echo "  $0 --tasks humaneval-ts"
    echo ""
    echo "  # Both benchmarks"
    echo "  $0 --tasks humaneval-ts,mbpp-ts --num-samples 10"
    echo ""
    echo "  # Custom endpoint"
    echo "  $0 --base-url http://remote-server:8000/v1 --model-name my-model"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --out)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Apply defaults
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
BASE_URL="${BASE_URL:-$DEFAULT_BASE_URL}"
API_KEY="${API_KEY:-$DEFAULT_API_KEY}"
TASKS="${TASKS:-$DEFAULT_TASKS}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/nemo_eval.yaml}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NeMo Evaluator${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/run_$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Model: $MODEL_NAME"
echo "  Base URL: $BASE_URL"
echo "  Tasks: $TASKS"
echo "  Output: $RUN_DIR"
echo ""

# Check if server is reachable
echo -e "${YELLOW}Checking server connectivity...${NC}"
HEALTH_URL="${BASE_URL%/v1}/health"
MODEL_URL="$BASE_URL/models"

if ! curl -s --connect-timeout 5 "$MODEL_URL" > /dev/null 2>&1; then
    # Try health endpoint
    if ! curl -s --connect-timeout 5 "$HEALTH_URL" > /dev/null 2>&1; then
        echo -e "${RED}Error: Cannot connect to server at $BASE_URL${NC}"
        echo "Make sure vLLM server is running: make serve-vllm"
        exit 1
    fi
fi
echo -e "${GREEN}  Server is reachable${NC}"

# Verify model is available
echo -e "${YELLOW}Checking available models...${NC}"
MODELS_RESPONSE=$(curl -s "$MODEL_URL" -H "Authorization: Bearer $API_KEY" 2>/dev/null || echo "{}")
echo "$MODELS_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = data.get('data', [])
    if models:
        for m in models:
            print(f\"  - {m.get('id', 'unknown')}\")
    else:
        print('  No models found')
except:
    print('  Could not parse model list')
"

# Check for NeMo toolkit or use Python evaluation runner
echo -e "\n${YELLOW}Checking evaluation framework...${NC}"

USE_NEMO=false
USE_BIGCODE=false

if python3 -c "import nemo" 2>/dev/null; then
    echo -e "${GREEN}  NeMo toolkit available${NC}"
    USE_NEMO=true
elif python3 -c "import lm_eval" 2>/dev/null; then
    echo -e "${GREEN}  lm-evaluation-harness available${NC}"
    USE_BIGCODE=true
else
    echo -e "${YELLOW}  Using built-in Python evaluator${NC}"
fi

# Build evaluation command
if [ "$USE_DOCKER" = true ]; then
    echo -e "\n${YELLOW}Running in Docker...${NC}"
    EVAL_CMD="docker run --rm --network host"
    EVAL_CMD+=" -v $RUN_DIR:/outputs"
    EVAL_CMD+=" -e OPENAI_API_KEY=$API_KEY"
    EVAL_CMD+=" gptoss-ts-eval:latest"
    EVAL_CMD+=" python -m src.eval.nemo_eval_runner"
elif [ "$USE_NEMO" = true ]; then
    # NeMo Evaluator command
    EVAL_CMD="python3 -m src.eval.nemo_eval_runner"
else
    # Fallback to built-in evaluator
    EVAL_CMD="python3 -m src.eval.nemo_eval_runner"
fi

# Add arguments
EVAL_CMD+=" --model-name $MODEL_NAME"
EVAL_CMD+=" --base-url $BASE_URL"
EVAL_CMD+=" --api-key $API_KEY"
EVAL_CMD+=" --tasks $TASKS"
EVAL_CMD+=" --output-dir $RUN_DIR"

if [ -n "$NUM_SAMPLES" ]; then
    EVAL_CMD+=" --num-samples $NUM_SAMPLES"
fi

if [ -n "$TEMPERATURE" ]; then
    EVAL_CMD+=" --temperature $TEMPERATURE"
fi

if [ -n "$MAX_TOKENS" ]; then
    EVAL_CMD+=" --max-tokens $MAX_TOKENS"
fi

if [ -f "$CONFIG_FILE" ]; then
    EVAL_CMD+=" --config $CONFIG_FILE"
fi

echo -e "\n${YELLOW}Evaluation Command:${NC}"
echo "$EVAL_CMD"

if [ "$DRY_RUN" = true ]; then
    echo -e "\n${YELLOW}Dry run - not executing${NC}"
    exit 0
fi

# Save run configuration
cat > "$RUN_DIR/config.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "model_name": "$MODEL_NAME",
    "base_url": "$BASE_URL",
    "tasks": "$TASKS",
    "num_samples": "${NUM_SAMPLES:-null}",
    "temperature": "${TEMPERATURE:-null}",
    "max_tokens": "${MAX_TOKENS:-null}",
    "config_file": "$CONFIG_FILE"
}
EOF

echo -e "\n${GREEN}Starting evaluation...${NC}"
echo -e "Results will be saved to: $RUN_DIR"
echo ""

# Execute evaluation
set +e
eval $EVAL_CMD
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  Evaluation Complete${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Results saved to: $RUN_DIR"

    # Print summary if available
    if [ -f "$RUN_DIR/summary.json" ]; then
        echo -e "\n${YELLOW}Summary:${NC}"
        python3 -c "
import json
with open('$RUN_DIR/summary.json') as f:
    data = json.load(f)
    for task, results in data.get('results', {}).items():
        print(f'  {task}:')
        for metric, value in results.items():
            if isinstance(value, float):
                print(f'    {metric}: {value:.4f}')
            else:
                print(f'    {metric}: {value}')
" 2>/dev/null || true
    fi
else
    echo -e "\n${RED}Evaluation failed with exit code $EXIT_CODE${NC}"
    echo "Check logs in: $RUN_DIR"
    exit $EXIT_CODE
fi
