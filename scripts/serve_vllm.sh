#!/usr/bin/env bash
# serve_vllm.sh - Start vLLM OpenAI-compatible server
# Optimized for GPT-OSS-20B and fine-tuned models with LoRA adapters

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

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
DEFAULT_MODEL="${MODEL_ID:-unsloth/gpt-oss-20b}"
DEFAULT_PORT="${VLLM_PORT:-8000}"
DEFAULT_HOST="${VLLM_HOST:-0.0.0.0}"
DEFAULT_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
DEFAULT_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
DEFAULT_DTYPE="auto"
DEFAULT_SERVED_NAME="${VLLM_SERVED_NAME:-gpt-oss}"

# Parse arguments
MODEL=""
LORA_PATH=""
DTYPE="$DEFAULT_DTYPE"
GPU_MEMORY_UTIL="$DEFAULT_GPU_MEMORY_UTIL"
MAX_MODEL_LEN="$DEFAULT_MAX_MODEL_LEN"
PORT="$DEFAULT_PORT"
HOST="$DEFAULT_HOST"
TENSOR_PARALLEL_SIZE=""
TRUST_REMOTE_CODE=true
QUANTIZATION=""
ENABLE_PREFIX_CACHING=true
SERVED_NAME="$DEFAULT_SERVED_NAME"
# GPT-OSS uses the Harmony format, which requires the 'openai' tool parser (vLLM >= 0.10.2)
ENABLE_TOOL_CHOICE="${VLLM_ENABLE_TOOL_CHOICE:-true}"
TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-openai}"
REASONING_PARSER="${VLLM_REASONING_PARSER:-}"
DRY_RUN=false

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Start a vLLM OpenAI-compatible server for GPT-OSS or fine-tuned models."
    echo ""
    echo "Options:"
    echo "  --model PATH          Model ID or local path (default: $DEFAULT_MODEL)"
    echo "  --lora PATH           Path to LoRA adapter (optional)"
    echo "  --dtype TYPE          Data type: auto, bfloat16, float16 (default: $DEFAULT_DTYPE)"
    echo "  --gpu-memory-util N   GPU memory utilization 0-1 (default: $DEFAULT_GPU_MEMORY_UTIL)"
    echo "  --max-model-len N     Maximum sequence length (default: $DEFAULT_MAX_MODEL_LEN)"
    echo "  --port PORT           Server port (default: $DEFAULT_PORT)"
    echo "  --host HOST           Server host (default: $DEFAULT_HOST)"
    echo "  --served-name NAME    Model name in API (default: $DEFAULT_SERVED_NAME)"
    echo "  --tensor-parallel N   Number of GPUs for tensor parallelism"
    echo "  --quantization TYPE   Quantization: awq, gptq, squeezellm, fp8 (optional)"
    echo "  --tool-call-parser P  Tool call parser: openai (for gpt-oss), hermes, llama3_json, etc."
    echo "  --reasoning-parser P  Reasoning parser: deepseek_r1 (optional)"
    echo "  --no-tool-choice      Disable auto tool choice"
    echo "  --no-prefix-caching   Disable prefix caching"
    echo "  --dry-run             Print command without executing"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Serve GPT-OSS base model"
    echo "  $0 --model unsloth/gpt-oss-20b"
    echo ""
    echo "  # Serve merged fine-tuned model"
    echo "  $0 --model ./outputs/checkpoints/dapt/merged"
    echo ""
    echo "  # Serve with LoRA adapter"
    echo "  $0 --model unsloth/gpt-oss-20b --lora ./outputs/checkpoints/lora"
    echo ""
    echo "  # Serve with reduced memory"
    echo "  $0 --gpu-memory-util 0.8 --max-model-len 2048"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --lora)
            LORA_PATH="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --gpu-memory-util)
            GPU_MEMORY_UTIL="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --tensor-parallel)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --served-name)
            SERVED_NAME="$2"
            shift 2
            ;;
        --tool-call-parser)
            TOOL_CALL_PARSER="$2"
            shift 2
            ;;
        --reasoning-parser)
            REASONING_PARSER="$2"
            shift 2
            ;;
        --no-tool-choice)
            ENABLE_TOOL_CHOICE=false
            shift
            ;;
        --no-prefix-caching)
            ENABLE_PREFIX_CACHING=false
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

# Use default model if not specified
MODEL="${MODEL:-$DEFAULT_MODEL}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  vLLM Server for GPT-OSS${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create log directory
LOG_DIR="$PROJECT_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_$(date +%Y%m%d_%H%M%S).log"

# Check vLLM installation
echo -e "${YELLOW}Checking vLLM installation...${NC}"
if ! python -c "import vllm" 2>/dev/null; then
    echo -e "${RED}Error: vLLM is not installed${NC}"
    echo "Install with: pip install vllm"
    echo "Or use: make install-vllm"
    exit 1
fi
VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)")
echo -e "${GREEN}  vLLM version: $VLLM_VERSION${NC}"

# Check GPU availability
echo -e "\n${YELLOW}Checking GPU availability...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found - no GPU available${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "${GREEN}  Found $GPU_COUNT GPU(s)${NC}"

# Auto-detect tensor parallel size if not specified
if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    TENSOR_PARALLEL_SIZE=1
    # GPT-OSS-20B fits comfortably on 16GB+ VRAM
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    echo -e "${GREEN}  GPU memory: ${GPU_MEMORY}MB${NC}"
fi

# Check model compatibility and provide info
echo -e "\n${YELLOW}Checking model compatibility...${NC}"
check_model_support() {
    local model_name="$1"

    # GPT-OSS-20B detection
    if [[ "$model_name" == *"gpt-oss"* ]]; then
        echo -e "${GREEN}  Detected: GPT-OSS-20B (OpenAI)${NC}"
        echo -e "  Architecture: 21B total params, 3.6B active (MoE)"
        echo -e "  Context: 4K tokens (default)"
        echo -e "  Estimated VRAM: ~16GB"
        return 0
    fi

    # Unsloth model detection
    if [[ "$model_name" == *"unsloth"* ]]; then
        echo -e "${GREEN}  Detected: Unsloth model${NC}"
        echo -e "  Optimized for efficient inference"
        return 0
    fi

    echo -e "${GREEN}  Model: $model_name${NC}"
    return 0
}

check_model_support "$MODEL"

# Verify model access
echo -e "\n${YELLOW}Verifying model access...${NC}"
if [[ "$MODEL" == /* ]] || [[ "$MODEL" == ./* ]]; then
    # Local path
    if [ ! -d "$MODEL" ]; then
        echo -e "${RED}Error: Local model path not found: $MODEL${NC}"
        exit 1
    fi
    echo -e "${GREEN}  Local model: $MODEL${NC}"
else
    # HuggingFace model - check if we can access it
    echo -e "  HuggingFace model: $MODEL"
    if [ -n "${HF_TOKEN:-}" ]; then
        echo -e "${GREEN}  HF_TOKEN is set${NC}"
    else
        echo -e "${YELLOW}  Warning: HF_TOKEN not set - may fail for gated models${NC}"
    fi
fi


# Build vLLM command
VLLM_CMD="python -m vllm.entrypoints.openai.api_server"
VLLM_CMD+=" --model $MODEL"
VLLM_CMD+=" --host $HOST"
VLLM_CMD+=" --port $PORT"
VLLM_CMD+=" --dtype $DTYPE"
VLLM_CMD+=" --gpu-memory-utilization $GPU_MEMORY_UTIL"
VLLM_CMD+=" --max-model-len $MAX_MODEL_LEN"

if [ "$TENSOR_PARALLEL_SIZE" -gt 1 ]; then
    VLLM_CMD+=" --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
fi

if [ "$TRUST_REMOTE_CODE" = true ]; then
    VLLM_CMD+=" --trust-remote-code"
fi

if [ "$ENABLE_PREFIX_CACHING" = true ]; then
    VLLM_CMD+=" --enable-prefix-caching"
fi

if [ -n "$LORA_PATH" ]; then
    if [ ! -d "$LORA_PATH" ]; then
        echo -e "${RED}Error: LoRA path not found: $LORA_PATH${NC}"
        exit 1
    fi
    VLLM_CMD+=" --enable-lora"
    VLLM_CMD+=" --lora-modules gptoss-lora=$LORA_PATH"
fi

if [ -n "$QUANTIZATION" ]; then
    VLLM_CMD+=" --quantization $QUANTIZATION"
fi

# Tool calling options (useful for agent workflows)
if [ "$ENABLE_TOOL_CHOICE" = true ]; then
    VLLM_CMD+=" --enable-auto-tool-choice"
    VLLM_CMD+=" --tool-call-parser $TOOL_CALL_PARSER"
fi

if [ -n "$REASONING_PARSER" ]; then
    VLLM_CMD+=" --reasoning-parser $REASONING_PARSER"
fi

# Add served model name for API compatibility
VLLM_CMD+=" --served-model-name $SERVED_NAME"

echo -e "\n${YELLOW}vLLM Command:${NC}"
echo "$VLLM_CMD"

if [ "$DRY_RUN" = true ]; then
    echo -e "\n${YELLOW}Dry run - not starting server${NC}"
    exit 0
fi

echo -e "\n${GREEN}Starting vLLM server...${NC}"
echo -e "  API endpoint: http://$HOST:$PORT/v1"
echo -e "  Log file: $LOG_FILE"
echo -e "  Press Ctrl+C to stop"
echo ""

# Start server with logging
exec $VLLM_CMD 2>&1 | tee "$LOG_FILE"
