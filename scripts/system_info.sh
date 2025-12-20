#!/usr/bin/env bash
# system_info.sh - Collect system information for reproducibility
# Outputs: ./outputs/system_info.json and ./outputs/system_info.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  System Information Collection${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Initialize JSON structure
JSON_OUTPUT="{"

# Function to escape JSON strings
escape_json() {
    echo -n "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

# Collect timestamp
TIMESTAMP=$(date -Iseconds)
JSON_OUTPUT+="\"timestamp\": \"$TIMESTAMP\","
echo -e "${GREEN}Timestamp:${NC} $TIMESTAMP"

# Collect OS information
echo -e "\n${YELLOW}--- OS Information ---${NC}"
UNAME_OUTPUT=$(uname -a)
echo "$UNAME_OUTPUT"

if [ -f /etc/os-release ]; then
    OS_RELEASE=$(cat /etc/os-release)
    OS_NAME=$(grep "^NAME=" /etc/os-release | cut -d'"' -f2 2>/dev/null || echo "Unknown")
    OS_VERSION=$(grep "^VERSION=" /etc/os-release | cut -d'"' -f2 2>/dev/null || echo "Unknown")
else
    OS_RELEASE=""
    OS_NAME="Unknown"
    OS_VERSION="Unknown"
fi

JSON_OUTPUT+="\"os\": {"
JSON_OUTPUT+="\"uname\": $(escape_json "$UNAME_OUTPUT"),"
JSON_OUTPUT+="\"name\": \"$OS_NAME\","
JSON_OUTPUT+="\"version\": \"$OS_VERSION\""
JSON_OUTPUT+="},"

echo "OS Name: $OS_NAME"
echo "OS Version: $OS_VERSION"

# Collect Python version
echo -e "\n${YELLOW}--- Python Version ---${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    PYTHON_PATH=$(which python3)
    echo "$PYTHON_VERSION"
    echo "Path: $PYTHON_PATH"
    JSON_OUTPUT+="\"python\": {"
    JSON_OUTPUT+="\"version\": $(escape_json "$PYTHON_VERSION"),"
    JSON_OUTPUT+="\"path\": \"$PYTHON_PATH\""
    JSON_OUTPUT+="},"
else
    echo -e "${RED}Python 3 not found${NC}"
    JSON_OUTPUT+="\"python\": null,"
fi

# Collect Node.js version
echo -e "\n${YELLOW}--- Node.js Version ---${NC}"
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version 2>&1)
    NODE_PATH=$(which node)
    echo "$NODE_VERSION"
    echo "Path: $NODE_PATH"
    JSON_OUTPUT+="\"node\": {"
    JSON_OUTPUT+="\"version\": \"$NODE_VERSION\","
    JSON_OUTPUT+="\"path\": \"$NODE_PATH\""
    JSON_OUTPUT+="},"
else
    echo -e "${YELLOW}Node.js not installed${NC}"
    JSON_OUTPUT+="\"node\": null,"
fi

# Collect NVIDIA GPU information
echo -e "\n${YELLOW}--- NVIDIA GPU Information ---${NC}"
JSON_OUTPUT+="\"nvidia\": {"

if command -v nvidia-smi &> /dev/null; then
    # Full nvidia-smi output
    NVIDIA_SMI_FULL=$(nvidia-smi 2>&1)
    echo "$NVIDIA_SMI_FULL"

    # Extract driver version
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo "Unknown")
    echo -e "\n${GREEN}Driver Version:${NC} $DRIVER_VERSION"

    # Extract CUDA version from nvidia-smi header
    CUDA_VERSION_SMI=$(nvidia-smi 2>&1 | grep -oP "CUDA Version: \K[0-9.]+" || echo "Unknown")
    echo -e "${GREEN}CUDA Version (driver-supported):${NC} $CUDA_VERSION_SMI"

    # Extract GPU names
    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ';' | sed 's/;$//' || echo "Unknown")
    echo -e "${GREEN}GPU(s):${NC} $GPU_NAMES"

    # Extract GPU memory
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | tr '\n' ';' | sed 's/;$//' || echo "Unknown")
    echo -e "${GREEN}GPU Memory:${NC} $GPU_MEMORY"

    JSON_OUTPUT+="\"available\": true,"
    JSON_OUTPUT+="\"driver_version\": \"$DRIVER_VERSION\","
    JSON_OUTPUT+="\"cuda_version_driver\": \"$CUDA_VERSION_SMI\","
    JSON_OUTPUT+="\"gpus\": \"$GPU_NAMES\","
    JSON_OUTPUT+="\"gpu_memory\": \"$GPU_MEMORY\","
    JSON_OUTPUT+="\"nvidia_smi_full\": $(escape_json "$NVIDIA_SMI_FULL")"
else
    echo -e "${RED}nvidia-smi not found - no NVIDIA GPU detected${NC}"
    JSON_OUTPUT+="\"available\": false,"
    JSON_OUTPUT+="\"error\": \"nvidia-smi not found\""
fi

JSON_OUTPUT+="},"

# Collect CUDA Toolkit (nvcc) information
echo -e "\n${YELLOW}--- CUDA Toolkit (nvcc) ---${NC}"
JSON_OUTPUT+="\"cuda_toolkit\": {"

if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version 2>&1)
    NVCC_PATH=$(which nvcc)
    echo "$NVCC_VERSION"
    echo "Path: $NVCC_PATH"

    # Extract version number
    CUDA_TOOLKIT_VERSION=$(echo "$NVCC_VERSION" | grep -oP "release \K[0-9.]+" || echo "Unknown")
    echo -e "${GREEN}CUDA Toolkit Version:${NC} $CUDA_TOOLKIT_VERSION"

    JSON_OUTPUT+="\"installed\": true,"
    JSON_OUTPUT+="\"version\": \"$CUDA_TOOLKIT_VERSION\","
    JSON_OUTPUT+="\"path\": \"$NVCC_PATH\","
    JSON_OUTPUT+="\"nvcc_full\": $(escape_json "$NVCC_VERSION")"
else
    echo -e "${YELLOW}nvcc not found - CUDA toolkit not installed locally${NC}"
    echo -e "This is OK if using containerized workflows or PyTorch with bundled CUDA"
    JSON_OUTPUT+="\"installed\": false,"
    JSON_OUTPUT+="\"note\": \"CUDA toolkit not installed locally - will use PyTorch bundled CUDA\""
fi

JSON_OUTPUT+="},"

# Check for PyTorch CUDA availability (if Python available)
echo -e "\n${YELLOW}--- PyTorch CUDA Status ---${NC}"
JSON_OUTPUT+="\"pytorch\": {"

if command -v python3 &> /dev/null; then
    PYTORCH_INFO=$(python3 -c "
import json
try:
    import torch
    info = {
        'installed': True,
        'version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    if torch.cuda.is_available():
        info['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(json.dumps(info))
except ImportError:
    print(json.dumps({'installed': False, 'error': 'PyTorch not installed'}))
except Exception as e:
    print(json.dumps({'installed': False, 'error': str(e)}))
" 2>/dev/null || echo '{"installed": false, "error": "Failed to check PyTorch"}')

    echo "$PYTORCH_INFO" | python3 -m json.tool 2>/dev/null || echo "$PYTORCH_INFO"

    # Remove the outer braces for embedding
    PYTORCH_JSON=$(echo "$PYTORCH_INFO" | sed 's/^{//' | sed 's/}$//')
    JSON_OUTPUT+="$PYTORCH_JSON"
else
    JSON_OUTPUT+="\"installed\": false"
fi

JSON_OUTPUT+="}"

# Close JSON
JSON_OUTPUT+="}"

# Write JSON output
echo "$JSON_OUTPUT" | python3 -m json.tool > "$OUTPUT_DIR/system_info.json" 2>/dev/null || echo "$JSON_OUTPUT" > "$OUTPUT_DIR/system_info.json"
echo -e "\n${GREEN}JSON output written to:${NC} $OUTPUT_DIR/system_info.json"

# Write text output
{
    echo "=========================================="
    echo "  System Information Report"
    echo "  Generated: $TIMESTAMP"
    echo "=========================================="
    echo ""
    echo "OS Information:"
    echo "  Name: $OS_NAME"
    echo "  Version: $OS_VERSION"
    echo "  Kernel: $UNAME_OUTPUT"
    echo ""
    echo "Python:"
    if command -v python3 &> /dev/null; then
        echo "  Version: $PYTHON_VERSION"
        echo "  Path: $PYTHON_PATH"
    else
        echo "  Not installed"
    fi
    echo ""
    echo "Node.js:"
    if command -v node &> /dev/null; then
        echo "  Version: $NODE_VERSION"
        echo "  Path: $NODE_PATH"
    else
        echo "  Not installed"
    fi
    echo ""
    echo "NVIDIA GPU:"
    if command -v nvidia-smi &> /dev/null; then
        echo "  Driver Version: $DRIVER_VERSION"
        echo "  CUDA Version (driver-supported): $CUDA_VERSION_SMI"
        echo "  GPU(s): $GPU_NAMES"
        echo "  GPU Memory: $GPU_MEMORY"
    else
        echo "  Not available"
    fi
    echo ""
    echo "CUDA Toolkit (nvcc):"
    if command -v nvcc &> /dev/null; then
        echo "  Version: $CUDA_TOOLKIT_VERSION"
        echo "  Path: $NVCC_PATH"
    else
        echo "  Not installed (using PyTorch bundled CUDA)"
    fi
    echo ""
    echo "=========================================="
} > "$OUTPUT_DIR/system_info.txt"

echo -e "${GREEN}Text output written to:${NC} $OUTPUT_DIR/system_info.txt"

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Collection Complete${NC}"
echo -e "${BLUE}========================================${NC}"
