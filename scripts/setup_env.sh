#!/usr/bin/env bash
# setup_env.sh - Set up the development environment
# Creates virtual environment, installs dependencies, validates GPU

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Environment Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

cd "$PROJECT_ROOT"

# Configuration
PYTHON_VERSION="${PYTHON_VERSION:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-dev}"  # Options: dev, vllm, nemo, all

# Check for required tools
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Python
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo -e "${RED}Error: $PYTHON_VERSION not found${NC}"
    echo "Please install Python 3.11 or set PYTHON_VERSION environment variable"
    exit 1
fi
PYTHON_ACTUAL=$($PYTHON_VERSION --version)
echo -e "${GREEN}  Python:${NC} $PYTHON_ACTUAL"

# Check for uv or pip
USE_UV=false
if command -v uv &> /dev/null; then
    USE_UV=true
    UV_VERSION=$(uv --version)
    echo -e "${GREEN}  uv:${NC} $UV_VERSION"
else
    echo -e "${YELLOW}  uv: not found (will use pip)${NC}"
fi

# Check for git
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version)
    echo -e "${GREEN}  Git:${NC} $GIT_VERSION"
else
    echo -e "${YELLOW}  Git: not found${NC}"
fi

# Check for NVIDIA GPU
echo -e "\n${YELLOW}Checking GPU availability...${NC}"
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -n1)
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
        CUDA_VERSION=$(nvidia-smi 2>&1 | grep -oP "CUDA Version: \K[0-9.]+" || echo "Unknown")
        echo -e "${GREEN}  GPU:${NC} $GPU_NAME"
        echo -e "${GREEN}  Memory:${NC} $GPU_MEMORY"
        echo -e "${GREEN}  Driver:${NC} $DRIVER_VERSION"
        echo -e "${GREEN}  CUDA (driver):${NC} $CUDA_VERSION"
    else
        echo -e "${RED}  nvidia-smi failed - GPU not accessible${NC}"
    fi
else
    echo -e "${RED}  No NVIDIA GPU detected (nvidia-smi not found)${NC}"
fi

if [ "$GPU_AVAILABLE" = false ]; then
    echo -e "${YELLOW}Warning: No GPU detected. Training and inference will be slow or may fail.${NC}"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create .env if it doesn't exist
echo -e "\n${YELLOW}Setting up environment files...${NC}"
if [ ! -f .env ]; then
    echo -e "  Creating .env from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}  Please edit .env with your credentials (GITHUB_PAT, HF_TOKEN)${NC}"
else
    echo -e "${GREEN}  .env already exists${NC}"
fi

# Create output directories
echo -e "\n${YELLOW}Creating directories...${NC}"
mkdir -p data outputs/logs outputs/checkpoints outputs/eval
echo -e "${GREEN}  Created: data/, outputs/{logs,checkpoints,eval}/${NC}"

# Create .gitkeep files
touch data/.gitkeep

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ "$USE_UV" = true ]; then
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
else
    $PYTHON_VERSION -m venv "$VENV_DIR"
fi
echo -e "${GREEN}  Created: $VENV_DIR${NC}"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
if [ "$USE_UV" = true ]; then
    case "$INSTALL_EXTRAS" in
        all)
            uv pip install -e ".[dev,vllm,nemo]"
            ;;
        vllm)
            uv pip install -e ".[dev,vllm]"
            ;;
        nemo)
            uv pip install -e ".[dev,nemo]"
            ;;
        *)
            uv pip install -e ".[dev]"
            ;;
    esac
else
    pip install --upgrade pip wheel setuptools
    case "$INSTALL_EXTRAS" in
        all)
            pip install -e ".[dev,vllm,nemo]"
            ;;
        vllm)
            pip install -e ".[dev,vllm]"
            ;;
        nemo)
            pip install -e ".[dev,nemo]"
            ;;
        *)
            pip install -e ".[dev]"
            ;;
    esac
fi
echo -e "${GREEN}  Dependencies installed${NC}"

# Verify PyTorch CUDA
echo -e "\n${YELLOW}Verifying PyTorch CUDA...${NC}"
python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  cuDNN version: {torch.backends.cudnn.version()}')
    print(f'  Device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
else:
    print('  Warning: CUDA not available in PyTorch')
"

# Set up pre-commit hooks
echo -e "\n${YELLOW}Setting up pre-commit hooks...${NC}"
if command -v pre-commit &> /dev/null || [ -f "$VENV_DIR/bin/pre-commit" ]; then
    "$VENV_DIR/bin/pre-commit" install 2>/dev/null || echo "  Pre-commit hooks not configured (no .pre-commit-config.yaml)"
else
    echo "  Pre-commit not installed"
fi

# Collect system info
echo -e "\n${YELLOW}Collecting system information...${NC}"
bash "$SCRIPT_DIR/system_info.sh" > /dev/null 2>&1
echo -e "${GREEN}  System info saved to outputs/system_info.json${NC}"

# Print next steps
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo ""
echo "1. Activate the virtual environment:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo "2. Edit .env with your credentials:"
echo "   - GITHUB_PAT: GitHub Personal Access Token"
echo "   - HF_TOKEN: Hugging Face Token"
echo ""
echo "3. Collect training data:"
echo "   make collect-repos"
echo "   make build-corpus"
echo ""
echo "4. Run training:"
echo "   make train-dapt   # Domain-adaptive pre-training"
echo "   make train-sft    # Supervised fine-tuning"
echo ""
echo "5. Serve and evaluate:"
echo "   make serve-vllm   # Start inference server"
echo "   make eval         # Run evaluations"
echo ""
echo -e "${YELLOW}For Docker workflows:${NC}"
echo "   make docker-build  # Build all images"
echo "   make docker-up     # Start services"
echo ""
echo -e "${BLUE}See README.md for detailed documentation.${NC}"
