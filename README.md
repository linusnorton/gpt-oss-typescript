# Nemotron TypeScript Post-Training Harness

A reproducible post-training and evaluation harness for **NVIDIA Nemotron 3 Nano 30B-A3B Base BF16**, optimized for TypeScript/JavaScript code assistant agents. Designed for single-GPU training using QLoRA-based workflows.

## Overview

This project provides a complete pipeline for:

1. **Domain-Adaptive Pre-Training (DAPT)** - Continued pre-training on TypeScript/JavaScript corpora using next-token prediction
2. **Supervised Fine-Tuning (SFT)** - Instruction tuning using Nemotron-Agentic-v1 for tool use and agent trajectories
3. **Evaluation** - Benchmark evaluation via NeMo Evaluator and repo-based "patch tasks" for agent realism

### Why Nemotron 3 Nano 30B-A3B?

- **Mixture of Experts (MoE)**: 30B total parameters with only 3B active per forward pass
- **Efficient inference**: MoE architecture enables fast inference despite large parameter count
- **128K context**: Native support for long code files and multi-file context
- **Strong base**: Built on NVIDIA's Nemotron architecture with excellent code understanding

## Evaluation Layers

### A) Benchmark Evaluation

Uses NeMo Evaluator with BigCode harness tasks:
- **HumanEval** - Python function completion
- **MBPP** - Python programming problems
- **HumanEval-TS** - TypeScript function completion (when available)
- **MultiPL-E TypeScript** - Multi-language benchmark

### B) Repo-Based Patch Tasks

Real-world agent evaluation:
- Clone actual TypeScript repositories
- Apply model-generated patches
- Validate via compile, test, and lint checks
- Measure practical agent effectiveness

## Training Flows

### A) Domain-Adaptive Pre-Training (DAPT)

Continued pre-training on TypeScript/JavaScript code:
- Collects permissively-licensed repositories from GitHub
- Builds deduplicated, sanitized training corpus
- Uses next-token prediction objective
- QLoRA for memory-efficient single-GPU training

### B) Supervised Fine-Tuning (SFT)

Instruction tuning for agent capabilities:
- Uses Nemotron-Agentic-v1 dataset
- Trains on tool use and instruction trajectories
- ChatML format for conversation structure
- Optional TypeScript-specific filtering

### C) Preference Tuning (Future)

DPO/ORPO for alignment:
- Collect preference data from model outputs
- Train reward model or direct preference optimization
- Integrate with existing pipeline (documented in `configs/`)

## Quick Start

### Prerequisites

- Ubuntu 22.04+
- NVIDIA GPU with 48GB+ VRAM (80GB recommended)
- NVIDIA Driver 580+
- Python 3.11
- Node.js 24 (for repo-based evaluation)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/nemotron-ts-posttrain-harness.git
cd nemotron-ts-posttrain-harness

# Run setup (creates venv, installs deps, validates GPU)
make setup

# Or manually:
cp .env.example .env
# Edit .env with your GITHUB_PAT and HF_TOKEN
source .venv/bin/activate
```

### Collect System Information

```bash
make system-info
# Results saved to outputs/system_info.json
```

### Full Workflow

```bash
# 1. Collect training data
make collect-repos    # Collect GitHub repositories
make build-corpus     # Build training corpus

# 2. Train
make train-dapt       # Domain-adaptive pre-training
make train-sft        # Supervised fine-tuning

# 3. Serve and evaluate
make serve-vllm       # Start inference server
make eval             # Run benchmarks
```

## CUDA Compatibility

### Understanding CUDA Versions

There are three distinct CUDA-related versions to consider:

| Component | Description | How to Check |
|-----------|-------------|--------------|
| **Driver CUDA** | Maximum CUDA version supported by your NVIDIA driver | `nvidia-smi` header |
| **Toolkit (nvcc)** | Locally installed CUDA development toolkit | `nvcc --version` |
| **PyTorch CUDA** | CUDA version bundled with PyTorch wheels | `python -c "import torch; print(torch.version.cuda)"` |

**Key insight**: You don't need nvcc installed locally. PyTorch and vLLM ship with their own CUDA runtime. The driver just needs to support a compatible CUDA version.

### Compatibility Matrix

| Driver Version | Max CUDA | Recommended PyTorch | vLLM Support |
|----------------|----------|---------------------|--------------|
| 580.x          | 12.8     | 2.4+ (CUDA 12.4)    | 0.6+         |
| 550.x          | 12.4     | 2.4+ (CUDA 12.4)    | 0.6+         |
| 535.x          | 12.2     | 2.2+ (CUDA 12.1)    | 0.5+         |
| 525.x          | 12.0     | 2.1+ (CUDA 11.8)    | 0.4+         |

### Docker CUDA Configuration

All Dockerfiles accept a `CUDA_VERSION` build argument:

```bash
# Default (CUDA 12.4.1)
make docker-build

# Custom CUDA version
make docker-build CUDA_VERSION=12.1.1

# Or directly
docker build --build-arg CUDA_VERSION=12.1.1 -f docker/Dockerfile.train .
```

### Detecting Your System

```bash
# Run system info collection
./scripts/system_info.sh

# View results
cat outputs/system_info.json | jq '.nvidia'
```

## Repository Structure

```
nemotron-ts-posttrain-harness/
├── README.md                 # This file
├── .gitignore
├── .env.example              # Environment template
├── Makefile                  # Common tasks
├── pyproject.toml            # Python dependencies
│
├── scripts/
│   ├── setup_env.sh          # Environment setup
│   ├── system_info.sh        # System information collection
│   ├── serve_vllm.sh         # vLLM inference server
│   ├── run_nemo_eval.sh      # Evaluation runner
│   ├── collect_github_repos.py
│   ├── build_dapt_corpus.py
│   ├── run_dapt_train.py
│   └── apply_agentic_sft_dataset.py
│
├── configs/
│   ├── github_collection.yaml
│   ├── dapt_train.yaml
│   ├── sft_data.yaml
│   ├── nemo_eval.yaml
│   └── repo_tasks.yaml
│
├── docker/
│   ├── Dockerfile.train
│   ├── Dockerfile.eval
│   ├── Dockerfile.vllm
│   └── docker-compose.yml
│
├── src/
│   ├── common/               # Shared utilities
│   │   ├── logging.py
│   │   ├── paths.py
│   │   └── subprocess_utils.py
│   │
│   ├── eval/                 # Evaluation
│   │   └── nemo_eval_runner.py
│   │
│   ├── data/                 # Data processing
│   │   ├── hf_download.py
│   │   ├── github/
│   │   │   ├── github_client.py
│   │   │   └── repo_filtering.py
│   │   └── corpus/
│   │       ├── pack_jsonl.py
│   │       ├── dedupe.py
│   │       ├── sanitize.py
│   │       └── chunking.py
│   │
│   └── train/                # Training
│       ├── dapt/
│       │   ├── data_collator.py
│       │   └── train_hf_peft.py
│       └── sft/
│           └── format_agentic.py
│
├── tests/
│   ├── test_corpus_builder.py
│   └── test_github_collector.py
│
├── data/                     # Training data (gitignored)
└── outputs/                  # Checkpoints, logs (gitignored)
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required
GITHUB_PAT=ghp_...           # GitHub token for repo collection
HF_TOKEN=hf_...              # HuggingFace token for gated models

# Paths
DATA_DIR=./data
OUTPUT_DIR=./outputs

# Model
MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16

# Training
TRAIN_BATCH_SIZE=1
TRAIN_GRADIENT_ACCUMULATION_STEPS=8
TRAIN_LEARNING_RATE=2e-5
QLORA_R=64
QLORA_ALPHA=16
```

### Training Configuration

Edit `configs/dapt_train.yaml`:

```yaml
# LoRA settings
lora:
  r: 64                    # Rank (higher = more capacity, more memory)
  alpha: 16                # Scaling factor
  dropout: 0.05

# Training
training:
  batch_size: 1
  gradient_accumulation_steps: 8  # Effective batch = 8
  learning_rate: 2.0e-5
  num_epochs: 3
```

## Serving with vLLM

### Basic Usage

```bash
# Serve base model
./scripts/serve_vllm.sh --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16

# Serve with LoRA adapter
./scripts/serve_vllm.sh \
  --model ./outputs/checkpoints/dapt/final \
  --lora ./outputs/checkpoints/sft/final

# Reduced memory usage
./scripts/serve_vllm.sh \
  --gpu-memory-util 0.8 \
  --max-model-len 4096
```

### vLLM Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | From .env | Model ID or path |
| `--lora` | - | LoRA adapter path |
| `--dtype` | bfloat16 | Data type |
| `--gpu-memory-util` | 0.90 | GPU memory fraction |
| `--max-model-len` | 8192 | Maximum sequence length |
| `--port` | 8000 | Server port |
| `--tensor-parallel` | 1 | Number of GPUs |

### Nemotron MoE Notes

Nemotron 3 Nano 30B-A3B is a Mixture of Experts model:
- Total parameters: 30B
- Active parameters: 3B per forward pass
- Memory requirement: ~60GB in BF16

For GPUs with less than 60GB VRAM:
- Use `--quantization awq` or `--quantization gptq` (~20-30GB)
- Use `--tensor-parallel 2` across multiple GPUs

## Docker Workflow

### Build Images

```bash
# Build all images
make docker-build

# Or individually
make docker-build-train
make docker-build-vllm
make docker-build-eval
```

### Run Services

```bash
# Start vLLM server
docker compose -f docker/docker-compose.yml up -d vllm

# Run evaluation
docker compose -f docker/docker-compose.yml run eval

# Run training
docker compose -f docker/docker-compose.yml --profile training run train

# Collect data
docker compose -f docker/docker-compose.yml --profile data run collect
```

## Data Collection

### License Filtering

Only permissively-licensed code is collected:

| License | SPDX ID | Status |
|---------|---------|--------|
| MIT | `mit` | Allowed |
| Apache 2.0 | `apache-2.0` | Allowed |
| BSD 2-Clause | `bsd-2-clause` | Allowed |
| BSD 3-Clause | `bsd-3-clause` | Allowed |
| ISC | `isc` | Allowed |
| Unlicense | `unlicense` | Allowed |
| CC0 | `cc0-1.0` | Allowed |
| GPL | `gpl-*` | **Excluded** |
| LGPL | `lgpl-*` | **Excluded** |
| AGPL | `agpl-*` | **Excluded** |

Configure in `configs/github_collection.yaml`.

### Sanitization

The corpus pipeline removes:
- API keys and tokens
- Private keys
- Passwords in config files
- Email addresses (optional)
- Database connection strings

See `src/data/corpus/sanitize.py` for patterns.

## Development

### Running Tests

```bash
make test           # Run all tests
make test-cov       # Run with coverage
```

### Code Quality

```bash
make lint           # Check code style
make format         # Auto-format code
make typecheck      # Run mypy
```

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce `--gpu-memory-util` to 0.8 or lower
- Reduce `--max-model-len`
- Use `--quantization awq`

**"Model not found" for Nemotron**
- Ensure `HF_TOKEN` is set
- Accept model license on HuggingFace website
- Check: `huggingface-cli whoami`

**vLLM fails to start**
- Check vLLM version: `pip show vllm`
- Ensure CUDA compatibility
- Try: `pip install -U vllm`

**GitHub API rate limit**
- Reduce `max_repos` in config
- Use authenticated requests (set `GITHUB_PAT`)

### Getting Help

1. Check `outputs/logs/` for detailed logs
2. Run `make system-info` and include in bug reports
3. Open an issue with reproduction steps

## Reproducibility

This project emphasizes reproducibility:

- **Pinned dependencies**: `pyproject.toml` with version constraints
- **Docker images**: Consistent environments across machines
- **Configuration files**: All parameters in version-controlled YAML
- **System info collection**: Document exact hardware/software setup
- **Random seeds**: Configurable in all scripts

## License

This project is licensed under the Apache License 2.0.

**Note**: The Nemotron model has its own license. Check the model card on HuggingFace before use.

## Acknowledgments

- NVIDIA for Nemotron models and NeMo toolkit
- HuggingFace for Transformers and PEFT libraries
- vLLM team for efficient serving infrastructure
- BigCode for evaluation benchmarks
