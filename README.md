# GPT-OSS TypeScript Post-Training Harness

A reproducible post-training and evaluation harness for **OpenAI GPT-OSS-20B** using Unsloth, optimized for TypeScript/JavaScript code assistant agents. Designed for single-GPU training using QLoRA workflows.

## Overview

This project provides a complete pipeline for:

1. **Domain-Adaptive Pre-Training (DAPT)** - Continued pre-training on TypeScript/JavaScript corpora using next-token prediction
2. **Supervised Fine-Tuning (SFT)** - Instruction tuning for tool use and agent trajectories
3. **Evaluation** - Benchmark evaluation via HumanEval/MBPP and repo-based "patch tasks" for agent realism

### Why GPT-OSS-20B?

- **Mixture of Experts (MoE)**: 21B total parameters with only 3.6B active per forward pass
- **Efficient training**: Fits in ~14GB VRAM with Unsloth QLoRA
- **Fast inference**: ~16GB VRAM, 30-50 tokens/second without CPU offloading
- **Apache 2.0**: Permissive open-source license
- **Strong base**: Built on OpenAI's architecture with excellent code understanding

## Evaluation Layers

### A) Code Benchmark Evaluation

TypeScript code generation benchmarks (via MultiPL-E):
- **HumanEval-TS** - TypeScript function completion
- **MBPP-TS** - TypeScript programming problems

### B) SWE-bench TypeScript

Real-world agent evaluation using Multi-SWE-bench TypeScript repositories:
- Clone actual TypeScript repositories (DarkReader, Material-UI, Vue.js)
- Agent-based bug fixing with tool use (bash, file editing)
- Validate patches against gold solutions
- Measure practical agent effectiveness on real issues

## Training Flows

### A) Domain-Adaptive Pre-Training (DAPT)

Continued pre-training on TypeScript/JavaScript code:
- Collects permissively-licensed repositories from GitHub
- Builds deduplicated, sanitized training corpus
- Uses next-token prediction objective
- QLoRA via Unsloth for memory-efficient single-GPU training

### B) Supervised Fine-Tuning (SFT)

Instruction tuning for agent capabilities:
- Uses agentic instruction datasets
- Trains on tool use and instruction trajectories
- ChatML format for conversation structure
- Optional TypeScript-specific filtering

## Quick Start

### Prerequisites

- Ubuntu 22.04+ (24.04 recommended)
- NVIDIA GPU with 16GB+ VRAM (24GB+ recommended for larger context)
  - RTX 4090/4080/3090: Well supported
  - RTX 5090/5080 (Blackwell): Requires CUDA 13.0+
- Python 3.11+
- Node.js 24 (for repo-based evaluation)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/gptoss-ts-posttrain-harness.git
cd gptoss-ts-posttrain-harness

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

## Memory Requirements

### Training (QLoRA via Unsloth)

| Configuration | VRAM Required | Notes |
|---------------|---------------|-------|
| QLoRA (r=64) | ~14GB | Default, fits RTX 4090 |
| QLoRA (r=128) | ~18GB | Higher capacity |
| Full LoRA (BF16) | ~44GB | Requires A100/H100 |

### Inference (vLLM)

| Configuration | VRAM Required | Notes |
|---------------|---------------|-------|
| Default | ~16GB | 4K context |
| Extended context | ~20GB | 8K context |

## Repository Structure

```
gptoss-ts-posttrain-harness/
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
│       │   └── train_unsloth.py
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
MODEL_ID=unsloth/gpt-oss-20b

# Training (via Unsloth)
TRAIN_BATCH_SIZE=2
TRAIN_GRADIENT_ACCUMULATION_STEPS=4
TRAIN_LEARNING_RATE=2e-4
QLORA_R=64
QLORA_ALPHA=16
```

### Training Configuration

Edit `configs/dapt_train.yaml`:

```yaml
# Model (Unsloth's gpt-oss-20b for QLoRA)
model_id: "unsloth/gpt-oss-20b"

# LoRA settings
lora:
  r: 64                    # Rank (higher = more capacity, more memory)
  alpha: 16                # Scaling factor
  dropout: 0.0             # Unsloth optimized (0 dropout)

# Training
training:
  batch_size: 2
  gradient_accumulation_steps: 4  # Effective batch = 8
  learning_rate: 2.0e-4
  num_epochs: 3
```

## Serving with vLLM

### Basic Usage

```bash
# Serve base model
./scripts/serve_vllm.sh --model unsloth/gpt-oss-20b

# Serve merged fine-tuned model
./scripts/serve_vllm.sh --model ./outputs/checkpoints/dapt/merged

# Serve with LoRA adapter
./scripts/serve_vllm.sh \
  --model unsloth/gpt-oss-20b \
  --lora ./outputs/checkpoints/dapt/final

# Reduced memory usage
./scripts/serve_vllm.sh \
  --gpu-memory-util 0.8 \
  --max-model-len 2048
```

### vLLM Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | From .env | Model ID or path |
| `--lora` | - | LoRA adapter path |
| `--dtype` | auto | Data type |
| `--gpu-memory-util` | 0.90 | GPU memory fraction |
| `--max-model-len` | 4096 | Maximum sequence length |
| `--port` | 8000 | Server port |
| `--served-name` | gpt-oss | Model name in API |

### GPT-OSS Notes

GPT-OSS-20B is a Mixture of Experts model:
- Total parameters: 21B
- Active parameters: 3.6B per forward pass
- Memory requirement: ~16GB in MXFP4

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
- Use smaller batch size in training

**"Model not found"**
- Ensure `HF_TOKEN` is set
- Check: `huggingface-cli whoami`

**vLLM fails to start**
- Check vLLM version: `pip show vllm`
- Ensure CUDA compatibility
- Try: `pip install -U vllm`

**Unsloth import error**
- Ensure Unsloth is installed: `pip install unsloth`
- Check for version compatibility

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

**Note**: GPT-OSS-20B is licensed under Apache 2.0. Check the model card on HuggingFace for details.

## Acknowledgments

- OpenAI for GPT-OSS model
- Unsloth for efficient training framework
- HuggingFace for Transformers and PEFT libraries
- vLLM team for efficient serving infrastructure
- BigCode for evaluation benchmarks
