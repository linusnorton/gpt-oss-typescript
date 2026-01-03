#!/bin/bash
# Run SWE-bench TypeScript evaluation using the official Multi-SWE-bench harness.
#
# Usage:
#   ./scripts/run_swe_bench_eval.sh [--predictions PATH] [--max-workers N]
#
# This script:
# 1. Downloads the Multi-SWE-bench TypeScript dataset if needed
# 2. Runs the official evaluation harness
# 3. Generates a final report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/outputs/swe-bench-ts"
DATA_DIR="${PROJECT_DIR}/data/swe-bench"
WORKDIR="${DATA_DIR}/workdir"
REPO_DIR="${DATA_DIR}/repos"
LOG_DIR="${DATA_DIR}/logs"

# Default values
PREDICTIONS="${OUTPUT_DIR}/predictions.jsonl"
MAX_WORKERS=8

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --predictions)
            PREDICTIONS="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--predictions PATH] [--max-workers N]"
            echo ""
            echo "Options:"
            echo "  --predictions PATH  Path to predictions.jsonl (default: outputs/swe-bench-ts/predictions.jsonl)"
            echo "  --max-workers N     Number of parallel workers (default: 8)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check predictions file exists
if [ ! -f "$PREDICTIONS" ]; then
    echo "Error: Predictions file not found: $PREDICTIONS"
    echo "Run 'make eval-swe-bench-predict' first to generate predictions."
    exit 1
fi

# Create directories
mkdir -p "$DATA_DIR" "$WORKDIR" "$REPO_DIR" "$LOG_DIR" "$OUTPUT_DIR"

# Download TypeScript dataset files if needed
DATASET_DIR="${DATA_DIR}/datasets"
mkdir -p "$DATASET_DIR"

echo "Downloading Multi-SWE-bench TypeScript datasets..."
for repo in "darkreader__darkreader" "mui__material-ui" "vuejs__core"; do
    dataset_file="${DATASET_DIR}/${repo}_dataset.jsonl"
    if [ ! -f "$dataset_file" ]; then
        echo "  Downloading ${repo}..."
        python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='ByteDance-Seed/Multi-SWE-bench',
    filename='ts/${repo}_dataset.jsonl',
    repo_type='dataset',
)
import shutil
shutil.copy(path, '${dataset_file}')
print(f'  Downloaded to ${dataset_file}')
"
    else
        echo "  ${repo} already downloaded"
    fi
done

# Create evaluation config
CONFIG_FILE="${OUTPUT_DIR}/eval_config.json"
cat > "$CONFIG_FILE" << EOF
{
    "mode": "evaluation",
    "workdir": "${WORKDIR}",
    "patch_files": ["${PREDICTIONS}"],
    "dataset_files": [
        "${DATASET_DIR}/darkreader__darkreader_dataset.jsonl",
        "${DATASET_DIR}/mui__material-ui_dataset.jsonl",
        "${DATASET_DIR}/vuejs__core_dataset.jsonl"
    ],
    "force_build": false,
    "output_dir": "${OUTPUT_DIR}/harness_results",
    "specifics": [],
    "skips": [],
    "repo_dir": "${REPO_DIR}",
    "need_clone": true,
    "global_env": [],
    "clear_env": true,
    "stop_on_error": false,
    "max_workers": ${MAX_WORKERS},
    "max_workers_build_image": ${MAX_WORKERS},
    "max_workers_run_instance": ${MAX_WORKERS},
    "log_dir": "${LOG_DIR}",
    "log_level": "INFO",
    "log_to_console": true
}
EOF

echo ""
echo "Running Multi-SWE-bench evaluation harness..."
echo "  Config: $CONFIG_FILE"
echo "  Predictions: $PREDICTIONS"
echo "  Workers: $MAX_WORKERS"
echo ""

# Activate venv and run evaluation
source "${PROJECT_DIR}/.venv/bin/activate"
python -m multi_swe_bench.harness.run_evaluation --config "$CONFIG_FILE"

# Show results
REPORT_FILE="${OUTPUT_DIR}/harness_results/final_report.json"
if [ -f "$REPORT_FILE" ]; then
    echo ""
    echo "============================================================"
    echo "Evaluation Complete!"
    echo "============================================================"
    python3 -c "
import json
with open('${REPORT_FILE}') as f:
    report = json.load(f)
resolved = len(report.get('resolved_instances', []))
unresolved = len(report.get('unresolved_instances', []))
total = resolved + unresolved
rate = (resolved / total * 100) if total > 0 else 0
print(f'  Resolved:   {resolved}/{total} ({rate:.1f}%)')
print(f'  Unresolved: {unresolved}/{total}')
"
    echo ""
    echo "Full report: $REPORT_FILE"
else
    echo "Warning: No final report generated. Check logs in $LOG_DIR"
fi
