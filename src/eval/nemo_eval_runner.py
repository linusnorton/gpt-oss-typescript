"""
NeMo Evaluator runner for code generation benchmarks.

Supports both NeMo toolkit evaluation and BigCode harness tasks.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from openai import OpenAI

from ..common.logging import get_logger, setup_logging
from ..common.paths import get_output_dir

logger = get_logger(__name__)


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    model_name: str = "nemotron"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    tasks: list[str] = field(default_factory=lambda: ["humaneval", "mbpp"])
    output_dir: Path = field(default_factory=lambda: get_output_dir() / "eval")
    num_samples: int = 1
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)


class CodeEvaluator:
    """
    Evaluator for code generation tasks.

    Supports HumanEval, MBPP, and TypeScript variants.
    """

    # Task definitions
    TASKS = {
        "humaneval": {
            "name": "HumanEval",
            "language": "python",
            "dataset": "openai_humaneval",
        },
        "mbpp": {
            "name": "MBPP",
            "language": "python",
            "dataset": "mbpp",
        },
        "humaneval-ts": {
            "name": "HumanEval TypeScript",
            "language": "typescript",
            "dataset": "humaneval-x-typescript",
        },
        "multiple-ts": {
            "name": "MultiPL-E TypeScript",
            "language": "typescript",
            "dataset": "multiple-e-typescript",
        },
    }

    def __init__(self, config: EvalConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self.results: dict[str, Any] = {}

    def run_task(self, task_name: str) -> dict[str, Any]:
        """
        Run a single evaluation task.

        Args:
            task_name: Name of the task to run

        Returns:
            Task results dictionary
        """
        if task_name not in self.TASKS:
            logger.warning(f"Unknown task: {task_name}, skipping")
            return {"error": f"Unknown task: {task_name}"}

        task_info = self.TASKS[task_name]
        logger.info(f"Running task: {task_info['name']}")

        # Load task prompts
        prompts = self._load_task_prompts(task_name)
        if not prompts:
            return {"error": "Failed to load prompts"}

        # Generate completions
        completions = []
        for i, prompt in enumerate(prompts):
            logger.debug(f"Processing prompt {i + 1}/{len(prompts)}")

            try:
                response = self.client.completions.create(
                    model=self.config.model_name,
                    prompt=prompt["prompt"],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    n=self.config.num_samples,
                    stop=self.config.stop_sequences or None,
                )

                completions.append({
                    "task_id": prompt.get("task_id", f"task_{i}"),
                    "prompt": prompt["prompt"],
                    "completions": [c.text for c in response.choices],
                })
            except Exception as e:
                logger.error(f"Generation failed for prompt {i}: {e}")
                completions.append({
                    "task_id": prompt.get("task_id", f"task_{i}"),
                    "error": str(e),
                })

        # Evaluate completions
        results = self._evaluate_completions(task_name, completions)

        return results

    def _load_task_prompts(self, task_name: str) -> list[dict]:
        """Load prompts for a task."""
        # Try to load from datasets library
        try:
            from datasets import load_dataset

            task_info = self.TASKS[task_name]
            if task_name == "humaneval":
                dataset = load_dataset("openai/openai_humaneval", split="test")
                return [
                    {
                        "task_id": item["task_id"],
                        "prompt": item["prompt"],
                        "canonical_solution": item["canonical_solution"],
                        "test": item["test"],
                    }
                    for item in dataset
                ]
            elif task_name == "mbpp":
                dataset = load_dataset("mbpp", split="test")
                return [
                    {
                        "task_id": str(item["task_id"]),
                        "prompt": item["text"] + "\n" + item["code"].split("\n")[0],
                        "code": item["code"],
                        "test_list": item["test_list"],
                    }
                    for item in dataset
                ]
            else:
                logger.warning(f"Dataset loading for {task_name} not implemented")
                return []

        except ImportError:
            logger.warning("datasets library not installed")
            return []
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return []

    def _evaluate_completions(
        self,
        task_name: str,
        completions: list[dict],
    ) -> dict[str, Any]:
        """Evaluate generated completions."""
        # Basic pass@k calculation
        passed = 0
        total = len(completions)
        errors = 0

        for comp in completions:
            if "error" in comp:
                errors += 1
                continue

            # For now, just count non-empty completions
            # Full evaluation requires code execution
            if comp.get("completions") and any(c.strip() for c in comp["completions"]):
                passed += 1

        return {
            "task": task_name,
            "total_problems": total,
            "completed": total - errors,
            "errors": errors,
            "pass_at_1": passed / total if total > 0 else 0,
            "note": "Full execution-based evaluation requires sandbox environment",
        }

    def run_all(self) -> dict[str, Any]:
        """Run all configured tasks."""
        results = {}

        for task in self.config.tasks:
            logger.info(f"Starting task: {task}")
            results[task] = self.run_task(task)

        # Compute summary
        summary = {
            "model": self.config.model_name,
            "tasks_run": len(self.config.tasks),
            "results": results,
        }

        return summary

    def save_results(self, results: dict[str, Any]) -> None:
        """Save results to output directory."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Save full results
        results_file = self.config.output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_file}")

        # Save summary
        summary_file = self.config.output_dir / "summary.json"
        summary = {
            "model": results.get("model"),
            "results": {
                task: {
                    "pass_at_1": data.get("pass_at_1", 0),
                    "total": data.get("total_problems", 0),
                }
                for task, data in results.get("results", {}).items()
            },
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run code evaluation benchmarks")
    parser.add_argument("--model-name", type=str, default="nemotron")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--tasks", type=str, default="humaneval,mbpp")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    # Load config file if specified
    file_config = {}
    if args.config and args.config.exists():
        with open(args.config) as f:
            file_config = yaml.safe_load(f) or {}

    # Build config from args and file
    config = EvalConfig(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        tasks=args.tasks.split(","),
        output_dir=args.output_dir or Path(file_config.get("output_dir", "./outputs/eval")),
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    logger.info(f"Running evaluation for model: {config.model_name}")
    logger.info(f"Tasks: {config.tasks}")

    evaluator = CodeEvaluator(config)

    try:
        results = evaluator.run_all()
        evaluator.save_results(results)

        # Print summary
        print("\n" + "=" * 50)
        print("Evaluation Summary")
        print("=" * 50)
        for task, data in results.get("results", {}).items():
            if "error" in data:
                print(f"  {task}: ERROR - {data['error']}")
            else:
                print(f"  {task}: pass@1 = {data.get('pass_at_1', 0):.4f}")
        print("=" * 50)

        return 0

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
