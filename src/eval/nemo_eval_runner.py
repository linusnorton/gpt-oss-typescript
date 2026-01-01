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
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from ..common.logging import get_logger, setup_logging
from ..common.paths import get_output_dir

console = Console()
logger = get_logger(__name__)


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    model_name: str = "gpt-oss"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    tasks: list[str] = field(default_factory=lambda: ["humaneval-ts", "mbpp-ts"])
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
        "humaneval-ts": {
            "name": "HumanEval TypeScript",
            "language": "typescript",
            "dataset": "THUDM/humaneval-x",
        },
        "mbpp-ts": {
            "name": "MBPP TypeScript",
            "language": "typescript",
            "dataset": "nuprl/MultiPL-E",
        },
    }

    def __init__(self, config: EvalConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self.results: dict[str, Any] = {}

    # Example prefix - model follows patterns better than instructions
    SYSTEM_PROMPT = """// Example: Check if all elements are positive
function all_positive(numbers: number[]): boolean {
    for (let i = 0; i < numbers.length; i++) {
        if (numbers[i] <= 0) {
            return false;
        }
    }
    return true;
}

"""

    def _extract_function_body(self, completion: str) -> str:
        """Extract function body up to the closing brace, handling nested braces."""
        depth = 1  # We start inside the opening brace from the prompt
        result = []
        i = 0
        while i < len(completion) and depth > 0:
            char = completion[i]
            result.append(char)
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
            i += 1
        return ''.join(result)

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

        # Load task prompts
        prompts = self._load_task_prompts(task_name)
        if not prompts:
            return {"error": "Failed to load prompts"}

        # Generate completions with progress bar
        completions = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"  {task_info['name']}", total=len(prompts))

            for prompt in prompts:
                try:
                    # Use stop tokens from dataset if available, else config
                    stop_tokens = prompt.get("stop_tokens") or self.config.stop_sequences or None
                    # Prepend system prompt to help model understand expected format
                    full_prompt = self.SYSTEM_PROMPT + prompt["prompt"]
                    response = self.client.completions.create(
                        model=self.config.model_name,
                        prompt=full_prompt,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        n=self.config.num_samples,
                        stop=stop_tokens,
                    )

                    task_id = prompt.get("task_id", f"task_{len(completions)}")
                    # Extract just the function body (up to closing brace)
                    raw_completions = [c.text for c in response.choices]
                    cleaned_completions = [self._extract_function_body(c) for c in raw_completions]
                    completions.append({
                        "task_id": task_id,
                        "prompt": prompt["prompt"],
                        "completions": cleaned_completions,
                        "tests": prompt.get("tests", ""),
                    })
                except Exception as e:
                    task_id = prompt.get("task_id", f"task_{len(completions)}")
                    logger.error(f"Generation failed for {task_id}: {e}")
                    completions.append({
                        "task_id": task_id,
                        "error": str(e),
                    })

                progress.advance(task)

        # Evaluate completions
        results = self._evaluate_completions(task_name, completions)

        return results

    def _load_task_prompts(self, task_name: str) -> list[dict]:
        """Load prompts for a task."""
        try:
            from datasets import load_dataset

            if task_name == "humaneval-ts":
                # MultiPL-E HumanEval TypeScript
                dataset = load_dataset("nuprl/MultiPL-E", "humaneval-ts", split="test")
                return [
                    {
                        "task_id": item["name"],
                        "prompt": item["prompt"],
                        "tests": item["tests"],
                        "stop_tokens": item["stop_tokens"],
                    }
                    for item in dataset
                ]
            elif task_name == "mbpp-ts":
                # MultiPL-E MBPP TypeScript
                dataset = load_dataset("nuprl/MultiPL-E", "mbpp-ts", split="test")
                return [
                    {
                        "task_id": item["name"],
                        "prompt": item["prompt"],
                        "tests": item["tests"],
                        "stop_tokens": item["stop_tokens"],
                    }
                    for item in dataset
                ]
            else:
                logger.warning(f"Unknown task: {task_name}")
                return []

        except ImportError:
            logger.warning("datasets library not installed")
            return []
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return []

    def _execute_code(self, code: str, timeout: int = 10) -> tuple[bool, str]:
        """
        Execute TypeScript code in a subprocess and return success/failure.

        Returns:
            Tuple of (passed, error_message)
        """
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
            f.write(code)
            f.flush()
            try:
                # Use npx tsx for fast TypeScript execution (no tsconfig needed)
                result = subprocess.run(
                    ['npx', 'tsx', f.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if result.returncode == 0:
                    return True, ""
                else:
                    return False, result.stderr[:500]
            except subprocess.TimeoutExpired:
                return False, "Timeout"
            except Exception as e:
                return False, str(e)
            finally:
                os.unlink(f.name)

    def _evaluate_completions(
        self,
        task_name: str,
        completions: list[dict],
    ) -> dict[str, Any]:
        """Evaluate generated completions by executing code."""
        passed = 0
        total = len(completions)
        errors = 0
        failed_tasks = []

        task_info = self.TASKS.get(task_name, {"name": task_name})

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            eval_task = progress.add_task(f"  Evaluating {task_info['name']}", total=total)

            for comp in completions:
                if "error" in comp:
                    errors += 1
                    progress.advance(eval_task)
                    continue

                if not comp.get("completions"):
                    errors += 1
                    progress.advance(eval_task)
                    continue

                task_id = comp.get("task_id", "unknown")
                prompt = comp.get("prompt", "")
                generated = comp["completions"][0]  # Use first completion for pass@1

                # Build full code to execute: prompt + completion + tests
                tests = comp.get("tests", "")
                full_code = prompt + generated + "\n" + tests

                success, error = self._execute_code(full_code)
                if success:
                    passed += 1
                else:
                    failed_tasks.append({"task_id": task_id, "error": error[:200]})

                progress.advance(eval_task)

        pass_rate = passed / total if total > 0 else 0

        return {
            "task": task_name,
            "total_problems": total,
            "passed": passed,
            "failed": total - passed - errors,
            "errors": errors,
            "pass_at_1": pass_rate,
            "failed_samples": failed_tasks[:10],  # First 10 failures for debugging
        }

    def run_all(self) -> dict[str, Any]:
        """Run all configured tasks."""
        results = {}
        total_tasks = len(self.config.tasks)

        for idx, task in enumerate(self.config.tasks, 1):
            console.print(f"\n[bold cyan][{idx}/{total_tasks}][/bold cyan] Running task: [bold]{task}[/bold]")
            results[task] = self.run_task(task)

            # Show task result immediately
            if "error" in results[task]:
                console.print(f"  [red]✗[/red] {task}: [red]ERROR[/red] - {results[task]['error']}")
            else:
                pass_rate = results[task].get('pass_at_1', 0)
                console.print(f"  [green]✓[/green] {task}: pass@1 = [bold]{pass_rate:.4f}[/bold]")

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
    parser.add_argument("--model-name", type=str, default="gpt-oss")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--tasks", type=str, default="humaneval-ts,mbpp-ts")
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

    # Extract nested config values
    model_config = file_config.get("model", {})
    generation_config = file_config.get("generation", {})
    output_config = file_config.get("output", {})

    # Build config from args and file (args override file config)
    config = EvalConfig(
        model_name=args.model_name if args.model_name != "gpt-oss" else model_config.get("name", args.model_name),
        base_url=args.base_url if args.base_url != "http://localhost:8000/v1" else model_config.get("base_url", args.base_url),
        api_key=args.api_key if args.api_key != "EMPTY" else model_config.get("api_key", args.api_key),
        tasks=args.tasks.split(","),
        output_dir=args.output_dir or Path(output_config.get("dir", "./outputs/eval")),
        num_samples=args.num_samples,
        temperature=args.temperature if args.temperature != 0.0 else generation_config.get("temperature", args.temperature),
        max_tokens=args.max_tokens if args.max_tokens != 512 else generation_config.get("max_tokens", args.max_tokens),
        top_p=generation_config.get("top_p", 1.0),
        stop_sequences=generation_config.get("stop_sequences", []),
    )

    console.print(f"Running evaluation for model: [bold]{config.model_name}[/bold]")
    console.print(f"Tasks: {', '.join(config.tasks)}")

    evaluator = CodeEvaluator(config)

    try:
        results = evaluator.run_all()
        evaluator.save_results(results)

        # Print summary
        console.print("\n" + "=" * 50)
        console.print("[bold]Evaluation Summary[/bold]")
        console.print("=" * 50)
        for task, data in results.get("results", {}).items():
            if "error" in data:
                console.print(f"  {task}: [red]ERROR[/red] - {data['error']}")
            else:
                console.print(f"  {task}: pass@1 = [bold]{data.get('pass_at_1', 0):.4f}[/bold]")
        console.print("=" * 50)

        return 0

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
