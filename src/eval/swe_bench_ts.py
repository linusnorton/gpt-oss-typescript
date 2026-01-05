"""
SWE-bench TypeScript Evaluator.

Generates predictions for Multi-SWE-bench TypeScript instances using an agentic loop.
Uses the official Multi-SWE-bench harness for evaluation.

Uses the GPT-OSS Responses API (harmony format) for proper tool calling.
"""

import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from openai import OpenAI
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from ..common.logging import get_logger
from ..common.paths import get_output_dir

console = Console()
logger = get_logger(__name__)


# =============================================================================
# Tool Definitions for GPT-OSS Responses API
# =============================================================================

AGENT_TOOLS = [
    {
        "type": "function",
        "name": "bash",
        "description": "Execute a bash command in the repository. Use this for file exploration (find, grep, cat), editing (sed -i), and verification (git diff).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                }
            },
            "required": ["command"]
        }
    },
    {
        "type": "function",
        "name": "submit",
        "description": "Submit the solution when you have finished fixing the bug. Call this after verifying your changes with git diff.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]

SYSTEM_PROMPT = """You are an expert software engineer fixing a bug in a TypeScript repository.

IMPORTANT: You can ONLY use these tools:
- bash: Execute shell commands (grep, find, cat, sed, git, etc.)
- submit: Call when done fixing the bug

To edit files, use sed -i. For example:
  sed -i 's/oldPattern/newPattern/g' path/to/file.ts

Do NOT try to use apply_patch - it does not exist. Use sed -i instead.

Workflow:
1. Use grep/find to locate relevant files
2. Use cat/sed -n to view code
3. Use sed -i to make the fix
4. Use git diff to verify changes
5. Call submit when done

Rules:
- Make minimal, focused changes
- Do NOT modify test files
- Always verify with git diff before submitting"""


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SWEBenchConfig:
    """Configuration for SWE-bench evaluation."""

    model_name: str = "gpt-oss"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_turns: int = 30  # Max agent turns
    max_workers: int = 8  # Parallel repo evaluations
    timeout: int = 300  # 5 min per instance
    output_dir: Path = field(
        default_factory=lambda: get_output_dir() / "swe-bench-ts"
    )
    repo_cache_dir: Path = field(
        default_factory=lambda: Path("data/swe-bench/repo-cache")
    )


# =============================================================================
# SWE-Bench Prediction Generator
# =============================================================================


class SWEBenchTSPredictor:
    """
    Generates predictions for SWE-bench TypeScript tasks.

    Uses an agentic loop to generate patches, then outputs in
    Multi-SWE-bench format for evaluation with the official harness.
    """

    # TypeScript repos in Multi-SWE-bench
    TS_REPOS = [
        "ts/darkreader__darkreader_dataset.jsonl",
        "ts/mui__material-ui_dataset.jsonl",
        "ts/vuejs__core_dataset.jsonl",
    ]

    def __init__(self, config: SWEBenchConfig):
        self.config = config
        self.instances: list[dict] = []
        self.predictions: list[dict] = []
        self._repo_locks: dict[str, Any] = {}  # Locks for repo cache operations
        import threading
        self._cache_lock = threading.Lock()

    def _ensure_repo_cached(self, org: str, repo: str) -> Path:
        """Ensure repo is cloned to cache. Returns cache path."""
        cache_dir = self.config.repo_cache_dir / org / repo

        # Use lock to prevent parallel clones of same repo
        repo_key = f"{org}/{repo}"
        with self._cache_lock:
            if repo_key not in self._repo_locks:
                import threading
                self._repo_locks[repo_key] = threading.Lock()

        with self._repo_locks[repo_key]:
            # Check for bare repo (has HEAD file, not .git directory)
            if cache_dir.exists() and (cache_dir / "HEAD").exists():
                # Already cached as bare repo
                logger.debug(f"Repo {repo_key} already cached")
                return cache_dir

            # Clean up incomplete clone if directory exists but isn't a valid bare repo
            if cache_dir.exists():
                import shutil
                logger.warning(f"Cleaning up incomplete cache for {repo_key}")
                shutil.rmtree(cache_dir)

            # Clone with full history to cache
            logger.info(f"Caching repo {repo_key}...")
            cache_dir.parent.mkdir(parents=True, exist_ok=True)

            clone_result = subprocess.run(
                [
                    "git", "clone", "--bare",
                    f"https://github.com/{org}/{repo}.git",
                    str(cache_dir),
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min for full clone
            )
            if clone_result.returncode != 0:
                raise RuntimeError(f"Clone failed: {clone_result.stderr[:200]}")

            logger.info(f"Cached {repo_key}")
            return cache_dir

    def load_instances(self) -> int:
        """Load all TypeScript instances from Multi-SWE-bench."""
        self.instances = []

        for repo_file in self.TS_REPOS:
            try:
                path = hf_hub_download(
                    repo_id="ByteDance-Seed/Multi-SWE-bench",
                    filename=repo_file,
                    repo_type="dataset",
                )
                with open(path) as f:
                    for line in f:
                        instance = json.loads(line)
                        self.instances.append(instance)
                logger.info(f"Loaded {repo_file}")
            except Exception as e:
                logger.error(f"Failed to load {repo_file}: {e}")

        logger.info(f"Total instances: {len(self.instances)}")
        return len(self.instances)

    def _build_prompt(self, instance: dict) -> str:
        """Build the prompt for an instance."""
        issue = (
            instance["resolved_issues"][0] if instance["resolved_issues"] else {}
        )
        title = issue.get("title", "No title")
        body = issue.get("body", "No description")

        # Truncate very long issue bodies
        if len(body) > 4000:
            body = body[:4000] + "\n... (truncated)"

        return f"Fix this bug:\n\n## {title}\n\n{body}"

    def _run_agent(
        self, instance: dict, repo_dir: Path
    ) -> tuple[bool, str, list[dict]]:
        """
        Run the agent to generate a fix using GPT-OSS Responses API.

        Returns:
            (submitted, patch, trajectory)
        """
        submitted = {"value": False}

        def execute_bash(command: str) -> str:
            """Execute a bash command."""
            if not command.strip():
                return "Error: Empty command"

            # Security: Block dangerous commands
            dangerous = ["rm -rf /", ":(){ :|:& };:", "mkfs", "> /dev/", "sudo"]
            if any(d in command for d in dangerous):
                return "Error: Command blocked for security reasons"

            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                output = result.stdout + result.stderr
                if len(output) > 4000:
                    output = output[:4000] + "\n... (truncated)"
                return output if output.strip() else "(no output)"
            except subprocess.TimeoutExpired:
                return "Error: Command timed out after 60s"
            except Exception as e:
                return f"Error: {str(e)[:200]}"

        # Create OpenAI client pointing to vLLM
        client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

        # Build the initial prompt
        user_prompt = f"{SYSTEM_PROMPT}\n\n{self._build_prompt(instance)}"

        # Conversation items for multi-turn
        conversation: list = [
            {"type": "message", "role": "user", "content": user_prompt}
        ]
        trajectory = []

        for turn in range(self.config.max_turns):
            try:
                # Call the Responses API (GPT-OSS harmony format)
                response = client.responses.create(
                    model=self.config.model_name,
                    input=conversation,
                    tools=AGENT_TOOLS,
                    temperature=1.0,
                    top_p=1.0,
                    max_output_tokens=1024,
                )

                # Process response output items
                function_call_item = None
                reasoning_text = ""
                message_text = ""

                for item in response.output:
                    if item.type == "reasoning":
                        # Extract reasoning text for trajectory
                        if hasattr(item, 'content') and item.content:
                            for c in item.content:
                                if hasattr(c, 'text'):
                                    reasoning_text = c.text[:200]
                    elif item.type in ("function_call", "mcp_call"):
                        # Handle both function_call and mcp_call (vLLM returns both)
                        function_call_item = item
                    elif item.type == "message":
                        # Final message (no more tool calls)
                        if hasattr(item, 'content') and item.content:
                            for c in item.content:
                                if hasattr(c, 'text'):
                                    message_text = c.text[:200]

                # Log trajectory
                trajectory.append({
                    "turn": turn + 1,
                    "reasoning": reasoning_text,
                    "tool_calls": [{
                        "name": function_call_item.name,
                        "args": function_call_item.arguments[:100]
                    }] if function_call_item else [],
                    "message": message_text,
                })

                # If no function call, the model is done
                if not function_call_item:
                    logger.debug(f"Turn {turn + 1}: No function call, stopping")
                    break

                # Clean up tool name (model sometimes adds harmony tokens like <|channel|>)
                tool_name = function_call_item.name.split("<|")[0].strip()

                # Parse arguments with error handling
                try:
                    tool_args = json.loads(function_call_item.arguments)
                except json.JSONDecodeError as e:
                    logger.warning(f"Turn {turn + 1}: Invalid JSON in tool args: {e}")
                    # Try to extract command from malformed JSON
                    cmd_match = re.search(r'"command"\s*:\s*"([^"]*)"', function_call_item.arguments)
                    if cmd_match:
                        tool_args = {"command": cmd_match.group(1)}
                    else:
                        tool_args = {}

                # Execute the tool
                if tool_name == "bash":
                    result = execute_bash(tool_args.get("command", ""))
                elif tool_name == "submit":
                    submitted["value"] = True
                    result = "Solution submitted successfully."
                else:
                    result = f"Unknown tool: {tool_name}"

                logger.debug(f"Turn {turn + 1}: {tool_name}() -> {result[:100]}...")

                # Get call_id from different possible attributes
                call_id = getattr(function_call_item, 'call_id', None) or \
                          getattr(function_call_item, 'id', None) or \
                          f"call_{turn}"
                item_id = getattr(function_call_item, 'id', call_id)

                # Add function call and result to conversation for next turn
                conversation.append({
                    "type": "function_call",
                    "id": item_id,
                    "call_id": call_id,
                    "name": tool_name,  # Use cleaned tool name
                    "arguments": function_call_item.arguments,
                })
                conversation.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                })

                if submitted["value"]:
                    break

            except Exception as e:
                logger.error(f"Agent error on turn {turn + 1}: {e}")
                trajectory.append({"turn": turn + 1, "error": str(e)[:500]})
                break

        # Get the patch
        patch = ""
        try:
            diff_result = subprocess.run(
                ["git", "diff"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            patch = diff_result.stdout
        except Exception as e:
            logger.error(f"Failed to get git diff: {e}")

        return submitted["value"], patch, trajectory

    def _generate_prediction(self, instance: dict) -> dict:
        """Generate a prediction for a single instance."""
        org = instance["org"]
        repo = instance["repo"]
        number = instance["number"]
        base_sha = instance.get("base", {}).get("sha")

        prediction = {
            "org": org,
            "repo": repo,
            "number": number,
            "fix_patch": "",
            "submitted": False,
            "error": None,
            "trajectory": [],
        }

        if not base_sha:
            prediction["error"] = "No base commit SHA"
            return prediction

        try:
            # Ensure repo is cached (clones once, reuses after)
            cache_dir = self._ensure_repo_cached(org, repo)
        except Exception as e:
            prediction["error"] = f"Cache failed: {str(e)[:200]}"
            return prediction

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / repo

            try:
                # Clone from local cache (fast, uses hardlinks)
                clone_result = subprocess.run(
                    [
                        "git", "clone",
                        "--shared",  # Use objects from cache
                        str(cache_dir),
                        str(repo_dir),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if clone_result.returncode != 0:
                    prediction["error"] = f"Local clone failed: {clone_result.stderr[:200]}"
                    return prediction

                # Checkout base commit (should already have it from full cache)
                checkout_result = subprocess.run(
                    ["git", "checkout", base_sha],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if checkout_result.returncode != 0:
                    prediction["error"] = f"Checkout failed: {checkout_result.stderr[:200]}"
                    return prediction

                # Run agent
                submitted, patch, trajectory = self._run_agent(instance, repo_dir)

                prediction["submitted"] = submitted
                prediction["fix_patch"] = patch
                prediction["trajectory"] = trajectory

            except subprocess.TimeoutExpired:
                prediction["error"] = "Timeout"
            except Exception as e:
                prediction["error"] = str(e)[:200]

        return prediction

    def _generate_prediction_safe(self, instance: dict) -> dict:
        """Wrapper to catch exceptions."""
        try:
            return self._generate_prediction(instance)
        except Exception as e:
            return {
                "org": instance.get("org", "unknown"),
                "repo": instance.get("repo", "unknown"),
                "number": instance.get("number", 0),
                "fix_patch": "",
                "error": str(e)[:200],
            }

    def run(self, max_instances: int | None = None) -> dict[str, Any]:
        """Generate predictions for all instances."""
        if not self.instances:
            self.load_instances()

        instances = (
            self.instances[:max_instances] if max_instances else self.instances
        )
        total = len(instances)

        console.print("\n[bold]Generating SWE-bench TypeScript predictions[/bold]")
        console.print(f"  Model: {self.config.model_name}")
        console.print(f"  Instances: {total}")
        console.print(f"  Max turns: {self.config.max_turns}")
        console.print(f"  Workers: {self.config.max_workers}")
        console.print()

        self.predictions = []
        submitted_count = 0
        patch_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating", total=total)

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._generate_prediction_safe, inst): inst
                    for inst in instances
                }

                for future in as_completed(futures):
                    pred = future.result()
                    self.predictions.append(pred)

                    if pred.get("submitted"):
                        submitted_count += 1
                    if pred.get("fix_patch"):
                        patch_count += 1

                    progress.advance(task)
                    progress.update(
                        task,
                        description=f"Generating ({patch_count}/{len(self.predictions)} patches)",
                    )

        summary = {
            "total": total,
            "submitted": submitted_count,
            "patches_generated": patch_count,
            "errors": sum(1 for p in self.predictions if p.get("error")),
        }

        return summary

    def save_predictions(self) -> Path:
        """Save predictions in Multi-SWE-bench format."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Save in JSONL format for the harness
        predictions_file = self.config.output_dir / "predictions.jsonl"
        with open(predictions_file, "w") as f:
            for pred in self.predictions:
                # Only include required fields for harness
                harness_pred = {
                    "org": pred["org"],
                    "repo": pred["repo"],
                    "number": pred["number"],
                    "fix_patch": pred.get("fix_patch", ""),
                }
                f.write(json.dumps(harness_pred) + "\n")

        # Also save full results with trajectories
        results_file = self.config.output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "config": {
                        "model_name": self.config.model_name,
                        "max_turns": self.config.max_turns,
                        "max_workers": self.config.max_workers,
                    },
                    "summary": {
                        "total": len(self.predictions),
                        "submitted": sum(
                            1 for p in self.predictions if p.get("submitted")
                        ),
                        "patches": sum(
                            1 for p in self.predictions if p.get("fix_patch")
                        ),
                        "errors": sum(
                            1 for p in self.predictions if p.get("error")
                        ),
                    },
                    "predictions": self.predictions,
                },
                f,
                indent=2,
            )

        logger.info(f"Predictions saved to {predictions_file}")
        logger.info(f"Full results saved to {results_file}")
        return predictions_file


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SWE-bench TypeScript predictions"
    )
    parser.add_argument("--model-name", type=str, default="gpt-oss")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Directory to cache git repos (default: data/swe-bench/repo-cache)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    config = SWEBenchConfig(
        model_name=args.model_name,
        base_url=args.base_url,
        max_turns=args.max_turns,
        max_workers=args.max_workers,
        timeout=args.timeout,
        output_dir=args.output_dir or get_output_dir() / "swe-bench-ts",
        repo_cache_dir=args.cache_dir or Path("data/swe-bench/repo-cache"),
    )

    predictor = SWEBenchTSPredictor(config)
    predictor.load_instances()

    summary = predictor.run(max_instances=args.max_instances)

    console.print("\n" + "=" * 60)
    console.print("[bold]SWE-bench TypeScript Prediction Results[/bold]")
    console.print("=" * 60)
    console.print(f"  Total instances:     {summary['total']}")
    console.print(f"  Submitted:           {summary['submitted']}")
    console.print(f"  Patches generated:   {summary['patches_generated']}")
    console.print(f"  Errors:              {summary['errors']}")
    console.print("=" * 60)

    predictions_file = predictor.save_predictions()

    console.print(f"\n[bold green]Predictions saved to:[/bold green] {predictions_file}")
    console.print("\nTo evaluate with official harness:")
    console.print(
        f"  python -m multi_swe_bench.harness.run_evaluation --config eval_config.json"
    )


if __name__ == "__main__":
    main()
