"""
SWE-bench TypeScript Evaluator.

Runs Multi-SWE-bench TypeScript instances with agentic evaluation.
Uses the Responses API directly with custom tool execution loop.
"""

import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import hf_hub_download
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from ..common.logging import get_logger
from ..common.paths import get_output_dir

console = Console()
logger = get_logger(__name__)


# =============================================================================
# Agent Instructions
# =============================================================================

AGENT_INSTRUCTIONS = """You are an expert software engineer fixing a bug in a TypeScript repository.

Workflow:
1. Use grep/find to locate the relevant code files
2. Use cat or sed -n to view the buggy code
3. Use sed -i to make the fix (in-place edit)
4. Use git diff to verify your changes are correct
5. Call submit() when done fixing the bug

Rules:
- Make minimal, focused changes
- Do NOT modify test files
- Always verify changes with git diff before submitting
- If stuck, try a different approach"""


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
    output_dir: Path = field(default_factory=lambda: get_output_dir() / "swe-bench-ts")


# =============================================================================
# SWE-Bench Evaluator
# =============================================================================

class SWEBenchTSEvaluator:
    """
    Evaluator for SWE-bench TypeScript tasks.

    Uses the OpenAI Agents SDK with gpt-oss for proper tool calling.
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
        self.results: list[dict] = []

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

    def _build_input(self, instance: dict) -> str:
        """Build the input message for an instance."""
        issue = instance["resolved_issues"][0] if instance["resolved_issues"] else {}
        title = issue.get("title", "No title")
        body = issue.get("body", "No description")

        # Truncate very long issue bodies
        if len(body) > 4000:
            body = body[:4000] + "\n... (truncated)"

        return f"Fix this bug:\n\n## {title}\n\n{body}"

    def _create_tools(self, repo_dir: Path) -> tuple[dict[str, Callable], dict]:
        """Create tool functions for the agent."""
        submitted = {"value": False}

        def bash(command: str) -> str:
            """Execute a bash command in the repository to explore and modify code."""
            logger.info(f"BASH TOOL CALLED: {command[:100]}")

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
                logger.info(f"BASH OUTPUT: {output[:200]}")
                return output if output.strip() else "(no output)"
            except subprocess.TimeoutExpired:
                return "Error: Command timed out after 60s"
            except Exception as e:
                return f"Error: {str(e)[:200]}"

        def submit() -> str:
            """Call this when you have finished fixing the bug."""
            submitted["value"] = True
            return "Solution submitted successfully."

        tools = {"bash": bash, "submit": submit}
        return tools, submitted

    def _get_tool_definitions(self) -> list[dict]:
        """Get Responses API-format tool definitions."""
        return [
            {
                "type": "function",
                "name": "bash",
                "description": "Execute a bash command in the repository to explore and modify code.",
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
                "description": "Call this when you have finished fixing the bug.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]

    def _run_agent(
        self,
        instance: dict,
        repo_dir: Path,
        debug: bool = False
    ) -> tuple[bool, str, list[dict]]:
        """
        Run the agent to fix the bug using the Responses API directly.

        Returns:
            (submitted, patch, trajectory)
        """
        # Create tools bound to this repo
        tools, submitted_flag = self._create_tools(repo_dir)
        tool_defs = self._get_tool_definitions()

        # Create OpenAI client pointing to vLLM
        client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

        # Build the input with instructions
        input_text = AGENT_INSTRUCTIONS + "\n\n" + self._build_input(instance)

        # Run the agent loop - vLLM Responses API uses string input, so we build
        # a conversation string that includes previous turns
        trajectory = []
        conversation = input_text

        for turn in range(self.config.max_turns):
            try:
                # Call the Responses API
                response = client.responses.create(
                    model=self.config.model_name,
                    input=conversation,
                    tools=tool_defs,
                )

                # Extract tool calls from response output
                tool_calls = []
                text_output = ""

                for item in response.output:
                    item_type = getattr(item, 'type', None)

                    # Handle function_call (ResponseFunctionToolCall)
                    if item_type == 'function_call':
                        tool_calls.append({
                            "id": getattr(item, 'call_id', getattr(item, 'id', f'call_{turn}')),
                            "name": item.name,
                            "arguments": item.arguments,
                        })

                    # Handle mcp_call (McpCall) - same as function_call for our purposes
                    elif item_type == 'mcp_call':
                        tool_calls.append({
                            "id": getattr(item, 'id', f'mcp_{turn}'),
                            "name": item.name,
                            "arguments": item.arguments,
                        })

                    # Handle text output
                    elif item_type == 'message':
                        text_output = getattr(item, 'content', '')

                # Log trajectory
                traj_entry = {
                    "turn": turn + 1,
                    "tool_calls": [{"name": tc["name"], "args": tc["arguments"][:100]} for tc in tool_calls],
                    "text": text_output[:200] if text_output else "",
                }
                trajectory.append(traj_entry)

                if debug:
                    logger.info(f"Turn {turn + 1}: {len(tool_calls)} tool calls")

                # If no tool calls, agent is done
                if not tool_calls:
                    if debug:
                        logger.info(f"Agent finished after {turn + 1} turns (no tool calls)")
                    break

                # Execute tool calls and collect results
                tool_results = []
                for tc in tool_calls:
                    tool_name = tc["name"]
                    try:
                        args = json.loads(tc["arguments"])
                    except json.JSONDecodeError:
                        args = {}

                    if tool_name in tools:
                        if tool_name == "bash":
                            result = tools["bash"](args.get("command", ""))
                        elif tool_name == "submit":
                            result = tools["submit"]()
                        else:
                            result = f"Unknown tool: {tool_name}"
                    else:
                        result = f"Tool not found: {tool_name}"

                    tool_results.append({
                        "name": tool_name,
                        "args": tc["arguments"],
                        "output": result,
                    })

                # Check if submitted
                if submitted_flag["value"]:
                    if debug:
                        logger.info(f"Agent submitted after {turn + 1} turns")
                    break

                # Append tool calls and results to conversation for next turn
                for tc, tr in zip(tool_calls, tool_results):
                    conversation += f"\n\n[Tool Call: {tc['name']}({tc['arguments']})]\n"
                    conversation += f"[Tool Result: {tr['output'][:2000]}]\n"

            except Exception as e:
                logger.error(f"Agent error on turn {turn + 1}: {e}")
                trajectory.append({"turn": turn + 1, "error": str(e)[:500]})
                break

        # Generate the patch from git diff
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

        return submitted_flag["value"], patch, trajectory

    def _evaluate_instance(self, instance: dict) -> dict:
        """Evaluate a single instance with agentic patch generation."""
        instance_id = instance["instance_id"]
        org = instance["org"]
        repo = instance["repo"]
        gold_patch = instance.get("fix_patch", "")

        result = {
            "instance_id": instance_id,
            "org": org,
            "repo": repo,
            "resolved": False,
            "submitted": False,
            "patch_valid": False,
            "files_match": False,
            "iterations": 0,
            "error": None,
            "generated_patch": "",
            "gold_patch": gold_patch,
            "trajectory": [],
        }

        # Clone repo for agentic access
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / repo

            try:
                # Get the base commit (the version BEFORE the fix)
                base_info = instance.get("base", {})
                base_sha = base_info.get("sha")

                if not base_sha:
                    result["error"] = "No base commit SHA in instance"
                    return result

                # Clone repo (need enough depth to reach base commit)
                clone_result = subprocess.run(
                    ["git", "clone", "--depth", "100", f"https://github.com/{org}/{repo}.git", str(repo_dir)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if clone_result.returncode != 0:
                    result["error"] = f"Clone failed: {clone_result.stderr[:200]}"
                    return result

                # Checkout the base commit (the buggy version)
                checkout_result = subprocess.run(
                    ["git", "checkout", base_sha],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if checkout_result.returncode != 0:
                    # If shallow clone doesn't have the commit, fetch it
                    fetch_result = subprocess.run(
                        ["git", "fetch", "--depth=1", "origin", base_sha],
                        cwd=repo_dir,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if fetch_result.returncode == 0:
                        subprocess.run(
                            ["git", "checkout", base_sha],
                            cwd=repo_dir,
                            capture_output=True,
                            text=True,
                        )
                    else:
                        logger.warning(f"Could not checkout base commit {base_sha[:8]}, using HEAD")

                # Run the agent
                submitted, patch, trajectory = self._run_agent(instance, repo_dir, debug=False)

                result["submitted"] = submitted
                result["generated_patch"] = patch
                result["trajectory"] = trajectory
                result["iterations"] = len(trajectory)  # Number of agent turns

            except subprocess.TimeoutExpired:
                result["error"] = "Clone timeout"
                return result
            except Exception as e:
                result["error"] = f"Error: {str(e)[:200]}"
                return result

        # Evaluate the generated patch
        if not patch or "diff" not in patch.lower():
            result["error"] = "No changes made"
            return result

        # Check 1: Is the patch syntactically valid?
        lines = patch.split("\n")
        has_diff = any(l.startswith("diff --git") or l.startswith("diff ") for l in lines)
        has_changes = any(
            l.startswith("-") or l.startswith("+")
            for l in lines
            if not l.startswith("---") and not l.startswith("+++")
        )

        if has_diff and has_changes:
            result["patch_valid"] = True
        else:
            result["error"] = "Invalid patch format"
            return result

        # Check 2: Does it modify the same files as gold patch?
        def extract_files(patch_text: str) -> set:
            files = set()
            for line in patch_text.split("\n"):
                if line.startswith("diff --git"):
                    parts = line.split(" b/")
                    if len(parts) >= 2:
                        files.add(parts[1].strip())
            return files

        gen_files = extract_files(patch)
        gold_files = extract_files(gold_patch)

        if gen_files & gold_files:  # Any overlap
            result["files_match"] = True

        # Check 3: Consider it "resolved" if patch is valid and modifies right files
        if result["patch_valid"] and result["files_match"]:
            result["resolved"] = True

        return result

    def _evaluate_instance_safe(self, instance: dict) -> dict:
        """Wrapper to catch any exceptions."""
        try:
            return self._evaluate_instance(instance)
        except Exception as e:
            return {
                "instance_id": instance.get("instance_id", "unknown"),
                "resolved": False,
                "error": str(e)[:200],
            }

    def run(self, max_instances: int | None = None) -> dict[str, Any]:
        """Run evaluation on all instances."""
        if not self.instances:
            self.load_instances()

        instances = self.instances[:max_instances] if max_instances else self.instances
        total = len(instances)

        console.print(f"\n[bold]Running SWE-bench TypeScript evaluation[/bold]")
        console.print(f"  Model: {self.config.model_name}")
        console.print(f"  Instances: {total}")
        console.print(f"  Max turns per instance: {self.config.max_turns}")
        console.print(f"  Workers: {self.config.max_workers}")
        console.print()

        self.results = []
        resolved_count = 0
        submitted_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating", total=total)

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._evaluate_instance_safe, inst): inst
                    for inst in instances
                }

                for future in as_completed(futures):
                    result = future.result()
                    self.results.append(result)

                    if result.get("resolved"):
                        resolved_count += 1
                    if result.get("submitted"):
                        submitted_count += 1

                    progress.advance(task)
                    progress.update(
                        task,
                        description=f"Evaluating ({resolved_count}/{len(self.results)} resolved)"
                    )

        # Compute summary
        valid_patches = sum(1 for r in self.results if r.get("patch_valid"))
        files_match = sum(1 for r in self.results if r.get("files_match"))
        avg_iterations = sum(r.get("iterations", 0) for r in self.results) / len(self.results) if self.results else 0

        summary = {
            "total": total,
            "resolved": resolved_count,
            "resolve_rate": resolved_count / total if total > 0 else 0,
            "submitted": submitted_count,
            "submit_rate": submitted_count / total if total > 0 else 0,
            "valid_patches": valid_patches,
            "valid_patch_rate": valid_patches / total if total > 0 else 0,
            "files_match": files_match,
            "files_match_rate": files_match / total if total > 0 else 0,
            "avg_iterations": avg_iterations,
            "errors": sum(1 for r in self.results if r.get("error")),
        }

        return summary

    def save_results(self) -> Path:
        """Save results to output directory."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_file = self.config.output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump({
                "config": {
                    "model_name": self.config.model_name,
                    "max_turns": self.config.max_turns,
                    "max_workers": self.config.max_workers,
                },
                "summary": {
                    "total": len(self.results),
                    "resolved": sum(1 for r in self.results if r.get("resolved")),
                    "resolve_rate": sum(1 for r in self.results if r.get("resolved")) / len(self.results) if self.results else 0,
                },
                "results": self.results,
            }, f, indent=2)

        logger.info(f"Results saved to {results_file}")
        return results_file


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run SWE-bench TypeScript evaluation")
    parser.add_argument("--model-name", type=str, default="gpt-oss")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--max-turns", type=int, default=30, help="Max agent turns")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    config = SWEBenchConfig(
        model_name=args.model_name,
        base_url=args.base_url,
        max_turns=args.max_turns,
        max_workers=args.max_workers,
        timeout=args.timeout,
        output_dir=args.output_dir or get_output_dir() / "swe-bench-ts",
    )

    evaluator = SWEBenchTSEvaluator(config)
    evaluator.load_instances()

    summary = evaluator.run(max_instances=args.max_instances)

    console.print("\n" + "=" * 60)
    console.print("[bold]SWE-bench TypeScript Results[/bold]")
    console.print("=" * 60)
    console.print(f"  Total instances:    {summary['total']}")
    console.print(f"  Submitted:          {summary['submitted']} ({summary['submit_rate']:.1%})")
    console.print(f"  Valid patches:      {summary['valid_patches']} ({summary['valid_patch_rate']:.1%})")
    console.print(f"  Correct files:      {summary['files_match']} ({summary['files_match_rate']:.1%})")
    console.print(f"  [bold green]Resolved:           {summary['resolved']} ({summary['resolve_rate']:.1%})[/bold green]")
    console.print(f"  Avg iterations:     {summary['avg_iterations']:.1f}")
    console.print(f"  Errors:             {summary['errors']}")
    console.print("=" * 60)

    if args.verbose and evaluator.results:
        console.print("\n[bold]Sample Trajectories:[/bold]")
        for r in evaluator.results[:3]:
            console.print(f"\n  {r['instance_id']}:")
            console.print(f"    Resolved: {r.get('resolved', False)}")
            console.print(f"    Iterations: {r.get('iterations', 0)}")
            if r.get("error"):
                console.print(f"    Error: {r['error'][:100]}")
            if r.get("trajectory"):
                for t in r["trajectory"][:3]:
                    if t.get("command"):
                        console.print(f"    [{t['iteration']}] {t['command'][:60]}...")

    evaluator.save_results()


if __name__ == "__main__":
    main()
