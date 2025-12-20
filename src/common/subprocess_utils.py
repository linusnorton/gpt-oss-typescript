"""
Subprocess utilities for running external commands.
"""

import asyncio
import subprocess
import shlex
from pathlib import Path
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class CommandResult:
    """Result of a command execution."""

    returncode: int
    stdout: str
    stderr: str
    command: str

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def check(self) -> None:
        """Raise an exception if the command failed."""
        if not self.success:
            raise subprocess.CalledProcessError(
                self.returncode,
                self.command,
                self.stdout,
                self.stderr,
            )


def run_command(
    command: Union[str, list[str]],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    timeout: Optional[float] = None,
    capture_output: bool = True,
    check: bool = False,
) -> CommandResult:
    """
    Run a command and return the result.

    Args:
        command: Command to run (string or list)
        cwd: Working directory
        env: Environment variables
        timeout: Timeout in seconds
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise on non-zero exit

    Returns:
        CommandResult with returncode, stdout, stderr
    """
    if isinstance(command, str):
        cmd_str = command
        cmd_list = shlex.split(command)
    else:
        cmd_str = " ".join(command)
        cmd_list = command

    logger.debug(f"Running command: {cmd_str}")

    try:
        result = subprocess.run(
            cmd_list,
            cwd=cwd,
            env=env,
            timeout=timeout,
            capture_output=capture_output,
            text=True,
        )

        cmd_result = CommandResult(
            returncode=result.returncode,
            stdout=result.stdout if capture_output else "",
            stderr=result.stderr if capture_output else "",
            command=cmd_str,
        )

        if check:
            cmd_result.check()

        return cmd_result

    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out: {cmd_str}")
        return CommandResult(
            returncode=-1,
            stdout=e.stdout or "" if hasattr(e, "stdout") else "",
            stderr=f"Timeout after {timeout}s",
            command=cmd_str,
        )


async def run_command_async(
    command: Union[str, list[str]],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> CommandResult:
    """
    Run a command asynchronously.

    Args:
        command: Command to run (string or list)
        cwd: Working directory
        env: Environment variables
        timeout: Timeout in seconds

    Returns:
        CommandResult with returncode, stdout, stderr
    """
    if isinstance(command, str):
        cmd_str = command
    else:
        cmd_str = " ".join(command)
        command = " ".join(command)

    logger.debug(f"Running async command: {cmd_str}")

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return CommandResult(
                returncode=-1,
                stdout="",
                stderr=f"Timeout after {timeout}s",
                command=cmd_str,
            )

        return CommandResult(
            returncode=process.returncode or 0,
            stdout=stdout.decode() if stdout else "",
            stderr=stderr.decode() if stderr else "",
            command=cmd_str,
        )

    except Exception as e:
        logger.error(f"Command failed: {e}")
        return CommandResult(
            returncode=-1,
            stdout="",
            stderr=str(e),
            command=cmd_str,
        )


def git_clone(
    url: str,
    dest: Path,
    depth: int = 1,
    branch: Optional[str] = None,
) -> CommandResult:
    """
    Clone a git repository.

    Args:
        url: Repository URL
        dest: Destination directory
        depth: Clone depth (1 for shallow)
        branch: Branch to clone

    Returns:
        CommandResult
    """
    cmd = ["git", "clone", "--depth", str(depth)]

    if branch:
        cmd.extend(["--branch", branch])

    cmd.extend([url, str(dest)])

    return run_command(cmd)


def check_command_exists(command: str) -> bool:
    """Check if a command exists in PATH."""
    result = run_command(f"which {command}")
    return result.success
