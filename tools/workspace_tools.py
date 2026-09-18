"""Safe workspace tools used by the AutoPR agent."""

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv


# Load .env before reading AUTOPR_WORKSPACE.
load_dotenv()


WORKSPACE = Path(
    os.getenv("AUTOPR_WORKSPACE", "dummy_repo")
).resolve()

COMMAND_TIMEOUT = int(
    os.getenv("COMMAND_TIMEOUT", "60")
)


def _safe_path(filename: str) -> Path:
    """Resolve a workspace-relative path and prevent path traversal."""

    if not filename or Path(filename).is_absolute():
        raise ValueError(
            "filename must be a non-empty relative path"
        )

    path = (WORKSPACE / filename).resolve()

    if path != WORKSPACE and WORKSPACE not in path.parents:
        raise ValueError(
            "path escapes the configured workspace"
        )

    return path


def read_file(filename: str) -> str:
    """Read a UTF-8 text file from the workspace."""

    return _safe_path(filename).read_text(
        encoding="utf-8"
    )


def write_file(
    filename: str,
    content: str,
) -> None:
    """Write a UTF-8 text file inside the workspace."""

    path = _safe_path(filename)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content,
        encoding="utf-8",
    )


def list_files() -> list[str]:
    """List workspace files while hiding Git/cache internals."""

    return sorted(
        str(p.relative_to(WORKSPACE))
        for p in WORKSPACE.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts
    )


def run_command(command: str) -> dict:
    """Run a shell command inside the configured workspace."""

    if not command or not command.strip():
        raise ValueError(
            "command cannot be empty"
        )

    try:

        result = subprocess.run(
            command,
            cwd=WORKSPACE,
            shell=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
        )

        return {
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "success": result.returncode == 0,
            "timed_out": False,
        }

    except subprocess.TimeoutExpired as exc:

        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        return {
            "command": command,
            "stdout": stdout,
            "stderr": (
                stderr
                + f"\nCommand timed out after "
                f"{COMMAND_TIMEOUT}s."
            ),
            "return_code": None,
            "success": False,
            "timed_out": True,
        }


# ============================================================
# GIT TOOLS
# ============================================================


def create_branch(
    branch_name: str,
) -> dict:
    """
    Create and switch to a new Git branch.

    The command always runs inside AUTOPR_WORKSPACE.
    """

    if not branch_name or not branch_name.strip():
        raise ValueError(
            "branch_name cannot be empty"
        )

    branch_name = branch_name.strip()

    result = subprocess.run(
        [
            "git",
            "checkout",
            "-b",
            branch_name,
        ],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
    )

    return {
        "operation": "create_branch",
        "workspace": str(WORKSPACE),
        "branch": branch_name,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.returncode,
        "success": result.returncode == 0,
    }


def commit_changes(
    commit_message: str,
) -> dict:
    """
    Stage relevant source changes and create a Git commit.

    Environment files, caches, and Python bytecode are excluded.
    """

    if (
        not commit_message
        or not commit_message.strip()
    ):
        raise ValueError(
            "commit_message cannot be empty"
        )

    commit_message = commit_message.strip()

    add_result = subprocess.run(
        [
            "git",
            "add",
            "-A",
            "--",
            ".",
            ":(exclude).env",
            ":(exclude).env.*",
            ":(exclude)__pycache__",
            ":(exclude).pytest_cache",
        ],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
    )

    if add_result.returncode != 0:

        return {
            "operation": "commit_changes",
            "workspace": str(WORKSPACE),
            "success": False,
            "stage_stdout": add_result.stdout,
            "stage_stderr": add_result.stderr,
            "commit_stdout": "",
            "commit_stderr": "",
            "return_code": add_result.returncode,
        }

    commit_result = subprocess.run(
        [
            "git",
            "commit",
            "-m",
            commit_message,
        ],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
    )

    return {
        "operation": "commit_changes",
        "workspace": str(WORKSPACE),
        "success": commit_result.returncode == 0,
        "stage_stdout": add_result.stdout,
        "stage_stderr": add_result.stderr,
        "commit_stdout": commit_result.stdout,
        "commit_stderr": commit_result.stderr,
        "return_code": commit_result.returncode,
    }


def push_branch(
    branch_name: str,
) -> dict:
    """
    Push a branch to the repository's origin remote.

    The origin remote comes from the target repository itself.
    """

    if not branch_name or not branch_name.strip():
        raise ValueError(
            "branch_name cannot be empty"
        )

    branch_name = branch_name.strip()

    result = subprocess.run(
        [
            "git",
            "push",
            "--set-upstream",
            "origin",
            branch_name,
        ],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
    )

    return {
        "operation": "push_branch",
        "workspace": str(WORKSPACE),
        "branch": branch_name,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.returncode,
        "success": result.returncode == 0,
    }
