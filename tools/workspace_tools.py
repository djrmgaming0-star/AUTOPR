"""Safe workspace tools used by the AutoPR agent."""

import os
import subprocess
from pathlib import Path

WORKSPACE = Path(os.getenv("WORKSPACE_DIR", "dummy_repo")).resolve()
COMMAND_TIMEOUT = int(os.getenv("COMMAND_TIMEOUT", "60"))


def _safe_path(filename: str) -> Path:
    """Resolve a workspace-relative path and prevent path traversal."""
    if not filename or Path(filename).is_absolute():
        raise ValueError("filename must be a non-empty relative path")
    path = (WORKSPACE / filename).resolve()
    if path != WORKSPACE and WORKSPACE not in path.parents:
        raise ValueError("path escapes the configured workspace")
    return path


def read_file(filename: str) -> str:
    return _safe_path(filename).read_text(encoding="utf-8")


def write_file(filename: str, content: str) -> None:
    path = _safe_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def list_files() -> list[str]:
    return sorted(
        str(p.relative_to(WORKSPACE))
        for p in WORKSPACE.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts
    )


def run_command(command: str) -> dict:
    """Run a command inside the workspace with bounded execution time."""
    if not command or not command.strip():
        raise ValueError("command cannot be empty")
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
        return {
            "command": command,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + f"\nCommand timed out after {COMMAND_TIMEOUT}s.",
            "return_code": None,
            "success": False,
            "timed_out": True,
        }
