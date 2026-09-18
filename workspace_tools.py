import subprocess
from pathlib import Path

WORKSPACE = Path("dummy_repo")


def read_file(filename):
    path = WORKSPACE / filename
    return path.read_text()


def write_file(filename, content):
    path = WORKSPACE / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def list_files():
    return [
        str(p.relative_to(WORKSPACE))
        for p in WORKSPACE.rglob("*")
        if p.is_file()
    ]


def run_command(command):
    result = subprocess.run(
        command,
        cwd=WORKSPACE,
        shell=True,
        capture_output=True,
        text=True
    )

    return {
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.returncode,
        "success": result.returncode == 0
    }
