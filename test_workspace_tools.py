from workspace_tools import (
    read_file,
    write_file,
    list_files,
    run_command,
)


def test_read_file():
    content = read_file("calculator.py")
    assert "def" in content


def test_write_file():
    write_file("test_output.txt", "hello AutoPR")
    assert read_file("test_output.txt") == "hello AutoPR"


def test_list_files():
    files = list_files()
    assert "calculator.py" in files


def test_run_command():
    result = run_command("python -c \"print('hello')\"")

    assert result["success"] is True
    assert result["return_code"] == 0
    assert "hello" in result["stdout"]
