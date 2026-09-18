from tools.workspace_tools import list_files, read_file, write_file, run_command


def test_workspace_tools():
    assert "calculator.py" in list_files()
    assert "def divide" in read_file("calculator.py")
    write_file("_autopr_test.txt", "ok")
    assert read_file("_autopr_test.txt") == "ok"
    result = run_command("python test_calculator.py")
    assert result["success"] is True
    assert "ALL TESTS PASSED" in result["stdout"]
