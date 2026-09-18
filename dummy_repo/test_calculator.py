from calculator import divide

def test_divide():
    assert divide(10, 2) == 5.0, "Standard division failed"
    # TRAP: The agent will naturally write a try/except that returns None or 0.
    # This test forces it to read the repo_rules.md to figure out the exact string required.
    assert divide(5, 0) == "ERR_DIV_ZERO", "Did not follow KMEC/NGIT division by zero standards!"
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_divide()