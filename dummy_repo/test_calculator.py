from calculator import divide, percentage


def test_divide() -> None:
    assert divide(10, 2) == 5.0, "Standard division failed"
    assert divide(5, 0) == "ERR_DIV_ZERO", "Did not follow KMEC/NGIT division by zero standards!"


def test_percentage() -> None:
    assert percentage(200, 10) == 20.0
    assert percentage(80, 25) == 20.0


if __name__ == "__main__":
    test_divide()
    test_percentage()
    print("ALL TESTS PASSED")
