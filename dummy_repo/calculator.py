def divide(a: float, b: float) -> float | str:
    if b == 0:
        return "ERR_DIV_ZERO"
    return a / b


def percentage(value: float, percent: float) -> float:
    """Return the given percentage of a value."""
    return value * percent / 100
