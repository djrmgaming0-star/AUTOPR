# KMEC / NGIT Engineering Standards
1. All Python functions MUST include type hints (e.g., `a: int, b: int -> float`).
2. Division by zero must NEVER throw a standard Python `ZeroDivisionError`. It must explicitly return the exact string: `"ERR_DIV_ZERO"`.