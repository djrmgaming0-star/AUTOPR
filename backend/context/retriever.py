from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

BRD_PATH = BASE_DIR / "demo_data" / "BRD.md"
RULES_PATH = BASE_DIR / "demo_data" / "repo_rules.md"


def get_context():
    brd = BRD_PATH.read_text(encoding="utf-8")
    repo_rules = RULES_PATH.read_text(encoding="utf-8")

    return {
        "brd": brd,
        "repo_rules": repo_rules,
    }
