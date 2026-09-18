"""AutoPR entry point: run one Work Item through the autonomous agent."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent import AutoPRAgent

from tools.github_tools import (
    add_issue_comment,
    create_pull_request,
    get_issue_details,
)

from tools.slack_tools import send_slack_notification

from tools.workspace_tools import (
    create_branch,
    commit_changes,
    list_files,
    push_branch,
    read_file,
    run_command,
    write_file,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# OPENAI CLIENT
# ============================================================

class OpenAIClient:
    """Small wrapper around the OpenAI API."""

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set."
            )

        self.client = OpenAI(api_key=api_key)

        self.model = os.getenv(
            "OPENAI_MODEL",
            "gpt-5.6-luna",
        )

    def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=messages,
        )

        return response.output_text


# ============================================================
# HELPERS
# ============================================================

def get_environment_value(
    name: str,
    required: bool = True,
) -> str:
    value = os.getenv(name, "").strip()

    if required and not value:
        raise RuntimeError(
            f"{name} is not set."
        )

    return value


def build_task_prompt(
    issue_number: int,
    issue_title: str,
    issue_body: str,
    repo_rules: str,
) -> str:
    return f"""
You are working on GitHub Issue #{issue_number}.

ISSUE TITLE:
{issue_title}

ISSUE DESCRIPTION:
{issue_body}

REPOSITORY RULES:
{repo_rules}

Complete this coding task in the repository.

Important:
- Inspect the existing code before modifying it.
- Keep unrelated code unchanged.
- Make the smallest appropriate change.
- Add or update tests when required by the issue.
- Run the relevant tests.
- Fix test failures if necessary.
- When the implementation is complete, create a branch, commit the changes, push the branch, and create a pull request.
""".strip()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 60)
    print("        AUTOPR MASTER AGENT")
    print("=" * 60)

    workspace = get_environment_value(
        "AUTOPR_WORKSPACE"
    )

    github_repo = get_environment_value(
        "GITHUB_REPO"
    )

    work_item_id = get_environment_value(
        "WORK_ITEM_ID"
    )

    base_branch = os.getenv(
        "BASE_BRANCH",
        "main",
    )

    print(
        f"[SYSTEM] Workspace: {workspace}"
    )

    print(
        f"[SYSTEM] GitHub Repository: {github_repo}"
    )

    print(
        f"[SYSTEM] Base Branch: {base_branch}"
    )

    print(
        f"[SYSTEM] Work Item: #{work_item_id}"
    )

    # --------------------------------------------------------
    # OPENAI
    # --------------------------------------------------------

    llm_client = OpenAIClient()

    # --------------------------------------------------------
    # AGENT
    # --------------------------------------------------------

    agent = AutoPRAgent(
        llm_client=llm_client,
        max_retries=10,
    )

    # --------------------------------------------------------
    # REGISTER TOOLS
    # --------------------------------------------------------

    tools = {
        "read_file": read_file,
        "write_file": write_file,
        "list_files": list_files,
        "run_command": run_command,
        "create_branch": create_branch,
        "commit_changes": commit_changes,
        "push_branch": push_branch,
        "get_issue_details": get_issue_details,
        "create_pull_request": create_pull_request,
        "add_issue_comment": add_issue_comment,
        "send_slack_notification": send_slack_notification,
    }

    for name, func in tools.items():
        try:
            agent.register_tool(
                name=name,
                func=func,
                schema={
                    "type": "object",
                    "description": (
                        f"Execute the {name} operation."
                    ),
                },
            )

            print(f"  ✓ {name}")

        except Exception as exc:
            print(
                f"  ✗ {name}: {exc}"
            )

    print()
    print("--- STARTING MASTER AGENT ---")
    print()

    # --------------------------------------------------------
    # GET ISSUE
    # --------------------------------------------------------

    try:
        issue_number = int(work_item_id)

        issue = get_issue_details(
            issue_number
        )

    except Exception as exc:
        print(
            f"[ERROR] Could not retrieve issue: {exc}"
        )
        raise

    # --------------------------------------------------------
    # EXTRACT ISSUE DATA
    # --------------------------------------------------------

    if isinstance(issue, dict):
        issue_title = str(
            issue.get("title", "")
        )

        issue_body = str(
            issue.get("body", "")
        )

    else:
        issue_title = ""
        issue_body = str(issue)

    if not issue_title and not issue_body:
        raise RuntimeError(
            "GitHub issue did not contain a title or description."
        )

    print(
        f"[ISSUE] #{issue_number}: {issue_title}"
    )

    # --------------------------------------------------------
    # REPOSITORY RULES
    # --------------------------------------------------------

    repo_rules = ""

    rules_path = Path(
        workspace
    ) / "repo_rules.md"

    if rules_path.exists():
        try:
            repo_rules = rules_path.read_text(
                encoding="utf-8"
            )
        except Exception as exc:
            print(
                f"[WARN] Could not read repo rules: {exc}"
            )

    # --------------------------------------------------------
    # BUILD TASK
    # --------------------------------------------------------

    task_prompt = build_task_prompt(
        issue_number=issue_number,
        issue_title=issue_title,
        issue_body=issue_body,
        repo_rules=repo_rules,
    )

    # --------------------------------------------------------
    # RUN AGENT
    # --------------------------------------------------------

    result = agent.run(
        initial_prompt=task_prompt
    )

    print()
    print("--- AGENT RESULT ---")
    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # SLACK
    # --------------------------------------------------------

    try:
        send_slack_notification(
            json.dumps(
                {
                    "repository": github_repo,
                    "issue": issue_number,
                    "result": result,
                }
            )
        )
    except Exception as exc:
        print(
            f"[WARN] Slack notification failed: {exc}"
        )

    print()
    print("--- AUTOPR FINISHED ---")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
