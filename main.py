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
# OPENAI
# ============================================================

class OpenAIClient:
    """OpenAI client used by AutoPR."""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        self.client = OpenAI(api_key=api_key)

        self.model = os.getenv(
            "OPENAI_MODEL",
            "gpt-5.6-luna",
        )

    def generate(self, system_prompt, messages):
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=messages,
        )

        return response.output_text


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("        AUTOPR MASTER AGENT")
    print("=" * 60)

    workspace = os.getenv("AUTOPR_WORKSPACE")
    github_repo = os.getenv("GITHUB_REPO")
    work_item_id = os.getenv("WORK_ITEM_ID")
    base_branch = os.getenv("BASE_BRANCH", "main")

    if not workspace:
        raise RuntimeError("AUTOPR_WORKSPACE is not set.")

    if not github_repo:
        raise RuntimeError("GITHUB_REPO is not set.")

    if not work_item_id:
        raise RuntimeError("WORK_ITEM_ID is not set.")

    issue_number = int(work_item_id)

    print(f"[SYSTEM] Workspace: {workspace}")
    print(f"[SYSTEM] GitHub Repository: {github_repo}")
    print(f"[SYSTEM] Base Branch: {base_branch}")
    print(f"[SYSTEM] Work Item: #{issue_number}")

    # ========================================================
    # OPENAI CLIENT
    # ========================================================

    llm_client = OpenAIClient()

    # ========================================================
    # CREATE AGENT
    # ========================================================

    agent = AutoPRAgent(
        llm_client=llm_client,
        max_retries=10,
    )

    # ========================================================
    # REGISTER TOOLS
    # ========================================================

    agent.register_tool(
        "read_file",
        read_file,
        {
            "type": "object",
            "description": "Read a file from the repository.",
        },
    )

    agent.register_tool(
        "write_file",
        write_file,
        {
            "type": "object",
            "description": "Write content to a repository file.",
        },
    )

    agent.register_tool(
        "list_files",
        list_files,
        {
            "type": "object",
            "description": "List files in the repository.",
        },
    )

    agent.register_tool(
        "run_command",
        run_command,
        {
            "type": "object",
            "description": "Run a command in the repository.",
        },
    )

    agent.register_tool(
        "create_branch",
        create_branch,
        {
            "type": "object",
            "description": "Create a Git branch.",
        },
    )

    agent.register_tool(
        "commit_changes",
        commit_changes,
        {
            "type": "object",
            "description": "Commit repository changes.",
        },
    )

    agent.register_tool(
        "push_branch",
        push_branch,
        {
            "type": "object",
            "description": "Push a branch to GitHub.",
        },
    )

    agent.register_tool(
        "get_issue_details",
        get_issue_details,
        {
            "type": "object",
            "description": "Get GitHub issue details.",
        },
    )

    agent.register_tool(
        "create_pull_request",
        create_pull_request,
        {
            "type": "object",
            "description": "Create a GitHub pull request.",
        },
    )

    agent.register_tool(
        "add_issue_comment",
        add_issue_comment,
        {
            "type": "object",
            "description": "Add a comment to a GitHub issue.",
        },
    )

    agent.register_tool(
        "send_slack_notification",
        send_slack_notification,
        {
            "type": "object",
            "description": "Send a Slack notification.",
        },
    )

    print("[SYSTEM] Registered tools:")

    for name in agent.tools:
        print(f"  ✓ {name}")

    print()
    print("--- STARTING MASTER AGENT ---")
    print()

    # ========================================================
    # GET ISSUE
    # ========================================================

    try:
        issue = get_issue_details(
            github_repo,
            issue_number,
        )
    except Exception as exc:
        print(
            f"[ERROR] Could not retrieve issue: {exc}"
        )
        raise

    # ========================================================
    # EXTRACT ISSUE
    # ========================================================

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

    print(
        f"[ISSUE] Title: {issue_title}"
    )

    print(
        f"[ISSUE] Body: {issue_body}"
    )

    # ========================================================
    # REPOSITORY RULES
    # ========================================================

    repo_rules = ""

    rules_path = Path(workspace) / "repo_rules.md"

    if rules_path.exists():
        repo_rules = rules_path.read_text(
            encoding="utf-8"
        )

    # ========================================================
    # TASK PROMPT
    # ========================================================

    task_prompt = f"""
You are AutoPR, an autonomous coding agent.

Complete GitHub Issue #{issue_number}.

ISSUE TITLE:
{issue_title}

ISSUE DESCRIPTION:
{issue_body}

REPOSITORY RULES:
{repo_rules}

Your job is to:
1. Inspect the repository.
2. Understand the issue.
3. Modify the required files.
4. Add or update tests when required.
5. Run the relevant tests.
6. Fix failures.
7. Create a branch.
8. Commit the changes.
9. Push the branch.
10. Create a pull request.
11. Comment on the issue when appropriate.

Keep unrelated existing code unchanged.

When finished, return DONE.
""".strip()

    # ========================================================
    # RUN AGENT
    # ========================================================

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

    # ========================================================
    # SLACK
    # ========================================================

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
