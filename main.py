"""AutoPR entry point: run one Work Item through the autonomous agent."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agent import AutoPRAgent
from gemini_client import GeminiClient

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
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("        AUTOPR MASTER AGENT")
    print("        Gemini 3.5 Flash-Lite")
    print("=" * 60)

    # ========================================================
    # ENVIRONMENT VARIABLES
    # ========================================================

    workspace = os.getenv("AUTOPR_WORKSPACE")
    github_repo = os.getenv("GITHUB_REPO")
    work_item_id = os.getenv("WORK_ITEM_ID")
    base_branch = os.getenv(
        "BASE_BRANCH",
        "main",
    )

    if not workspace:
        raise RuntimeError(
            "AUTOPR_WORKSPACE is not set."
        )

    if not github_repo:
        raise RuntimeError(
            "GITHUB_REPO is not set."
        )

    if not work_item_id:
        raise RuntimeError(
            "WORK_ITEM_ID is not set."
        )

    try:
        issue_number = int(work_item_id)
    except ValueError as exc:
        raise RuntimeError(
            "WORK_ITEM_ID must be a GitHub issue number."
        ) from exc

    print(
        f"[SYSTEM] Workspace: {workspace}"
    )

    print(
        f"[SYSTEM] GitHub Repository: "
        f"{github_repo}"
    )

    print(
        f"[SYSTEM] Base Branch: "
        f"{base_branch}"
    )

    print(
        f"[SYSTEM] Work Item: "
        f"#{issue_number}"
    )

    # ========================================================
    # GEMINI CLIENT
    # ========================================================

    print()
    print(
        "[SYSTEM] Initializing Gemini..."
    )

    gemini_model = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite",
    )

    print(
        f"[SYSTEM] Gemini model: "
        f"{gemini_model}"
    )

    llm_client = GeminiClient(
        model=gemini_model,
    )

    print(
        "[SYSTEM] Gemini client initialized."
    )

    # ========================================================
    # CREATE AGENT
    # ========================================================

    agent = AutoPRAgent(
        llm_client=llm_client,
        max_retries=10,
    )

    # ========================================================
    # REGISTER WORKSPACE TOOLS
    # ========================================================

    agent.register_tool(
        "read_file",
        read_file,
        {
            "type": "object",
            "description": (
                "Read a file from the repository."
            ),
        },
    )

    agent.register_tool(
        "write_file",
        write_file,
        {
            "type": "object",
            "description": (
                "Write content to a repository file."
            ),
        },
    )

    agent.register_tool(
        "list_files",
        list_files,
        {
            "type": "object",
            "description": (
                "List files in the repository."
            ),
        },
    )

    agent.register_tool(
        "run_command",
        run_command,
        {
            "type": "object",
            "description": (
                "Run a command in the repository."
            ),
        },
    )

    agent.register_tool(
        "create_branch",
        create_branch,
        {
            "type": "object",
            "description": (
                "Create a Git branch."
            ),
        },
    )

    agent.register_tool(
        "commit_changes",
        commit_changes,
        {
            "type": "object",
            "description": (
                "Commit repository changes."
            ),
        },
    )

    agent.register_tool(
        "push_branch",
        push_branch,
        {
            "type": "object",
            "description": (
                "Push a branch to GitHub."
            ),
        },
    )

    # ========================================================
    # REGISTER GITHUB TOOLS
    # ========================================================

    agent.register_tool(
        "get_issue_details",
        get_issue_details,
        {
            "type": "object",
            "description": (
                "Get GitHub issue details."
            ),
        },
    )

    agent.register_tool(
        "create_pull_request",
        create_pull_request,
        {
            "type": "object",
            "description": (
                "Create a GitHub pull request."
            ),
        },
    )

    agent.register_tool(
        "add_issue_comment",
        add_issue_comment,
        {
            "type": "object",
            "description": (
                "Add a comment to a GitHub issue."
            ),
        },
    )

    # ========================================================
    # REGISTER SLACK TOOL
    # ========================================================

    agent.register_tool(
        "send_slack_notification",
        send_slack_notification,
        {
            "type": "object",
            "description": (
                "Send a Slack notification."
            ),
        },
    )

    # ========================================================
    # SHOW REGISTERED TOOLS
    # ========================================================

    print()
    print(
        "[SYSTEM] Registered tools:"
    )

    for name in agent.tools:
        print(
            f"  ✓ {name}"
        )

    # ========================================================
    # GET GITHUB ISSUE
    # ========================================================

    print()
    print(
        "--- RETRIEVING GITHUB ISSUE ---"
    )

    try:
        issue = get_issue_details(
            github_repo,
            issue_number,
        )

    except Exception as exc:
        print(
            "[ERROR] Could not retrieve "
            f"issue: {exc}"
        )
        raise

    # ========================================================
    # EXTRACT ISSUE INFORMATION
    # ========================================================

    if isinstance(
        issue,
        dict,
    ):
        issue_title = str(
            issue.get(
                "title",
                "",
            )
        )

        issue_body = str(
            issue.get(
                "body",
                "",
            )
        )

    else:
        issue_title = ""
        issue_body = str(issue)

    print(
        f"[ISSUE] Title: "
        f"{issue_title}"
    )

    print(
        f"[ISSUE] Body: "
        f"{issue_body}"
    )

    # ========================================================
    # READ REPOSITORY RULES
    # ========================================================

    repo_rules = ""

    rules_path = (
        Path(workspace)
        / "repo_rules.md"
    )

    if rules_path.exists():
        try:
            repo_rules = (
                rules_path.read_text(
                    encoding="utf-8"
                )
            )

            print(
                "[SYSTEM] Repository rules loaded."
            )

        except Exception as exc:
            print(
                "[WARN] Could not read "
                f"repo_rules.md: {exc}"
            )

    else:
        print(
            "[SYSTEM] No repo_rules.md found."
        )

    # ========================================================
    # READ BRD IF PRESENT
    # ========================================================

    brd = ""

    brd_path = (
        Path(workspace)
        / "BRD.md"
    )

    if brd_path.exists():
        try:
            brd = (
                brd_path.read_text(
                    encoding="utf-8"
                )
            )

            print(
                "[SYSTEM] BRD.md loaded."
            )

        except Exception as exc:
            print(
                "[WARN] Could not read "
                f"BRD.md: {exc}"
            )

    # ========================================================
    # BUILD TASK PROMPT
    # ========================================================

    task_prompt = f"""
You are AutoPR, an autonomous software engineering agent.

Complete GitHub Issue #{issue_number}.

TARGET REPOSITORY:
{github_repo}

BASE BRANCH:
{base_branch}

ISSUE TITLE:
{issue_title}

ISSUE DESCRIPTION:
{issue_body}

REPOSITORY RULES:
{repo_rules}

BUSINESS REQUIREMENTS / BRD:
{brd}

AUTOPR WORKSPACE:
{workspace}

Your workflow must be evidence-driven.

First inspect the repository before making changes.

Then:

1. Inspect the repository structure.
2. Read README and relevant documentation.
3. Read repo_rules.md if present.
4. Read BRD.md if present.
5. Understand the GitHub issue.
6. Find the relevant source files.
7. Inspect existing tests.
8. Create an implementation plan internally.
9. Modify only the required files.
10. Add or update tests when appropriate.
11. Run the repository's relevant validation commands.
12. If validation fails, inspect the failure.
13. Fix the implementation.
14. Run validation again.
15. Continue until validation succeeds or genuine human
    input is required.
16. Create a feature branch.
17. Commit the changes.
18. Push the branch.
19. Create a pull request.
20. Comment on the issue when appropriate.

IMPORTANT:

- Do not invent repository information.
- Do not invent test results.
- Do not claim validation passed unless a tool actually
  reports success.
- Do not modify unrelated files.
- Follow repository rules.
- Prefer existing project conventions.
- Inspect before editing.
- Use the available tools to perform actual work.
- If a tool fails, analyze the error and recover when possible.
- If genuinely required information is missing, return
  NEEDS_INPUT.
- When the implementation, validation, commit, push, and PR
  workflow is complete, return DONE.

The goal is a real repository change and a traceable GitHub PR,
not a simulated result.
""".strip()

    # ========================================================
    # START AGENT
    # ========================================================

    print()
    print(
        "--- STARTING GEMINI AUTOPR AGENT ---"
    )
    print()

    result = agent.run(
        initial_prompt=task_prompt
    )

    # ========================================================
    # AGENT RESULT
    # ========================================================

    print()
    print(
        "--- AGENT RESULT ---"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    # ========================================================
    # SLACK NOTIFICATION
    # ========================================================

    print()
    print(
        "--- SLACK NOTIFICATION ---"
    )

    try:
        send_slack_notification(
            json.dumps(
                {
                    "repository": github_repo,
                    "issue": issue_number,
                    "result": result,
                },
                ensure_ascii=False,
            )
        )

        print(
            "[SLACK] Notification sent."
        )

    except Exception as exc:
        print(
            "[WARN] Slack notification failed: "
            f"{exc}"
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print(
        "--- AUTOPR FINISHED ---"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
