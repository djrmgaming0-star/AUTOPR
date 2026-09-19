"""AutoPR entry point: run one Work Item through the autonomous agent."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agent import AutoPRAgent
from gemini_client import GeminiClient
from mcp_client import search_knowledge_via_mcp

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
# TOOL SCHEMAS
# ============================================================


def build_tool_schemas():
    """
    Exact JSON schemas exposed to Gemini.

    These schemas intentionally match the actual Python
    function signatures used by AutoPR.
    """

    return {
        "read_file": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": (
                        "Repository-relative file name, "
                        "for example calculator.py."
                    ),
                }
            },
            "required": ["filename"],
        },

        "write_file": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": (
                        "Repository-relative file name."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Complete content to write."
                    ),
                },
            },
            "required": [
                "filename",
                "content",
            ],
        },

        "list_files": {
            "type": "object",
            "properties": {},
        },

        "run_command": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Command to run in the repository."
                    ),
                }
            },
            "required": ["command"],
        },

        "create_branch": {
            "type": "object",
            "properties": {
                "branch_name": {
                    "type": "string",
                    "description": (
                        "Git branch name."
                    ),
                }
            },
            "required": ["branch_name"],
        },

        # ====================================================
        # FIXED: commit_changes schema
        # ====================================================

        "commit_changes": {
            "type": "object",
            "properties": {
                "commit_message": {
                    "type": "string",
                    "description": (
                        "Git commit message."
                    ),
                }
            },
            "required": [
                "commit_message"
            ],
        },

        "push_branch": {
            "type": "object",
            "properties": {
                "branch_name": {
                    "type": "string",
                    "description": (
                        "Branch to push."
                    ),
                }
            },
            "required": ["branch_name"],
        },

        "get_issue_details": {
            "type": "object",
            "properties": {
                "repo_name": {
                    "type": "string",
                    "description": (
                        "GitHub repository in owner/name format."
                    ),
                },
                "issue_number": {
                    "type": "integer",
                    "description": (
                        "GitHub issue number."
                    ),
                },
            },
            "required": [
                "repo_name",
                "issue_number",
            ],
        },

        "create_pull_request": {
            "type": "object",
            "properties": {
                "repo_name": {
                    "type": "string",
                },
                "title": {
                    "type": "string",
                },
                "body": {
                    "type": "string",
                },
                "head": {
                    "type": "string",
                    "description": (
                        "Source branch."
                    ),
                },
                "base": {
                    "type": "string",
                    "description": (
                        "Target branch."
                    ),
                },
            },
            "required": [
                "repo_name",
                "title",
                "body",
                "head",
                "base",
            ],
        },

        "add_issue_comment": {
            "type": "object",
            "properties": {
                "repo_name": {
                    "type": "string",
                },
                "issue_number": {
                    "type": "integer",
                },
                "comment": {
                    "type": "string",
                },
            },
            "required": [
                "repo_name",
                "issue_number",
                "comment",
            ],
        },

        "send_slack_notification": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                }
            },
            "required": ["message"],
        },

        # ====================================================
        # MCP + RAG TOOL
        # ====================================================

        "search_knowledge": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Question or topic to search in the "
                        "AutoPR knowledge base through MCP."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Number of relevant knowledge chunks "
                        "to retrieve."
                    ),
                },
            },
            "required": ["query"],
        },
    }


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

    workspace = os.getenv(
        "AUTOPR_WORKSPACE"
    )

    github_repo = os.getenv(
        "GITHUB_REPO"
    )

    work_item_id = os.getenv(
        "WORK_ITEM_ID"
    )

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
        issue_number = int(
            work_item_id
        )

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
    # GEMINI
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
        model=gemini_model
    )

    print(
        "[SYSTEM] Gemini client initialized."
    )

    # ========================================================
    # AGENT
    # ========================================================

    agent = AutoPRAgent(
        llm_client=llm_client,
        max_retries=25,
    )

    # ========================================================
    # SCHEMAS
    # ========================================================

    schemas = build_tool_schemas()

    # ========================================================
    # REGISTER WORKSPACE TOOLS
    # ========================================================

    agent.register_tool(
        "read_file",
        read_file,
        schemas["read_file"],
    )

    agent.register_tool(
        "write_file",
        write_file,
        schemas["write_file"],
    )

    agent.register_tool(
        "list_files",
        list_files,
        schemas["list_files"],
    )

    agent.register_tool(
        "run_command",
        run_command,
        schemas["run_command"],
    )

    agent.register_tool(
        "create_branch",
        create_branch,
        schemas["create_branch"],
    )

    agent.register_tool(
        "commit_changes",
        commit_changes,
        schemas["commit_changes"],
    )

    agent.register_tool(
        "push_branch",
        push_branch,
        schemas["push_branch"],
    )

    # ========================================================
    # REGISTER GITHUB TOOLS
    # ========================================================

    agent.register_tool(
        "get_issue_details",
        get_issue_details,
        schemas["get_issue_details"],
    )

    agent.register_tool(
        "create_pull_request",
        create_pull_request,
        schemas["create_pull_request"],
    )

    agent.register_tool(
        "add_issue_comment",
        add_issue_comment,
        schemas["add_issue_comment"],
    )

    # ========================================================
    # REGISTER SLACK
    # ========================================================

    agent.register_tool(
        "send_slack_notification",
        send_slack_notification,
        schemas["send_slack_notification"],
    )

    # ========================================================
    # REGISTER MCP + RAG
    # ========================================================

    agent.register_tool(
        "search_knowledge",
        search_knowledge_via_mcp,
        schemas["search_knowledge"],
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
    # GET ISSUE
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
    # ISSUE INFORMATION
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

        issue_body = str(
            issue
        )

    print(
        f"[ISSUE] Title: "
        f"{issue_title}"
    )

    print(
        f"[ISSUE] Body: "
        f"{issue_body}"
    )

    # ========================================================
    # REPOSITORY RULES
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
    # BRD
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
    # TASK PROMPT
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

=========================================================
REQUIRED WORKFLOW
=========================================================

1. Inspect repository structure.
2. Read relevant documentation.
3. Read repo_rules.md.
4. Read BRD.md if present.
5. Understand the GitHub issue.
6. Use search_knowledge when repository rules,
   coding standards, testing rules, architecture,
   or previous PR patterns are relevant.
7. Inspect relevant source files.
8. Inspect existing tests.
9. Implement the requested feature.
10. Add/update tests when required.
11. Run repository validation.
12. Analyze failures.
13. Fix failures.
14. Re-run validation.
15. Create a feature branch.
16. Commit changes using the commit_changes tool.
   IMPORTANT: The required argument is "commit_message".
17. Push branch.
18. Create GitHub pull request.
19. Comment on the GitHub issue.
20. Finish with DONE only after the work is actually
    implemented and validated.

=========================================================
KNOWLEDGE / MCP
=========================================================

The search_knowledge tool is backed by the AutoPR
Model Context Protocol (MCP) server.

When repository knowledge is required:

1. Call search_knowledge.
2. MCP routes the request to the AutoPR MCP server.
3. The MCP server invokes the RAG retriever.
4. RAG searches the AutoPR knowledge base.
5. Use the retrieved context when making implementation
   decisions.

Do not bypass MCP for knowledge retrieval.

=========================================================
IMPORTANT
=========================================================

Do NOT spend all attempts only reading files.

After sufficient inspection, IMPLEMENT the requested change.

Use search_knowledge to retrieve relevant knowledge
from the AutoPR knowledge base before making decisions
about repository conventions.

Do not invent test results.

Do not claim validation succeeded unless a tool actually
returned successful validation output.

Do not modify unrelated files.

Respect repo_rules.md.

Use existing project conventions.

If a command fails, diagnose it and recover.

If genuinely blocked, return NEEDS_INPUT.

The goal is a REAL repository change and REAL GitHub PR,
not a simulated result.
""".strip()

    # ========================================================
    # RUN AGENT
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
    # RESULT
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
    # SLACK
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