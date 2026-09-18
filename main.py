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
# OPENAI LLM
# ============================================================


class OpenAILLM:
    """OpenAI LLM adapter used by AutoPR."""

    def __init__(self) -> None:

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=api_key
        )

        self.model = os.getenv(
            "OPENAI_MODEL",
            "gpt-5.6-luna",
        )

    def generate(
        self,
        system_prompt: str,
        history: list[dict],
    ) -> str:
        """Generate exactly one AutoPR decision."""

        prompt = (
            f"SYSTEM INSTRUCTIONS:\n"
            f"{system_prompt}\n\n"
            f"EXECUTION HISTORY:\n"
        )

        for msg in history:

            prompt += (
                f"[{msg['role'].upper()}]\n"
                f"{msg['content']}\n\n"
            )

        prompt += """
IMPORTANT OUTPUT RULES:

Return exactly ONE JSON object.

Required keys:

thought
status
action
action_input

Valid status values:

CONTINUE
SUCCESS
NEEDS_INPUT

The response must contain one JSON object only.

Do not return:
- multiple JSON objects
- a JSON array
- markdown
- explanations outside JSON
- code fences

Use exactly ONE tool action per response.

If status is SUCCESS:
action must be an empty string
action_input must be {}

If status is NEEDS_INPUT:
action must be an empty string
action_input must be {}

If status is CONTINUE:
action must contain exactly one registered tool name.
"""

        try:

            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )

            output = (
                response.output_text or ""
            ).strip()

            if not output:
                raise ValueError(
                    "LLM returned an empty response."
                )

            # Find the first JSON object.
            decoder = json.JSONDecoder()

            start = output.find("{")

            if start == -1:
                raise ValueError(
                    "LLM response did not contain a JSON object."
                )

            parsed, _ = decoder.raw_decode(
                output[start:]
            )

            if not isinstance(
                parsed,
                dict,
            ):
                raise ValueError(
                    "LLM response was not a JSON object."
                )

            required_keys = {
                "thought",
                "status",
                "action",
                "action_input",
            }

            missing_keys = (
                required_keys
                - set(parsed.keys())
            )

            if missing_keys:

                raise ValueError(
                    "LLM response is missing keys: "
                    + ", ".join(
                        sorted(missing_keys)
                    )
                )

            if parsed["status"] not in {
                "CONTINUE",
                "SUCCESS",
                "NEEDS_INPUT",
            }:

                raise ValueError(
                    "Invalid status: "
                    + str(parsed["status"])
                )

            if not isinstance(
                parsed["action_input"],
                dict,
            ):

                raise ValueError(
                    "action_input must be a JSON object."
                )

            # Terminal states must not contain an action.
            if parsed["status"] in {
                "SUCCESS",
                "NEEDS_INPUT",
            }:

                parsed["action"] = ""
                parsed["action_input"] = {}

            return json.dumps(
                parsed
            )

        except Exception as exc:

            return json.dumps(
                {
                    "thought": (
                        f"LLM error: {exc}"
                    ),
                    "status": "NEEDS_INPUT",
                    "action": "",
                    "action_input": {},
                }
            )


# ============================================================
# LOCAL FALLBACK TOOLS
# ============================================================


def local_ticket(
    ticket_id: str,
) -> str:
    """Offline fallback for the bundled demo workspace."""

    workspace = Path(
        os.getenv(
            "AUTOPR_WORKSPACE",
            "dummy_repo",
        )
    )

    path = workspace / "tickets.json"

    if not path.exists():

        return (
            "No local ticket database is available. "
            "Use get_issue_details for the GitHub issue."
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return json.dumps(
        data.get(
            str(ticket_id),
            {
                "error": (
                    f"Ticket {ticket_id} "
                    "not found"
                )
            },
        ),
        indent=2,
    )


def local_rules() -> str:
    """Read repository rules."""

    workspace = Path(
        os.getenv(
            "AUTOPR_WORKSPACE",
            "dummy_repo",
        )
    )

    path = workspace / "repo_rules.md"

    if path.exists():

        return path.read_text(
            encoding="utf-8"
        )

    return (
        "No local repo_rules.md found; "
        "inspect repository documentation."
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    work_item_id = os.getenv(
        "WORK_ITEM_ID"
    )

    repo_name = os.getenv(
        "GITHUB_REPO"
    )

    base_branch = os.getenv(
        "BASE_BRANCH",
        "main",
    )

    if not work_item_id:

        raise ValueError(
            "WORK_ITEM_ID is not configured."
        )

    if repo_name:

        if "/" not in repo_name:

            raise ValueError(
                "GITHUB_REPO must be "
                "owner/repository."
            )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = OpenAILLM()

    # --------------------------------------------------------
    # AGENT
    # --------------------------------------------------------

    agent = AutoPRAgent(
        llm_client=llm,
        max_retries=int(
            os.getenv(
                "MAX_AGENT_ITERATIONS",
                "10",
            )
        ),
    )

    # ========================================================
    # WORKSPACE TOOLS
    # ========================================================

    agent.register_tool(
        "read_file",
        read_file,
        {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string"
                }
            },
            "required": [
                "filename"
            ],
        },
    )

    agent.register_tool(
        "write_file",
        write_file,
        {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string"
                },
                "content": {
                    "type": "string"
                },
            },
            "required": [
                "filename",
                "content",
            ],
        },
    )

    agent.register_tool(
        "list_files",
        list_files,
        {
            "type": "object",
            "properties": {},
        },
    )

    agent.register_tool(
        "run_command",
        run_command,
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string"
                }
            },
            "required": [
                "command"
            ],
        },
    )

    # ========================================================
    # GIT TOOLS
    # ========================================================

    agent.register_tool(
        "create_branch",
        create_branch,
        {
            "type": "object",
            "properties": {
                "branch_name": {
                    "type": "string"
                }
            },
            "required": [
                "branch_name"
            ],
        },
    )

    agent.register_tool(
        "commit_changes",
        commit_changes,
        {
            "type": "object",
            "properties": {
                "commit_message": {
                    "type": "string"
                }
            },
            "required": [
                "commit_message"
            ],
        },
    )

    agent.register_tool(
        "push_branch",
        push_branch,
        {
            "type": "object",
            "properties": {
                "branch_name": {
                    "type": "string"
                }
            },
            "required": [
                "branch_name"
            ],
        },
    )

    # ========================================================
    # ISSUE / CONTEXT TOOLS
    # ========================================================

    if repo_name:

        agent.register_tool(
            "get_issue_details",
            get_issue_details,
            {
                "type": "object",
                "properties": {
                    "repo_name": {
                        "type": "string"
                    },
                    "issue_number": {
                        "type": "integer"
                    },
                },
                "required": [
                    "repo_name",
                    "issue_number",
                ],
            },
        )

    agent.register_tool(
        "fetch_ticket",
        local_ticket,
        {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string"
                }
            },
            "required": [
                "ticket_id"
            ],
        },
    )

    agent.register_tool(
        "read_repo_rules",
        local_rules,
        {
            "type": "object",
            "properties": {},
        },
    )

    # ========================================================
    # GITHUB DELIVERY TOOLS
    # ========================================================

    if repo_name:

        agent.register_tool(
            "create_pull_request",
            create_pull_request,
            {
                "type": "object",
                "properties": {
                    "repo_name": {
                        "type": "string"
                    },
                    "branch_name": {
                        "type": "string"
                    },
                    "title": {
                        "type": "string"
                    },
                    "body": {
                        "type": "string"
                    },
                    "base_branch": {
                        "type": "string"
                    },
                },
                "required": [
                    "repo_name",
                    "branch_name",
                    "title",
                    "body",
                ],
            },
        )

        agent.register_tool(
            "add_issue_comment",
            add_issue_comment,
            {
                "type": "object",
                "properties": {
                    "repo_name": {
                        "type": "string"
                    },
                    "issue_number": {
                        "type": "integer"
                    },
                    "comment": {
                        "type": "string"
                    },
                },
                "required": [
                    "repo_name",
                    "issue_number",
                    "comment",
                ],
            },
        )

    # ========================================================
    # SLACK
    # ========================================================

    if os.getenv(
        "SLACK_WEBHOOK_URL"
    ):

        agent.register_tool(
            "send_slack_notification",
            send_slack_notification,
            {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string"
                    }
                },
                "required": [
                    "message"
                ],
            },
        )

    # ========================================================
    # TASK
    # ========================================================

    source = (
        "GitHub Issue"
        if repo_name
        else "local demo ticket"
    )

    task = f"""
You are AutoPR, an autonomous software engineering agent.

Complete Work Item {work_item_id}
from requirement to verified result.

Source:
{source}

Repository:
{repo_name or "local demo workspace"}

Base branch:
{base_branch}

============================================================
OPERATING RULES
============================================================

- The Work Item is the source of truth.
- Read repository rules before coding.
- Inspect the repository before modifying anything.
- The task may require NEW code.
- The task may require modifying existing code.
- The task may require fixing a bug.
- The task may require adding tests.
- Do not assume the task is calculator-related.
- Do not assume a particular filename.
- Choose files based on the actual requirement.
- Preserve existing behavior unless the Work Item requires a change.
- Follow repository conventions.
- Use type hints where appropriate.
- Add or update tests when necessary.
- Run the repository's appropriate validation command.
- Prefer the repository's documented test command.
- Otherwise use a sensible command such as pytest.
- If tests fail, inspect the failure.
- Make a targeted fix.
- Run the tests again.
- Do not declare SUCCESS just because code was written.
- Do not modify unrelated files.
- Do not commit .env files.
- Do not commit secrets.
- Do not commit credentials.
- Do not commit caches.
- Do not commit generated artifacts.

============================================================
GITHUB DELIVERY
============================================================

For a real GitHub Issue:

1. Create a feature branch.
2. Make sure only relevant source/test changes are included.
3. Commit the changes.
4. Push the feature branch to origin.
5. Create exactly one Pull Request against {base_branch}.
6. Comment on the original Issue.
7. Send a Slack notification when available.

The Git tools operate inside the configured
AUTOPR_WORKSPACE.

The GitHub repository is:

{repo_name}

Do NOT push to any other repository.

============================================================
EXECUTION
============================================================

1. Retrieve the Work Item using get_issue_details.
2. Read repository rules.
3. List repository files.
4. Inspect relevant source and test files.
5. Implement or repair the requested functionality.
6. Run validation.
7. If validation fails, inspect and fix it.
8. Run validation again.
9. If source changes exist:
   create branch,
   commit,
   push,
   create PR.
10. Comment on the Issue.
11. Notify Slack when available.
12. Return SUCCESS only after verification.

============================================================
IMPORTANT
============================================================

Use EXACTLY ONE tool action per response.

Never output multiple JSON objects.

After each tool observation,
choose the NEXT SINGLE action.

Do not repeat a successful action
unless a retry is genuinely necessary.
"""

    # ========================================================
    # START
    # ========================================================

    print("=" * 60)
    print("        AUTOPR MASTER AGENT")
    print("=" * 60)

    print(
        f"[SYSTEM] Workspace: "
        f"{os.getenv('AUTOPR_WORKSPACE', 'dummy_repo')}"
    )

    print(
        f"[SYSTEM] GitHub Repository: "
        f"{repo_name or 'local'}"
    )

    print(
        f"[SYSTEM] Base Branch: "
        f"{base_branch}"
    )

    print(
        "[SYSTEM] Registered tools:"
    )

    for name in agent.tools:
        print(f"  ✓ {name}")

    print(
        "\n--- STARTING MASTER AGENT ---\n"
    )

    result = agent.run(task)

    print(result)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()