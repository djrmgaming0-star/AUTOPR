"""AutoPR entry point: run one Work Item through the autonomous agent."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from agent import AutoPRAgent
from tools.github_tools import add_issue_comment, create_pull_request, get_issue_details
from tools.slack_tools import send_slack_notification
from tools.workspace_tools import list_files, read_file, run_command, write_file

load_dotenv()


class GeminiLLM:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured")
        self.client = genai.Client(api_key=api_key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    def generate(self, system_prompt: str, history: list[dict]) -> str:
        prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nEXECUTION HISTORY:\n"
        for msg in history:
            prompt += f"[{msg['role'].upper()}]\n{msg['content']}\n\n"
        prompt += (
            "Return ONLY one valid JSON object with exactly these keys: "
            "thought, status, action, action_input. "
            "Valid status values: CONTINUE, SUCCESS, NEEDS_INPUT. "
            "No markdown and no text outside JSON."
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return (response.text or "").strip()
        except Exception as exc:
            return json.dumps({
                "thought": f"LLM error: {exc}",
                "status": "NEEDS_INPUT",
                "action": "",
                "action_input": {},
            })


def local_ticket(ticket_id: str) -> str:
    """Offline fallback for the bundled demo workspace."""
    path = Path(os.getenv("WORKSPACE_DIR", "dummy_repo")) / "tickets.json"
    if not path.exists():
        return "No local ticket database is available. Use get_issue_details for the GitHub issue."
    data = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(data.get(ticket_id, {"error": f"Ticket {ticket_id} not found"}), indent=2)


def local_rules() -> str:
    path = Path(os.getenv("WORKSPACE_DIR", "dummy_repo")) / "repo_rules.md"
    return path.read_text(encoding="utf-8") if path.exists() else "No local repo_rules.md found; inspect repository documentation."


def main() -> None:
    work_item_id = os.getenv("WORK_ITEM_ID")
    repo_name = os.getenv("GITHUB_REPO")
    base_branch = os.getenv("BASE_BRANCH", "main")
    if not work_item_id:
        raise ValueError("WORK_ITEM_ID is not configured")

    llm = GeminiLLM()
    agent = AutoPRAgent(llm_client=llm, max_retries=int(os.getenv("MAX_AGENT_ITERATIONS", "20")))

    # Workspace
    agent.register_tool("read_file", read_file, {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]})
    agent.register_tool("write_file", write_file, {"type": "object", "properties": {"filename": {"type": "string"}, "content": {"type": "string"}}, "required": ["filename", "content"]})
    agent.register_tool("list_files", list_files, {"type": "object", "properties": {}})
    agent.register_tool("run_command", run_command, {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]})

    # Context / issue
    if repo_name:
        agent.register_tool("get_issue_details", get_issue_details, {"type": "object", "properties": {"repo_name": {"type": "string"}, "issue_number": {"type": "integer"}}, "required": ["repo_name", "issue_number"]})
    agent.register_tool("fetch_ticket", local_ticket, {"type": "object", "properties": {"ticket_id": {"type": "string"}}, "required": ["ticket_id"]})
    agent.register_tool("read_repo_rules", local_rules, {"type": "object", "properties": {}})

    # Delivery
    if repo_name:
        agent.register_tool("create_pull_request", create_pull_request, {"type": "object", "properties": {"repo_name": {"type": "string"}, "branch_name": {"type": "string"}, "title": {"type": "string"}, "body": {"type": "string"}, "base_branch": {"type": "string"}}, "required": ["repo_name", "branch_name", "title", "body"]})
        agent.register_tool("add_issue_comment", add_issue_comment, {"type": "object", "properties": {"repo_name": {"type": "string"}, "issue_number": {"type": "integer"}, "comment": {"type": "string"}}, "required": ["repo_name", "issue_number", "comment"]})
    if os.getenv("SLACK_WEBHOOK_URL"):
        agent.register_tool("send_slack_notification", send_slack_notification, {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]})

    source = "GitHub Issue" if repo_name else "local demo ticket"
    task = f"""You are AutoPR, an autonomous software engineering agent.

Complete Work Item {work_item_id} from requirement to verified result.
Source: {source}.
Repository: {repo_name or 'local demo workspace'}.
Base branch: {base_branch}.

OPERATING RULES
- The Work Item is the source of truth for requested behavior.
- Read repository rules and relevant documentation before coding.
- Inspect the existing repository before changing anything.
- The task may require NEW code, a modification, a bug fix, or tests.
- Do not assume the task is calculator-related or tied to any particular filename.
- Choose relevant files based on the actual requirement and repository structure.
- Preserve existing behavior unless the Work Item requires changing it.
- Use type hints and repository conventions when applicable.
- If implementation is missing, create it. If implementation is broken, repair it.
- Add or update tests when needed to verify acceptance criteria.
- Run the repository's appropriate validation commands. Prefer the repo's documented test command; otherwise use a sensible command such as pytest.
- If validation fails, inspect the complete failure, make a targeted fix, and validate again.
- Never declare SUCCESS merely because code was written.
- Do not modify unrelated files.
- Do not commit generated artifacts, secrets, .env files, caches, or credentials.
- Only create a branch/commit/PR when an actual source change is required.
- For a real GitHub issue, after validation create a feature branch, commit the source/test changes, push it, create one PR against {base_branch}, and optionally comment on the issue and notify Slack.
- If a required external operation cannot be completed, return NEEDS_INPUT with the exact blocker.

EXECUTION
1. Retrieve the Work Item using get_issue_details for the GitHub repository when available; otherwise use fetch_ticket.
2. Read repository rules.
3. List the workspace files and inspect the files relevant to the requirement and tests.
4. Implement or repair the requested functionality.
5. Run validation and self-debug until it passes or a genuine blocker remains.
6. If there is a real source change, use a branch name such as feature-{work_item_id}, commit only relevant files, push, and create the PR.
7. Report the concrete change, validation result, branch/commit/PR information, and notification status.

Return SUCCESS only when the requested work is actually verified."""

    print("=" * 60)
    print("        AUTOPR MASTER AGENT")
    print("=" * 60)
    print("[SYSTEM] Registered tools:")
    for name in agent.tools:
        print(f"  ✓ {name}")
    print("\n--- STARTING MASTER AGENT ---\n")
    print(agent.run(task))


if __name__ == "__main__":
    main()
