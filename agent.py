from __future__ import annotations

import inspect
import json
from typing import Any, Callable


class AutoPRAgent:
    """
    Context-aware agent for AutoPR.

    The agent:
    1. Reads the task/context.
    2. Chooses tools using Gemini.
    3. Executes tools safely.
    4. Normalizes common LLM argument-name mistakes.
    5. Feeds tool results back to Gemini.
    6. Continues until the task is complete or retries are exhausted.
    """

    def __init__(
        self,
        llm_client,
        max_retries: int = 25,
    ):
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.tools: dict[str, dict[str, Any]] = {}
        self.history: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # TOOL REGISTRATION
    # ------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        schema: dict[str, Any] | None = None,
    ):
        self.tools[name] = {
            "func": func,
            "description": description,
            "schema": schema or {},
        }

    # ------------------------------------------------------------------
    # TOOL DESCRIPTION FOR GEMINI
    # ------------------------------------------------------------------

    def _build_tool_descriptions(self) -> str:
        descriptions = []

        for name, info in self.tools.items():
            func = info["func"]
            schema = info.get("schema", {})

            try:
                signature = str(inspect.signature(func))
            except Exception:
                signature = "(unknown signature)"

            descriptions.append(
                f"""
TOOL: {name}

DESCRIPTION:
{info["description"]}

JSON SCHEMA:
{json.dumps(schema, indent=2)}

PYTHON SIGNATURE:
{signature}
""".strip()
            )

        return "\n\n" + "\n\n".join(descriptions)

    # ------------------------------------------------------------------
    # NORMALIZE LLM ARGUMENTS
    # ------------------------------------------------------------------

    def _normalize_action_input(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Gemini sometimes returns semantically correct but differently
        named arguments.

        Example:

            {"file_path": "calculator.py"}

        when the actual tool expects:

            {"filename": "calculator.py"}

        Normalize those common variations before validation/execution.
        """

        if not isinstance(action_input, dict):
            raise ValueError(
                f"Tool input for '{action}' must be a JSON object."
            )

        normalized = dict(action_input)

        aliases: dict[str, dict[str, str]] = {
            "read_file": {
                "file_path": "filename",
                "path": "filename",
                "file": "filename",
                "target": "filename",
                "name": "filename",
            },
            "write_file": {
                "file_path": "filename",
                "path": "filename",
                "file": "filename",
                "target": "filename",
                "name": "filename",
            },
            "run_command": {
                "cmd": "command",
                "shell_command": "command",
                "command_line": "command",
            },
            "create_branch": {
                "branch": "branch_name",
                "name": "branch_name",
            },
            "push_branch": {
                "branch": "branch_name",
                "name": "branch_name",
            },
            "commit_changes": {
                "commit_message": "message",
                "msg": "message",
            },
            "get_issue_details": {
                "issue": "issue_number",
                "number": "issue_number",
                "repository": "repo_name",
                "repo": "repo_name",
            },
            "create_pull_request": {
                "repository": "repo_name",
                "repo": "repo_name",
                "pr_title": "title",
                "pr_body": "body",
                "source_branch": "head",
                "target_branch": "base",
            },
            "add_issue_comment": {
                "repository": "repo_name",
                "repo": "repo_name",
                "issue": "issue_number",
                "number": "issue_number",
                "body": "comment",
            },
        }

        tool_aliases = aliases.get(action, {})

        for old_key, new_key in tool_aliases.items():
            if old_key in normalized and new_key not in normalized:
                normalized[new_key] = normalized.pop(old_key)

        return normalized

    # ------------------------------------------------------------------
    # VALIDATE TOOL INPUT
    # ------------------------------------------------------------------

    def _validate_action_input(
        self,
        action: str,
        action_input: dict[str, Any],
    ):
        if action not in self.tools:
            raise ValueError(
                f"Unknown tool '{action}'. "
                f"Available tools: {list(self.tools.keys())}"
            )

        if not isinstance(action_input, dict):
            raise ValueError(
                f"Input for tool '{action}' must be a JSON object."
            )

        tool = self.tools[action]["func"]

        try:
            signature = inspect.signature(tool)
        except Exception:
            return

        parameters = signature.parameters

        required_names = [
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]

        missing_names = [
            name
            for name in required_names
            if name not in action_input
        ]

        unknown_names = [
            name
            for name in action_input
            if name not in parameters
        ]

        if missing_names:
            raise ValueError(
                f"Missing required argument(s): "
                f"{', '.join(sorted(missing_names))}."
            )

        if unknown_names:
            raise ValueError(
                f"Unknown argument(s): "
                f"{', '.join(sorted(unknown_names))}."
            )

    # ------------------------------------------------------------------
    # EXECUTE TOOL
    # ------------------------------------------------------------------

    def _execute_tool(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> Any:

        if action not in self.tools:
            raise ValueError(
                f"Unknown tool '{action}'. "
                f"Available tools: {list(self.tools.keys())}"
            )

        # Fix common Gemini argument-name mistakes.
        normalized_input = self._normalize_action_input(
            action,
            action_input,
        )

        # Validate AFTER normalization.
        self._validate_action_input(
            action,
            normalized_input,
        )

        tool = self.tools[action]["func"]

        print(
            f"[TOOL] {action}("
            f"{json.dumps(normalized_input, ensure_ascii=False)})"
        )

        result = tool(**normalized_input)

        return result

    # ------------------------------------------------------------------
    # RECENT HISTORY
    # ------------------------------------------------------------------

    def _build_recent_history(
        self,
        max_items: int = 12,
    ) -> list[dict[str, str]]:
        return self.history[-max_items:]

    # ------------------------------------------------------------------
    # SYSTEM PROMPT
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:

        tool_descriptions = self._build_tool_descriptions()

        return f"""
You are AutoPR, an autonomous software-engineering agent.

Your job is to take a work item and turn it into a validated GitHub pull
request.

You MUST inspect the repository before modifying code.

You MUST use the available tools to:
1. Understand the work item.
2. Inspect repository structure.
3. Read repository rules.
4. Read relevant source files.
5. Read relevant tests.
6. Implement the requested change.
7. Run the repository tests.
8. Debug failures if tests fail.
9. Re-run validation.
10. Create a branch.
11. Commit the implementation.
12. Push the branch.
13. Create a pull request.
14. Add a useful work-item comment.
15. Finish only after the implementation is actually validated.

IMPORTANT TOOL RULES:

- Use EXACT argument names from the JSON schema.
- For read_file the correct argument is:

  {{
    "filename": "calculator.py"
  }}

- Do NOT use "path".
- Do NOT use "file_path".
- Do NOT use "file".
- For write_file use "filename" and "content".
- For run_command use "command".
- For create_branch use "branch_name".
- For commit_changes use "message".
- For push_branch use "branch_name".
- For get_issue_details use "repo_name" and "issue_number".
- For create_pull_request use:
  "repo_name", "title", "body", "head", "base".
- For add_issue_comment use:
  "repo_name", "issue_number", "comment".

Do not repeatedly inspect the same files without a reason.

Do not claim a test passed unless you actually ran it.

Do not claim a PR exists unless the PR creation tool succeeded.

If a command fails:
- inspect the failure,
- determine the cause,
- modify the implementation,
- rerun the relevant validation.

Do not stop merely because one tool call failed.

You are allowed to use multiple tool calls.

When you need a tool, respond ONLY with valid JSON in this form:

{{
  "action": "tool_name",
  "action_input": {{
    "argument": "value"
  }}
}}

When the task is completely finished, respond ONLY with:

{{
  "action": "DONE",
  "action_input": {{
    "summary": "short completion summary"
  }}
}}

If you genuinely cannot continue without human input, respond ONLY with:

{{
  "action": "NEEDS_INPUT",
  "action_input": {{
    "question": "specific question for the human"
  }}
}}

AVAILABLE TOOLS:

{tool_descriptions}
""".strip()

    # ------------------------------------------------------------------
    # PARSE GEMINI RESPONSE
    # ------------------------------------------------------------------

    def _parse_llm_response(self, response: str) -> dict[str, Any]:

        if not response:
            raise ValueError("Gemini returned an empty response.")

        text = response.strip()

        # Remove accidental markdown JSON fences.
        if text.startswith("```"):
            lines = text.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemini returned invalid JSON: {exc}\n"
                f"Response:\n{text}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "Gemini response must be a JSON object."
            )

        if "action" not in parsed:
            raise ValueError(
                "Gemini response is missing the 'action' field."
            )

        if "action_input" not in parsed:
            parsed["action_input"] = {}

        if not isinstance(parsed["action_input"], dict):
            raise ValueError(
                "'action_input' must be a JSON object."
            )

        return parsed

    # ------------------------------------------------------------------
    # MAIN AGENT LOOP
    # ------------------------------------------------------------------

    def run(self, task_prompt: str) -> dict[str, Any]:

        self.history = [
            {
                "role": "user",
                "content": task_prompt,
            }
        ]

        system_prompt = self._build_system_prompt()

        for attempt in range(1, self.max_retries + 1):

            print(
                f"\n[AGENT] Attempt "
                f"{attempt}/{self.max_retries}"
            )

            try:
                response = self.llm_client.generate(
                    system_prompt=system_prompt,
                    history=self._build_recent_history(),
                )

                print(
                    f"[GEMINI] {response[:1000]}"
                )

                decision = self._parse_llm_response(response)

            except Exception as exc:

                error_message = (
                    f"LLM response error: {type(exc).__name__}: {exc}"
                )

                print(f"[AGENT ERROR] {error_message}")

                self.history.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

                continue

            action = decision["action"]
            action_input = decision.get(
                "action_input",
                {},
            )

            # ----------------------------------------------------------
            # DONE
            # ----------------------------------------------------------

            if action == "DONE":

                summary = action_input.get(
                    "summary",
                    "Task completed.",
                )

                print(
                    "\n[AGENT] Task completed."
                )

                print(
                    f"[SUMMARY] {summary}"
                )

                return {
                    "status": "DONE",
                    "summary": summary,
                    "attempts": attempt,
                }

            # ----------------------------------------------------------
            # NEEDS HUMAN INPUT
            # ----------------------------------------------------------

            if action == "NEEDS_INPUT":

                question = action_input.get(
                    "question",
                    "Human input is required.",
                )

                print(
                    "\n[AGENT] Human input required:"
                )
                print(question)

                return {
                    "status": "NEEDS_INPUT",
                    "question": question,
                    "attempts": attempt,
                }

            # ----------------------------------------------------------
            # UNKNOWN ACTION
            # ----------------------------------------------------------

            if action not in self.tools:

                error_message = (
                    f"Unknown action '{action}'. "
                    f"Available actions: "
                    f"{list(self.tools.keys())}"
                )

                print(
                    f"[AGENT ERROR] {error_message}"
                )

                self.history.append(
                    {
                        "role": "user",
                        "content": error_message,
                    }
                )

                continue

            # ----------------------------------------------------------
            # TOOL EXECUTION
            # ----------------------------------------------------------

            try:

                result = self._execute_tool(
                    action,
                    action_input,
                )

                if isinstance(result, str):
                    result_text = result
                else:
                    result_text = json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    )

                print(
                    f"[TOOL RESULT] "
                    f"{result_text[:2000]}"
                )

                self.history.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "action": action,
                                "action_input": action_input,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )

                self.history.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool '{action}' succeeded.\n"
                            f"Result:\n{result_text}"
                        ),
                    }
                )

            except Exception as exc:

                error_message = (
                    f"Tool '{action}' failed.\n"
                    f"Error type: {type(exc).__name__}\n"
                    f"Error: {exc}"
                )

                print(
                    f"[TOOL ERROR] {error_message}"
                )

                self.history.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "action": action,
                                "action_input": action_input,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )

                self.history.append(
                    {
                        "role": "user",
                        "content": (
                            f"{error_message}\n\n"
                            "Diagnose the failure and continue "
                            "the task. Do not repeat the exact "
                            "same invalid tool call."
                        ),
                    }
                )

        print(
            "\n[AGENT] Maximum retry limit reached."
        )

        return {
            "status": "MAX_RETRIES",
            "summary": (
                "Agent stopped after reaching "
                "the maximum retry limit."
            ),
            "attempts": self.max_retries,
        }
