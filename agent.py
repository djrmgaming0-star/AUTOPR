"""Core autonomous agent for AutoPR."""

import json
from typing import Any, Callable


class AutoPRAgent:
    """Bounded Reason-Act-Observe agent."""

    def __init__(
        self,
        llm_client,
        max_retries: int = 10,
    ) -> None:
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.history: list[dict[str, Any]] = []
        self.tools: dict[str, dict[str, Any]] = {}

    def register_tool(
        self,
        name: str,
        func: Callable,
        schema: dict[str, Any],
    ) -> None:
        self.tools[name] = {
            "func": func,
            "schema": schema,
        }

    def _build_tool_descriptions(self) -> str:
        lines = ["AVAILABLE TOOLS:"]

        for name, data in self.tools.items():
            lines.append(
                f"- {name}: "
                f"{json.dumps(data['schema'], separators=(',', ':'))}"
            )

        return "\n".join(lines)

    def _build_recent_history(
        self,
        max_messages: int = 4,
        max_chars_per_message: int = 3000,
    ) -> list[dict[str, str]]:
        recent = self.history[-max_messages:]

        compact = []

        for msg in recent:
            content = str(msg.get("content", ""))

            if len(content) > max_chars_per_message:
                content = (
                    content[:max_chars_per_message]
                    + "\n...[truncated]..."
                )

            compact.append(
                {
                    "role": str(msg.get("role", "user")),
                    "content": content,
                }
            )

        return compact

    def _build_system_prompt(self) -> str:
        tools = self._build_tool_descriptions()

        return f"""
You are AutoPR, an autonomous coding agent.

Your job is to modify the repository to satisfy the user's coding task.

Use the available tools to:
1. Inspect the repository.
2. Understand the requested change.
3. Modify the necessary files.
4. Run tests when appropriate.
5. Fix failures if needed.
6. Finish only when the task is complete.

Return ONLY valid JSON.

The JSON MUST contain exactly these four fields:

{{
  "thought": "short summary of the next action",
  "status": "CONTINUE or DONE or NEEDS_INPUT",
  "action": "tool name or empty string",
  "action_input": {{}}
}}

Rules:
- "thought" must be brief.
- "status" must be one of CONTINUE, DONE, or NEEDS_INPUT.
- "action" must be the name of an available tool, or "" when finished.
- "action_input" must contain the arguments for the selected tool.
- Do not include markdown.
- Do not include code fences.
- Do not include extra JSON fields.

{tools}
""".strip()

    def _parse_llm_response(
        self,
        response: str,
    ) -> dict[str, Any]:
        try:
            decision = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM response is not valid JSON: {exc}"
            ) from exc

        if not isinstance(decision, dict):
            raise ValueError("LLM response must be a JSON object.")

        required_keys = {
            "thought",
            "status",
            "action",
            "action_input",
        }

        missing = required_keys - set(decision.keys())

        if missing:
            raise ValueError(
                "LLM JSON is missing keys: "
                + ", ".join(sorted(missing))
            )

        if decision["status"] not in {
            "CONTINUE",
            "DONE",
            "NEEDS_INPUT",
        }:
            raise ValueError(
                "Invalid status: "
                + str(decision["status"])
            )

        if not isinstance(decision["action"], str):
            raise ValueError("action must be a string.")

        if not isinstance(decision["action_input"], dict):
            raise ValueError("action_input must be an object.")

        return decision

    def run(
        self,
        initial_prompt: str,
    ) -> dict[str, Any]:
        self.history = [
            {
                "role": "user",
                "content": initial_prompt,
            }
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                system_prompt = self._build_system_prompt()

                recent_history = self._build_recent_history()

                response = self.llm_client.generate(
                    system_prompt,
                    recent_history,
                )

                decision = self._parse_llm_response(response)

            except Exception as exc:
                error_message = (
                    f"LLM error: {str(exc)}"
                )

                self.history.append(
                    {
                        "role": "user",
                        "content": error_message,
                    }
                )

                if attempt >= self.max_retries:
                    return {
                        "thought": error_message,
                        "status": "NEEDS_INPUT",
                        "action": "",
                        "action_input": {},
                    }

                continue

            thought = str(
                decision.get(
                    "thought",
                    "Continuing the task.",
                )
            )

            status = decision["status"]
            action = decision["action"]
            action_input = decision["action_input"]

            if status == "DONE":
                return {
                    "thought": thought,
                    "status": "DONE",
                    "action": "",
                    "action_input": {},
                }

            if status == "NEEDS_INPUT":
                return {
                    "thought": thought,
                    "status": "NEEDS_INPUT",
                    "action": action,
                    "action_input": action_input,
                }

            if not action:
                self.history.append(
                    {
                        "role": "user",
                        "content": (
                            "No action was provided. "
                            "Choose an available tool or mark the task DONE."
                        ),
                    }
                )
                continue

            if action not in self.tools:
                self.history.append(
                    {
                        "role": "user",
                        "content": (
                            f"Unknown tool '{action}'. "
                            "Choose one of the available tools."
                        ),
                    }
                )
                continue

            try:
                tool = self.tools[action]["func"]
                result = tool(**action_input)

                self.history.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "thought": thought,
                                "status": status,
                                "action": action,
                                "action_input": action_input,
                            }
                        ),
                    }
                )

                self.history.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool '{action}' result:\n"
                            f"{result}"
                        ),
                    }
                )

            except Exception as exc:
                self.history.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool '{action}' failed:\n"
                            f"{exc}"
                        ),
                    }
                )

        return {
            "thought": "Maximum retry limit reached.",
            "status": "NEEDS_INPUT",
            "action": "",
            "action_input": {},
        }
