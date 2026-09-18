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

    def run(self, work_item_context: str) -> str:
        """Run the autonomous agent."""

        self.history = [
            {
                "role": "user",
                "content": work_item_context,
            }
        ]

        tool_descriptions = self._build_tool_descriptions()

        system_prompt = f"""
You are AutoPR, an autonomous software development agent.

Your job is to complete the user's software task by using the available
tools.

{tool_descriptions}

Follow this process:

1. Understand the requested task.
2. Inspect the repository before changing files.
3. Read relevant documentation and repository rules.
4. Modify only the necessary files.
5. Add or update tests when appropriate.
6. Run tests or other validation commands.
7. If validation fails, fix the problem and test again.
8. For a real GitHub task, create a branch, commit changes, push the
   branch, and create the pull request.
9. Do not claim success until the work has been validated.

IMPORTANT:

Return exactly one JSON object.

The JSON object must contain exactly these fields:

reason
status
action
action_input

The "reason" field must contain only a short one-sentence explanation
of the next action. Do not provide private chain-of-thought or detailed
internal reasoning.

Valid status values:

CONTINUE
SUCCESS
NEEDS_INPUT

If status is CONTINUE:
- action must be one of the available tools.
- action_input must be an object.

If status is SUCCESS:
- action must be an empty string.
- action_input must be an empty object.

If status is NEEDS_INPUT:
- action must be an empty string.
- action_input must be an empty object.

Return JSON only. No markdown. No text outside the JSON object.
"""

        print(
            f"[AGENT] Starting task. "
            f"Maximum iterations: {self.max_retries}"
        )

        for iteration in range(1, self.max_retries + 1):

            print(
                f"\n[AGENT] Iteration "
                f"{iteration}/{self.max_retries}"
            )

            recent_history = self._build_recent_history()

            raw_response = self.llm_client.generate(
                system_prompt,
                recent_history,
            )

            print(
                f"[AGENT] LLM response:\n"
                f"{raw_response}"
            )

            decision = self._parse_llm_response(raw_response)

            if decision.get("status") == "ERROR":

                message = decision.get(
                    "message",
                    "Unknown LLM error.",
                )

                print(
                    f"[AGENT] Invalid LLM response: "
                    f"{message}"
                )

                self.history.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response was invalid. "
                            "Return a valid JSON object with the "
                            "required fields."
                        ),
                    }
                )

                continue

            status = decision.get("status")

            if status == "SUCCESS":

                print(
                    "[AGENT] Task completed successfully."
                )

                return (
                    "SUCCESS: "
                    + decision.get(
                        "reason",
                        "Task complete.",
                    )
                )

            if status == "NEEDS_INPUT":

                return (
                    "NEEDS_INPUT: "
                    + decision.get(
                        "reason",
                        "Human input required.",
                    )
                )

            tool_name = decision.get("action")
            tool_args = decision.get(
                "action_input",
                {},
            )

            if not tool_name:

                observation = (
                    "System Error: action cannot be empty "
                    "when status is CONTINUE."
                )

                print(
                    f"[AGENT] {observation}"
                )

                self.history.append(
                    {
                        "role": "user",
                        "content": observation,
                    }
                )

                continue

            print(
                f"[AGENT] Action: {tool_name}"
            )

            print(
                f"[AGENT] Arguments: {tool_args}"
            )

            observation = self._execute_tool(
                tool_name,
                tool_args,
            )

            print(
                f"[AGENT] Observation:\n"
                f"{observation}"
            )

            self.history.append(
                {
                    "role": "assistant",
                    "content": raw_response,
                }
            )

            self.history.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool result from {tool_name}:\n"
                        f"{observation}"
                    ),
                }
            )

        return (
            "MAX_RETRIES_REACHED: "
            "Agent reached the maximum number of "
            "Reason-Act-Observe iterations."
        )

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> str:

        if tool_name not in self.tools:
            return (
                f"Error: Tool '{tool_name}' "
                f"is not registered. "
                f"Available tools: "
                f"{list(self.tools.keys())}"
            )

        try:
            if not isinstance(tool_args, dict):
                tool_args = {}

            result = self.tools[
                tool_name
            ]["func"](**tool_args)

            return str(result)

        except Exception as exc:
            return (
                f"Execution Error in "
                f"{tool_name}: {exc}"
            )

    def _parse_llm_response(
        self,
        response: str,
    ) -> dict[str, Any]:

        try:
            cleaned = response.strip()

            if not cleaned:
                raise ValueError(
                    "LLM returned an empty response."
                )

            if "```json" in cleaned:
                cleaned = (
                    cleaned
                    .split("```json", 1)[1]
                )

            if "```" in cleaned:
                cleaned = (
                    cleaned
                    .split("```", 1)[0]
                )

            start = cleaned.find("{")

            if start == -1:
                raise ValueError(
                    "No JSON object found."
                )

            decoder = json.JSONDecoder()

            decision, _ = decoder.raw_decode(
                cleaned[start:]
            )

            if not isinstance(decision, dict):
                raise ValueError(
                    "LLM response was not a JSON object."
                )

            required_keys = {
                "reason",
                "status",
                "action",
                "action_input",
            }

            missing = (
                required_keys
                - set(decision.keys())
            )

            if missing:
                return {
                    "status": "ERROR",
                    "message": (
                        "LLM JSON is missing keys: "
                        + ", ".join(sorted(missing))
                    ),
                }

            valid_statuses = {
                "CONTINUE",
                "SUCCESS",
                "NEEDS_INPUT",
            }

            if decision["status"] not in valid_statuses:
                return {
                    "status": "ERROR",
                    "message": (
                        "Invalid status: "
                        + str(decision["status"])
                    ),
                }

            if not isinstance(
                decision["action_input"],
                dict,
            ):
                return {
                    "status": "ERROR",
                    "message": (
                        "action_input must be a dictionary."
                    ),
                }

            if decision["status"] == "CONTINUE":
                if not decision["action"]:
                    return {
                        "status": "ERROR",
                        "message": (
                            "CONTINUE requires an action."
                        ),
                    }

            else:
                decision["action"] = ""
                decision["action_input"] = {}

            return decision

        except (
            json.JSONDecodeError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:

            return {
                "status": "ERROR",
                "message": (
                    "LLM output could not be parsed: "
                    + str(exc)
                ),
            }
