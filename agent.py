"""Core autonomous agent for AutoPR."""

import json
from typing import Any, Callable


class AutoPRAgent:
    """Reason-Act-Observe agent with bounded conversation history."""

    def __init__(
        self,
        llm_client,
        max_retries: int = 10,
    ) -> None:
        self.llm_client = llm_client
        self.max_retries = max_retries

        self.history: list[dict[str, Any]] = []
        self.tools: dict[str, dict[str, Any]] = {}

        self.base_prompt = """
You are AutoPR, an autonomous software development agent.

You operate in a strict Reason-Act-Observe loop.

Your response MUST ALWAYS be exactly ONE valid JSON object with these keys:

thought
status
action
action_input

Valid status values:

CONTINUE
SUCCESS
NEEDS_INPUT

CRITICAL RULES:

1. Return exactly ONE JSON object.
2. Do not output markdown.
3. Do not output multiple JSON objects.
4. Do not output explanations outside JSON.
5. ONLY use tools listed in AVAILABLE TOOLS.
6. Use tools to inspect files, modify files, execute commands,
   and interact with external systems.
7. After receiving a tool observation, choose the NEXT logical action.
8. Do not repeat a successful tool call with identical arguments.
9. Inspect files before modifying them when necessary.
10. After modifying code, run appropriate tests.
11. If tests fail, inspect the failure, fix the problem,
    and test again.
12. Do NOT declare SUCCESS merely because a file was edited.
13. Return SUCCESS only after the requested work is actually verified.
14. Return NEEDS_INPUT only when a genuine blocker requires human input.
"""

    def register_tool(
        self,
        name: str,
        func: Callable,
        schema: dict[str, Any],
    ) -> None:
        """Register a tool."""
        self.tools[name] = {
            "func": func,
            "schema": schema,
        }

    def _build_tool_descriptions(self) -> str:
        """Build a compact description of available tools."""

        lines = ["AVAILABLE TOOLS:"]

        for name, data in self.tools.items():
            lines.append(
                f"- {name}: {json.dumps(data['schema'], separators=(',', ':'))}"
            )

        return "\n".join(lines)

    def _build_recent_history(
        self,
        max_messages: int = 8,
        max_chars_per_message: int = 6000,
    ) -> list[dict[str, str]]:
        """
        Keep only a small recent window of history.

        This prevents the LLM prompt from growing indefinitely
        after many tool calls.
        """

        recent = self.history[-max_messages:]

        compact: list[dict[str, str]] = []

        for msg in recent:
            content = str(msg.get("content", ""))

            if len(content) > max_chars_per_message:
                content = (
                    content[:max_chars_per_message]
                    + "\n...[observation truncated]..."
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

        self.history = []

        self.history.append(
            {
                "role": "user",
                "content": work_item_context,
            }
        )

        tool_descriptions = self._build_tool_descriptions()

        system_prompt = (
            self.base_prompt
            + "\n\n"
            + tool_descriptions
        )

        print(
            f"[AGENT] Starting task. "
            f"Maximum iterations: {self.max_retries}"
        )

        for iteration in range(1, self.max_retries + 1):

            print(
                f"\n[AGENT] Iteration "
                f"{iteration}/{self.max_retries}"
            )

            # --------------------------------------------------
            # REASON
            # --------------------------------------------------

            recent_history = self._build_recent_history()

            raw_response = self.llm_client.generate(
                system_prompt,
                recent_history,
            )

            print(
                f"[AGENT] LLM response:\n"
                f"{raw_response}"
            )

            # Save assistant response.
            self.history.append(
                {
                    "role": "assistant",
                    "content": raw_response,
                }
            )

            # --------------------------------------------------
            # PARSE
            # --------------------------------------------------

            decision = self._parse_llm_response(
                raw_response
            )

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
                            "Observation: "
                            + message
                        ),
                    }
                )

                continue

            # --------------------------------------------------
            # TERMINAL STATES
            # --------------------------------------------------

            status = decision.get("status")

            if status == "SUCCESS":

                print(
                    "[AGENT] Task completed successfully."
                )

                return (
                    "SUCCESS: "
                    + decision.get(
                        "thought",
                        "Task complete.",
                    )
                )

            if status == "NEEDS_INPUT":

                return (
                    "NEEDS_INPUT: "
                    + decision.get(
                        "thought",
                        "Human input required.",
                    )
                )

            # --------------------------------------------------
            # ACT
            # --------------------------------------------------

            tool_name = decision.get("action")

            tool_args = decision.get(
                "action_input",
                {},
            )

            if not tool_name:

                observation = (
                    "System Error: action cannot be "
                    "empty unless status is SUCCESS "
                    "or NEEDS_INPUT."
                )

                print(
                    f"[AGENT] {observation}"
                )

                self.history.append(
                    {
                        "role": "user",
                        "content": (
                            "Observation: "
                            + observation
                        ),
                    }
                )

                continue

            print(
                f"[AGENT] Action: {tool_name}"
            )

            print(
                f"[AGENT] Arguments: {tool_args}"
            )

            # --------------------------------------------------
            # EXECUTE TOOL
            # --------------------------------------------------

            observation = self._execute_tool(
                tool_name,
                tool_args,
            )

            print(
                f"[AGENT] Observation:\n"
                f"{observation}"
            )

            # --------------------------------------------------
            # OBSERVE
            # --------------------------------------------------

            self.history.append(
                {
                    "role": "user",
                    "content": (
                        f"Observation from {tool_name}:\n"
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
        """Execute a registered tool safely."""

        if tool_name not in self.tools:

            return (
                f"Error: Tool '{tool_name}' "
                f"is not registered.\n"
                f"Available tools: "
                f"{list(self.tools.keys())}"
            )

        try:

            if not isinstance(
                tool_args,
                dict,
            ):
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
        """Parse exactly the first valid JSON object."""

        try:

            cleaned = response.strip()

            if not cleaned:
                raise ValueError(
                    "LLM returned an empty response."
                )

            # Remove markdown fences if present.
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

            # Find the first JSON object.
            start = cleaned.find("{")

            if start == -1:
                raise ValueError(
                    "No JSON object found."
                )

            decoder = json.JSONDecoder()

            decision, _ = decoder.raw_decode(
                cleaned[start:]
            )

            if not isinstance(
                decision,
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