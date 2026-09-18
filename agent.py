
import json
from typing import List, Dict, Any, Callable


class AutoPRAgent:

    def __init__(self, llm_client, max_retries: int = 15):
        """
        Initializes the autonomous agent.

        max_retries is the maximum number of Reason-Act-Observe
        iterations allowed for a single task.
        """

        self.llm_client = llm_client
        self.max_retries = max_retries

        self.history: List[Dict[str, Any]] = []
        self.tools: Dict[str, Dict[str, Any]] = {}

        self.base_prompt = (
            "You are AutoPR, an autonomous software development agent. "
            "You operate in a strict Reason-Act-Observe loop.\n\n"

            "Your response MUST ALWAYS be a single valid JSON object "
            "with exactly these keys:\n"
            "thought, status, action, action_input\n\n"

            "STATUS VALUES:\n"
            "- CONTINUE: You need to perform another action.\n"
            "- SUCCESS: The task is completely finished.\n"
            "- NEEDS_INPUT: Human input is required.\n\n"

            "CRITICAL RULES:\n"

            "1. Your entire response must be valid JSON. "
            "Do not output markdown, explanations, or conversational text.\n"

            "2. ONLY use tools listed in AVAILABLE TOOLS.\n"

            "3. Use tools to inspect files, modify files, execute commands, "
            "and interact with external systems.\n"

            "4. You can analyze tool observations yourself. "
            "Do not invent tool calls for analysis.\n"

            "5. After receiving an observation, determine the NEXT "
            "logical action required to complete the task.\n"

            "6. Do not repeatedly call the same tool with identical "
            "arguments unless the previous attempt failed or the "
            "underlying state has changed.\n"

            "7. Before modifying a file, inspect its current contents "
            "when necessary.\n"

            "8. After modifying code, run the appropriate tests or "
            "commands to verify the change.\n"

            "9. If tests fail, inspect the failure, fix the code, "
            "and run the tests again.\n"

            "10. Do NOT declare SUCCESS merely because a file was edited. "
            "The actual requested task must be verified.\n"

            "11. When the task is completely verified, return:\n"
            "{"
            "\"thought\":\"final summary\","
            "\"status\":\"SUCCESS\","
            "\"action\":\"\","
            "\"action_input\":{}"
            "}\n\n"
        )

    def register_tool(
        self,
        name: str,
        func: Callable,
        schema: Dict[str, Any]
    ):
        """Register a tool."""

        self.tools[name] = {
            "func": func,
            "schema": schema
        }

    def run(self, work_item_context: str) -> str:

        # Reset history for every new task
        self.history = []

        self.history.append({
            "role": "user",
            "content": work_item_context
        })

        iterations = 0

        # Build tool menu
        tool_descriptions = "AVAILABLE TOOLS:\n"

        for name, data in self.tools.items():
            tool_descriptions += (
                f"- {name}: "
                f"{json.dumps(data['schema'])}\n"
            )

        system_prompt = self.base_prompt + tool_descriptions

        print(
            f"[AGENT] Starting task. "
            f"Maximum iterations: {self.max_retries}"
        )

        while iterations < self.max_retries:

            iterations += 1

            print(
                f"\n[AGENT] Iteration "
                f"{iterations}/{self.max_retries}"
            )

            # -----------------------------------------
            # REASON
            # -----------------------------------------

            raw_response = self.llm_client.generate(
                system_prompt,
                self.history
            )

            print(
                f"[AGENT] LLM response:\n"
                f"{raw_response}"
            )

            self.history.append({
                "role": "assistant",
                "content": raw_response
            })

            # -----------------------------------------
            # PARSE
            # -----------------------------------------

            decision = self._parse_llm_response(raw_response)

            if decision.get("status") == "ERROR":

                message = decision.get(
                    "message",
                    "Unknown LLM error."
                )

                print(
                    f"[AGENT] Invalid LLM response: "
                    f"{message}"
                )

                self.history.append({
                    "role": "user",
                    "content":
                    f"Observation: {message}"
                })

                continue

            # -----------------------------------------
            # TERMINAL STATES
            # -----------------------------------------

            status = decision.get("status")

            if status == "SUCCESS":

                print(
                    "[AGENT] Task completed successfully."
                )

                return (
                    "SUCCESS: "
                    + decision.get(
                        "thought",
                        "Task complete."
                    )
                )

            if status == "NEEDS_INPUT":

                return (
                    "NEEDS_INPUT: "
                    + decision.get(
                        "thought",
                        "Human input required."
                    )
                )

            # -----------------------------------------
            # ACT
            # -----------------------------------------

            tool_name = decision.get("action")
            tool_args = decision.get(
                "action_input",
                {}
            )

            if not tool_name:

                observation = (
                    "System Error: action cannot be empty "
                    "unless status is SUCCESS or NEEDS_INPUT."
                )

                self.history.append({
                    "role": "user",
                    "content":
                    f"Observation: {observation}"
                })

                continue

            print(
                f"[AGENT] Action: {tool_name}"
            )

            print(
                f"[AGENT] Arguments: {tool_args}"
            )

            # -----------------------------------------
            # EXECUTE
            # -----------------------------------------

            observation = self._execute_tool(
                tool_name,
                tool_args
            )

            print(
                f"[AGENT] Observation:\n"
                f"{observation}"
            )

            # -----------------------------------------
            # OBSERVE
            # -----------------------------------------

            self.history.append({
                "role": "user",
                "content":
                f"Observation from {tool_name}:\n"
                f"{observation}"
            })

        # -----------------------------------------
        # MAX ITERATIONS
        # -----------------------------------------

        return (
            "MAX_RETRIES_REACHED: "
            "Agent reached the maximum number of "
            "Reason-Act-Observe iterations."
        )

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> str:

        if tool_name not in self.tools:

            return (
                f"Error: Tool '{tool_name}' "
                f"is not registered.\n"
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

        except Exception as e:

            return (
                f"Execution Error in "
                f"{tool_name}: {str(e)}"
            )

    def _parse_llm_response(
        self,
        response: str
    ) -> Dict[str, Any]:

        try:

            cleaned = response.strip()

            # Remove markdown fences if Gemini
            # accidentally produces them.

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

            decision = json.loads(
                cleaned.strip()
            )

            # Basic structure validation

            required_keys = {
                "thought",
                "status",
                "action",
                "action_input"
            }

            missing = (
                required_keys
                - set(decision.keys())
            )

            if missing:

                return {
                    "status": "ERROR",
                    "message":
                    "LLM JSON is missing keys: "
                    + ", ".join(missing)
                }

            valid_statuses = {
                "CONTINUE",
                "SUCCESS",
                "NEEDS_INPUT"
            }

            if decision["status"] not in valid_statuses:

                return {
                    "status": "ERROR",
                    "message":
                    f"Invalid status: "
                    f"{decision['status']}"
                }

            if not isinstance(
                decision["action_input"],
                dict
            ):

                return {
                    "status": "ERROR",
                    "message":
                    "action_input must be a dictionary."
                }

            return decision

        except (
            json.JSONDecodeError,
            IndexError,
            TypeError
        ):

            return {
                "status": "ERROR",
                "message":
                "LLM output was not valid JSON. "
                "Return ONLY a JSON object with "
                "thought, status, action, action_input."
            }

