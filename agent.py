"""
Core autonomous agent for AutoPR.

Implements a bounded:

    Reason -> Act -> Observe -> Repeat

loop using Gemini as the reasoning model.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable


class AutoPRAgent:
    """
    Bounded autonomous coding agent.

    The agent:
        1. Sends the task and recent history to Gemini.
        2. Gemini chooses the next tool.
        3. AutoPR executes that tool.
        4. The result is returned to Gemini.
        5. The process repeats until DONE or NEEDS_INPUT.
    """

    VALID_STATUSES = {
        "CONTINUE",
        "DONE",
        "NEEDS_INPUT",
    }

    def __init__(
        self,
        llm_client: Any,
        max_retries: int = 10,
    ) -> None:
        if llm_client is None:
            raise ValueError(
                "llm_client cannot be None."
            )

        if max_retries < 1:
            raise ValueError(
                "max_retries must be at least 1."
            )

        self.llm_client = llm_client
        self.max_retries = max_retries

        self.history: list[dict[str, Any]] = []

        self.tools: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------
    # TOOL REGISTRATION
    # ---------------------------------------------------------

    def register_tool(
        self,
        name: str,
        func: Callable[..., Any],
        schema: dict[str, Any],
    ) -> None:
        """
        Register a tool that Gemini can select.

        The schema is shown directly to Gemini so Gemini knows
        the exact parameter names expected by the function.
        """

        if not name:
            raise ValueError(
                "Tool name cannot be empty."
            )

        if not callable(func):
            raise TypeError(
                f"Tool '{name}' must be callable."
            )

        if not isinstance(schema, dict):
            raise TypeError(
                f"Schema for '{name}' must be a dict."
            )

        self.tools[name] = {
            "func": func,
            "schema": schema,
        }

    # ---------------------------------------------------------
    # TOOL DESCRIPTION
    # ---------------------------------------------------------

    def _build_tool_descriptions(self) -> str:
        """
        Convert registered tools into a clear description
        that Gemini can understand.

        IMPORTANT:
        Gemini must use the exact argument names shown here.
        """

        if not self.tools:
            return (
                "AVAILABLE TOOLS:\n"
                "No tools are currently registered."
            )

        lines = [
            "AVAILABLE TOOLS:",
            "",
            "CRITICAL TOOL ARGUMENT RULE:",
            "Use ONLY the exact parameter names shown in each "
            "tool schema.",
            "Do NOT rename parameters.",
            "Do NOT invent parameters.",
            "",
        ]

        for name, data in self.tools.items():
            schema = data["schema"]

            try:
                schema_json = json.dumps(
                    schema,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            except TypeError:
                schema_json = "{}"

            lines.append(
                f"- {name}: {schema_json}"
            )

            # Add Python signature as a second source of truth.
            try:
                signature = inspect.signature(
                    data["func"]
                )

                parameters = []

                for parameter_name, parameter in (
                    signature.parameters.items()
                ):
                    if parameter.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue

                    if parameter.default is inspect.Parameter.empty:
                        required = "required"
                    else:
                        required = "optional"

                    parameters.append(
                        f"{parameter_name} ({required})"
                    )

                if parameters:
                    lines.append(
                        "  Exact Python parameters: "
                        + ", ".join(parameters)
                    )

            except Exception:
                pass

            lines.append("")

        return "\n".join(lines)

    # ---------------------------------------------------------
    # ARGUMENT VALIDATION
    # ---------------------------------------------------------

    def _validate_action_input(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> None:
        """
        Validate that Gemini supplied the correct argument names
        before attempting to execute a tool.

        This prevents errors such as:

            read_file(file_path="calculator.py")

        when the real function expects:

            read_file(filename="calculator.py")
        """

        if action not in self.tools:
            raise ValueError(
                f"Unknown tool '{action}'."
            )

        if not isinstance(action_input, dict):
            raise ValueError(
                "action_input must be a JSON object."
            )

        tool = self.tools[action]
        function = tool["func"]

        try:
            signature = inspect.signature(function)
        except Exception:
            return

        parameters = signature.parameters

        accepted_names = set()
        required_names = set()

        accepts_kwargs = False

        for name, parameter in parameters.items():

            if parameter.kind == (
                inspect.Parameter.VAR_KEYWORD
            ):
                accepts_kwargs = True
                continue

            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue

            accepted_names.add(name)

            if parameter.default is inspect.Parameter.empty:
                required_names.add(name)

        supplied_names = set(
            action_input.keys()
        )

        unknown_names = (
            supplied_names - accepted_names
        )

        missing_names = (
            required_names - supplied_names
        )

        if unknown_names and not accepts_kwargs:
            raise ValueError(
                f"Tool '{action}' received invalid "
                f"argument(s): "
                f"{', '.join(sorted(unknown_names))}. "
                f"Expected parameter(s): "
                f"{', '.join(sorted(accepted_names))}."
            )

        if missing_names:
            raise ValueError(
                f"Tool '{action}' is missing required "
                f"argument(s): "
                f"{', '.join(sorted(missing_names))}."
            )

    # ---------------------------------------------------------
    # HISTORY
    # ---------------------------------------------------------

    def _build_recent_history(
        self,
        max_messages: int = 8,
        max_chars_per_message: int = 5000,
    ) -> list[dict[str, str]]:
        """
        Return a bounded portion of the conversation.

        This prevents the prompt from growing without
        limits during long coding tasks.
        """

        recent = self.history[-max_messages:]

        compact: list[dict[str, str]] = []

        for message in recent:
            role = str(
                message.get(
                    "role",
                    "user",
                )
            )

            content = str(
                message.get(
                    "content",
                    "",
                )
            )

            if len(content) > max_chars_per_message:
                content = (
                    content[:max_chars_per_message]
                    + "\n...[truncated]..."
                )

            compact.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return compact

    # ---------------------------------------------------------
    # SYSTEM PROMPT
    # ---------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """
        Build the main instruction given to Gemini.
        """

        tools = self._build_tool_descriptions()

        return f"""
You are AutoPR, an autonomous software engineering agent.

Your job is to complete a coding work item safely and
traceably.

You operate using a bounded Reason -> Act -> Observe loop.

You must:

1. Understand the work item.
2. Inspect the repository before changing code.
3. Read repository instructions and coding rules.
4. Read relevant documentation and requirements.
5. Inspect existing source code.
6. Inspect relevant tests.
7. Make the smallest appropriate implementation.
8. Run the repository-approved validation commands.
9. Analyze failures.
10. Fix failures when possible.
11. Re-run validation.
12. Only report completion after validation succeeds.
13. Never claim that a test passed unless a validation tool
    actually reported success.
14. Never invent files, test results, PR URLs, or repository
    information.
15. If genuinely required information is missing, request
    human input.

IMPORTANT SAFETY RULES:

- Only use registered tools.
- Never invent a tool.
- Never invent tool arguments.
- Do not modify unrelated files.
- Respect repository coding rules.
- Prefer inspecting before modifying.
- Do not claim success without evidence.
- If a tool fails, inspect the error and recover when possible.
- If recovery is not possible, return NEEDS_INPUT.

CRITICAL TOOL ARGUMENT RULES:

- You MUST use the exact parameter names shown in
  AVAILABLE TOOLS.
- Never rename a parameter.
- Never use "path" when the tool expects "filename".
- Never use "file_path" when the tool expects "filename".
- For read_file, the parameter is EXACTLY "filename".
- Correct example:
  {{"filename":"calculator.py"}}
- Incorrect examples:
  {{"path":"calculator.py"}}
  {{"file_path":"calculator.py"}}
- For every other tool, use ONLY the exact parameter names
  shown in its schema and Python parameter list.
- If a previous tool call failed because of an argument name,
  correct the argument name before trying again.

{tools}

RESPONSE FORMAT:

Return ONLY valid JSON.

Do not use Markdown.
Do not use code fences.
Do not include explanations outside JSON.

For an action:

{{
  "thought": "short description of the next action",
  "status": "CONTINUE",
  "action": "registered_tool_name",
  "action_input": {{}}
}}

When the work item is completely finished:

{{
  "thought": "short completion summary",
  "status": "DONE",
  "action": "",
  "action_input": {{}}
}}

When human input is genuinely required:

{{
  "thought": "short explanation of what information is missing",
  "status": "NEEDS_INPUT",
  "action": "",
  "action_input": {{}}
}}

VALID STATUS VALUES:

- CONTINUE
- DONE
- NEEDS_INPUT

The "action" field must contain a registered tool name
when status is CONTINUE.

The "action_input" field must always be a JSON object.

The "thought" field must remain short.
""".strip()

    # ---------------------------------------------------------
    # RESPONSE PARSING
    # ---------------------------------------------------------

    def _parse_llm_response(
        self,
        response: str,
    ) -> dict[str, Any]:
        """
        Parse and validate Gemini's JSON response.
        """

        if not response:
            raise ValueError(
                "Gemini returned an empty response."
            )

        response = response.strip()

        # Handle accidental markdown fences even though
        # the system prompt explicitly forbids them.
        if response.startswith("```"):
            response = (
                response
                .replace("```json", "", 1)
                .replace("```", "", 1)
                .strip()
            )

        try:
            decision = json.loads(response)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Gemini response was not valid JSON: "
                f"{exc}"
            ) from exc

        if not isinstance(decision, dict):
            raise ValueError(
                "Gemini response must be a JSON object."
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
            raise ValueError(
                "Gemini JSON is missing required keys: "
                + ", ".join(sorted(missing))
            )

        status = decision["status"]

        if status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status: {status}"
            )

        if not isinstance(
            decision["thought"],
            str,
        ):
            raise ValueError(
                "thought must be a string."
            )

        if not isinstance(
            decision["action"],
            str,
        ):
            raise ValueError(
                "action must be a string."
            )

        if not isinstance(
            decision["action_input"],
            dict,
        ):
            raise ValueError(
                "action_input must be an object."
            )

        return decision

    # ---------------------------------------------------------
    # TOOL EXECUTION
    # ---------------------------------------------------------

    def _execute_tool(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> Any:
        """
        Execute a registered tool safely.
        """

        if action not in self.tools:
            raise ValueError(
                f"Unknown tool '{action}'. "
                "Available tools: "
                + ", ".join(self.tools.keys())
            )

        # Validate argument names BEFORE execution.
        self._validate_action_input(
            action,
            action_input,
        )

        tool = self.tools[action]["func"]

        return tool(**action_input)

    # ---------------------------------------------------------
    # MAIN AGENT LOOP
    # ---------------------------------------------------------

    def run(
        self,
        initial_prompt: str,
    ) -> dict[str, Any]:
        """
        Execute the autonomous agent loop.
        """

        if not initial_prompt.strip():
            raise ValueError(
                "initial_prompt cannot be empty."
            )

        self.history = [
            {
                "role": "user",
                "content": initial_prompt,
            }
        ]

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
            print(
                f"\n[AGENT] Attempt {attempt}/"
                f"{self.max_retries}"
            )

            # -------------------------------------------------
            # ASK GEMINI
            # -------------------------------------------------

            try:
                system_prompt = (
                    self._build_system_prompt()
                )

                recent_history = (
                    self._build_recent_history()
                )

                print(
                    "[AGENT] Asking Gemini..."
                )

                response = (
                    self.llm_client.generate(
                        system_prompt,
                        recent_history,
                    )
                )

                print(
                    "[AGENT] Gemini response received."
                )

                decision = (
                    self._parse_llm_response(
                        response
                    )
                )

            except Exception as exc:
                error_message = (
                    "LLM error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(
                    f"[ERROR] {error_message}"
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

            # -------------------------------------------------
            # READ GEMINI DECISION
            # -------------------------------------------------

            thought = str(
                decision.get(
                    "thought",
                    "",
                )
            )

            status = decision["status"]
            action = decision["action"]
            action_input = decision["action_input"]

            print(
                f"[AGENT] Status: {status}"
            )

            if thought:
                print(
                    f"[AGENT] Thought: {thought}"
                )

            if action:
                print(
                    f"[AGENT] Action: {action}"
                )

            # -------------------------------------------------
            # DONE
            # -------------------------------------------------

            if status == "DONE":
                print(
                    "[AGENT] Task completed."
                )

                return {
                    "thought": thought,
                    "status": "DONE",
                    "action": "",
                    "action_input": {},
                }

            # -------------------------------------------------
            # NEEDS INPUT
            # -------------------------------------------------

            if status == "NEEDS_INPUT":
                print(
                    "[AGENT] Human input required."
                )

                return {
                    "thought": thought,
                    "status": "NEEDS_INPUT",
                    "action": action,
                    "action_input": action_input,
                }

            # -------------------------------------------------
            # CONTINUE VALIDATION
            # -------------------------------------------------

            if not action:
                error_message = (
                    "No action was provided. "
                    "Choose one of the available tools."
                )

                print(
                    f"[ERROR] {error_message}"
                )

                self.history.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            decision
                        ),
                    }
                )

                self.history.append(
                    {
                        "role": "user",
                        "content": error_message,
                    }
                )

                continue

            if action not in self.tools:
                error_message = (
                    f"Unknown tool '{action}'. "
                    "Choose one of the available tools."
                )

                print(
                    f"[ERROR] {error_message}"
                )

                self.history.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            decision
                        ),
                    }
                )

                self.history.append(
                    {
                        "role": "user",
                        "content": error_message,
                    }
                )

                continue

            # -------------------------------------------------
            # RECORD GEMINI DECISION
            # -------------------------------------------------

            self.history.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        decision,
                        ensure_ascii=False,
                    ),
                }
            )

            # -------------------------------------------------
            # EXECUTE TOOL
            # -------------------------------------------------

            try:
                print(
                    f"[AGENT] "
                    f"Executing tool: {action}"
                )

                result = (
                    self._execute_tool(
                        action,
                        action_input,
                    )
                )

                print(
                    f"[AGENT] "
                    f"Tool '{action}' completed."
                )

                # Convert tool result safely.
                try:
                    result_text = json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    )

                except Exception:
                    result_text = str(result)

                # Keep extremely large tool outputs
                # from exploding the conversation.
                max_result_chars = 10000

                if len(result_text) > max_result_chars:
                    result_text = (
                        result_text[:max_result_chars]
                        + "\n...[result truncated]..."
                    )

                self.history.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool '{action}' result:\n"
                            f"{result_text}"
                        ),
                    }
                )

            except Exception as exc:
                error_message = (
                    f"Tool '{action}' failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(
                    f"[ERROR] {error_message}"
                )

                self.history.append(
                    {
                        "role": "user",
                        "content": error_message,
                    }
                )

                # Do not immediately stop.
                # Gemini gets the error and can
                # attempt another strategy.
                continue

        # -----------------------------------------------------
        # RETRY LIMIT
        # -----------------------------------------------------

        print(
            "[AGENT] Maximum retry limit reached."
        )

        return {
            "thought": (
                "Maximum retry limit reached "
                "before completion."
            ),
            "status": "NEEDS_INPUT",
            "action": "",
            "action_input": {},
        }
