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
    """Bounded autonomous coding agent."""

    VALID_STATUSES = {
        "CONTINUE",
        "DONE",
        "NEEDS_INPUT",
    }

    def __init__(
        self,
        llm_client: Any,
        max_retries: int = 25,
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

        self.tools: dict[
            str,
            dict[str, Any],
        ] = {}

    # =========================================================
    # TOOL REGISTRATION
    # =========================================================

    def register_tool(
        self,
        name: str,
        func: Callable[..., Any],
        schema: dict[str, Any],
    ) -> None:

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

    # =========================================================
    # TOOL DESCRIPTION
    # =========================================================

    def _build_tool_descriptions(self) -> str:

        if not self.tools:
            return (
                "AVAILABLE TOOLS:\n"
                "No tools are currently registered."
            )

        lines = [
            "AVAILABLE TOOLS:",
            "",
            "CRITICAL TOOL ARGUMENT RULE:",
            "Use ONLY the exact parameter names shown.",
            "Never rename parameters.",
            "Never invent parameters.",
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

            # Also show the actual Python signature.
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

                    if (
                        parameter.default
                        is inspect.Parameter.empty
                    ):
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

    # =========================================================
    # ARGUMENT VALIDATION
    # =========================================================

    def _validate_action_input(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> None:

        if action not in self.tools:
            raise ValueError(
                f"Unknown tool '{action}'."
            )

        if not isinstance(
            action_input,
            dict,
        ):
            raise ValueError(
                "action_input must be a JSON object."
            )

        function = self.tools[action]["func"]

        try:
            signature = inspect.signature(
                function
            )
        except Exception:
            return

        accepted_names = set()
        required_names = set()

        accepts_kwargs = False

        for name, parameter in (
            signature.parameters.items()
        ):

            if parameter.kind == (
                inspect.Parameter.VAR_KEYWORD
            ):
                accepts_kwargs = True
                continue

            if parameter.kind == (
                inspect.Parameter.VAR_POSITIONAL
            ):
                continue

            accepted_names.add(name)

            if (
                parameter.default
                is inspect.Parameter.empty
            ):
                required_names.add(name)

        supplied_names = set(
            action_input.keys()
        )

        unknown_names = (
            supplied_names
            - accepted_names
        )

        missing_names = (
            required_names
            - supplied_names
        )

        if (
            unknown_names
            and not accepts_kwargs
        ):
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
                f"{', ".join(sorted(missing_names))}."
            )

    # =========================================================
    # NORMALIZE GEMINI ARGUMENTS
    # =========================================================

    def _normalize_action_input(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Repair common argument-name mistakes from Gemini.

        Example:

            {"file_path": "calculator.py"}

        becomes:

            {"filename": "calculator.py"}
        """

        normalized = dict(
            action_input
        )

        # -----------------------------------------------------
        # read_file
        # -----------------------------------------------------

        if action == "read_file":

            if "filename" not in normalized:

                aliases = [
                    "file_path",
                    "path",
                    "file",
                    "target",
                    "name",
                ]

                for alias in aliases:

                    if alias in normalized:

                        normalized[
                            "filename"
                        ] = normalized.pop(
                            alias
                        )

                        print(
                            f"[AGENT] Normalized "
                            f"read_file argument "
                            f"'{alias}' -> 'filename'"
                        )

                        break

        # -----------------------------------------------------
        # write_file
        # -----------------------------------------------------

        if action == "write_file":

            if "filename" not in normalized:

                aliases = [
                    "file_path",
                    "path",
                    "file",
                    "target",
                    "name",
                ]

                for alias in aliases:

                    if alias in normalized:

                        normalized[
                            "filename"
                        ] = normalized.pop(
                            alias
                        )

                        print(
                            f"[AGENT] Normalized "
                            f"write_file argument "
                            f"'{alias}' -> 'filename'"
                        )

                        break

        # -----------------------------------------------------
        # run_command
        # -----------------------------------------------------

        if action == "run_command":

            if "command" not in normalized:

                aliases = [
                    "cmd",
                    "shell_command",
                    "command_line",
                ]

                for alias in aliases:

                    if alias in normalized:

                        normalized[
                            "command"
                        ] = normalized.pop(
                            alias
                        )

                        print(
                            f"[AGENT] Normalized "
                            f"run_command argument "
                            f"'{alias}' -> 'command'"
                        )

                        break

        # -----------------------------------------------------
        # create_branch
        # -----------------------------------------------------

        if action == "create_branch":

            if "branch_name" not in normalized:

                aliases = [
                    "branch",
                    "name",
                ]

                for alias in aliases:

                    if alias in normalized:

                        normalized[
                            "branch_name"
                        ] = normalized.pop(
                            alias
                        )

                        print(
                            f"[AGENT] Normalized "
                            f"create_branch argument "
                            f"'{alias}' -> 'branch_name'"
                        )

                        break

        # -----------------------------------------------------
        # push_branch
        # -----------------------------------------------------

        if action == "push_branch":

            if "branch_name" not in normalized:

                aliases = [
                    "branch",
                    "name",
                ]

                for alias in aliases:

                    if alias in normalized:

                        normalized[
                            "branch_name"
                        ] = normalized.pop(
                            alias
                        )

                        print(
                            f"[AGENT] Normalized "
                            f"push_branch argument "
                            f"'{alias}' -> 'branch_name'"
                        )

                        break

        # -----------------------------------------------------
        # commit_changes
        # -----------------------------------------------------

        if action == "commit_changes":

            if "message" not in normalized:

                aliases = [
                    "commit_message",
                    "msg",
                ]

                for alias in aliases:

                    if alias in normalized:

                        normalized[
                            "message"
                        ] = normalized.pop(
                            alias
                        )

                        print(
                            f"[AGENT] Normalized "
                            f"commit_changes argument "
                            f"'{alias}' -> 'message'"
                        )

                        break

        return normalized

    # =========================================================
    # HISTORY
    # =========================================================

    def _build_recent_history(
        self,
        max_messages: int = 10,
        max_chars_per_message: int = 6000,
    ) -> list[dict[str, str]]:

        recent = self.history[
            -max_messages:
        ]

        compact = []

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

            if len(content) > (
                max_chars_per_message
            ):

                content = (
                    content[
                        :max_chars_per_message
                    ]
                    + "\n...[truncated]..."
                )

            compact.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return compact

    # =========================================================
    # SYSTEM PROMPT
    # =========================================================

    def _build_system_prompt(self) -> str:

        tools = (
            self._build_tool_descriptions()
        )

        return f"""
You are AutoPR, an autonomous software engineering agent.

Your job is to complete a coding work item safely and
traceably.

You operate using a bounded:

Reason -> Act -> Observe -> Repeat

loop.

You MUST:

1. Understand the work item.
2. Inspect the repository.
3. Read repository rules.
4. Read relevant documentation.
5. Inspect existing source code.
6. Inspect existing tests.
7. Implement the smallest correct change.
8. Run real validation commands.
9. Inspect failures.
10. Fix failures when possible.
11. Re-run validation.
12. Only report DONE after real validation succeeds.
13. Never invent test results.
14. Never invent repository information.
15. Never claim a PR exists unless the PR tool actually
    succeeds.
16. Ask for human input only when genuinely required.

IMPORTANT TOOL RULES:

- Only use registered tools.
- Never invent tools.
- Never invent parameters.
- Use the EXACT parameter names shown in AVAILABLE TOOLS.
- If a previous call failed because of a parameter name,
  correct it immediately.
- Do not repeatedly retry the same invalid argument.

SPECIAL READ_FILE RULE:

The read_file tool uses:

{{"filename":"calculator.py"}}

It does NOT use:

{{"path":"calculator.py"}}

It does NOT use:

{{"file_path":"calculator.py"}}

AVAILABLE TOOLS:

{tools}

WORKFLOW:

First inspect.

Then understand.

Then implement.

Then test.

Then debug if necessary.

Then create a feature branch.

Then commit.

Then push.

Then create the pull request.

Then optionally comment on the issue.

Finally return DONE only when the work is actually complete.

RESPONSE FORMAT:

Return ONLY valid JSON.

For CONTINUE:

{{
  "thought": "short description",
  "status": "CONTINUE",
  "action": "registered_tool_name",
  "action_input": {{}}
}}

For DONE:

{{
  "thought": "short completion summary",
  "status": "DONE",
  "action": "",
  "action_input": {{}}
}}

For NEEDS_INPUT:

{{
  "thought": "short explanation",
  "status": "NEEDS_INPUT",
  "action": "",
  "action_input": {{}}
}}

VALID STATUS VALUES:

CONTINUE
DONE
NEEDS_INPUT
""".strip()

    # =========================================================
    # RESPONSE PARSING
    # =========================================================

    def _parse_llm_response(
        self,
        response: str,
    ) -> dict[str, Any]:

        if not response:
            raise ValueError(
                "Gemini returned an empty response."
            )

        response = response.strip()

        if response.startswith("```"):

            response = (
                response
                .replace(
                    "```json",
                    "",
                    1,
                )
                .replace(
                    "```",
                    "",
                    1,
                )
                .strip()
            )

        try:

            decision = json.loads(
                response
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Gemini response was not valid JSON: "
                f"{exc}"
            ) from exc

        if not isinstance(
            decision,
            dict,
        ):

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
                + ", ".join(
                    sorted(missing)
                )
            )

        status = decision[
            "status"
        ]

        if status not in (
            self.VALID_STATUSES
        ):

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

    # =========================================================
    # TOOL EXECUTION
    # =========================================================

    def _execute_tool(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> Any:

        if action not in self.tools:

            raise ValueError(
                f"Unknown tool '{action}'. "
                "Available tools: "
                + ", ".join(
                    self.tools.keys()
                )
            )

        # Normalize Gemini's argument names first.
        normalized_input = (
            self._normalize_action_input(
                action,
                action_input,
            )
        )

        # Validate after normalization.
        self._validate_action_input(
            action,
            normalized_input,
        )

        tool = self.tools[action][
            "func"
        ]

        return tool(
            **normalized_input
        )

    # =========================================================
    # MAIN LOOP
    # =========================================================

    def run(
        self,
        initial_prompt: str,
    ) -> dict[str, Any]:

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
                f"\n[AGENT] Attempt "
                f"{attempt}/"
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
                    f"[ERROR] "
                    f"{error_message}"
                )

                self.history.append(
                    {
                        "role": "user",
                        "content": error_message,
                    }
                )

                if attempt >= (
                    self.max_retries
                ):

                    return {
                        "thought": error_message,
                        "status": "NEEDS_INPUT",
                        "action": "",
                        "action_input": {},
                    }

                continue

            # -------------------------------------------------
            # DECISION
            # -------------------------------------------------

            thought = str(
                decision.get(
                    "thought",
                    "",
                )
            )

            status = decision[
                "status"
            ]

            action = decision[
                "action"
            ]

            action_input = decision[
                "action_input"
            ]

            print(
                f"[AGENT] Status: "
                f"{status}"
            )

            if thought:

                print(
                    f"[AGENT] Thought: "
                    f"{thought}"
                )

            if action:

                print(
                    f"[AGENT] Action: "
                    f"{action}"
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
            # VALIDATE ACTION
            # -------------------------------------------------

            if not action:

                error_message = (
                    "No action was provided. "
                    "Choose one of the available tools."
                )

                print(
                    f"[ERROR] "
                    f"{error_message}"
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
                    f"[ERROR] "
                    f"{error_message}"
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
            # RECORD DECISION
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
                    f"Executing tool: "
                    f"{action}"
                )

                result = (
                    self._execute_tool(
                        action,
                        action_input,
                    )
                )

                print(
                    f"[AGENT] Tool "
                    f"'{action}' completed."
                )

                try:

                    result_text = json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    )

                except Exception:

                    result_text = str(
                        result
                    )

                max_result_chars = 10000

                if len(result_text) > (
                    max_result_chars
                ):

                    result_text = (
                        result_text[
                            :max_result_chars
                        ]
                        + "\n...[result truncated]..."
                    )

                self.history.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool '{action}' "
                            "result:\n"
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
                    f"[ERROR] "
                    f"{error_message}"
                )

                self.history.append(
                    {
                        "role": "user",
                        "content": error_message,
                    }
                )

                continue

        # =====================================================
        # RETRY LIMIT
        # =====================================================

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
