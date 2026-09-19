"""
Core autonomous agent for AutoPR.

Bounded:

    Reason -> Act -> Observe -> Repeat

Uses Gemini as the reasoning model and registered Python
tools for repository inspection, editing, testing, GitHub,
PR creation, and notifications.
"""

from __future__ import annotations

import inspect
import json
import time
from collections import Counter
from typing import Any, Callable


class AutoPRAgent:

    VALID_STATUSES = {
        "CONTINUE",
        "DONE",
        "NEEDS_INPUT",
    }

    INSPECTION_ACTIONS = {
        "list_files",
        "read_file",
        "get_issue_details",
    }

    VALIDATION_ACTIONS = {
        "run_command",
    }

    WRITE_ACTIONS = {
        "write_file",
    }

    GIT_ACTIONS = {
        "create_branch",
        "commit_changes",
        "push_branch",
        "create_pull_request",
        "add_issue_comment",
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

        self.history: list[
            dict[str, Any]
        ] = []

        self.tools: dict[
            str,
            dict[str, Any],
        ] = {}

        # Runtime tracking
        self.action_counts: Counter[str] = Counter()
        self.action_input_counts: Counter[str] = Counter()

        self.last_action: str | None = None
        self.last_action_input: dict[str, Any] | None = None
        self.last_action_result: str | None = None

        self.consecutive_inspections = 0
        self.consecutive_validation_runs = 0

        self.has_written_changes = False
        self.has_created_branch = False
        self.has_committed = False
        self.has_pushed = False
        self.has_created_pr = False

        self.llm_failures = 0

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
            "AVAILABLE TOOLS:"
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

            try:
                signature = str(
                    inspect.signature(
                        data["func"]
                    )
                )
            except Exception:
                signature = "(unknown)"

            lines.append(
                f"- {name}"
            )

            lines.append(
                f"  JSON schema: {schema_json}"
            )

            lines.append(
                f"  Python signature: {signature}"
            )

        return "\n".join(lines)

    # =========================================================
    # HISTORY
    # =========================================================

    def _build_recent_history(
        self,
        max_messages: int = 14,
        max_chars_per_message: int = 5000,
    ) -> list[dict[str, str]]:

        recent = self.history[
            -max_messages:
        ]

        compact: list[
            dict[str, str]
        ] = []

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

    # =========================================================
    # NORMALIZE ACTION INPUT
    # =========================================================

    def _normalize_action_input(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> dict[str, Any]:

        if not isinstance(
            action_input,
            dict,
        ):
            raise ValueError(
                f"Tool input for '{action}' "
                "must be a JSON object."
            )

        normalized = dict(
            action_input
        )

        aliases = {

            # -------------------------------------------------
            # FILE TOOLS
            # -------------------------------------------------

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

            # -------------------------------------------------
            # COMMAND
            # -------------------------------------------------

            "run_command": {
                "cmd": "command",
                "shell_command": "command",
                "command_line": "command",
            },

            # -------------------------------------------------
            # BRANCH
            # -------------------------------------------------

            "create_branch": {
                "branch": "branch_name",
                "name": "branch_name",
            },

            "push_branch": {
                "branch": "branch_name",
                "name": "branch_name",
            },

            # -------------------------------------------------
            # COMMIT
            # -------------------------------------------------

            "commit_changes": {
                "message": "commit_message",
                "msg": "commit_message",
            },

            # -------------------------------------------------
            # ISSUE
            # -------------------------------------------------

            "get_issue_details": {
                "issue": "issue_number",
                "number": "issue_number",
                "repository": "repo_name",
                "repo": "repo_name",
            },

            # -------------------------------------------------
            # PULL REQUEST
            # -------------------------------------------------

            "create_pull_request": {
                "repository": "repo_name",
                "repo": "repo_name",

                "pr_title": "title",
                "pr_body": "body",

                # Gemini/GitHub terminology -> actual tool arg
                "source_branch": "branch_name",
                "head": "branch_name",

                "target_branch": "base_branch",
            },

            # -------------------------------------------------
            # ISSUE COMMENT
            # -------------------------------------------------

            "add_issue_comment": {
                "repository": "repo_name",
                "repo": "repo_name",
                "issue": "issue_number",
                "number": "issue_number",
                "body": "comment",
            },
        }

        tool_aliases = aliases.get(
            action,
            {},
        )

        for old_key, new_key in (
            tool_aliases.items()
        ):

            if (
                old_key in normalized
                and new_key not in normalized
            ):
                normalized[new_key] = (
                    normalized.pop(old_key)
                )

        return normalized

    # =========================================================
    # VALIDATE ACTION INPUT
    # =========================================================

    def _validate_action_input(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> None:

        if action not in self.tools:

            raise ValueError(
                f"Unknown tool '{action}'. "
                "Available tools: "
                + ", ".join(
                    self.tools.keys()
                )
            )

        tool = self.tools[action]["func"]

        signature = inspect.signature(
            tool
        )

        parameters = signature.parameters

        required_names = []

        for name, parameter in parameters.items():

            if (
                parameter.default
                is inspect.Parameter.empty
                and parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ):
                required_names.append(
                    name
                )

        missing_names = [
            name
            for name in required_names
            if name not in action_input
        ]

        if missing_names:

            raise ValueError(
                "Missing required argument(s): "
                + ", ".join(
                    sorted(missing_names)
                )
            )

        unknown_names = [
            name
            for name in action_input
            if name not in parameters
        ]

        if unknown_names:

            raise ValueError(
                "Unknown argument(s): "
                + ", ".join(
                    sorted(unknown_names)
                )
            )

    # =========================================================
    # EXECUTE TOOL
    # =========================================================

    def _execute_tool(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> Any:
        """
        Normalize, validate and execute a registered tool.
        """

        if action not in self.tools:

            raise ValueError(
                f"Unknown tool '{action}'. "
                "Available tools: "
                + ", ".join(
                    self.tools.keys()
                )
            )

        normalized_input = (
            self._normalize_action_input(
                action,
                action_input,
            )
        )

        self._validate_action_input(
            action,
            normalized_input,
        )

        tool = self.tools[action]["func"]

        print(
            f"[TOOL] {action}({json.dumps(normalized_input, ensure_ascii=False)})"
        )

        try:
            return tool(
                **normalized_input
            )
        except Exception as exc:
            # GitHub returns 422 when a PR already exists for the
            # requested head branch. Treat that as a successful
            # end-state because the PR is already present.
            if (
                action == "create_pull_request"
                and "pull request already exists" in str(exc).lower()
            ):
                self.has_created_pr = True

                existing_result = {
                    "operation": "create_pull_request",
                    "success": True,
                    "already_exists": True,
                    "message": str(exc),
                    "branch_name": normalized_input.get(
                        "branch_name"
                    ),
                    "base_branch": normalized_input.get(
                        "base_branch",
                        "main",
                    ),
                }

                print(
                    "[GITHUB] Pull request already exists; "
                    "treating it as the completed PR state."
                )

                return existing_result

            raise

    # =========================================================
    # RUNTIME GUIDANCE
    # =========================================================

    def _build_runtime_guidance(self) -> str:

        if self.has_created_pr:

            next_step = (
                "The pull request exists. "
                "Add/update the work-item comment, "
                "then finish with DONE."
            )

        elif self.has_pushed:

            next_step = (
                "The branch has been pushed. "
                "Create the pull request now."
            )

        elif self.has_committed:

            next_step = (
                "The changes are committed. "
                "Push the branch now."
            )

        elif self.has_created_branch:

            if self.has_written_changes:

                next_step = (
                    "A branch exists and code has changed. "
                    "Validate if necessary, then commit."
                )

            else:

                next_step = (
                    "A branch exists but no implementation "
                    "has been made. Implement the work item."
                )

        elif self.has_written_changes:

            next_step = (
                "Code has changed. Run validation, "
                "fix failures if needed, then create the "
                "branch/commit/push/PR workflow."
            )

        elif self.consecutive_inspections >= 4:

            next_step = (
                "STOP INSPECTING. "
                "IMPLEMENT THE REQUEST NOW."
            )

        else:

            next_step = (
                "Inspect only the minimum context needed, "
                "then implement."
            )

        return f"""
=========================================================
RUNTIME PROGRESS
=========================================================

Inspection actions in a row:
{self.consecutive_inspections}

Validation actions in a row:
{self.consecutive_validation_runs}

Files changed:
{self.has_written_changes}

Branch created:
{self.has_created_branch}

Commit created:
{self.has_committed}

Branch pushed:
{self.has_pushed}

Pull request created:
{self.has_created_pr}

NEXT STEP:

{next_step}

Do not repeat identical inspections.

Do not repeatedly run validation without a code/config
change.

If a Git operation fails because of an argument mismatch,
correct the registered tool arguments and retry the same
registered tool.

The exact registered arguments are:
- commit_changes: commit_message
- create_pull_request: repo_name, branch_name, title, body, base_branch

Do NOT replace commit_changes or create_pull_request with
raw git commands or raw GitHub API requests.
""".strip()

    # =========================================================
    # SYSTEM PROMPT
    # =========================================================

    def _build_system_prompt(self) -> str:

        tools = (
            self._build_tool_descriptions()
        )

        runtime = (
            self._build_runtime_guidance()
        )

        return f"""
You are AutoPR, an autonomous software engineering agent.

Your job is to complete a coding work item safely,
correctly and traceably.

Use:

Reason -> Act -> Observe -> Repeat

=========================================================
WORKFLOW
=========================================================

1. Understand the work item.
2. Inspect repository structure.
3. Read repository rules.
4. Read relevant documentation.
5. Inspect relevant source.
6. Inspect relevant tests.
7. IMPLEMENT the requested change.
8. Run repository-approved validation.
9. Fix failures when possible.
10. Re-run validation after changes.
11. Create a branch.
12. Commit the changes.
13. Push the branch.
14. Create the pull request.
15. Add the work-item comment.
16. Finish with DONE.

=========================================================
IMPORTANT
=========================================================

Do NOT spend the whole run reading files.

Once you have:

- issue requirements
- repository rules
- relevant source
- relevant tests

IMPLEMENT THE REQUEST.

Do not repeatedly read the same file.

Do not repeatedly run the same test command without a
change.

Do not claim validation passed unless run_command actually
reports success.

Do not claim a PR exists unless create_pull_request actually
succeeds.

=========================================================
TOOL ARGUMENTS
=========================================================

Use the EXACT argument names from the registered Python
function signatures.

The AutoPR runtime normalizes common aliases automatically.

read_file:

{{
  "filename": "calculator.py"
}}

write_file:

{{
  "filename": "calculator.py",
  "content": "..."
}}

run_command:

{{
  "command": "python -m pytest -q"
}}

create_branch:

{{
  "branch_name": "autopr/issue-1-percentage"
}}

commit_changes:

{{
  "message": "Add percentage calculation"
}}

push_branch:

{{
  "branch_name": "autopr/issue-1-percentage"
}}

get_issue_details:

{{
  "repo_name": "djrmgaming0-star/AUTOPR-DEMO",
  "issue_number": 1
}}

create_pull_request:

Use the actual registered function arguments.

{{
  "repo_name": "djrmgaming0-star/AUTOPR-DEMO",
  "title": "Add percentage calculation",
  "body": "Implements #1",
  "branch_name": "autopr/issue-1-percentage",
  "base": "main"
}}

add_issue_comment:

{{
  "repo_name": "djrmgaming0-star/AUTOPR-DEMO",
  "issue_number": 1,
  "comment": "Implemented and validated."
}}

=========================================================
GIT TOOL RECOVERY
=========================================================

If commit_changes fails:

1. Read the exact error.
2. Inspect the registered signature/schema already
   provided above.
3. Use the exact argument name "commit_message".
4. Retry commit_changes.

Do NOT use raw git commit commands to bypass the tool.

If create_pull_request fails:

1. Read the exact error.
2. Use the exact registered arguments:
   repo_name, branch_name, title, body, base_branch.
3. Retry create_pull_request.
4. If GitHub explicitly reports that a pull request already
   exists for the branch, treat that existing PR as the
   completed PR state and continue with the work-item comment.

Do NOT use a raw GitHub API request to bypass the tool.

=========================================================
CODING RULES
=========================================================

Respect repository rules.

Respect existing project conventions.

Make the smallest appropriate change.

Do not modify unrelated files.

Do not invent requirements.

Do not claim tests passed without actual validation.

Do not claim a PR exists without actual PR creation.

If a tool fails:

1. Read the error.
2. Correct the action.
3. Continue.

=========================================================
RESPONSE FORMAT
=========================================================

Return ONLY valid JSON.

No Markdown.

No code fences.

No explanation outside JSON.

For a tool action:

{{
  "thought": "short description",
  "status": "CONTINUE",
  "action": "tool_name",
  "action_input": {{}}
}}

When completely finished:

{{
  "thought": "short completion summary",
  "status": "DONE",
  "action": "",
  "action_input": {{}}
}}

When human input is genuinely required:

{{
  "thought": "what information is missing",
  "status": "NEEDS_INPUT",
  "action": "",
  "action_input": {{}}
}}

VALID STATUS VALUES:

CONTINUE
DONE
NEEDS_INPUT

The action must be a registered tool when status is CONTINUE.

action_input must always be a JSON object.

Keep thought short.

=========================================================
RUNTIME GUIDANCE
=========================================================

{runtime}

=========================================================
AVAILABLE TOOLS
=========================================================

{tools}
""".strip()

    # =========================================================
    # PARSE GEMINI RESPONSE
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

        # Remove Markdown fences.
        if response.startswith("```"):

            lines = response.splitlines()

            if lines:
                lines = lines[1:]

            if (
                lines
                and lines[-1].strip() == "```"
            ):
                lines = lines[:-1]

            response = "\n".join(
                lines
            ).strip()

        # First attempt: direct JSON.
        try:

            decision = json.loads(
                response
            )

        except json.JSONDecodeError:

            # Gemini sometimes puts text around JSON.
            start = response.find("{")
            end = response.rfind("}")

            if (
                start == -1
                or end == -1
                or end <= start
            ):

                raise ValueError(
                    "Gemini response was not valid JSON."
                )

            try:

                decision = json.loads(
                    response[
                        start:end + 1
                    ]
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

        required = {
            "thought",
            "status",
            "action",
            "action_input",
        }

        missing = (
            required
            - set(decision.keys())
        )

        if missing:

            raise ValueError(
                "Gemini JSON is missing: "
                + ", ".join(
                    sorted(missing)
                )
            )

        if decision["status"] not in (
            self.VALID_STATUSES
        ):

            raise ValueError(
                f"Invalid status: "
                f"{decision['status']}"
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
    # ACTION KEY
    # =========================================================

    def _action_key(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> str:

        return (
            action
            + ":"
            + json.dumps(
                action_input,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
        )

    # =========================================================
    # REPEATED ACTION GUARD
    # =========================================================

    def _check_repeated_action(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> str | None:

        key = self._action_key(
            action,
            action_input,
        )

        count = (
            self.action_input_counts.get(
                key,
                0,
            )
        )

        # Do not repeatedly inspect the same thing.
        if (
            action in self.INSPECTION_ACTIONS
            and count >= 2
        ):

            return (
                f"The exact action '{action}' "
                f"was already executed {count} times. "
                "Do not repeat it. Move forward."
            )

        # Do not repeatedly run tests without changing code.
        if (
            action in self.VALIDATION_ACTIONS
            and count >= 2
            and not self.has_written_changes
        ):

            return (
                "The same validation command was already "
                "executed multiple times without code changes. "
                "Implement the requested change first."
            )

        return None

    # =========================================================
    # RESULT TEXT
    # =========================================================

    def _result_to_text(
        self,
        result: Any,
        max_chars: int = 9000,
    ) -> str:

        try:

            text = json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            )

        except Exception:

            text = str(
                result
            )

        if len(text) > max_chars:

            text = (
                text[:max_chars]
                + "\n...[result truncated]..."
            )

        return text

    # =========================================================
    # HISTORY HELPERS
    # =========================================================

    def _record_assistant(
        self,
        decision: dict[str, Any],
    ) -> None:

        self.history.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    decision,
                    ensure_ascii=False,
                ),
            }
        )

    def _record_user(
        self,
        content: str,
    ) -> None:

        self.history.append(
            {
                "role": "user",
                "content": content,
            }
        )

    # =========================================================
    # UPDATE RUNTIME STATE
    # =========================================================

    def _update_state(
        self,
        action: str,
        action_input: dict[str, Any],
        result: Any,
    ) -> None:

        self.action_counts[action] += 1

        key = self._action_key(
            action,
            action_input,
        )

        self.action_input_counts[key] += 1

        self.last_action = action

        self.last_action_input = dict(
            action_input
        )

        self.last_action_result = (
            self._result_to_text(
                result,
                max_chars=2000,
            )
        )

        if action in self.INSPECTION_ACTIONS:

            self.consecutive_inspections += 1

        else:

            self.consecutive_inspections = 0

        if action in self.VALIDATION_ACTIONS:

            self.consecutive_validation_runs += 1

        else:

            self.consecutive_validation_runs = 0

        if action in self.WRITE_ACTIONS:

            self.has_written_changes = True

        if action == "create_branch":

            self.has_created_branch = True

        elif action == "commit_changes":

            self.has_committed = True

        elif action == "push_branch":

            self.has_pushed = True

        elif action == "create_pull_request":

            self.has_created_pr = True

    # =========================================================
    # GEMINI BACKOFF
    # =========================================================

    def _backoff(
        self,
        error: Exception,
    ) -> None:

        message = str(
            error
        ).lower()

        if (
            "429" in message
            or "resource_exhausted" in message
            or "quota" in message
            or "rate limit" in message
        ):

            # Respect free-tier rate limits.
            delay = min(
                max(
                    15,
                    10 * self.llm_failures,
                ),
                60,
            )

        else:

            delay = min(
                2 * self.llm_failures,
                10,
            )

        print(
            f"[GEMINI] Waiting {delay}s before retry..."
        )

        time.sleep(
            delay
        )

    # =========================================================
    # MAIN AGENT LOOP
    # =========================================================

    def run(
        self,
        initial_prompt: str,
    ) -> dict[str, Any]:

        if not initial_prompt.strip():

            raise ValueError(
                "initial_prompt cannot be empty."
            )

        # Reset conversation.
        self.history = [
            {
                "role": "user",
                "content": initial_prompt,
            }
        ]

        # Reset runtime state.
        self.action_counts.clear()
        self.action_input_counts.clear()

        self.last_action = None
        self.last_action_input = None
        self.last_action_result = None

        self.consecutive_inspections = 0
        self.consecutive_validation_runs = 0

        self.has_written_changes = False
        self.has_created_branch = False
        self.has_committed = False
        self.has_pushed = False
        self.has_created_pr = False

        self.llm_failures = 0

        # =====================================================
        # AGENT LOOP
        # =====================================================

        for attempt in range(
            1,
            self.max_retries + 1,
        ):

            print(
                f"\n[AGENT] Attempt "
                f"{attempt}/"
                f"{self.max_retries}"
            )

            # =================================================
            # ASK GEMINI
            # =================================================

            try:

                print(
                    "[AGENT] Asking Gemini..."
                )

                response = (
                    self.llm_client.generate(
                        self._build_system_prompt(),
                        self._build_recent_history(),
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

                # Successful Gemini request.
                self.llm_failures = 0

            except Exception as exc:

                self.llm_failures += 1

                error_message = (
                    f"LLM error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(
                    f"[ERROR] {error_message}"
                )

                self._record_user(
                    error_message
                )

                self._backoff(
                    exc
                )

                continue

            # =================================================
            # READ DECISION
            # =================================================

            thought = str(
                decision["thought"]
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
                f"[AGENT] Status: {status}"
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

            # =================================================
            # DONE
            # =================================================

            if status == "DONE":

                if not self.has_created_pr:

                    error = (
                        "You cannot finish yet. "
                        "The pull request has not been created."
                    )

                    print(
                        f"[GUARD] {error}"
                    )

                    self._record_assistant(
                        decision
                    )

                    self._record_user(
                        error
                    )

                    continue

                print(
                    "[AGENT] Task completed."
                )

                return {
                    "thought": thought,
                    "status": "DONE",
                    "action": "",
                    "action_input": {},
                }

            # =================================================
            # NEEDS INPUT
            # =================================================

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

            # =================================================
            # ACTION VALIDATION
            # =================================================

            if not action:

                error = (
                    "No action was provided. "
                    "Choose a registered tool."
                )

                print(
                    f"[ERROR] {error}"
                )

                self._record_assistant(
                    decision
                )

                self._record_user(
                    error
                )

                continue

            if action not in self.tools:

                error = (
                    f"Unknown tool '{action}'. "
                    "Choose a registered tool."
                )

                print(
                    f"[ERROR] {error}"
                )

                self._record_assistant(
                    decision
                )

                self._record_user(
                    error
                )

                continue

            # =================================================
            # NORMALIZE
            # =================================================

            try:

                normalized_input = (
                    self._normalize_action_input(
                        action,
                        action_input,
                    )
                )

            except Exception as exc:

                error = (
                    f"Invalid tool input: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(
                    f"[ERROR] {error}"
                )

                self._record_assistant(
                    decision
                )

                self._record_user(
                    error
                )

                continue

            # =================================================
            # REPEATED ACTION GUARD
            # =================================================

            repeat_error = (
                self._check_repeated_action(
                    action,
                    normalized_input,
                )
            )

            if repeat_error:

                print(
                    f"[GUARD] {repeat_error}"
                )

                self._record_assistant(
                    decision
                )

                self._record_user(
                    "ACTION BLOCKED:\n"
                    + repeat_error
                )

                continue

            # =================================================
            # RECORD DECISION
            # =================================================

            self._record_assistant(
                decision
            )

            # =================================================
            # EXECUTE TOOL
            # =================================================

            try:

                print(
                    f"[AGENT] Executing tool: "
                    f"{action}"
                )

                result = (
                    self._execute_tool(
                        action,
                        normalized_input,
                    )
                )

                print(
                    f"[AGENT] Tool "
                    f"'{action}' completed."
                )

                self._update_state(
                    action,
                    normalized_input,
                    result,
                )

                result_text = (
                    self._result_to_text(
                        result
                    )
                )

                self._record_user(
                    f"Tool '{action}' result:\n"
                    f"{result_text}"
                )

            except Exception as exc:

                error = (
                    f"Tool '{action}' failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(
                    f"[ERROR] {error}"
                )

                self._record_user(
                    error
                )

                continue

        # =====================================================
        # MAX RETRIES
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
