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

    main()
