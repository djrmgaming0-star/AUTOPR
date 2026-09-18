# AutoPR — Autonomous Software Engineering Agent

AutoPR turns a software Work Item / GitHub Issue into a validated code change.

**Workflow**

`Issue → Context & Rules → Inspect → Implement → Test → Self-Debug → Git Branch → Commit → Push → Pull Request → Slack`

The included `dummy_repo/` is only a safe demo workspace. The agent is **not hardcoded to calculator functionality**. Set `WORK_ITEM_ID` and `GITHUB_REPO` to run it against a different issue/repository.

## 1. Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your own credentials. **Never commit `.env`.**

Required:
- `GEMINI_API_KEY`
- `GITHUB_TOKEN`
- `GITHUB_REPO` (for example `owner/repository`)
- `WORK_ITEM_ID`

Optional:
- `SLACK_WEBHOOK_URL`
- `BASE_BRANCH` (default `main`)
- `GEMINI_MODEL`
- `WORKSPACE_DIR`
- `MAX_AGENT_ITERATIONS`
- `COMMAND_TIMEOUT`

## 2. Run

```powershell
python main.py
```

For a GitHub Issue, AutoPR retrieves the issue title/body/labels, reads repository rules, inspects the workspace, implements or repairs code, validates it, and only creates a PR after a real source change has been verified.

## 3. Demo

The demo issue is in `dummy_repo/tickets.json`. To test implementation/debugging locally, change the demo repository or add a ticket and run the agent with a demo configuration.

## Security

Credentials belong in environment variables only. If a real credential was ever committed or shared, revoke/rotate it before using the repository.
