"""
AutoPR - Role 2: GitHub Integration Tools

Responsibilities:
- Fetch GitHub Issue details
- Create Pull Requests
- Provide small, deterministic wrappers for the agent

Credentials are read from environment variables.
"""

import os
from typing import Any, Dict, Optional

from github import Github
from github.GithubException import GithubException


class GitHubToolError(Exception):
    """Raised when a GitHub operation cannot be completed."""


def _get_client() -> Github:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise GitHubToolError(
            "GITHUB_TOKEN is not configured. Add it to the environment."
        )
    return Github(token, timeout=20)


def _get_repo(repo_name: str):
    if not repo_name or "/" not in repo_name:
        raise GitHubToolError(
            "repo_name must be in the format 'owner/repository'."
        )

    try:
        return _get_client().get_repo(repo_name)
    except GithubException as exc:
        raise GitHubToolError(
            f"Unable to access repository '{repo_name}': "
            f"GitHub returned {exc.status}."
        ) from exc


def get_issue_details(repo_name: str, issue_number: int) -> Dict[str, Any]:
    """
    Fetch a GitHub Issue and return agent-friendly structured data.

    Returns:
        {
            "id": int,
            "number": int,
            "title": str,
            "body": str,
            "acceptance_criteria": str,
            "state": str,
            "url": str,
            "labels": list[str]
        }
    """
    try:
        issue_number = int(issue_number)
    except (TypeError, ValueError) as exc:
        raise GitHubToolError("issue_number must be an integer.") from exc

    if issue_number <= 0:
        raise GitHubToolError("issue_number must be greater than zero.")

    repo = _get_repo(repo_name)

    try:
        issue = repo.get_issue(issue_number)
    except GithubException as exc:
        raise GitHubToolError(
            f"Unable to fetch issue #{issue_number}: "
            f"GitHub returned {exc.status}."
        ) from exc

    # For the MVP, acceptance criteria are normally part of the Issue body.
    # Keep the complete body so the LLM can interpret its exact structure.
    body = issue.body or ""

    return {
        "id": issue.id,
        "number": issue.number,
        "title": issue.title,
        "body": body,
        "acceptance_criteria": body,
        "state": issue.state,
        "url": issue.html_url,
        "labels": [label.name for label in issue.labels],
    }


def create_pull_request(
    repo_name: str,
    branch_name: str,
    title: str,
    body: str,
    base_branch: str = "main",
) -> Dict[str, Any]:
    """
    Create a GitHub Pull Request.

    The branch must already exist on the remote repository.
    """

    if not branch_name:
        raise GitHubToolError("branch_name is required.")

    if not title.strip():
        raise GitHubToolError("PR title cannot be empty.")

    if not base_branch:
        raise GitHubToolError("base_branch is required.")

    repo = _get_repo(repo_name)

    try:
        # Explicitly identify the owner of the source branch.
        head = f"{repo.owner.login}:{branch_name}"

        print(f"[GITHUB] Repository: {repo.full_name}")
        print(f"[GITHUB] Owner: {repo.owner.login}")
        print(f"[GITHUB] Head: {head}")
        print(f"[GITHUB] Base: {base_branch}")
        pr = repo.create_pull(
            title=title.strip(),
            body=body or "",
            head=head,
            base=base_branch,
        )

        return {
            "success": True,
            "pr_number": pr.number,
            "pr_url": pr.html_url,
            "title": pr.title,
            "head": head,
            "base": base_branch,
        }

    except GithubException as exc:
        raise GitHubToolError(
            f"Unable to create Pull Request: "
            f"GitHub returned {exc.status}. "
            f"Response: {exc.data}"
        ) from exc

    return {
        "number": pr.number,
        "title": pr.title,
        "url": pr.html_url,
        "state": pr.state,
        "head_branch": pr.head.ref,
        "base_branch": pr.base.ref,
    }


def add_issue_comment(
    repo_name: str,
    issue_number: int,
    comment: str,
) -> Dict[str, Any]:
    """Add a status/result comment to the original GitHub Issue."""
    if not comment.strip():
        raise GitHubToolError("comment cannot be empty.")

    repo = _get_repo(repo_name)

    try:
        issue = repo.get_issue(int(issue_number))
        created = issue.create_comment(comment)
    except (GithubException, ValueError) as exc:
        raise GitHubToolError(
            f"Unable to comment on issue #{issue_number}."
        ) from exc

    return {
        "comment_id": created.id,
        "issue_number": int(issue_number),
        "url": created.html_url,
    }


def close() -> None:
    """Optional cleanup hook for callers that want to explicitly close the client."""
    # PyGithub handles its underlying requests session; no explicit action is
    # required for the simple MVP wrapper.
    return None
