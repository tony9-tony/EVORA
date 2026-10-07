"""GitHub tools. Reading is free; creating issues/comments always needs the owner's approval."""

from __future__ import annotations

import base64
import os
from typing import Any, Optional

import httpx

from evora.security import PermissionLevel, PermissionManager
from evora.tools import Tool, ToolResult

API = "https://api.github.com"


def _headers(vault) -> dict:
    token = None
    try:
        token = vault.get("github_token") if vault is not None else None
    except Exception:
        token = None
    token = token or os.environ.get("GITHUB_TOKEN")
    h = {"Accept": "application/vnd.github+json", "User-Agent": "EVORA"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


class _GithubBase(Tool):
    transport: Optional[httpx.AsyncBaseTransport] = None  # injected in tests

    def __init__(self, security: PermissionManager, logger=None, vault=None):
        super().__init__(security, logger)
        self.vault = vault

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=30.0, headers=_headers(self.vault), transport=self.transport)


class GithubReadTool(_GithubBase):
    name = "github_read"
    description = (
        "Read from GitHub. kind: 'repo' (info), 'issues', 'pulls', 'file' (needs path), 'tree' (list a folder), "
        "'search_code' (needs query). repo is 'owner/name'."
    )
    permission = PermissionLevel.SAFE
    parameters = {
        "kind": {"type": "string", "description": "repo | issues | pulls | file | tree | search_code", "required": True},
        "repo": {"type": "string", "description": "owner/name", "required": False},
        "path": {"type": "string", "description": "File or folder path inside the repo.", "required": False},
        "ref": {"type": "string", "description": "Branch, tag or commit.", "required": False},
        "query": {"type": "string", "description": "Search text for search_code.", "required": False},
    }

    async def execute(self, kind: str, repo: str = "", path: str = "", ref: str = "", query: str = "") -> ToolResult:
        try:
            async with self._client() as client:
                if kind == "search_code":
                    r = await client.get(f"{API}/search/code", params={"q": f"{query} repo:{repo}" if repo else query})
                    r.raise_for_status()
                    items = r.json().get("items", [])[:10]
                    return ToolResult(success=True, output="\n".join(f"- {i['repository']['full_name']}: {i['path']}" for i in items) or "No matches")
                if "/" not in repo:
                    return ToolResult(success=False, error="repo must look like owner/name")
                params = {"ref": ref} if ref else None
                if kind == "repo":
                    r = await client.get(f"{API}/repos/{repo}")
                    r.raise_for_status()
                    d = r.json()
                    return ToolResult(success=True, output=(
                        f"{d['full_name']} — {d.get('description') or 'no description'}\n"
                        f"default branch: {d['default_branch']} · stars: {d['stargazers_count']} · "
                        f"open issues: {d['open_issues_count']} · private: {d['private']}"))
                if kind in ("issues", "pulls"):
                    r = await client.get(f"{API}/repos/{repo}/{'issues' if kind == 'issues' else 'pulls'}", params={"state": "open", "per_page": 15})
                    r.raise_for_status()
                    rows = [i for i in r.json() if kind == "pulls" or "pull_request" not in i]
                    return ToolResult(success=True, output="\n".join(f"#{i['number']} {i['title']} ({i['user']['login']})" for i in rows) or "None open")
                if kind in ("file", "tree"):
                    r = await client.get(f"{API}/repos/{repo}/contents/{path}".rstrip("/"), params=params)
                    r.raise_for_status()
                    d = r.json()
                    if isinstance(d, list):
                        return ToolResult(success=True, output="\n".join(f"{i['type'][0]} {i['path']}" for i in d))
                    text = base64.b64decode(d["content"]).decode("utf-8", "replace") if d.get("encoding") == "base64" else d.get("content", "")
                    return ToolResult(success=True, output=text[:8000], data={"sha": d.get("sha")})
                return ToolResult(success=False, error=f"Unknown kind '{kind}'")
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, error=f"GitHub said {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            return ToolResult(success=False, error=f"GitHub request failed: {e}")


class GithubWriteTool(_GithubBase):
    name = "github_write"
    description = "Create a GitHub issue or comment. kind: 'create_issue' (title, body) or 'comment' (number, body). Always needs the owner's approval."
    permission = PermissionLevel.DANGEROUS
    parameters = {
        "kind": {"type": "string", "description": "create_issue | comment", "required": True},
        "repo": {"type": "string", "description": "owner/name", "required": True},
        "title": {"type": "string", "description": "Issue title.", "required": False},
        "body": {"type": "string", "description": "Text.", "required": True},
        "number": {"type": "integer", "description": "Issue/PR number for comments.", "required": False},
    }

    async def execute(self, kind: str, repo: str, body: str, title: str = "", number: int = 0) -> ToolResult:
        if "Authorization" not in _headers(self.vault):
            return ToolResult(success=False, error="No GitHub token. Add 'github_token' in the vault (Keys tab) or set GITHUB_TOKEN.")
        try:
            async with self._client() as client:
                if kind == "create_issue":
                    r = await client.post(f"{API}/repos/{repo}/issues", json={"title": title, "body": body})
                elif kind == "comment":
                    r = await client.post(f"{API}/repos/{repo}/issues/{int(number)}/comments", json={"body": body})
                else:
                    return ToolResult(success=False, error=f"Unknown kind '{kind}'")
                r.raise_for_status()
                return ToolResult(success=True, output=f"Done: {r.json().get('html_url')}")
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, error=f"GitHub said {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            return ToolResult(success=False, error=f"GitHub request failed: {e}")
