"""Extra tools that only make sense under owner governance."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from evora.security import PermissionLevel, PermissionManager
from evora.tools import Tool, ToolResult, ToolRegistry


class DeletePathTool(Tool):
    """Deleting moves the file or folder to the trash so the owner can restore it."""

    name = "delete_path"
    description = "Delete a file or folder. It is moved to a recoverable trash, never destroyed. Always needs the owner's approval."
    permission = PermissionLevel.DANGEROUS
    parameters = {
        "path": {"type": "string", "description": "File or folder to delete.", "required": True},
    }

    def __init__(self, security: PermissionManager, logger=None, trash=None):
        super().__init__(security, logger)
        self.trash = trash

    async def execute(self, path: str) -> ToolResult:
        if self.trash is None:
            return ToolResult(success=False, error="Trash is not configured")
        try:
            full = self.security.check_workspace_path(path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))
        if not full.exists():
            return ToolResult(success=False, error=f"Not found: {path}")
        try:
            manifest = self.trash.soft_delete(full)
        except Exception as e:
            return ToolResult(success=False, error=f"Could not move to trash: {e}")
        return ToolResult(success=True, output=f"Moved to trash (id {manifest['id']}): {path}", data=manifest)


def register_owner_tools(registry: ToolRegistry, governance, browser_session=None) -> None:
    registry.register(DeletePathTool(registry.security, registry.logger, trash=governance.trash))
    from evora.connectors.browser import BrowserActTool, BrowserReadTool, BrowserSearchTool, BrowserSession
    from evora.connectors.email_tools import EmailReadTool, EmailSendTool
    from evora.connectors.github import GithubReadTool, GithubWriteTool
    session = browser_session or BrowserSession()
    vault = governance.vault
    sec, log = registry.security, registry.logger
    registry.register(BrowserSearchTool(sec, log, session=session))
    registry.register(BrowserReadTool(sec, log, session=session))
    registry.register(BrowserActTool(sec, log, session=session, vault=vault))
    registry.register(GithubReadTool(sec, log, vault=vault))
    registry.register(GithubWriteTool(sec, log, vault=vault))
    registry.register(EmailReadTool(sec, log, vault=vault))
    registry.register(EmailSendTool(sec, log, vault=vault))
