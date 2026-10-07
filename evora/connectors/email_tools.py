"""
Email tools over IMAP/SMTP with an app password kept in the vault
(names: email_address, email_app_password; optional imap_host, smtp_host).

email_read asks the owner (can be allowed for the session); email_send always asks and shows the
recipient, subject and body on the approval card.
"""

from __future__ import annotations

import asyncio
import email
import imaplib
import smtplib
from email.header import decode_header, make_header
from email.message import EmailMessage
from typing import Optional

from evora.security import PermissionLevel, PermissionManager
from evora.tools import Tool, ToolResult


def _creds(vault):
    if vault is None:
        raise RuntimeError("No vault configured")
    address = vault.get("email_address")
    password = vault.get("email_app_password")
    if not address or not password:
        raise RuntimeError("Add 'email_address' and 'email_app_password' to the vault (Keys tab). "
                           "For Gmail use an App Password.")
    return (address, password,
            vault.get("imap_host") or "imap.gmail.com", vault.get("smtp_host") or "smtp.gmail.com")


def _h(value) -> str:
    try:
        return str(make_header(decode_header(value or "")))
    except Exception:
        return str(value or "")


class EmailReadTool(Tool):
    name = "email_read"
    description = "Read the owner's mailbox. action='list' shows recent messages; action='read' with uid shows one message."
    permission = PermissionLevel.ASK
    parameters = {
        "action": {"type": "string", "description": "list | read", "required": True},
        "limit": {"type": "integer", "description": "How many recent messages to list (default 10).", "required": False},
        "uid": {"type": "string", "description": "Message id from the list, for action='read'.", "required": False},
        "unread_only": {"type": "boolean", "description": "List only unread messages.", "required": False},
    }

    def __init__(self, security: PermissionManager, logger=None, vault=None):
        super().__init__(security, logger)
        self.vault = vault

    def _run(self, action: str, limit: int, uid: str, unread_only: bool) -> str:
        address, password, imap_host, _ = _creds(self.vault)
        box = imaplib.IMAP4_SSL(imap_host)
        try:
            box.login(address, password)
            box.select("INBOX", readonly=True)
            if action == "read":
                typ, data = box.fetch(str(uid), "(BODY.PEEK[])")
                msg = email.message_from_bytes(data[0][1])
                body = ""
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", "replace")
                        break
                return (f"From: {_h(msg['From'])}\nTo: {_h(msg['To'])}\nSubject: {_h(msg['Subject'])}\n"
                        f"Date: {msg['Date']}\nMessage-ID: {msg['Message-ID']}\n\n{body[:6000]}")
            typ, data = box.search(None, "UNSEEN" if unread_only else "ALL")
            ids = data[0].split()[-int(limit or 10):][::-1]
            lines = []
            for i in ids:
                typ, d = box.fetch(i, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
                msg = email.message_from_bytes(d[0][1])
                lines.append(f"{i.decode()} · {_h(msg['From'])} · {_h(msg['Subject'])} · {msg['Date']}")
            return "\n".join(lines) or "No messages."
        finally:
            try:
                box.logout()
            except Exception:
                pass

    async def execute(self, action: str, limit: int = 10, uid: str = "", unread_only: bool = False) -> ToolResult:
        if action not in ("list", "read") or (action == "read" and not uid):
            return ToolResult(success=False, error="Use action='list', or action='read' with a uid.")
        try:
            out = await asyncio.to_thread(self._run, action, limit, uid, unread_only)
            return ToolResult(success=True, output=out)
        except Exception as e:
            return ToolResult(success=False, error=f"Email read failed: {e}")


class EmailSendTool(Tool):
    name = "email_send"
    description = ("Send an email as the owner. For a reply, pass in_reply_to (the Message-ID). "
                   "Always needs the owner's approval, who sees the full text first.")
    permission = PermissionLevel.DANGEROUS
    parameters = {
        "to": {"type": "string", "description": "Recipient address.", "required": True},
        "subject": {"type": "string", "description": "Subject line.", "required": True},
        "body": {"type": "string", "description": "Plain-text message.", "required": True},
        "in_reply_to": {"type": "string", "description": "Message-ID being replied to.", "required": False},
    }

    def __init__(self, security: PermissionManager, logger=None, vault=None):
        super().__init__(security, logger)
        self.vault = vault

    def _send(self, to: str, subject: str, body: str, in_reply_to: Optional[str]) -> None:
        address, password, _, smtp_host = _creds(self.vault)
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = address, to, subject
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
            msg["References"] = in_reply_to
        msg.set_content(body)
        with smtplib.SMTP_SSL(smtp_host, 465, timeout=30) as server:
            server.login(address, password)
            server.send_message(msg)

    async def execute(self, to: str, subject: str, body: str, in_reply_to: str = "") -> ToolResult:
        if "@" not in to:
            return ToolResult(success=False, error="Recipient must be an email address.")
        try:
            await asyncio.to_thread(self._send, to, subject, body, in_reply_to or None)
            return ToolResult(success=True, output=f"Email sent to {to}.")
        except Exception as e:
            return ToolResult(success=False, error=f"Email send failed: {e}")
