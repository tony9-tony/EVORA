"""
Live browser tools (Playwright). The window is visible on the owner's PC unless
EVORA_BROWSER_HEADLESS=1, and every step returns a small screenshot so the web console can show it.

browser_search / browser_read only look (no approval needed at any level).
browser_act clicks, types and signs in; it always needs the owner's approval, and secrets are
referenced by vault name ({"secret": "github_password"}) so the model never sees them.
"""

from __future__ import annotations

import asyncio
import base64
import os
from typing import Any, Optional
from urllib.parse import quote_plus, urlparse

from evora.security import PermissionLevel, PermissionManager
from evora.tools import Tool, ToolResult


class BrowserSession:
    """One shared browser page for all browser tools."""

    def __init__(self):
        self._pw = None
        self._browser = None
        self._page = None
        self._lock = asyncio.Lock()

    async def page(self):
        async with self._lock:
            if self._page is not None and not self._page.is_closed():
                return self._page
            try:
                from playwright.async_api import async_playwright
            except ImportError as e:
                raise RuntimeError("Playwright is not installed. Run: pip install playwright") from e
            if self._pw is None:
                self._pw = await async_playwright().start()
            headless = os.environ.get("EVORA_BROWSER_HEADLESS") == "1"
            channel = os.environ.get("EVORA_BROWSER_CHANNEL", "chrome")
            try:
                self._browser = await self._pw.chromium.launch(channel=channel, headless=headless)
            except Exception:
                self._browser = await self._pw.chromium.launch(headless=headless)
            context = await self._browser.new_context(viewport={"width": 1100, "height": 720})
            self._page = await context.new_page()
            return self._page

    async def screenshot(self) -> str:
        try:
            page = await self.page()
            raw = await page.screenshot(type="jpeg", quality=45)
            return base64.b64encode(raw).decode()
        except Exception:
            return ""

    async def close(self):
        try:
            if self._browser:
                await self._browser.close()
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass
        self._page = self._browser = self._pw = None


def _check_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "Only http(s) pages can be opened."
    return None


class BrowserSearchTool(Tool):
    name = "browser_search"
    description = "Search the web in a live browser window the owner can watch. Returns result titles and links."
    permission = PermissionLevel.SAFE
    parameters = {
        "query": {"type": "string", "description": "What to search for.", "required": True},
    }

    def __init__(self, security: PermissionManager, logger=None, session: Optional[BrowserSession] = None):
        super().__init__(security, logger)
        self.session = session or BrowserSession()

    async def execute(self, query: str) -> ToolResult:
        try:
            page = await self.session.page()
            await page.goto("https://html.duckduckgo.com/html/?q=" + quote_plus(query), timeout=30000)
            results = await page.eval_on_selector_all(
                ".result__a", "els => els.slice(0, 8).map(e => ({title: e.innerText.trim(), url: e.href}))")
            shot = await self.session.screenshot()
            if not results:
                text = (await page.inner_text("body"))[:1500]
                return ToolResult(success=False, error=f"No results parsed. Page said: {text}", data={"screenshot": shot})
            out = "\n".join(f"- {r['title']}: {r['url']}" for r in results)
            return ToolResult(success=True, output=out, data={"results": results, "screenshot": shot})
        except Exception as e:
            return ToolResult(success=False, error=f"Browser search failed: {e}")


class BrowserReadTool(Tool):
    name = "browser_read"
    description = "Open a web page in the live browser and return its text."
    permission = PermissionLevel.SAFE
    parameters = {
        "url": {"type": "string", "description": "http(s) address to open.", "required": True},
        "max_length": {"type": "integer", "description": "Maximum characters to return.", "required": False},
    }

    def __init__(self, security: PermissionManager, logger=None, session: Optional[BrowserSession] = None):
        super().__init__(security, logger)
        self.session = session or BrowserSession()

    async def execute(self, url: str, max_length: int = 6000) -> ToolResult:
        problem = _check_url(url)
        if problem:
            return ToolResult(success=False, error=problem)
        try:
            page = await self.session.page()
            await page.goto(url, timeout=30000)
            text = (await page.inner_text("body")).strip()
            shot = await self.session.screenshot()
            title = await page.title()
            return ToolResult(success=True, output=f"{title}\n\n{text[:max_length]}",
                              data={"url": page.url, "title": title, "screenshot": shot})
        except Exception as e:
            return ToolResult(success=False, error=f"Could not read {url}: {e}")


class BrowserActTool(Tool):
    name = "browser_act"
    description = (
        "Do things on a web page: goto, click, fill, press, wait. Steps are a list like "
        '[{"action":"goto","url":"https://..."},{"action":"fill","selector":"#user","text":"me"},'
        '{"action":"fill","selector":"#pw","secret":"vault_name"},{"action":"click","selector":"button[type=submit]"}]. '
        "Use \"secret\" to type a stored password without seeing it. Always needs the owner's approval."
    )
    permission = PermissionLevel.DANGEROUS
    parameters = {
        "steps": {"type": "array", "items": {"type": "object"}, "description": "Ordered browser steps.", "required": True},
    }

    def __init__(self, security: PermissionManager, logger=None, session: Optional[BrowserSession] = None, vault=None):
        super().__init__(security, logger)
        self.session = session or BrowserSession()
        self.vault = vault

    async def execute(self, steps: list) -> ToolResult:
        if not isinstance(steps, list) or not steps:
            return ToolResult(success=False, error="steps must be a non-empty list")
        log = []
        try:
            page = await self.session.page()
            for i, step in enumerate(steps[:30], 1):
                action = str(step.get("action", "")).lower()
                sel = step.get("selector")
                if action == "goto":
                    problem = _check_url(str(step.get("url", "")))
                    if problem:
                        return ToolResult(success=False, error=f"step {i}: {problem}")
                    await page.goto(step["url"], timeout=30000)
                    log.append(f"{i}. opened {urlparse(step['url']).netloc}")
                elif action == "click":
                    await page.click(sel, timeout=10000)
                    log.append(f"{i}. clicked {sel}")
                elif action == "fill":
                    if "secret" in step:
                        if self.vault is None:
                            return ToolResult(success=False, error="No vault configured")
                        value = self.vault.get(str(step["secret"]))
                        if value is None:
                            return ToolResult(success=False, error=f"step {i}: no secret named '{step['secret']}' in the vault")
                        await page.fill(sel, value, timeout=10000)
                        log.append(f"{i}. filled {sel} from vault ({step['secret']})")
                    else:
                        await page.fill(sel, str(step.get("text", "")), timeout=10000)
                        log.append(f"{i}. filled {sel}")
                elif action == "press":
                    await page.press(sel or "body", str(step.get("key", "Enter")), timeout=10000)
                    log.append(f"{i}. pressed {step.get('key', 'Enter')}")
                elif action == "wait":
                    await page.wait_for_timeout(min(int(step.get("ms", 1000)), 10000))
                    log.append(f"{i}. waited")
                else:
                    return ToolResult(success=False, error=f"step {i}: unknown action '{action}'")
            text = (await page.inner_text("body"))[:1500]
            shot = await self.session.screenshot()
            return ToolResult(success=True, output="\n".join(log) + f"\n\nPage now shows:\n{text}",
                              data={"url": page.url, "screenshot": shot})
        except Exception as e:
            return ToolResult(success=False, error=f"{'; '.join(log)} | failed: {e}",
                              data={"screenshot": await self.session.screenshot()})
