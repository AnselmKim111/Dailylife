"""Browser automation via Playwright headless Chromium.

If Playwright or Chromium isn't available at runtime (e.g. dev shells),
the public functions return structured error dicts so callers degrade
gracefully without blowing up the bot.

Vault: AES-GCM encrypted passwords keyed on `vault_passphrase` fact.
Passphrase never travels to OpenRouter — encryption/decryption happens
locally in this module only.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try-import — bot stays alive if Playwright/Chromium missing.
try:
    from playwright.async_api import async_playwright  # type: ignore
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False
    logger.info("playwright not installed — browser tools degraded")

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
    CRYPTO_AVAILABLE = True
except Exception:
    CRYPTO_AVAILABLE = False
    logger.info("cryptography not installed — vault encryption disabled")


# ---------------- Vault (AES-GCM) ----------------


def _derive_key(passphrase: str) -> bytes:
    """passphrase → 32-byte key via SHA-256. Deterministic so same passphrase
    yields same key across restarts (no salt — single-user bot, low threat)."""
    return hashlib.sha256(passphrase.encode("utf-8")).digest()


def encrypt_password(plaintext: str, passphrase: str) -> Optional[Dict[str, bytes]]:
    if not CRYPTO_AVAILABLE:
        return None
    key = _derive_key(passphrase)
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return {"ciphertext": ct, "nonce": nonce}


def decrypt_password(ciphertext: bytes, nonce: bytes, passphrase: str) -> Optional[str]:
    if not CRYPTO_AVAILABLE:
        return None
    key = _derive_key(passphrase)
    try:
        pt = AESGCM(key).decrypt(nonce, ciphertext, None)
        return pt.decode("utf-8")
    except Exception:
        return None


# ---------------- Browser primitives ----------------


_browser = None
_browser_lock = asyncio.Lock()


async def _get_browser():
    global _browser
    if not PLAYWRIGHT_AVAILABLE:
        return None
    async with _browser_lock:
        if _browser is None:
            try:
                pw = await async_playwright().start()
                exe = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
                launch_kwargs = {"headless": True}
                if exe:
                    launch_kwargs["executable_path"] = exe
                _browser = await pw.chromium.launch(**launch_kwargs)
            except Exception as e:
                logger.exception("chromium launch failed: %s", e)
                _browser = None
        return _browser


async def scrape_url(
    url: str, wait_for_selector: Optional[str] = None, max_chars: int = 6000,
) -> Dict[str, Any]:
    """Headless browser fetch — JS rendered. Returns text content (HTML stripped)."""
    if not PLAYWRIGHT_AVAILABLE:
        return {"ok": False, "error": "playwright not installed"}
    if not (url.startswith("http://") or url.startswith("https://")):
        return {"ok": False, "error": "url must be http(s)"}
    browser = await _get_browser()
    if browser is None:
        return {"ok": False, "error": "chromium unavailable"}
    context = None
    try:
        context = await browser.new_context(locale="ko-KR")
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        if wait_for_selector:
            try:
                await page.wait_for_selector(wait_for_selector, timeout=8000)
            except Exception:
                pass
        text = await page.evaluate(
            "() => document.body ? document.body.innerText : ''")
        return {
            "ok": True, "url_final": page.url,
            "text": (text or "")[:max_chars],
            "truncated": len(text or "") > max_chars,
        }
    except Exception as e:
        logger.exception("scrape_url failed: %s", url)
        return {"ok": False, "error": str(e)}
    finally:
        if context is not None:
            try:
                await context.close()
            except Exception:
                pass


async def screenshot_url(url: str, full_page: bool = True) -> Dict[str, Any]:
    """Headless screenshot — returns PNG bytes for Telegram send_photo."""
    if not PLAYWRIGHT_AVAILABLE:
        return {"ok": False, "error": "playwright not installed"}
    browser = await _get_browser()
    if browser is None:
        return {"ok": False, "error": "chromium unavailable"}
    context = None
    try:
        context = await browser.new_context(locale="ko-KR", viewport={"width": 1280, "height": 800})
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        png = await page.screenshot(full_page=full_page)
        return {"ok": True, "png": png, "url_final": page.url}
    except Exception as e:
        logger.exception("screenshot_url failed: %s", url)
        return {"ok": False, "error": str(e)}
    finally:
        if context is not None:
            try:
                await context.close()
            except Exception:
                pass


async def fill_and_submit(
    url: str, fields: Dict[str, str], submit_selector: str,
    wait_for_after: Optional[str] = None,
) -> Dict[str, Any]:
    """선택자 → 값 매핑으로 폼 채우고 submit 후 결과 page 텍스트 + screenshot."""
    if not PLAYWRIGHT_AVAILABLE:
        return {"ok": False, "error": "playwright not installed"}
    browser = await _get_browser()
    if browser is None:
        return {"ok": False, "error": "chromium unavailable"}
    context = None
    try:
        context = await browser.new_context(locale="ko-KR")
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        for selector, value in fields.items():
            try:
                await page.fill(selector, value)
            except Exception as e:
                logger.warning("fill failed for %s: %s", selector, e)
        await page.click(submit_selector, timeout=8000)
        if wait_for_after:
            try:
                await page.wait_for_selector(wait_for_after, timeout=10000)
            except Exception:
                pass
        text = await page.evaluate(
            "() => document.body ? document.body.innerText : ''")
        png = await page.screenshot(full_page=False)
        return {
            "ok": True, "url_final": page.url,
            "text": (text or "")[:4000], "png": png,
        }
    except Exception as e:
        logger.exception("fill_and_submit failed: %s", url)
        return {"ok": False, "error": str(e)}
    finally:
        if context is not None:
            try:
                await context.close()
            except Exception:
                pass


async def shutdown() -> None:
    """Graceful close on bot shutdown."""
    global _browser
    if _browser is not None:
        try:
            await _browser.close()
        except Exception:
            pass
        _browser = None
