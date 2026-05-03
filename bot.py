"""Telegram bot that proxies user messages to an OpenRouter LLM."""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Deque, Dict, List

import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dailylife")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are Dailylife, a warm, concise daily-life assistant on Telegram. "
    "Default to the user's language. Keep replies under 1500 characters unless asked.",
)
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", "12"))
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HTTP_REFERER = os.environ.get("OPENROUTER_REFERER", "https://t.me/AnselmsSlave7bot")
APP_TITLE = os.environ.get("OPENROUTER_TITLE", "Dailylife Telegram Bot")

# In-memory per-chat history. Resets on restart.
chat_history: Dict[int, Deque[Dict[str, str]]] = {}


def history_for(chat_id: int) -> Deque[Dict[str, str]]:
    if chat_id not in chat_history:
        chat_history[chat_id] = deque(maxlen=HISTORY_LIMIT * 2)
    return chat_history[chat_id]


async def call_openrouter(messages: List[Dict[str, str]]) -> str:
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": HTTP_REFERER,
        "X-Title": APP_TITLE,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.error("OpenRouter %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "안녕하세요! Dailylife 봇입니다. 무엇이든 물어보세요.\n"
        "Hi! I'm Dailylife. Send me a message and I'll reply.\n\n"
        "/reset — clear conversation memory\n"
        "/model — show the current model"
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_history.pop(update.effective_chat.id, None)
    await update.message.reply_text("Conversation memory cleared.")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"Model: {OPENROUTER_MODEL}")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    user_text = update.message.text
    logger.info("msg from %s: %r", chat_id, user_text[:120])

    history = history_for(chat_id)
    history.append({"role": "user", "content": user_text})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        reply = await call_openrouter(messages)
    except Exception as exc:
        logger.exception("OpenRouter call failed")
        await update.message.reply_text(f"⚠️ Error talking to the model: {exc}")
        # Don't keep a half-broken turn in history.
        history.pop()
        return

    history.append({"role": "assistant", "content": reply})
    # Telegram message limit is 4096 chars; chunk if needed.
    for i in range(0, len(reply), 4000):
        await update.message.reply_text(reply[i : i + 4000])


def main() -> None:
    logger.info("Starting Dailylife bot — model=%s", OPENROUTER_MODEL)
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
