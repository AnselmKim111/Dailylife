# Dailylife — Telegram AI Bot

A Telegram bot powered by OpenRouter LLMs, deployed on Railway as a worker.

## Bot

- Username: [@AnselmsSlave7bot](http://t.me/AnselmsSlave7bot)
- Display name: 컨텍스트봇
- Bot ID: `8689694881`
- Token: stored in Railway as `TELEGRAM_BOT_TOKEN`
  (also tracked in the Claude Code memory note `.claude/secrets.local.md`).

## Railway

- Project name: `practical-strength`
- Project ID: `1046a615-9ab8-4575-a0a0-64ddadceb3bd`
- Service name: `Dailylife`
- Service ID: `3aa42043-5786-4c15-b09c-4c7811d439c9`
- Environment: `production` (`4836258d-d1e2-4d4e-a036-ee65b6cfe10d`)
- Project token: stored locally in `.claude/secrets.local.md` (do **not** commit).
  Use it via `RAILWAY_TOKEN=<token> railway ...`.

## OpenRouter

- API key: stored in Railway as `OPENROUTER_API_KEY` (and in `.claude/secrets.local.md`).
- Endpoint: `https://openrouter.ai/api/v1/chat/completions`
- Default model: `anthropic/claude-haiku-4.5` (override with `OPENROUTER_MODEL`).

## Kakao Developers

- REST API key: stored in Railway as `KAKAO_REST_API_KEY` (and in `.claude/secrets.local.md`).
- Used for Kakao Local API (place search, address ↔ coords).
- Auth header: `Authorization: KakaoAK <key>`.

## Required Railway env vars

| Name                 | Purpose                          |
|----------------------|----------------------------------|
| `TELEGRAM_BOT_TOKEN` | Telegram Bot API HTTP token      |
| `OPENROUTER_API_KEY` | OpenRouter API key               |
| `OPENROUTER_MODEL`   | Optional model id override       |
| `SYSTEM_PROMPT`      | Optional system prompt override  |
| `HISTORY_LIMIT`      | Max conversation turns kept (default 12) |

## Common commands

```bash
# Tail deploy logs
RAILWAY_TOKEN=$RAILWAY_TOKEN railway logs --service Dailylife --deployment

# Tail build logs
RAILWAY_TOKEN=$RAILWAY_TOKEN railway logs --service Dailylife --build

# Show / set env vars
RAILWAY_TOKEN=$RAILWAY_TOKEN railway variables --service Dailylife
RAILWAY_TOKEN=$RAILWAY_TOKEN railway variables --service Dailylife --set KEY=value

# Re-deploy current dir
RAILWAY_TOKEN=$RAILWAY_TOKEN railway up --service Dailylife --ci
```

## Local dev

```bash
pip install -r requirements.txt
TELEGRAM_BOT_TOKEN=... OPENROUTER_API_KEY=... python bot.py
```

## Branch

Active development branch: `claude/deploy-telegram-bot-7cYNh`.

## Secrets policy

GitHub push protection blocks commits that contain raw API keys. Keep all
credentials in Railway env vars + the gitignored `.claude/secrets.local.md`.
Never put raw tokens in tracked files.
