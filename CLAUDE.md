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

## Naver Developers

- Client ID + secret: stored in Railway as `NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET` (and in `.claude/secrets.local.md`).
- Used for Naver Search API (local/blog/web — business hours, phone numbers).
- Auth headers: `X-Naver-Client-Id`, `X-Naver-Client-Secret`.

## Google Calendar OAuth

- Client ID + Secret: stored in Railway as `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` (and in `.claude/secrets.local.md`).
- Required scope: `https://www.googleapis.com/auth/calendar`.
- Per-user `access_token` + `refresh_token` stored in `oauth_tokens` table (chat-scoped).
- Bot exposes a small HTTP callback (`/oauth/google/callback`) on the Railway public domain to receive the OAuth redirect.

## Required Railway env vars

| Name                 | Purpose                          |
|----------------------|----------------------------------|
| `TELEGRAM_BOT_TOKEN` | Telegram Bot API HTTP token      |
| `OPENROUTER_API_KEY` | OpenRouter API key               |
| `OPENROUTER_MODEL`   | Optional model id override (default `anthropic/claude-haiku-4.5`) |
| `OPENROUTER_VISION_MODEL` | Optional vision model id (default same as OPENROUTER_MODEL) |
| `KAKAO_REST_API_KEY` | Kakao Local + Mobility           |
| `NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET` | Naver Search API |
| `OPENAI_API_KEY`     | Whisper STT for Telegram voice notes (optional — voice falls back gracefully if missing) |
| `USER_TZ`            | Default `Asia/Seoul`             |
| `HISTORY_LIMIT`      | Max conversation turns kept (default 16) |
| `DAILYLIFE_DB_PATH`  | Default `/data/dailylife.db` (Railway volume) |

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

## Bot capabilities (current state)

- **Memory**: schedule events (with reminders), free-form notes (FTS5 trigram + LIKE fallback), facts (key/value), full chat-log search.
- **Long-horizon goals** with auto-armed proactive crons:
    - Sun 09:00 KST: full agent-driven weekly goal review (web_search/fetch_url for `watch_query`, can add sub-tasks/events).
    - Daily 08:00 KST: silent unless any open goal is within D-7.
    - Opt-out via fact `goal_review_enabled=false`.
- **Multi-modal input**: voice (Whisper, needs `OPENAI_API_KEY`), photo (Haiku 4.5 vision via OpenRouter — extracts events/expenses/places/notes from images).
- **External tools**: Naver Search, Kakao Local + Address, Kakao Mobility (driving), generic `fetch_url`.
- **Recurring tasks**: arbitrary daily prompt at HH:MM KST runs the agent loop with full tool access and pushes the answer.
- **Cost tracking**: every OpenRouter call logged with usage + cost; `/cost` rolls up today / this-month + per-model.
- **Inline-keyboard undo**: every destructive tool call (delete event/fact/recurring) snapshots the row and surfaces an `↩️ 취소` button on the bot's reply.
- **/setup**: one-shot guided onboarding that walks the user through name, home, unit, recurring patterns, and long-horizon goals.

## Branch

Active development branch: `claude/deploy-telegram-bot-7cYNh`.

## Secrets policy

GitHub push protection blocks commits that contain raw API keys. Keep all
credentials in Railway env vars + the gitignored `.claude/secrets.local.md`.
Never put raw tokens in tracked files.
