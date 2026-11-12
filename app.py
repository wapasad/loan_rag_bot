import logging

from fastapi import FastAPI, HTTPException, Request
from telegram import Update
from telegram.error import InvalidToken

from bot import build_application, get_env

app = FastAPI(title="Loan RAG Telegram Bot")


@app.get("/")
async def health():
    return {
        "status": "ok",
        "bot_configured": bool(get_env("BOT_TOKEN")),
        "gemini_configured": bool(get_env("GEMINI_API_KEY")),
    }


@app.post("/webhook")
async def telegram_webhook(request: Request):
    expected_secret = get_env("TELEGRAM_WEBHOOK_SECRET")
    if expected_secret:
        received_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if received_secret != expected_secret:
            raise HTTPException(status_code=403, detail="Invalid webhook secret token.")

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    try:
        async with build_application(for_webhook=True) as ptb_app:
            update = Update.de_json(payload, ptb_app.bot)
            if update is not None:
                await ptb_app.process_update(update)
    except InvalidToken as exc:
        logging.exception("Invalid BOT_TOKEN: %s", exc)
        raise HTTPException(status_code=500, detail="Invalid BOT_TOKEN.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"ok": True}
