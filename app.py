import asyncio
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, Request
from telegram import Update
from telegram.ext import Application

from bot import build_application

_ptb_app: Application | None = None
_init_lock = asyncio.Lock()


async def get_initialized_ptb_app() -> Application:
    """Return a lazily initialized Telegram Application for webhook processing."""
    global _ptb_app
    async with _init_lock:
        if _ptb_app is None:
            _ptb_app = build_application(for_webhook=True)
        await _ptb_app.initialize()
        return _ptb_app


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("BOT_TOKEN"):
        await get_initialized_ptb_app()
    try:
        yield
    finally:
        if _ptb_app is not None:
            await _ptb_app.shutdown()


app = FastAPI(title="Loan RAG Telegram Bot", lifespan=lifespan)


@app.get("/")
async def health():
    return {
        "status": "ok",
        "bot_configured": bool(os.environ.get("BOT_TOKEN")),
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
    }


@app.post("/webhook")
async def telegram_webhook(request: Request):
    expected_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET")
    if expected_secret:
        received_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if received_secret != expected_secret:
            raise HTTPException(status_code=403, detail="Invalid webhook secret token.")

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    try:
        ptb_app = await get_initialized_ptb_app()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    update = Update.de_json(payload, ptb_app.bot)
    if update is not None:
        await ptb_app.process_update(update)

    return {"ok": True}
