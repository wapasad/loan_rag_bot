import asyncio
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_PATH = BASE_DIR / "embeddings.json"

_client: genai.Client | None = None
_embeddings_cache: list[dict] | None = None


def get_genai_client() -> genai.Client:
    """Return a cached Gemini client initialized from GEMINI_API_KEY."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def get_embeddings() -> list[dict]:
    """Load and cache precomputed document chunks and embeddings."""
    global _embeddings_cache
    if _embeddings_cache is None:
        with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
            _embeddings_cache = json.load(f)
    return _embeddings_cache


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def answer_question(question):
    client = get_genai_client()
    data = get_embeddings()

    q_embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    ).embeddings[0].values

    scored = [(cosine_similarity(q_embedding, item["embedding"]), item["text"]) for item in data]
    scored.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [text for _, text in scored[:3]]
    relevant_chunks = "\n\n".join(top_chunks)

    prompt = f"""You are a helpful assistant answering questions about a personal loan policy.
Use only the following context to answer. If the answer is not in the context, say you don't know.

Context:
{relevant_chunks}

Question: {question}

Answer:"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < 2:
                print(f"চেষ্টা {attempt+1} ব্যর্থ, আবার চেষ্টা করছি...")
                time.sleep(1)
            else:
                raise e


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    question = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        answer = await asyncio.to_thread(answer_question, question)
        await update.message.reply_text(answer)
    except Exception as e:
        logging.exception("Error answering question: %s", e)
        await update.message.reply_text("দুঃখিত, এই মুহূর্তে উত্তর দিতে সমস্যা হচ্ছে। একটু পর আবার চেষ্টা করুন।")


def build_application(token: str | None = None, *, for_webhook: bool = False) -> Application:
    """Create and configure the Telegram Application with message handlers."""
    bot_token = token or os.environ.get("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("BOT_TOKEN environment variable is not set.")

    builder = Application.builder().token(bot_token)
    if for_webhook:
        builder = builder.updater(None)

    app = builder.build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return app


def main():
    app = build_application()
    print("Bot started...")
    app.run_polling()


if __name__ == "__main__":
    main()