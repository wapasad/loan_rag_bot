import os
import json
import time
import logging
import numpy as np
from dotenv import load_dotenv
from google import genai
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters


load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
BOT_TOKEN = os.environ.get("BOT_TOKEN")

with open("embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def answer_question(question):
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
                time.sleep(3)
            else:
                raise e

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        answer = answer_question(question)
        await update.message.reply_text(answer)
    except Exception as e:
        print(f"ERROR: {e}")
        await update.message.reply_text("দুঃখিত, এই মুহূর্তে উত্তর দিতে সমস্যা হচ্ছে। একটু পর আবার চেষ্টা করুন।")

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot started...")
    app.run_polling()

if __name__ == "__main__":
    main()