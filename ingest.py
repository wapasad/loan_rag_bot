import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ডকুমেন্ট পড়া
with open("policy.md", "r", encoding="utf-8") as f:
    text = f.read()

# ছোট ছোট অংশে ভাগ করা
chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

print(f"মোট {len(chunks)}টা অংশ পাওয়া গেছে।")

data = []
for i, chunk in enumerate(chunks):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )
    embedding = result.embeddings[0].values
    data.append({"id": i, "text": chunk, "embedding": embedding})
    print(f"অংশ {i+1}/{len(chunks)} সংরক্ষণ হয়েছে।")

# JSON ফাইলে সংরক্ষণ
with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print("সম্পূর্ণ! ডেটাবেজ প্রস্তুত।")