import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ডকুমেন্ট পড়া
with open("policy.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ছোট ছোট অংশে ভাগ করা
chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

print(f"মোট {len(chunks)}টা অংশ পাওয়া গেছে।")

# আগের কালেকশন থাকলে মুছে নতুন করে বানানো
try:
    chroma_client.delete_collection("loan_policy")
except Exception:
    pass

collection = chroma_client.create_collection("loan_policy")

# প্রতিটা chunk-এর embedding বানিয়ে ডেটাবেজে জমা রাখা
for i, chunk in enumerate(chunks):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )
    embedding = result.embeddings[0].values

    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding],
        documents=[chunk]
    )
    print(f"অংশ {i+1}/{len(chunks)} সংরক্ষণ হয়েছে।")

print("সম্পূর্ণ! ডেটাবেজ প্রস্তুত।")