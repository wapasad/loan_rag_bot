import os
import json
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# সংরক্ষিত embedding লোড করা
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

    # প্রতিটা chunk-এর সাথে similarity বের করে সবচেয়ে প্রাসঙ্গিক ৩টা নেওয়া
    scored = [(cosine_similarity(q_embedding, item["embedding"]), item["text"]) for item in data]
    scored.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [text for _, text in scored[:3]]
    relevant_chunks = "\n\n".join(top_chunks)

    prompt = f"""You are a helpful assistant answering questions about a loan policy document.
Use only the following context to answer. If the answer is not in the context, say you don't know.

Context:
{relevant_chunks}

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    return response.text

if __name__ == "__main__":
    print("Loan Policy Q&A — টাইপ করুন 'exit' বেরিয়ে যেতে")
    while True:
        q = input("\nআপনার প্রশ্ন: ")
        if q.lower() == "exit":
            break
        print("\nউত্তর:", answer_question(q))