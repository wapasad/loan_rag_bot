import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("loan_policy")

def answer_question(question):
    # প্রশ্নটাকেও embedding-এ রূপান্তর করা
    q_embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    ).embeddings[0].values

    # সবচেয়ে প্রাসঙ্গিক ৩টা অংশ খুঁজে বের করা
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=3
    )
    relevant_chunks = "\n\n".join(results["documents"][0])

    # প্রাসঙ্গিক তথ্য + প্রশ্ন একসাথে Gemini-কে পাঠিয়ে উত্তর তৈরি করা
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

# টার্মিনালে সরাসরি প্রশ্ন করার লুপ
if __name__ == "__main__":
    print("Loan Policy Q&A — টাইপ করুন 'exit' বেরিয়ে যেতে")
    while True:
        q = input("\nআপনার প্রশ্ন: ")
        if q.lower() == "exit":
            break
        print("\nউত্তর:", answer_question(q))