import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import add_document, search
from groq import Groq

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

app = Flask(__name__)
CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5500").split(","))

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

client = Groq(api_key=api_key)
MODEL = "llama3-8b-8192"


@app.route('/')
def home():
    return "AI Study Assistant Backend Running "


@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    add_document(data['text'])
    return jsonify({"message": "Document added successfully "})


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400

    query = data['query']
    context = search(query)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful study assistant. Answer ONLY from the provided context. If the answer is not in the context, say 'Not found in uploaded notes'."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"answer": f"Error: {str(e)}"}), 200

    return jsonify({"answer": answer})


if __name__ == '__main__':
    app.run(
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
        host="0.0.0.0",
        port=5000
    )
