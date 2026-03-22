from sentence_transformers import SentenceTransformer
import numpy as np

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    raise RuntimeError(f"Failed to load embedding model: {e}")

documents = []
embeddings = []

def add_document(text):
    if not text or not text.strip():
        raise ValueError("Document text cannot be empty")
    try:
        emb = model.encode([text])[0]
        documents.append(text)
        embeddings.append(emb)
    except Exception as e:
        raise RuntimeError(f"Failed to encode document: {e}")

def search(query):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if len(embeddings) == 0:
        return "No documents uploaded yet."
    try:
        q_emb = model.encode([query])[0]
        scores = np.dot(np.array(embeddings), q_emb)
        best_idx = int(np.argmax(scores))
        return documents[best_idx]
    except Exception as e:
        raise RuntimeError(f"Search failed: {e}")