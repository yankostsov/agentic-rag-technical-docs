"""Find the chunks whose meaning is closest to a question."""

import sys

import chromadb
from openai import OpenAI

import config

# Created once when the module is imported, then reused for every question
_client = OpenAI(base_url=config.EMBEDDING_BASE_URL, api_key="ollama")
_collection = chromadb.PersistentClient(path=config.CHROMA_DIR).get_collection(config.COLLECTION_NAME)


def retrieve(question, k=5):
    """Return the k closest chunks as a list of dicts: text, page, distance."""
    # The question must be embedded with the same model as the chunks
    query_embedding = _client.embeddings.create(model=config.EMBEDDING_MODEL, input=[question]).data[0].embedding

    results = _collection.query(query_embeddings=[query_embedding], n_results=k)

    # Chroma returns one list per query; we sent one query, so we take index [0]
    return [
        {"text": text, "page": meta["page"], "distance": dist}
        for text, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]


if __name__ == "__main__":
    # Usage: uv run python retriever.py "your question"
    question = sys.argv[1]
    for i, chunk in enumerate(retrieve(question, k=3), start=1):
        print(f"\n#{i}  page {chunk['page']}  distance {chunk['distance']:.3f}")
        print(chunk["text"][:300], "...")
