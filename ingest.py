"""Read the PDF, split it into chunks, embed each chunk and store everything in Chroma."""

import chromadb
from openai import OpenAI
from pypdf import PdfReader

import config

BATCH_SIZE = 32  # number of chunks sent to the embedding model in one request


def split_into_chunks(text, size, overlap):
    """Cut text into pieces of `size` characters; neighbours share `overlap` characters."""
    chunks = []
    start = 0
    while True:
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def main():
    # 1. PDF -> text of each page -> chunks with page metadata
    reader = PdfReader(config.PDF_PATH)
    texts, metadatas, ids = [], [], []

    for page_index, page in enumerate(reader.pages):
        # Collapse line breaks and repeated spaces into single spaces
        text = " ".join(page.extract_text().split())
        if not text:
            continue  # skip empty pages (e.g. blank or image-only)

        page_number = page_index + config.FIRST_PAGE_IN_FULL_GUIDE
        for i, chunk in enumerate(split_into_chunks(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            texts.append(chunk)
            metadatas.append({"page": page_number})
            ids.append(f"p{page_number}_c{i}")

    print(f"Created {len(texts)} chunks from {len(reader.pages)} pages")

    # 2. Chunks -> embeddings (Ollama ignores the API key, but the library requires one)
    client = OpenAI(base_url=config.EMBEDDING_BASE_URL, api_key="ollama")
    embeddings = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in response.data)
        print(f"Embedded {len(embeddings)}/{len(texts)} chunks")

    # 3. Store text + embedding + metadata in Chroma (saved on disk in CHROMA_DIR)
    db = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        # Start from scratch each run so old chunks never mix with new ones
        db.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # the collection does not exist yet on the first run

    collection = db.create_collection(config.COLLECTION_NAME)
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print(f"Stored {collection.count()} chunks in '{config.CHROMA_DIR}/'")

    # Show one example so we can see what a chunk looks like
    print("\nExample chunk:", ids[0], metadatas[0])
    print(texts[0][:300], "...")


if __name__ == "__main__":
    main()
