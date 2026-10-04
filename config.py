"""Central settings for the project. Change values here, not in the other files."""

from pathlib import Path

# --- Documents ---
PDF_PATH = Path("data/femp_om_guide_ch3-8.pdf")
FIRST_PAGE_IN_FULL_GUIDE = 29  # page 1 of our extract = page 29 of the official PDF

# --- Chunking ---
CHUNK_SIZE = 1000     # characters per chunk
CHUNK_OVERLAP = 200   # characters shared by neighbouring chunks

# --- Embeddings (local, through Ollama's OpenAI-compatible endpoint) ---
EMBEDDING_BASE_URL = "http://localhost:11434/v1"
EMBEDDING_MODEL = "bge-m3"

# --- Vector database ---
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "femp_om_guide"

# --- LLM: switch between "ollama" (local) and "openai" (API) by changing one line ---

LLM_PROVIDER = "ollama"

LLM_SETTINGS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen3.5:9b", "reasoning_effort": "none"},
    # gpt-5-mini, not the judge model (gpt-5.4-mini), so the judge never grades its own answers
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-5-mini", "reasoning_effort": "minimal"},
}

#REASONING_EFFORT = "none"  # turn off hidden "thinking" to keep answers fast

# Fixed refusal sentence, so the evaluation can detect refusals automatically
REFUSAL_MESSAGE = "The documents do not contain enough information to answer this question."