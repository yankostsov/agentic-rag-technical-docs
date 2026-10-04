"""Baseline RAG: retrieve the top-k chunks, then answer in a single LLM call."""

import sys
import time

import config
from llm import ask_llm
from retriever import retrieve

TOP_K = 5

SYSTEM_PROMPT = (
    "You answer questions about technical maintenance documentation. "
    "Use only the context provided. Answer concisely, in at most 5 sentences. "
    "Cite the pages you used, for example (page 60). " 
    f"If the context does not contain the answer, reply exactly: {config.REFUSAL_MESSAGE}"
)


def format_context(chunks):
    """Join chunks into one text block, each labelled with its page number."""
    return "\n\n".join(f"[page {c['page']}] {c['text']}" for c in chunks)


def answer(question):
    """Answer one question. Return a dict with the answer and measurements for evaluation."""
    start = time.time()

    chunks = retrieve(question, k=TOP_K)
    prompt = f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"
    text, usage = ask_llm(prompt, system=SYSTEM_PROMPT)

    return {
        "answer": text,
        "pages": [c["page"] for c in chunks],
        "llm_calls": 1,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "seconds": round(time.time() - start, 1),
    }


if __name__ == "__main__":
    # Usage: uv run python baseline_rag.py "your question"
    result = answer(sys.argv[1])
    print("Answer:", result["answer"])
    print(f"\nRetrieved pages: {result['pages']} | time: {result['seconds']}s "
          f"| tokens in/out: {result['input_tokens']}/{result['output_tokens']}")
