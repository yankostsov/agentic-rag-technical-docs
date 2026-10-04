"""One function to call the chosen LLM, whatever the provider (Ollama or OpenAI)."""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

import config

load_dotenv()  # reads OPENAI_API_KEY from the .env file

_settings = config.LLM_SETTINGS[config.LLM_PROVIDER]
# Ollama needs no real key; OpenAI uses the secret key from .env
_api_key = os.getenv("OPENAI_API_KEY") if config.LLM_PROVIDER == "openai" else "ollama"
_client = OpenAI(base_url=_settings["base_url"], api_key=_api_key)


def ask_llm(prompt, system="You are a helpful assistant."):
    """Send one prompt to the LLM. Return (answer_text, token_usage)."""
    response = _client.chat.completions.create(
        model=_settings["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        reasoning_effort=_settings["reasoning_effort"],
    )
    return response.choices[0].message.content, response.usage


if __name__ == "__main__":
    # Quick check: which model answers, how fast, and how many tokens it used
    start = time.time()
    answer, usage = ask_llm("Explain in two sentences what preventive maintenance is.")
    print(f"Provider: {config.LLM_PROVIDER} | model: {_settings['model']}")
    print(f"Answer: {answer}")
    print(f"Time: {time.time() - start:.1f}s | input tokens: {usage.prompt_tokens} | output tokens: {usage.completion_tokens}")
