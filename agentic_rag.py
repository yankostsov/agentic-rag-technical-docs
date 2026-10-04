"""Agentic RAG: rewrite the query, retrieve, check if the context is sufficient, then answer or refuse."""

import sys
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import config
from baseline_rag import SYSTEM_PROMPT as ANSWER_PROMPT  # same answer prompt as the baseline
from baseline_rag import TOP_K, format_context
from llm import ask_llm
from retriever import retrieve

MAX_ATTEMPTS = 2

REWRITE_PROMPT = (
    "You turn a user question into a short search query for technical maintenance documentation. "
    "Use precise technical terms. Reply with the query only."
)

ASSESS_PROMPT = (
    "You check whether a context contains enough information to fully answer a question. "
    "Judge only the context, not your own knowledge. Reply in exactly two lines:\n"
    "DECISION: SUFFICIENT or INSUFFICIENT\n"
    "REASON: one short sentence"
)


class State(TypedDict):
    """The shared notebook passed from node to node."""
    question: str        # original user question (never changed)
    query: str           # current search query
    queries: list        # all queries tried, for debugging
    chunks: list         # retrieved chunks
    sufficient: bool     # decision of the context assessor
    reason: str          # assessor's one-sentence justification
    attempt: int         # number of retrieval attempts so far
    answer: str
    llm_calls: int
    input_tokens: int
    output_tokens: int


def _llm(state, prompt, system):
    """Call the LLM and return (text, updated counters) for the state."""
    text, usage = ask_llm(prompt, system=system)
    counters = {
        "llm_calls": state["llm_calls"] + 1,
        "input_tokens": state["input_tokens"] + usage.prompt_tokens,
        "output_tokens": state["output_tokens"] + usage.completion_tokens,
    }
    return text.strip(), counters


# --- Nodes: each one reads the state and returns only the fields it changes ---

def rewrite_query(state):
    prompt = f"Question: {state['question']}"
    if state["attempt"] > 0:
        # Second attempt: tell the model what failed so it searches differently
        prompt += (f"\nThe search query '{state['query']}' did not find enough information "
                   f"({state['reason']}). Write a different query.")
    query, counters = _llm(state, prompt, REWRITE_PROMPT)
    return {"query": query, "queries": state["queries"] + [query], "attempt": state["attempt"] + 1, **counters}


def retrieve_chunks(state):
    # v2: search with BOTH the original question (anchor) and the rewritten query (exploration).
    # A bad rewrite can no longer push out the chunks the original question finds.
    from_question = retrieve(state["question"], k=TOP_K)
    from_query = retrieve(state["query"], k=TOP_K)

    # Alternate between the two result lists, skip duplicates, keep TOP_K chunks (same as the baseline)
    chunks, seen = [], set()
    for chunk in [c for pair in zip(from_question, from_query) for c in pair]:
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            chunks.append(chunk)
    return {"chunks": chunks[:TOP_K]}


def assess_context(state):
    # The assessor judges the context against the ORIGINAL question, not the rewritten query
    prompt = f"Context:\n{format_context(state['chunks'])}\n\nQuestion: {state['question']}"
    reply, counters = _llm(state, prompt, ASSESS_PROMPT)
    upper = reply.upper()
    # "INSUFFICIENT" contains "SUFFICIENT", so check both; unclear replies count as insufficient
    sufficient = "SUFFICIENT" in upper and "INSUFFICIENT" not in upper
    reason = reply.split("REASON:", 1)[-1].strip()
    return {"sufficient": sufficient, "reason": reason, **counters}


def generate_answer(state):
    prompt = f"Context:\n{format_context(state['chunks'])}\n\nQuestion: {state['question']}"
    answer, counters = _llm(state, prompt, ANSWER_PROMPT)
    return {"answer": answer, **counters}


def refuse(state):
    return {"answer": config.REFUSAL_MESSAGE}


# --- Conditional edge: decides where to go after the assessment ---

def route_after_assessment(state):
    if state["sufficient"]:
        return "generate_answer"
    if state["attempt"] < MAX_ATTEMPTS:
        return "rewrite_query"
    return "refuse"


# --- Build the graph ---

graph = StateGraph(State)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("retrieve", retrieve_chunks)
graph.add_node("assess_context", assess_context)
graph.add_node("generate_answer", generate_answer)
graph.add_node("refuse", refuse)

graph.add_edge(START, "rewrite_query")
graph.add_edge("rewrite_query", "retrieve")
graph.add_edge("retrieve", "assess_context")
graph.add_conditional_edges("assess_context", route_after_assessment,
                            ["generate_answer", "rewrite_query", "refuse"])
graph.add_edge("generate_answer", END)
graph.add_edge("refuse", END)

agent = graph.compile()


def answer(question):
    """Answer one question. Same output keys as the baseline, plus agent-specific details."""
    start = time.time()
    initial_state = {
        "question": question, "query": "", "queries": [], "chunks": [],
        "sufficient": False, "reason": "", "attempt": 0, "answer": "",
        "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
    }
    final = agent.invoke(initial_state)
    return {
        "answer": final["answer"],
        "pages": [c["page"] for c in final["chunks"]],
        "llm_calls": final["llm_calls"],
        "input_tokens": final["input_tokens"],
        "output_tokens": final["output_tokens"],
        "seconds": round(time.time() - start, 1),
        "attempts": final["attempt"],
        "queries": final["queries"],
        "assessment_reason": final["reason"],
    }


if __name__ == "__main__":
    # Usage: uv run python agentic_rag.py "your question"
    result = answer(sys.argv[1])
    print("Answer:", result["answer"])
    print(f"\nQueries tried: {result['queries']}")
    print(f"Last assessment: {result['assessment_reason']}")
    print(f"Retrieved pages: {result['pages']} | attempts: {result['attempts']} | LLM calls: {result['llm_calls']} "
          f"| time: {result['seconds']}s | tokens in/out: {result['input_tokens']}/{result['output_tokens']}")
