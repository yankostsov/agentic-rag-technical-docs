"""Run both RAG systems on the evaluation questions, grade the answers with an LLM judge, save the results."""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import agentic_rag
import baseline_rag
import config
from llm import ask_llm

load_dotenv()  # the judge always uses OpenAI, so it needs the key from .env

JUDGE_MODEL = "gpt-5.4-mini"
RESULTS_DIR = Path("results")
QUICK_IDS = {"S01", "M01", "U01"}  # one question of each type, for a fast test run

JUDGE_PROMPT = (
    "You grade answers from a question-answering system about a technical maintenance guide. "
    "You get the question, a reference answer and the system answer. "
    "If the reference answer is REFUSE, the guide does not contain the answer.\n"
    "Reply with exactly one word:\n"
    "CORRECT - the system answer contains the key facts of the reference answer (extra correct details are fine).\n"
    "REFUSED - the system says the documents do not contain the information, in any wording, "
    "even if it adds related details.\n"
    "INCORRECT - anything else: wrong or missing key facts, or an invented answer when the reference is REFUSE."
)

judge_client = OpenAI()  # reads OPENAI_API_KEY from the environment


def judge(question, reference, answer):
    """Ask the judge model for a verdict: CORRECT, INCORRECT or REFUSED."""
    prompt = f"Question: {question}\nReference answer: {reference}\nSystem answer: {answer}"
    response = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": prompt}],
        reasoning_effort="low",
    )
    verdict = response.choices[0].message.content.upper()
    # "INCORRECT" contains "CORRECT", so it must be checked first
    for label in ("INCORRECT", "REFUSED", "CORRECT"):
        if label in verdict:
            return label
    return "UNKNOWN"


def rejudge(path):
    """Re-grade only the answers whose reference answer changed in eval_questions.json (no new system runs)."""
    references = {q["id"]: q["expected_answer"] for q in json.loads(Path("eval_questions.json").read_text())}
    records = json.loads(path.read_text())
    for r in records:
        if r["expected_answer"] != references[r["id"]]:
            r["expected_answer"] = references[r["id"]]
            old_verdict = r["verdict"]
            r["verdict"] = judge(r["question"], r["expected_answer"], r["answer"])
            print(f"{r['id']} {r['system']:8} {old_verdict} -> {r['verdict']}")
    path.write_text(json.dumps(records, indent=2))
    print(f"Updated {path}")


def main():
    if "rejudge" in sys.argv:
        rejudge(RESULTS_DIR / f"raw_{config.LLM_PROVIDER}.json")
        return

    quick = "quick" in sys.argv
    questions = json.loads(Path("eval_questions.json").read_text())
    if quick:
        questions = [q for q in questions if q["id"] in QUICK_IDS]

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"raw_{config.LLM_PROVIDER}{'_quick' if quick else ''}.json"

    # The first call loads the model into memory; do it before timing anything
    print("Warming up the model...")
    ask_llm("Hello")

    systems = {"baseline": baseline_rag.answer, "agentic": agentic_rag.answer}
    records = []

    for n, q in enumerate(questions, start=1):
        for system_name, answer_fn in systems.items():
            result = answer_fn(q["question"])
            record = {
                "id": q["id"],
                "type": q["type"],
                "system": system_name,
                "provider": config.LLM_PROVIDER,
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                **result,
                "verdict": judge(q["question"], q["expected_answer"], result["answer"]),
            }
            records.append(record)
            print(f"[{n}/{len(questions)}] {q['id']} {system_name:8} -> {record['verdict']:9} ({result['seconds']}s)")

        # Save after every question, so a crash does not lose the finished work
        out_path.write_text(json.dumps(records, indent=2))

    print(f"\nSaved {len(records)} results to {out_path}")


if __name__ == "__main__":
    main()
