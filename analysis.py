"""Load the evaluation results, compute the metrics, print a table and save two plots."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
# Each file contains both systems. The baseline is the same code in every run.
# v1 = original agent, v2 = agent with anchored retrieval (both with the local model qwen3.5:9b);
# openai = v2 code with gpt-5-mini through the OpenAI API.
RUNS = {"v1": "raw_ollama_v1.json", "v2": "raw_ollama_v2.json", "openai": "raw_openai_v2.json"}
ORDER = [(run, system) for run in RUNS for system in ("baseline", "agentic")]


def cited_pages(answer):
    """Extract page numbers cited in an answer, e.g. '(page 60)', '(pages 62-63)', '(pages 60, 66)'."""
    # Remove section numbers such as "6.6.4" first, so they are not mistaken for pages
    answer = re.sub(r"\d+(\.\d+)+", " ", answer)
    pages = set()
    range_pattern = r"(\d+)\s*[-–]\s*(\d+)"
    for group in re.findall(r"pages?\s+(\d[\d\s,\-–and]*)", answer, flags=re.IGNORECASE):
        for start, end in re.findall(range_pattern, group):
            low, high = sorted((int(start), int(end)))
            pages.update(range(low, high + 1))
        # Single numbers that are not part of a range
        pages.update(int(n) for n in re.findall(r"\d+", re.sub(range_pattern, " ", group)))
    return pages


def metrics(group):
    """Compute all metrics for one system in one run."""
    answerable = group[group["type"] != "unanswerable"]
    unanswerable = group[group["type"] == "unanswerable"]
    with_citations = group[group["citations_ok"].notna()]
    return pd.Series({
        "accuracy": (answerable["verdict"] == "CORRECT").mean(),
        "over_refusal": (answerable["verdict"] == "REFUSED").mean(),
        "correct_refusal": (unanswerable["verdict"] == "REFUSED").mean(),
        "hallucination": (unanswerable["verdict"] == "INCORRECT").mean(),
        "citation_accuracy": with_citations["citations_ok"].astype(bool).mean(),
        "avg_seconds": group["seconds"].mean(),
        "avg_llm_calls": group["llm_calls"].mean(),
        "avg_tokens": (group["input_tokens"] + group["output_tokens"]).mean(),
    })


def main():
    # 1. Load both runs into one table, with a "run" column
    rows = []
    for run, filename in RUNS.items():
        for record in json.loads((RESULTS_DIR / filename).read_text()):
            rows.append({**record, "run": run})
    df = pd.DataFrame(rows)

    # 2. Citation check: every cited page must be among the retrieved pages (None = no citation)
    df["citations_ok"] = [
        cited <= set(pages) if cited else None
        for cited, pages in zip(df["answer"].apply(cited_pages), df["pages"])
    ]

    # 3. One row of metrics per (run, system)
    columns = ["type", "verdict", "citations_ok", "seconds", "llm_calls", "input_tokens", "output_tokens"]
    summary = df.groupby(["run", "system"])[columns].apply(metrics).loc[ORDER]
    print(summary.round(2).to_string())
    summary.round(3).to_csv(RESULTS_DIR / "summary.csv")

    # 4. Plots (labels like "baseline\n(v1)")
    plot_df = summary.copy()
    plot_df.index = [f"{system}\n({run})" for run, system in summary.index]

    ax = plot_df[["accuracy", "correct_refusal", "over_refusal", "hallucination"]].mul(100).plot.bar(
        rot=0, figsize=(11, 5))
    ax.set_ylabel("%")
    ax.set_ylim(0, 115)  # leave room for the legend above the bars
    ax.legend(loc="upper center", ncol=4)
    ax.set_title("Answer quality: 21 answerable and 9 unanswerable questions")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "quality.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_df["avg_seconds"].plot.bar(ax=axes[0], rot=0, title="Average time per question (s)")
    plot_df["avg_llm_calls"].plot.bar(ax=axes[1], rot=0, title="LLM calls per question")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cost.png", dpi=150)

    print(f"\nSaved summary.csv, quality.png and cost.png to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
