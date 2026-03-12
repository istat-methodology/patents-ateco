from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.config import ANALYSIS_DIR

EVAL_DIR = ANALYSIS_DIR / "evaluation"
FIG_DIR = ANALYSIS_DIR / "paper_figures"
TAB_DIR = ANALYSIS_DIR / "paper_tables"

SUMMARY_PATH = EVAL_DIR / "bge_vs_llm_eval_summary.csv"
PATENT_LEVEL_PATH = EVAL_DIR / "bge_vs_llm_patent_level.parquet"
HIT_AT_K_PATH = EVAL_DIR / "bge_vs_llm_hit_at_k.csv"

TABLE_OUTPUT = TAB_DIR / "retrieval_metrics_summary.csv"
LATEX_OUTPUT = TAB_DIR / "retrieval_metrics_summary.tex"

HIT_K_FIG = FIG_DIR / "hit_at_k_curve.png"
CONFIDENCE_FIG = FIG_DIR / "accuracy_by_confidence_quantile.png"


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# TABLE FOR PAPER
# ---------------------------------------------------------


def build_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:

    row = summary_df.iloc[0]

    table = pd.DataFrame(
        {
            "Metric": [
                "Top-1 Accuracy",
                "Hit@3",
                "Hit@5",
                "Hit@10",
                "Mean Reciprocal Rank (MRR)",
                "Median rank of first relevant code",
            ],
            "Value": [
                row["top1_accuracy"],
                row["hit_at_3"],
                row["hit_at_5"],
                row["hit_at_10"],
                row["mrr"],
                row["median_first_relevant_rank"],
            ],
        }
    )

    return table


def export_tables():

    summary = pd.read_csv(SUMMARY_PATH)

    table = build_summary_table(summary)

    table.to_csv(TABLE_OUTPUT, index=False)

    latex = table.to_latex(
        index=False,
        float_format="%.3f",
        caption="Retrieval performance of BGE-M3 embeddings against LLM-labelled NACE codes.",
        label="tab:retrieval_performance",
    )

    with open(LATEX_OUTPUT, "w") as f:
        f.write(latex)

    print("Tables exported")


# ---------------------------------------------------------
# HIT@K CURVE
# ---------------------------------------------------------


def plot_hit_at_k():

    hit_df = pd.read_csv(HIT_AT_K_PATH)

    plt.figure(figsize=(7, 5))

    plt.plot(hit_df["k"], hit_df["hit_at_k"], linewidth=2)

    plt.xlabel("k")
    plt.ylabel("Hit@k")
    plt.title("Semantic retrieval performance (BGE-M3)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(HIT_K_FIG, dpi=300)
    plt.close()

    print("Hit@k figure saved")


# ---------------------------------------------------------
# ACCURACY VS SEMANTIC CONFIDENCE
# ---------------------------------------------------------


def plot_accuracy_vs_confidence():

    df = pd.read_parquet(PATENT_LEVEL_PATH)

    df["confidence"] = df["gap_top1_top2"]

    df = df[df["confidence"].notna()]

    df["quantile"] = pd.qcut(df["confidence"], 5, labels=False)

    accuracy_by_q = df.groupby("quantile")["top1_correct"].mean().reset_index()

    plt.figure(figsize=(7, 5))

    plt.plot(
        accuracy_by_q["quantile"] + 1,
        accuracy_by_q["top1_correct"],
        marker="o",
        linewidth=2,
    )

    plt.xlabel("Semantic confidence quintile")
    plt.ylabel("Top-1 accuracy")

    plt.title("Accuracy vs semantic confidence")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(CONFIDENCE_FIG, dpi=300)
    plt.close()

    print("Confidence figure saved")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():

    ensure_dirs()

    export_tables()

    plot_hit_at_k()

    plot_accuracy_vs_confidence()

    print("All paper outputs generated.")


if __name__ == "__main__":
    main()
