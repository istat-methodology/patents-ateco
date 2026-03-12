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
    df = df[df["confidence"].notna()].copy()

    df["quantile"] = pd.qcut(df["confidence"], 5, labels=False)

    accuracy_by_q = (
        df.groupby("quantile")["top1_correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
    )

    # standard error for a binomial proportion
    accuracy_by_q["se"] = np.sqrt(
        accuracy_by_q["accuracy"] * (1 - accuracy_by_q["accuracy"]) / accuracy_by_q["n"]
    )

    # 95% confidence interval
    accuracy_by_q["ci95"] = 1.96 * accuracy_by_q["se"]

    x = accuracy_by_q["quantile"] + 1
    y = accuracy_by_q["accuracy"]
    yerr = accuracy_by_q["ci95"]

    plt.figure(figsize=(7, 5))

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="-o",
        linewidth=2,
        capsize=4,
    )

    plt.xticks([1, 2, 3, 4, 5], ["Q1", "Q2", "Q3", "Q4", "Q5"])
    plt.xlim(0.8, 5.2)

    plt.xlabel("Semantic confidence quintile (Q1 = lowest, Q5 = highest)")
    plt.ylabel("Top-1 accuracy")
    plt.title("Top-1 accuracy vs semantic confidence (BGE-M3)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CONFIDENCE_FIG, dpi=300)
    plt.close()

    print("Confidence figure saved")


def plot_gap_distribution_correct_vs_wrong():

    df = pd.read_parquet(PATENT_LEVEL_PATH)

    df = df[df["gap_top1_top2"].notna()].copy()

    correct = df[df["top1_correct"]]["gap_top1_top2"]
    wrong = df[~df["top1_correct"]]["gap_top1_top2"]

    plt.figure(figsize=(7, 5))

    plt.hist(wrong, bins=50, alpha=0.5, density=True, label="Incorrect predictions")

    plt.hist(correct, bins=50, alpha=0.5, density=True, label="Correct predictions")

    plt.xlabel("Semantic confidence (top1 similarity − top2 similarity)")
    plt.ylabel("Density")

    plt.title(
        "Distribution of semantic confidence for correct vs incorrect predictions"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(FIG_DIR / "confidence_correct_vs_wrong.png", dpi=300)
    plt.close()

    print("Confidence distribution figure saved")


def plot_confidence_calibration():

    df = pd.read_parquet(PATENT_LEVEL_PATH)

    df["confidence"] = df["gap_top1_top2"]

    thresholds = np.linspace(0, df["confidence"].max(), 50)

    accuracies = []
    coverages = []

    for t in thresholds:

        subset = df[df["confidence"] >= t]

        if len(subset) == 0:
            accuracies.append(np.nan)
            coverages.append(0)
            continue

        accuracy = subset["top1_correct"].mean()
        coverage = len(subset) / len(df)

        accuracies.append(accuracy)
        coverages.append(coverage)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():

    ensure_dirs()

    export_tables()

    plot_hit_at_k()

    plot_accuracy_vs_confidence()

    plot_gap_distribution_correct_vs_wrong()

    plot_confidence_calibration()

    print("All paper outputs generated.")


if __name__ == "__main__":
    main()
