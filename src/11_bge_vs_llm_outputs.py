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


def plot_confidence_calibration() -> None:
    """
    Plot selective accuracy and coverage as a function of the semantic confidence threshold.

    Semantic confidence is defined as:
        gap_top1_top2 = top1_similarity - top2_similarity

    For each threshold t, the function computes:
        - accuracy among patents with confidence >= t
        - coverage, i.e. share of patents with confidence >= t
    """
    df = pd.read_parquet(PATENT_LEVEL_PATH)

    required_columns = {"gap_top1_top2", "top1_correct"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Patent-level evaluation file missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df = df[df["gap_top1_top2"].notna()].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows found for confidence calibration plot.")

    df["confidence"] = pd.to_numeric(df["gap_top1_top2"], errors="raise")
    df["top1_correct"] = df["top1_correct"].astype(bool)

    # Build thresholds over the observed confidence range
    max_conf = float(df["confidence"].max())
    thresholds = np.linspace(0.0, max_conf, 50)

    rows = []
    total_n = len(df)

    for t in thresholds:
        subset = df[df["confidence"] >= t]

        if subset.empty:
            rows.append(
                {
                    "threshold": float(t),
                    "accuracy": np.nan,
                    "coverage": 0.0,
                    "n_patents": 0,
                }
            )
            continue

        accuracy = float(subset["top1_correct"].mean())
        coverage = float(len(subset) / total_n)

        rows.append(
            {
                "threshold": float(t),
                "accuracy": accuracy,
                "coverage": coverage,
                "n_patents": int(len(subset)),
            }
        )

    calibration_df = pd.DataFrame(rows)

    # Optional: save table for later inspection
    calibration_output = TAB_DIR / "confidence_calibration_curve.csv"
    calibration_df.to_csv(calibration_output, index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(
        calibration_df["threshold"],
        calibration_df["accuracy"],
        linewidth=2,
        label="Selective accuracy",
    )
    ax1.set_xlabel("Semantic confidence threshold")
    ax1.set_ylabel("Top-1 accuracy")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        calibration_df["threshold"],
        calibration_df["coverage"],
        linestyle="--",
        linewidth=2,
        label="Coverage",
    )
    ax2.set_ylabel("Coverage")

    ax1.set_title("Confidence calibration curve")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="center right")

    fig.tight_layout()

    output_path = FIG_DIR / "confidence_calibration_curve.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Confidence calibration figure saved to: {output_path}")
    print(f"Calibration table saved to: {calibration_output}")


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
