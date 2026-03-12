from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.config import (
    ANALYSIS_DIR,
    PATENT_TOP12_GAP_PATH,
    SIMILARITY_GLOBAL_STATS_PATH,
    SIMILARITY_RANK_STATS_PATH,
    ensure_directories,
)


FIGURES_DIR = ANALYSIS_DIR / "figures"

GLOBAL_SIM_MEAN = 0.431622


def ensure_figure_directory() -> None:
    ensure_directories()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_rank_stats() -> pd.DataFrame:
    print(f"Loading rank statistics from: {SIMILARITY_RANK_STATS_PATH}")

    df = pd.read_csv(SIMILARITY_RANK_STATS_PATH)

    required_columns = {"rank", "dens_sim_mean", "dens_sim_std"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Rank stats file missing required columns: {sorted(missing)}")

    return df


def load_global_stats() -> pd.DataFrame:
    print(f"Loading global statistics from: {SIMILARITY_GLOBAL_STATS_PATH}")

    df = pd.read_csv(SIMILARITY_GLOBAL_STATS_PATH)

    required_columns = {"dens_sim_mean"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Global stats file missing required columns: {sorted(missing)}"
        )

    return df


def load_top12_gap_data() -> pd.DataFrame:
    print(f"Loading patent-level top1/top2 data from: {PATENT_TOP12_GAP_PATH}")

    df = pd.read_parquet(PATENT_TOP12_GAP_PATH)

    required_columns = {
        "id",
        "top1_code",
        "top1_similarity",
        "top2_code",
        "top2_similarity",
        "gap_top1_top2",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Top1/top2 gap file missing required columns: {sorted(missing)}"
        )

    return df


def plot_similarity_vs_rank(
    rank_df: pd.DataFrame,
    global_mean: float,
    max_rank: int = 200,
) -> None:
    plot_df = rank_df[rank_df["rank"] <= max_rank].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        plot_df["rank"],
        plot_df["dens_sim_mean"],
        linewidth=2,
        label="Mean similarity by rank",
    )

    ax.fill_between(
        plot_df["rank"],
        plot_df["dens_sim_mean"] - plot_df["dens_sim_std"],
        plot_df["dens_sim_mean"] + plot_df["dens_sim_std"],
        alpha=0.2,
        label="±1 std",
    )

    ax.axhline(
        y=global_mean,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Global mean = {global_mean:.3f}",
    )

    ax.set_title(f"Semantic ranking behaviour of BGE-M3 embeddings (Top-{max_rank})")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cosine similarity")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()

    output_path = FIGURES_DIR / f"semantic_ranking_behaviour_bge_m3_top{max_rank}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


def plot_top1_similarity_distribution(top12_df: pd.DataFrame) -> None:
    mean_top1 = top12_df["top1_similarity"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        top12_df["top1_similarity"],
        bins=40,
        alpha=0.8,
    )

    ax.axvline(
        x=mean_top1,
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
        label=f"Mean = {mean_top1:.3f}",
    )

    ax.set_title("Distribution of top-1 cosine similarity")
    ax.set_xlabel("Top-1 cosine similarity")
    ax.set_ylabel("Number of patents")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()

    output_path = FIGURES_DIR / "top1_similarity_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


def plot_gap_distribution(top12_df: pd.DataFrame) -> None:
    mean_gap = top12_df["gap_top1_top2"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        top12_df["gap_top1_top2"],
        bins=40,
        alpha=0.8,
    )

    ax.axvline(
        x=mean_gap,
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
        label=f"Mean gap = {mean_gap:.3f}",
    )

    ax.set_title("Distribution of semantic separation (top-1 vs top-2)")
    ax.set_xlabel("Gap: top-1 similarity - top-2 similarity")
    ax.set_ylabel("Number of patents")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()

    output_path = FIGURES_DIR / "gap_top1_top2_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


def plot_similarity_percentiles(rank_df: pd.DataFrame, max_rank: int = 200):

    plot_df = rank_df[rank_df["rank"] <= max_rank].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        plot_df["rank"],
        plot_df["dens_sim_median"],
        linewidth=2,
        label="Median similarity",
    )

    ax.fill_between(
        plot_df["rank"],
        plot_df["dens_sim_p25"],
        plot_df["dens_sim_p75"],
        alpha=0.25,
        label="Interquartile range (p25–p75)",
    )

    # baseline
    ax.axhline(
        GLOBAL_SIM_MEAN,
        linestyle="--",
        linewidth=1.5,
        label=f"Global similarity baseline = {GLOBAL_SIM_MEAN:.3f}",
    )

    ax.annotate(
        f"{plot_df['dens_sim_median'].iloc[0]:.3f}",
        (plot_df["rank"].iloc[0], plot_df["dens_sim_median"].iloc[0]),
        xytext=(10, 10),
        textcoords="offset points",
    )

    # marker rank1
    ax.scatter(
        plot_df["rank"].iloc[0],
        plot_df["dens_sim_median"].iloc[0],
        s=60,
        zorder=5,
    )

    ax.set_title(
        f"Semantic similarity percentiles across ranking positions (Top-{max_rank})"
    )
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cosine similarity")

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()

    output_path = FIGURES_DIR / "semantic_similarity_percentiles_top200.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


def plot_semantic_confidence_ecdf(top12_df: pd.DataFrame) -> None:
    gap = top12_df["gap_top1_top2"].sort_values().reset_index(drop=True)
    y = (gap.index + 1) / len(gap)

    median_gap = float(top12_df["gap_top1_top2"].median())
    mean_gap = float(top12_df["gap_top1_top2"].mean())
    p90_gap = float(top12_df["gap_top1_top2"].quantile(0.90))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        gap,
        y,
        linewidth=2,
        label="Empirical cumulative distribution",
    )

    ax.axvline(
        median_gap,
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        label=f"Median = {median_gap:.3f}",
    )

    ax.axvline(
        mean_gap,
        linestyle=":",
        linewidth=1.5,
        alpha=0.9,
        label=f"Mean = {mean_gap:.3f}",
    )

    ax.axvline(
        p90_gap,
        linestyle="-.",
        linewidth=1.5,
        alpha=0.9,
        label=f"P90 = {p90_gap:.3f}",
    )

    ax.set_title("Semantic confidence based on top-1 vs top-2 separation")
    ax.set_xlabel("Semantic confidence (top-1 similarity - top-2 similarity)")
    ax.set_ylabel("Cumulative share of patents")

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()

    output_path = FIGURES_DIR / "semantic_confidence_ecdf.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


def main() -> None:
    ensure_figure_directory()

    rank_df = load_rank_stats()
    global_stats_df = load_global_stats()
    top12_df = load_top12_gap_data()

    global_mean = float(global_stats_df.loc[0, "dens_sim_mean"])

    plot_similarity_vs_rank(rank_df, global_mean=global_mean, max_rank=200)
    plot_top1_similarity_distribution(top12_df)
    plot_gap_distribution(top12_df)

    plot_similarity_percentiles(rank_df, max_rank=200)

    plot_semantic_confidence_ecdf(top12_df)


if __name__ == "__main__":
    main()
