from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.config import (
    ANALYSIS_DIR,
    PATENT_TOP12_GAP_PATH,
    SIMILARITY_RANK_STATS_PATH,
    ensure_directories,
)


FIGURES_DIR = ANALYSIS_DIR / "figures"


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


def plot_similarity_vs_rank(rank_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        rank_df["rank"],
        rank_df["dens_sim_mean"],
        linewidth=2,
        label="Mean similarity",
    )

    ax.fill_between(
        rank_df["rank"],
        rank_df["dens_sim_mean"] - rank_df["dens_sim_std"],
        rank_df["dens_sim_mean"] + rank_df["dens_sim_std"],
        alpha=0.2,
        label="±1 std",
    )

    ax.set_title("Semantic ranking behaviour of BGE-M3 embeddings")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cosine similarity")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()

    output_path = FIGURES_DIR / "semantic_ranking_behaviour_bge_m3.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


def plot_similarity_vs_rank_zoom(rank_df: pd.DataFrame, max_rank: int = 50) -> None:
    zoom_df = rank_df[rank_df["rank"] <= max_rank].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        zoom_df["rank"],
        zoom_df["dens_sim_mean"],
        linewidth=2,
        label="Mean similarity",
    )

    ax.fill_between(
        zoom_df["rank"],
        zoom_df["dens_sim_mean"] - zoom_df["dens_sim_std"],
        zoom_df["dens_sim_mean"] + zoom_df["dens_sim_std"],
        alpha=0.2,
        label="±1 std",
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

    print(f"Zoom figure saved to: {output_path}")


def plot_top1_similarity_distribution(top12_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        top12_df["top1_similarity"],
        bins=40,
        alpha=0.8,
    )

    ax.set_title("Distribution of top-1 cosine similarity")
    ax.set_xlabel("Top-1 cosine similarity")
    ax.set_ylabel("Number of patents")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    output_path = FIGURES_DIR / "top1_similarity_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


def plot_gap_distribution(top12_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        top12_df["gap_top1_top2"],
        bins=40,
        alpha=0.8,
    )

    ax.set_title("Distribution of semantic separation (top-1 vs top-2)")
    ax.set_xlabel("Gap: top-1 similarity - top-2 similarity")
    ax.set_ylabel("Number of patents")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    output_path = FIGURES_DIR / "gap_top1_top2_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


def main() -> None:
    ensure_figure_directory()

    rank_df = load_rank_stats()
    top12_df = load_top12_gap_data()

    plot_similarity_vs_rank(rank_df)
    plot_similarity_vs_rank_zoom(rank_df, max_rank=50)
    plot_top1_similarity_distribution(top12_df)
    plot_gap_distribution(top12_df)


if __name__ == "__main__":
    main()
