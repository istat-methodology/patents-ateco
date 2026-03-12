from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.config import (
    ANALYSIS_DIR,
    NACE_PREPROCESSED_PATH,
    TOP1_CODE_CONCENTRATION_STATS_PATH,
    TOP1_CODE_DISTRIBUTION_PATH,
    ensure_directories,
)


FIGURES_DIR = ANALYSIS_DIR / "figures"
TABLES_DIR = ANALYSIS_DIR / "tables"


def ensure_output_directories() -> None:
    ensure_directories()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_top1_distribution() -> pd.DataFrame:
    print(f"Loading top1 code distribution from: {TOP1_CODE_DISTRIBUTION_PATH}")
    df = pd.read_csv(TOP1_CODE_DISTRIBUTION_PATH)

    required_columns = {"top1_code", "n_patents", "share", "cum_share"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Top1 distribution file missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["top1_code"] = df["top1_code"].astype(str)
    df["n_patents"] = pd.to_numeric(df["n_patents"], errors="raise").astype(int)
    df["share"] = pd.to_numeric(df["share"], errors="raise")
    df["cum_share"] = pd.to_numeric(df["cum_share"], errors="raise")

    return df


def load_nace_titles() -> pd.DataFrame:
    print(f"Loading NACE metadata from: {NACE_PREPROCESSED_PATH}")
    df = pd.read_parquet(NACE_PREPROCESSED_PATH, columns=["code", "title"])

    required_columns = {"code", "title"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"NACE preprocessed file missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["code"] = df["code"].astype(str)
    df["title"] = df["title"].astype(str)

    return df.drop_duplicates(subset=["code"]).reset_index(drop=True)


def load_concentration_stats() -> pd.DataFrame:
    print(f"Loading concentration stats from: {TOP1_CODE_CONCENTRATION_STATS_PATH}")
    return pd.read_csv(TOP1_CODE_CONCENTRATION_STATS_PATH)


def build_distribution_table(
    top1_dist_df: pd.DataFrame,
    nace_df: pd.DataFrame,
) -> pd.DataFrame:
    df = top1_dist_df.merge(
        nace_df,
        left_on="top1_code",
        right_on="code",
        how="left",
    )

    df["title"] = df["title"].fillna("Title not available")
    df = df.drop(columns=["code"])

    df["share_pct"] = df["share"] * 100
    df["cum_share_pct"] = df["cum_share"] * 100

    return df


def save_top_tables(df: pd.DataFrame, top_n: int = 20) -> None:
    top_df = df.head(top_n).copy()

    top_df.to_csv(TABLES_DIR / "top1_code_distribution_top20.csv", index=False)

    export_cols = [
        "top1_code",
        "title",
        "n_patents",
        "share_pct",
        "cum_share_pct",
    ]
    top_df[export_cols].to_csv(
        TABLES_DIR / "top1_code_distribution_top20_paper.csv",
        index=False,
    )

    print(f"Saved top-{top_n} tables to: {TABLES_DIR}")


def plot_top20_codes(df: pd.DataFrame, top_n: int = 20) -> None:
    plot_df = df.head(top_n).copy()

    fig_height = max(6, top_n * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.barh(
        plot_df["top1_code"][::-1],
        plot_df["share_pct"][::-1],
    )

    ax.set_title(f"Top-{top_n} NACE codes by top-1 assignment frequency")
    ax.set_xlabel("Share of patents (%)")
    ax.set_ylabel("NACE code")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()

    output_path = FIGURES_DIR / "top20_nace_codes_top1_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


def plot_pareto_distribution(
    df: pd.DataFrame,
    concentration_stats_df: pd.DataFrame,
    top_n: int = 50,
) -> None:
    plot_df = df.head(top_n).copy()
    plot_df["rank"] = range(1, len(plot_df) + 1)

    effective_n = None
    if "effective_number_of_codes" in concentration_stats_df.columns:
        effective_n = float(concentration_stats_df.loc[0, "effective_number_of_codes"])

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(
        plot_df["rank"],
        plot_df["share_pct"],
        alpha=0.8,
        label="Share by code",
    )
    ax1.set_xlabel("Rank of NACE code by frequency")
    ax1.set_ylabel("Share of patents (%)")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(
        plot_df["rank"],
        plot_df["cum_share_pct"],
        linewidth=2,
        label="Cumulative share",
    )
    ax2.set_ylabel("Cumulative share of patents (%)")
    ax2.set_ylim(0, 100)

    if effective_n is not None and effective_n <= top_n:
        ax1.axvline(
            effective_n,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"Effective number of codes = {effective_n:.1f}",
        )

    ax1.set_title("Pareto distribution of top-1 NACE code assignments")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right")

    fig.tight_layout()

    output_path = FIGURES_DIR / "pareto_top1_nace_code_distribution_top50.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


def plot_cumulative_share_curve(df: pd.DataFrame, max_rank: int = 100) -> None:
    plot_df = df.head(max_rank).copy()
    plot_df["rank"] = range(1, len(plot_df) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        plot_df["rank"],
        plot_df["cum_share_pct"],
        linewidth=2,
    )

    for threshold in [25, 50, 75]:
        reached = plot_df[plot_df["cum_share_pct"] >= threshold]
        if not reached.empty:
            first_rank = int(reached["rank"].iloc[0])
            ax.axhline(threshold, linestyle="--", linewidth=1, alpha=0.5)
            ax.axvline(first_rank, linestyle="--", linewidth=1, alpha=0.5)

    ax.set_title(
        f"Cumulative share of patents by top-1 NACE code rank (Top-{max_rank})"
    )
    ax.set_xlabel("Rank of NACE code by frequency")
    ax.set_ylabel("Cumulative share of patents (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    output_path = FIGURES_DIR / "cumulative_share_top1_nace_codes_top100.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


def main() -> None:
    ensure_output_directories()

    top1_dist_df = load_top1_distribution()
    nace_df = load_nace_titles()
    concentration_stats_df = load_concentration_stats()

    dist_table_df = build_distribution_table(top1_dist_df, nace_df)

    save_top_tables(dist_table_df, top_n=20)

    plot_top20_codes(dist_table_df, top_n=20)
    plot_pareto_distribution(
        dist_table_df,
        concentration_stats_df=concentration_stats_df,
        top_n=50,
    )
    plot_cumulative_share_curve(dist_table_df, max_rank=100)


if __name__ == "__main__":
    main()
