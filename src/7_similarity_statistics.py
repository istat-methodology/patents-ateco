# src/analysis/7_similarity_core_stats.py

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import (
    ANALYSIS_DIR,
    GAP_STATS_PATH,
    PATENT_CODE_SIMILARITY_PATH,
    PATENT_TOP12_GAP_PATH,
    SIMILARITY_GLOBAL_STATS_PATH,
    SIMILARITY_RANK_STATS_PATH,
    TOP1_CODE_CONCENTRATION_STATS_PATH,
    TOP1_CODE_DISTRIBUTION_PATH,
    TOP1_SIMILARITY_STATS_PATH,
    ensure_directories,
)


REQUIRED_COLUMNS = ["id", "code", "dens_sim", "rank"]


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def ensure_analysis_directories() -> None:
    ensure_directories()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def load_similarity_data() -> pd.DataFrame:
    """
    Carica il dataset delle similarità leggendo solo le colonne necessarie.
    """
    log(f"Loading similarity dataset from: {PATENT_CODE_SIMILARITY_PATH}")

    df = pd.read_parquet(
        PATENT_CODE_SIMILARITY_PATH,
        columns=REQUIRED_COLUMNS,
    )

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Il file di similarity non contiene le colonne richieste: {sorted(missing)}"
        )

    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["code"] = df["code"].astype(str)
    df["dens_sim"] = pd.to_numeric(df["dens_sim"], errors="raise").astype(np.float32)
    df["rank"] = pd.to_numeric(df["rank"], errors="raise").astype(np.int16)

    if df.empty:
        raise ValueError("Il dataset di similarity è vuoto.")

    log(f"Rows loaded: {len(df):,}")
    log(f"Unique patents: {df['id'].nunique():,}")
    log(f"Unique codes: {df['code'].nunique():,}")
    log(f"Rank min/max: {df['rank'].min()} / {df['rank'].max()}")

    return df


def summarize_global_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiche globali su tutte le similarity.
    """
    log("Computing global similarity statistics...")

    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    q = df["dens_sim"].quantile(quantiles)

    stats = {
        "n_rows": int(len(df)),
        "n_patents": int(df["id"].nunique()),
        "n_codes": int(df["code"].nunique()),
        "rank_min": int(df["rank"].min()),
        "rank_max": int(df["rank"].max()),
        "dens_sim_min": float(df["dens_sim"].min()),
        "dens_sim_max": float(df["dens_sim"].max()),
        "dens_sim_mean": float(df["dens_sim"].mean()),
        "dens_sim_std": float(df["dens_sim"].std(ddof=1)),
    }

    for qt in quantiles:
        stats[f"dens_sim_p{int(qt * 100):02d}"] = float(q.loc[qt])

    return pd.DataFrame([stats])


def summarize_similarity_by_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiche della similarity per rank.
    """
    log("Computing similarity statistics by rank...")

    rank_stats = (
        df.groupby("rank")["dens_sim"]
        .agg(
            count="size",
            dens_sim_mean="mean",
            dens_sim_std="std",
            dens_sim_min="min",
            dens_sim_p25=lambda s: s.quantile(0.25),
            dens_sim_median="median",
            dens_sim_p75=lambda s: s.quantile(0.75),
            dens_sim_max="max",
        )
        .reset_index()
        .sort_values("rank")
        .reset_index(drop=True)
    )

    return rank_stats


def build_top1_top2_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce un dataset patent-level con:
    - top1_code
    - top1_similarity
    - top2_code
    - top2_similarity
    - gap_top1_top2
    """
    log("Building patent-level top1/top2 dataset...")

    top12 = df[df["rank"].isin([1, 2])].copy()

    if top12.empty:
        raise ValueError("Nessuna osservazione con rank 1 o 2 trovata.")

    counts_per_id = top12.groupby("id").size()
    incomplete_ids = counts_per_id[counts_per_id != 2]

    if not incomplete_ids.empty:
        log(
            f"Warning: found {len(incomplete_ids):,} patents without exactly 2 rows in rank 1/2. "
            "These patents will be discarded from top1/top2 analysis."
        )
        valid_ids = counts_per_id[counts_per_id == 2].index
        top12 = top12[top12["id"].isin(valid_ids)].copy()

    top1 = (
        top12[top12["rank"] == 1][["id", "code", "dens_sim"]]
        .rename(
            columns={
                "code": "top1_code",
                "dens_sim": "top1_similarity",
            }
        )
        .reset_index(drop=True)
    )

    top2 = (
        top12[top12["rank"] == 2][["id", "code", "dens_sim"]]
        .rename(
            columns={
                "code": "top2_code",
                "dens_sim": "top2_similarity",
            }
        )
        .reset_index(drop=True)
    )

    top12_df = top1.merge(top2, on="id", how="inner")

    top12_df["top1_similarity"] = top12_df["top1_similarity"].astype(np.float32)
    top12_df["top2_similarity"] = top12_df["top2_similarity"].astype(np.float32)
    top12_df["gap_top1_top2"] = (
        top12_df["top1_similarity"] - top12_df["top2_similarity"]
    ).astype(np.float32)

    top12_df = top12_df.sort_values("id").reset_index(drop=True)

    log(f"Patent-level top1/top2 rows: {len(top12_df):,}")
    return top12_df


def summarize_top1_similarity(top12_df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiche sulla distribuzione della top1 similarity.
    """
    log("Computing top1 similarity statistics...")

    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    q = top12_df["top1_similarity"].quantile(quantiles)

    stats = {
        "n_patents": int(len(top12_df)),
        "top1_min": float(top12_df["top1_similarity"].min()),
        "top1_max": float(top12_df["top1_similarity"].max()),
        "top1_mean": float(top12_df["top1_similarity"].mean()),
        "top1_std": float(top12_df["top1_similarity"].std(ddof=1)),
    }

    for qt in quantiles:
        stats[f"top1_p{int(qt * 100):02d}"] = float(q.loc[qt])

    return pd.DataFrame([stats])


def summarize_gap_distribution(top12_df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiche sulla distribuzione del gap top1-top2.
    """
    log("Computing gap statistics...")

    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    q = top12_df["gap_top1_top2"].quantile(quantiles)

    stats = {
        "n_patents": int(len(top12_df)),
        "gap_min": float(top12_df["gap_top1_top2"].min()),
        "gap_max": float(top12_df["gap_top1_top2"].max()),
        "gap_mean": float(top12_df["gap_top1_top2"].mean()),
        "gap_std": float(top12_df["gap_top1_top2"].std(ddof=1)),
    }

    for qt in quantiles:
        stats[f"gap_p{int(qt * 100):02d}"] = float(q.loc[qt])

    return pd.DataFrame([stats])


def build_top1_code_distribution(top12_df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribuzione dei codici assegnati come top1.
    """
    log("Computing top1 code distribution...")

    counts = (
        top12_df.groupby("top1_code")
        .size()
        .reset_index(name="n_patents")
        .sort_values("n_patents", ascending=False)
        .reset_index(drop=True)
    )

    total = counts["n_patents"].sum()
    counts["share"] = counts["n_patents"] / total
    counts["cum_share"] = counts["share"].cumsum()

    return counts


def compute_entropy_and_herfindahl(top1_dist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola indici di concentrazione sulla distribuzione dei top1 code.
    """
    log("Computing concentration statistics for top1 code distribution...")

    p = top1_dist_df["share"].to_numpy(dtype=np.float64)

    entropy = float(-(p * np.log(p)).sum())
    herfindahl = float((p**2).sum())
    effective_n_codes = float(1.0 / herfindahl) if herfindahl > 0 else np.nan

    out = {
        "n_codes_with_top1": int(len(top1_dist_df)),
        "entropy_top1_code_distribution": entropy,
        "herfindahl_top1_code_distribution": herfindahl,
        "effective_number_of_codes": effective_n_codes,
    }

    return pd.DataFrame([out])


def save_outputs(
    global_stats: pd.DataFrame,
    rank_stats: pd.DataFrame,
    top12_df: pd.DataFrame,
    top1_stats: pd.DataFrame,
    gap_stats: pd.DataFrame,
    top1_dist: pd.DataFrame,
    concentration_stats: pd.DataFrame,
) -> None:
    """
    Salva tutti gli output dell'analisi.
    """
    log("Saving analysis outputs...")

    global_stats.to_csv(SIMILARITY_GLOBAL_STATS_PATH, index=False)
    rank_stats.to_csv(SIMILARITY_RANK_STATS_PATH, index=False)
    top12_df.to_parquet(PATENT_TOP12_GAP_PATH, index=False)
    top1_stats.to_csv(TOP1_SIMILARITY_STATS_PATH, index=False)
    gap_stats.to_csv(GAP_STATS_PATH, index=False)
    top1_dist.to_csv(TOP1_CODE_DISTRIBUTION_PATH, index=False)
    concentration_stats.to_csv(TOP1_CODE_CONCENTRATION_STATS_PATH, index=False)

    manifest = {
        "similarity_global_stats": str(SIMILARITY_GLOBAL_STATS_PATH),
        "similarity_rank_stats": str(SIMILARITY_RANK_STATS_PATH),
        "patent_top1_top2_gap": str(PATENT_TOP12_GAP_PATH),
        "top1_similarity_stats": str(TOP1_SIMILARITY_STATS_PATH),
        "gap_stats": str(GAP_STATS_PATH),
        "top1_code_distribution": str(TOP1_CODE_DISTRIBUTION_PATH),
        "top1_code_concentration_stats": str(TOP1_CODE_CONCENTRATION_STATS_PATH),
    }

    manifest_path = Path(ANALYSIS_DIR) / "similarity_analysis_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log(f"Manifest saved to: {manifest_path}")


def print_summary(
    global_stats: pd.DataFrame,
    top1_stats: pd.DataFrame,
    gap_stats: pd.DataFrame,
    concentration_stats: pd.DataFrame,
) -> None:
    log("=== GLOBAL SIMILARITY STATS ===")
    print(global_stats.to_string(index=False), flush=True)

    log("=== TOP1 SIMILARITY STATS ===")
    print(top1_stats.to_string(index=False), flush=True)

    log("=== GAP TOP1-TOP2 STATS ===")
    print(gap_stats.to_string(index=False), flush=True)

    log("=== TOP1 CODE CONCENTRATION STATS ===")
    print(concentration_stats.to_string(index=False), flush=True)


def main() -> None:
    start_time = time.time()

    ensure_analysis_directories()

    df = load_similarity_data()

    global_stats = summarize_global_similarity(df)
    rank_stats = summarize_similarity_by_rank(df)

    top12_df = build_top1_top2_gap(df)
    top1_stats = summarize_top1_similarity(top12_df)
    gap_stats = summarize_gap_distribution(top12_df)

    top1_dist = build_top1_code_distribution(top12_df)
    concentration_stats = compute_entropy_and_herfindahl(top1_dist)

    save_outputs(
        global_stats=global_stats,
        rank_stats=rank_stats,
        top12_df=top12_df,
        top1_stats=top1_stats,
        gap_stats=gap_stats,
        top1_dist=top1_dist,
        concentration_stats=concentration_stats,
    )

    print_summary(
        global_stats=global_stats,
        top1_stats=top1_stats,
        gap_stats=gap_stats,
        concentration_stats=concentration_stats,
    )

    elapsed = time.time() - start_time
    log(f"Done in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
