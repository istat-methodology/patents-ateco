from __future__ import annotations

import ast
import json
import time

import numpy as np
import pandas as pd

from utils.config import (
    ANALYSIS_DIR,
    GAP_STATS_PATH,
    LLM_LABELLED_PATH,
    PATENT_CODE_SIMILARITY_PATH,
    PATENT_TOP12_GAP_PATH,
    ensure_directories,
)


EVAL_DIR = ANALYSIS_DIR / "evaluation"

# opzionali, se vuoi poi metterli anche in config.py
EVAL_SUMMARY_PATH = EVAL_DIR / "bge_vs_llm_eval_summary.csv"
EVAL_PATENT_LEVEL_PATH = EVAL_DIR / "bge_vs_llm_patent_level.parquet"
EVAL_HIT_AT_K_PATH = EVAL_DIR / "bge_vs_llm_hit_at_k.csv"
EVAL_MANIFEST_PATH = EVAL_DIR / "bge_vs_llm_eval_manifest.json"


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def ensure_output_directories() -> None:
    ensure_directories()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def parse_code_list(value) -> list[str]:
    """
    Parse secondary_codes stored as Python-list-like strings, lists, or nulls.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return []

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except (ValueError, SyntaxError):
            pass

        # fallback: single code in string form
        return [value]

    return [str(value).strip()]


def load_labelled_data() -> pd.DataFrame:
    log(f"Loading labelled dataset from: {LLM_LABELLED_PATH}")

    df = pd.read_csv(LLM_LABELLED_PATH)

    required_columns = {"id", "primary_code", "secondary_codes"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Labelled dataset missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["primary_code"] = df["primary_code"].astype(str).str.strip()
    df["secondary_codes"] = df["secondary_codes"].apply(parse_code_list)

    # remove primary_code from secondary_codes if duplicated
    df["secondary_codes"] = df.apply(
        lambda row: [c for c in row["secondary_codes"] if c != row["primary_code"]],
        axis=1,
    )

    df["label_codes"] = df.apply(
        lambda row: [row["primary_code"]] + row["secondary_codes"],
        axis=1,
    )
    df["n_label_codes"] = df["label_codes"].apply(len).astype(int)

    df = df[df["primary_code"].str.len() > 0].reset_index(drop=True)

    if df.empty:
        raise ValueError("Labelled dataset is empty after preprocessing.")

    log(f"Labelled rows: {len(df):,}")
    log(f"Unique patents in labelled dataset: {df['id'].nunique():,}")

    return df[["id", "primary_code", "secondary_codes", "label_codes", "n_label_codes"]]


def load_similarity_data() -> pd.DataFrame:
    log(f"Loading BGE similarity dataset from: {PATENT_CODE_SIMILARITY_PATH}")

    df = pd.read_parquet(
        PATENT_CODE_SIMILARITY_PATH,
        columns=["id", "code", "dens_sim", "rank"],
    )

    required_columns = {"id", "code", "dens_sim", "rank"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Similarity dataset missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["code"] = df["code"].astype(str)
    df["dens_sim"] = pd.to_numeric(df["dens_sim"], errors="raise").astype(np.float32)
    df["rank"] = pd.to_numeric(df["rank"], errors="raise").astype(np.int16)

    if df.empty:
        raise ValueError("Similarity dataset is empty.")

    log(f"Similarity rows: {len(df):,}")
    log(f"Unique patents in similarity dataset: {df['id'].nunique():,}")

    return df


def build_relevant_pairs(labelled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode label_codes to obtain one row per (id, relevant_code).
    """
    relevant = (
        labelled_df[["id", "label_codes"]]
        .explode("label_codes")
        .rename(columns={"label_codes": "code"})
    )

    relevant["id"] = relevant["id"].astype(str)
    relevant["code"] = relevant["code"].astype(str)
    relevant = relevant.drop_duplicates().reset_index(drop=True)

    log(f"Relevant (id, code) pairs: {len(relevant):,}")
    return relevant


def find_first_relevant_rank(
    similarity_df: pd.DataFrame,
    relevant_pairs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join similarity ranking with relevant label codes and find, for each patent,
    the first rank at which a relevant code appears.
    """
    log("Joining BGE ranking with LLM label codes...")

    matched = similarity_df.merge(
        relevant_pairs_df,
        on=["id", "code"],
        how="inner",
    )

    if matched.empty:
        raise ValueError("No overlap found between similarity ranking and label codes.")

    first_hit = (
        matched.sort_values(["id", "rank"])
        .groupby("id", as_index=False)
        .first()[["id", "code", "rank", "dens_sim"]]
        .rename(
            columns={
                "code": "first_relevant_code",
                "rank": "first_relevant_rank",
                "dens_sim": "first_relevant_similarity",
            }
        )
    )

    log(f"Patents with at least one relevant code in ranking: {len(first_hit):,}")
    return first_hit


def load_top12_gap_data() -> pd.DataFrame:
    log(f"Loading patent-level top1/top2 data from: {PATENT_TOP12_GAP_PATH}")

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
            f"Top1/top2 gap dataset missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["id"] = df["id"].astype(str)
    return df


def build_patent_level_evaluation(
    labelled_df: pd.DataFrame,
    first_hit_df: pd.DataFrame,
    top12_df: pd.DataFrame,
) -> pd.DataFrame:
    log("Building patent-level evaluation dataset...")

    eval_df = labelled_df.merge(first_hit_df, on="id", how="left")
    eval_df = eval_df.merge(top12_df, on="id", how="left")

    eval_df["first_relevant_rank"] = pd.to_numeric(
        eval_df["first_relevant_rank"], errors="coerce"
    )

    eval_df["has_relevant_match"] = eval_df["first_relevant_rank"].notna()
    eval_df["top1_correct"] = eval_df["first_relevant_rank"].eq(1)

    for k in [3, 5, 10]:
        eval_df[f"hit_at_{k}"] = eval_df["first_relevant_rank"].le(k).fillna(False)

    eval_df["reciprocal_rank"] = np.where(
        eval_df["first_relevant_rank"].notna(),
        1.0 / eval_df["first_relevant_rank"],
        0.0,
    )

    return eval_df


def compute_summary_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    log("Computing summary evaluation metrics...")

    summary = {
        "n_patents": int(len(eval_df)),
        "n_patents_with_match": int(eval_df["has_relevant_match"].sum()),
        "top1_accuracy": float(eval_df["top1_correct"].mean()),
        "hit_at_3": float(eval_df["hit_at_3"].mean()),
        "hit_at_5": float(eval_df["hit_at_5"].mean()),
        "hit_at_10": float(eval_df["hit_at_10"].mean()),
        "mrr": float(eval_df["reciprocal_rank"].mean()),
        "mean_first_relevant_rank": float(
            eval_df.loc[eval_df["has_relevant_match"], "first_relevant_rank"].mean()
        ),
        "median_first_relevant_rank": float(
            eval_df.loc[eval_df["has_relevant_match"], "first_relevant_rank"].median()
        ),
        "mean_n_label_codes": float(eval_df["n_label_codes"].mean()),
    }

    return pd.DataFrame([summary])


def compute_hit_at_k_curve(eval_df: pd.DataFrame, max_k: int = 20) -> pd.DataFrame:
    log(f"Computing Hit@k curve up to k={max_k}...")

    rows = []
    for k in range(1, max_k + 1):
        hit_k = eval_df["first_relevant_rank"].le(k).fillna(False).mean()
        rows.append({"k": k, "hit_at_k": float(hit_k)})

    return pd.DataFrame(rows)


def save_outputs(
    summary_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    hit_at_k_df: pd.DataFrame,
) -> None:
    log("Saving evaluation outputs...")

    summary_df.to_csv(EVAL_SUMMARY_PATH, index=False)
    eval_df.to_parquet(EVAL_PATENT_LEVEL_PATH, index=False)
    hit_at_k_df.to_csv(EVAL_HIT_AT_K_PATH, index=False)

    manifest = {
        "summary_metrics": str(EVAL_SUMMARY_PATH),
        "patent_level_evaluation": str(EVAL_PATENT_LEVEL_PATH),
        "hit_at_k_curve": str(EVAL_HIT_AT_K_PATH),
    }

    with open(EVAL_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log(f"Manifest saved to: {EVAL_MANIFEST_PATH}")


def print_summary(summary_df: pd.DataFrame) -> None:
    log("=== BGE-M3 vs LLM LABELS: SUMMARY METRICS ===")
    print(summary_df.to_string(index=False), flush=True)


def main() -> None:
    start_time = time.time()

    ensure_output_directories()

    labelled_df = load_labelled_data()
    similarity_df = load_similarity_data()
    relevant_pairs_df = build_relevant_pairs(labelled_df)
    first_hit_df = find_first_relevant_rank(similarity_df, relevant_pairs_df)
    top12_df = load_top12_gap_data()

    eval_df = build_patent_level_evaluation(
        labelled_df=labelled_df,
        first_hit_df=first_hit_df,
        top12_df=top12_df,
    )

    summary_df = compute_summary_metrics(eval_df)
    hit_at_k_df = compute_hit_at_k_curve(eval_df, max_k=20)

    save_outputs(
        summary_df=summary_df,
        eval_df=eval_df,
        hit_at_k_df=hit_at_k_df,
    )

    print_summary(summary_df)

    elapsed = time.time() - start_time
    log(f"Done in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
