import ast
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel

from utils.config import (
    EMBEDDING_BATCH_SIZE,
    NACE_OPEN_EMBEDDINGS_PATH,
    NACE_PREPROCESSED_PATH,
    OPEN_EMBEDDING_MODEL_NAME,
    PATENT_CODE_SIMILARITY_PATH,
    PATENT_OPEN_EMBEDDINGS_PATH,
    PATENTS_FILTERED_PATH,
    RECOMPUTE_EMBEDDINGS,
    SIMILARITY_TOP_K,
    ensure_directories,
)

load_dotenv()


def _parse_embedding(value) -> np.ndarray:
    """Parse embeddings stored as list, numpy array, or stringified list."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)

    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)

    if isinstance(value, str):
        value = value.strip()
        try:
            parsed = ast.literal_eval(value)
            return np.asarray(parsed, dtype=np.float32)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                "Impossibile parsare una embedding salvata come stringa."
            ) from exc

    raise TypeError(f"Formato embedding non supportato: {type(value)}")


def load_patents_filtered() -> pd.DataFrame:
    print("Loading filtered patents dataset...")
    patents_df = pd.read_parquet(PATENTS_FILTERED_PATH)

    required_columns = {"id", "year", "abstract"}
    missing = required_columns - set(patents_df.columns)
    if missing:
        raise ValueError(
            f"Il file patents filtered non contiene le colonne richieste: {sorted(missing)}"
        )

    patents_df = patents_df.copy()
    patents_df["id"] = patents_df["id"].astype(str)
    patents_df["year"] = pd.to_numeric(patents_df["year"], errors="raise").astype(int)
    patents_df["abstract"] = patents_df["abstract"].astype(str).str.strip()

    patents_df = patents_df[patents_df["abstract"].str.len() > 0].reset_index(drop=True)

    if patents_df.empty:
        raise ValueError("Il dataset patents filtered è vuoto dopo il filtraggio.")

    print(f"Patents loaded: {len(patents_df)}")
    return patents_df


def load_nace_preprocessed() -> pd.DataFrame:
    print("Loading NACE preprocessed data...")
    nace_df = pd.read_parquet(NACE_PREPROCESSED_PATH)

    required_columns = {"CODE", "title", "text"}
    missing = required_columns - set(nace_df.columns)
    if missing:
        raise ValueError(
            f"Il file NACE preprocessed non contiene le colonne richieste: {sorted(missing)}"
        )

    nace_df = nace_df.copy()
    nace_df["CODE"] = nace_df["CODE"].astype(str)
    nace_df["title"] = nace_df["title"].astype(str).str.strip()
    nace_df["text"] = nace_df["text"].astype(str).str.strip()

    nace_df = nace_df[nace_df["text"].str.len() > 0].reset_index(drop=True)

    if nace_df.empty:
        raise ValueError("Il dataframe NACE preprocessato è vuoto.")

    print(f"NACE rows loaded: {len(nace_df)}")
    return nace_df


def load_model() -> BGEM3FlagModel:
    print(f"Loading open embedding model: {OPEN_EMBEDDING_MODEL_NAME}")
    return BGEM3FlagModel(
        OPEN_EMBEDDING_MODEL_NAME,
        use_fp16=False,
    )


def embed_texts(
    model: BGEM3FlagModel,
    texts: list[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
    max_length: int = 2048,
) -> np.ndarray:
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    embeddings = output["dense_vecs"]
    return np.asarray(embeddings, dtype=np.float32)


def build_patent_embeddings(
    model: BGEM3FlagModel,
    patents_df: pd.DataFrame,
) -> pd.DataFrame:
    print("Generating patent embeddings...")
    embeddings = embed_texts(
        model=model,
        texts=patents_df["abstract"].tolist(),
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    out_df = patents_df[["id", "year", "abstract"]].copy()
    out_df["embedding"] = [emb for emb in embeddings]
    out_df["embedding_model"] = OPEN_EMBEDDING_MODEL_NAME

    return out_df


def build_nace_embeddings(
    model: BGEM3FlagModel,
    nace_df: pd.DataFrame,
) -> pd.DataFrame:
    print("Generating NACE embeddings...")
    embeddings = embed_texts(
        model=model,
        texts=nace_df["text"].tolist(),
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    out_df = nace_df[["CODE", "title", "text"]].copy()
    out_df["embedding"] = [emb for emb in embeddings]
    out_df["embedding_model"] = OPEN_EMBEDDING_MODEL_NAME

    return out_df


def load_cached_patent_embeddings(path: Path) -> pd.DataFrame:
    print(f"Loading cached patent embeddings from: {path}")
    df = pd.read_parquet(path)

    required_columns = {"id", "year", "abstract", "embedding", "embedding_model"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Il file patent embeddings non contiene le colonne richieste: {sorted(missing)}"
        )

    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["abstract"] = df["abstract"].astype(str)
    df["embedding_model"] = df["embedding_model"].astype(str)
    df["embedding"] = df["embedding"].apply(_parse_embedding)

    return df


def load_cached_nace_embeddings(path: Path) -> pd.DataFrame:
    print(f"Loading cached NACE embeddings from: {path}")
    df = pd.read_parquet(path)

    required_columns = {"CODE", "title", "text", "embedding", "embedding_model"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Il file NACE embeddings non contiene le colonne richieste: {sorted(missing)}"
        )

    df = df.copy()
    df["CODE"] = df["CODE"].astype(str)
    df["title"] = df["title"].astype(str)
    df["text"] = df["text"].astype(str)
    df["embedding_model"] = df["embedding_model"].astype(str)
    df["embedding"] = df["embedding"].apply(_parse_embedding)

    return df


def get_or_build_patent_embeddings(model: BGEM3FlagModel) -> pd.DataFrame:
    if PATENT_OPEN_EMBEDDINGS_PATH.exists() and not RECOMPUTE_EMBEDDINGS:
        df = load_cached_patent_embeddings(PATENT_OPEN_EMBEDDINGS_PATH)

        if (
            df["embedding_model"].nunique() != 1
            or df["embedding_model"].iloc[0] != OPEN_EMBEDDING_MODEL_NAME
        ):
            print(
                "Cached patent embeddings found, but model name differs. Recomputing..."
            )
        else:
            return df

    patents_df = load_patents_filtered()
    patent_embeddings_df = build_patent_embeddings(model, patents_df)
    patent_embeddings_df.to_parquet(PATENT_OPEN_EMBEDDINGS_PATH, index=False)
    print(f"Saved patent embeddings to: {PATENT_OPEN_EMBEDDINGS_PATH}")
    return patent_embeddings_df


def get_or_build_nace_embeddings(model: BGEM3FlagModel) -> pd.DataFrame:
    if NACE_OPEN_EMBEDDINGS_PATH.exists() and not RECOMPUTE_EMBEDDINGS:
        df = load_cached_nace_embeddings(NACE_OPEN_EMBEDDINGS_PATH)

        if (
            df["embedding_model"].nunique() != 1
            or df["embedding_model"].iloc[0] != OPEN_EMBEDDING_MODEL_NAME
        ):
            print(
                "Cached NACE embeddings found, but model name differs. Recomputing..."
            )
        else:
            return df

    nace_df = load_nace_preprocessed()
    nace_embeddings_df = build_nace_embeddings(model, nace_df)
    nace_embeddings_df.to_parquet(NACE_OPEN_EMBEDDINGS_PATH, index=False)
    print(f"Saved NACE embeddings to: {NACE_OPEN_EMBEDDINGS_PATH}")
    return nace_embeddings_df


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms


def embeddings_df_to_matrix(
    df: pd.DataFrame, id_col: str
) -> tuple[list[str], np.ndarray]:
    ids = df[id_col].astype(str).tolist()
    matrix = np.vstack(df["embedding"].apply(_parse_embedding).to_list()).astype(
        np.float32
    )
    matrix = normalize_rows(matrix)
    return ids, matrix


def compute_similarity_topk(
    patent_ids: list[str],
    patent_embeddings: np.ndarray,
    nace_codes: list[str],
    nace_embeddings: np.ndarray,
    top_k: int = SIMILARITY_TOP_K,
) -> pd.DataFrame:
    print("Computing patent-code cosine similarity...")
    similarity_matrix = patent_embeddings @ nace_embeddings.T

    n_codes = similarity_matrix.shape[1]
    top_k = min(top_k, n_codes)

    topk_idx_unsorted = np.argpartition(
        -similarity_matrix,
        kth=top_k - 1,
        axis=1,
    )[:, :top_k]

    topk_scores_unsorted = np.take_along_axis(
        similarity_matrix,
        topk_idx_unsorted,
        axis=1,
    )

    order_within_topk = np.argsort(-topk_scores_unsorted, axis=1)
    topk_idx = np.take_along_axis(topk_idx_unsorted, order_within_topk, axis=1)
    topk_scores = np.take_along_axis(topk_scores_unsorted, order_within_topk, axis=1)

    rows = []
    for i, patent_id in enumerate(patent_ids):
        for rank, (code_idx, score) in enumerate(
            zip(topk_idx[i], topk_scores[i]),
            start=1,
        ):
            rows.append(
                {
                    "id": patent_id,
                    "CODE": nace_codes[int(code_idx)],
                    "dens_sim": float(score),
                    "rank": rank,
                }
            )

    return pd.DataFrame(rows)


def summarize_embeddings_df(df: pd.DataFrame, id_col: str) -> dict:
    first_emb = (
        _parse_embedding(df["embedding"].iloc[0]) if not df.empty else np.array([])
    )
    return {
        "n_rows": int(len(df)),
        "n_unique_ids": int(df[id_col].nunique()),
        "embedding_dim": int(len(first_emb)) if len(first_emb) > 0 else 0,
        "embedding_model": str(df["embedding_model"].iloc[0]) if not df.empty else None,
    }


def summarize_similarity_df(similarity_df: pd.DataFrame) -> dict:
    per_patent = (
        similarity_df.groupby("id").size()
        if not similarity_df.empty
        else pd.Series(dtype=int)
    )
    return {
        "n_rows": int(len(similarity_df)),
        "n_patents": (
            int(similarity_df["id"].nunique()) if not similarity_df.empty else 0
        ),
        "n_codes": (
            int(similarity_df["CODE"].nunique()) if not similarity_df.empty else 0
        ),
        "dens_sim_min": (
            float(similarity_df["dens_sim"].min()) if not similarity_df.empty else None
        ),
        "dens_sim_max": (
            float(similarity_df["dens_sim"].max()) if not similarity_df.empty else None
        ),
        "dens_sim_mean": (
            float(similarity_df["dens_sim"].mean()) if not similarity_df.empty else None
        ),
        "top_k_min": int(per_patent.min()) if not per_patent.empty else 0,
        "top_k_max": int(per_patent.max()) if not per_patent.empty else 0,
    }


def main() -> None:
    ensure_directories()

    model = load_model()

    patent_embeddings_df = get_or_build_patent_embeddings(model)
    nace_embeddings_df = get_or_build_nace_embeddings(model)

    print("\nPatent embeddings summary:")
    for key, value in summarize_embeddings_df(patent_embeddings_df, "id").items():
        print(f"- {key}: {value}")

    print("\nNACE embeddings summary:")
    for key, value in summarize_embeddings_df(nace_embeddings_df, "CODE").items():
        print(f"- {key}: {value}")

    patent_ids, patent_matrix = embeddings_df_to_matrix(patent_embeddings_df, "id")
    nace_codes, nace_matrix = embeddings_df_to_matrix(nace_embeddings_df, "CODE")

    similarity_df = compute_similarity_topk(
        patent_ids=patent_ids,
        patent_embeddings=patent_matrix,
        nace_codes=nace_codes,
        nace_embeddings=nace_matrix,
        top_k=SIMILARITY_TOP_K,
    )

    similarity_df.to_parquet(PATENT_CODE_SIMILARITY_PATH, index=False)

    print(f"\nSaved similarities to: {PATENT_CODE_SIMILARITY_PATH}")
    print("Similarity summary:")
    for key, value in summarize_similarity_df(similarity_df).items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
