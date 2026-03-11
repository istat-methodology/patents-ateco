from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import PATENT_CODE_SIMILARITY_PATH


def load_default_similarity_dataset():
    return load_similarity_dataset(PATENT_CODE_SIMILARITY_PATH)


REQUIRED_COLUMNS = {"id", "CODE", "dens_sim"}


@dataclass
class SimilarityDataset:
    results_df: pd.DataFrame
    sim_matrix_df: pd.DataFrame
    sim_matrix: np.ndarray
    patent_ids: list[str]
    codes: list[str]


def validate_results_df(results_df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(results_df.columns)
    if missing:
        raise ValueError(
            f"Il dataframe dei risultati non contiene le colonne richieste: {sorted(missing)}"
        )

    if results_df.empty:
        raise ValueError("Il dataframe dei risultati è vuoto.")

    if results_df["id"].isna().any():
        raise ValueError("La colonna 'id' contiene valori nulli.")

    if results_df["CODE"].isna().any():
        raise ValueError("La colonna 'CODE' contiene valori nulli.")

    if results_df["dens_sim"].isna().any():
        raise ValueError("La colonna 'dens_sim' contiene valori nulli.")


def load_results_df(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    if path.suffix.lower() == ".csv":
        results_df = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        results_df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Formato file non supportato per {path.name}. Usa CSV o Parquet."
        )

    validate_results_df(results_df)

    results_df = results_df.copy()
    results_df["id"] = results_df["id"].astype(str)
    results_df["CODE"] = results_df["CODE"].astype(str)
    results_df["dens_sim"] = pd.to_numeric(results_df["dens_sim"], errors="raise")

    return results_df


def build_similarity_matrix(results_df: pd.DataFrame) -> pd.DataFrame:
    validate_results_df(results_df)

    duplicated_pairs = results_df.duplicated(subset=["id", "CODE"], keep=False)
    if duplicated_pairs.any():
        duplicated_df = results_df.loc[
            duplicated_pairs, ["id", "CODE"]
        ].drop_duplicates()
        n_dup = len(duplicated_df)
        raise ValueError(
            f"Trovate coppie duplicate (id, CODE) nel dataframe: {n_dup} combinazioni duplicate."
        )

    sim_matrix_df = (
        results_df.pivot(index="id", columns="CODE", values="dens_sim")
        .fillna(0.0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    return sim_matrix_df


def prepare_similarity_dataset(results_df: pd.DataFrame) -> SimilarityDataset:
    sim_matrix_df = build_similarity_matrix(results_df)

    patent_ids = sim_matrix_df.index.astype(str).tolist()
    codes = sim_matrix_df.columns.astype(str).tolist()
    sim_matrix = sim_matrix_df.to_numpy(dtype=float)

    return SimilarityDataset(
        results_df=results_df,
        sim_matrix_df=sim_matrix_df,
        sim_matrix=sim_matrix,
        patent_ids=patent_ids,
        codes=codes,
    )


def load_similarity_dataset(path: str | Path) -> SimilarityDataset:
    results_df = load_results_df(path)
    return prepare_similarity_dataset(results_df)


def summarize_results_df(results_df: pd.DataFrame) -> dict:
    validate_results_df(results_df)

    return {
        "n_rows": int(len(results_df)),
        "n_patents": int(results_df["id"].nunique()),
        "n_codes": int(results_df["CODE"].nunique()),
        "dens_sim_min": float(results_df["dens_sim"].min()),
        "dens_sim_max": float(results_df["dens_sim"].max()),
        "dens_sim_mean": float(results_df["dens_sim"].mean()),
        "dens_sim_median": float(results_df["dens_sim"].median()),
    }


def summarize_similarity_dataset(dataset: SimilarityDataset) -> dict:
    return {
        "n_patents": len(dataset.patent_ids),
        "n_codes": len(dataset.codes),
        "matrix_shape": dataset.sim_matrix.shape,
        "non_zero_entries": int((dataset.sim_matrix > 0).sum()),
    }
