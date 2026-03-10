import os
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY non trovata nelle variabili ambiente.")

    return OpenAI(api_key=api_key)


def build_patent_text(title: str, abstract: str) -> str:

    title = str(title).strip() if title else ""
    abstract = str(abstract).strip() if abstract else ""

    return f"Title: {title}\nAbstract: {abstract}"


def embed_text(client: OpenAI, text: str) -> List[float]:

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding


def normalize(vectors: np.ndarray) -> np.ndarray:

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1

    return vectors / norms


def retrieve_top_k_nace(
    patent_text: str,
    nace_embeddings_df: pd.DataFrame,
    client: OpenAI,
    k: int = 15,
) -> pd.DataFrame:

    # embedding brevetto
    patent_embedding = embed_text(client, patent_text)

    patent_vec = np.array(patent_embedding).reshape(1, -1)

    # matrice embedding NACE
    nace_matrix = np.vstack(nace_embeddings_df["embedding"].values)

    # normalizzazione vettori
    patent_vec = normalize(patent_vec)
    nace_matrix = normalize(nace_matrix)

    # cosine similarity vettoriale
    similarities = np.dot(nace_matrix, patent_vec.T).flatten()

    work_df = nace_embeddings_df.copy()
    work_df["similarity"] = similarities

    top_k = (
        work_df.sort_values("similarity", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )

    return top_k
