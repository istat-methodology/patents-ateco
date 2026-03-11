import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

from utils.config import (
    NACE_PREPROCESSED_PATH,
    NACE_EMBEDDINGS_PATH,
    ensure_directories,
)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_texts(texts, batch_size=100):
    embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i : i + batch_size]

        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)

        embeddings.extend([e.embedding for e in resp.data])

        print(f"Embedded {i + len(batch)} / {len(texts)}")

    return embeddings


def main():

    ensure_directories()

    print("Loading NACE preprocessed data...")
    nace_df = pd.read_parquet(NACE_PREPROCESSED_PATH)

    texts = nace_df["text"].tolist()

    print("Generating embeddings...")
    embeddings = embed_texts(texts)

    nace_df["embedding"] = embeddings

    nace_df.to_parquet(NACE_EMBEDDINGS_PATH, index=False)

    print(f"Saved embeddings to: {NACE_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
