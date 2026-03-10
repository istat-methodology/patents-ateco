import json
import os

import time
from datetime import timedelta

from time import sleep

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from utils.config import (
    NACE_EMBEDDINGS_PATH,
    PATENTS_SAMPLE_PATH,
    PREDICTIONS_PATH,
    PREDICTIONS_TEST_PATH,
    ensure_directories,
)
from utils.prompting import generate_classification_prompt
from utils.retrieval import build_patent_text, retrieve_top_k_nace
from utils.validation import validate_classification_result


# -----------------------------------------------------------
# CONFIGURAZIONE
# -----------------------------------------------------------

load_dotenv()

MODEL = "gpt-5-mini"
TOP_K_CANDIDATES = 10
MAX_SECONDARY = 3

CHECKPOINT_EVERY = 5
RESUME_IF_EXISTS = True
SLEEP_SECONDS = 0.2

# test rapido: None = usa tutto il dataset
MAX_ROWS = 40

LOG_EVERY = 100

OUTPUT_PATH = PREDICTIONS_TEST_PATH if MAX_ROWS is not None else PREDICTIONS_PATH

COLUMN_ORDER = [
    "row_id",
    "id",
    "title",
    "year",
    "primary_code",
    "secondary_codes",
    "top_k_codes",
    "top1_code",
    "top1_similarity",
    "top2_code",
    "top2_similarity",
]


# -----------------------------------------------------------
# CLIENT
# -----------------------------------------------------------


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY non trovata nel file .env")
    return OpenAI(api_key=api_key)


# -----------------------------------------------------------
# FUNZIONI I/O
# -----------------------------------------------------------


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(columns=COLUMN_ORDER)


def load_existing_results(path: str):
    if os.path.exists(path):
        try:
            df = pd.read_csv(
                path,
                dtype={
                    "row_id": "Int64",
                    "id": str,
                    "title": str,
                    "abstract": str,
                    "year": str,
                    "primary_code": str,
                    "secondary_codes": str,
                    "top_k_codes": str,
                    "top1_code": str,
                    "top1_similarity": str,
                    "top2_code": str,
                    "top2_similarity": str,
                },
            )
            print(f"🟡 Resume da file esistente: {len(df)} righe")
            return reorder_columns(df)
        except Exception as e:
            print(f"⚠️ Errore caricando output esistente: {e}")
    return None


def save_checkpoint(df: pd.DataFrame, path: str) -> None:
    df = reorder_columns(df)
    df = df.drop_duplicates(subset=["row_id"], keep="last")
    df.to_csv(path, index=False)
    print(f"💾 Checkpoint salvato: {len(df)} righe")


def serialize_secondary_codes(value) -> str:
    """
    Converte la lista di secondary codes in stringa JSON
    per salvarla nel CSV.
    """
    if value is None:
        return "[]"

    if isinstance(value, list):
        return json.dumps([str(x).strip() for x in value], ensure_ascii=False)

    return "[]"


# -----------------------------------------------------------
# CLASSIFICAZIONE
# -----------------------------------------------------------


def call_llm_for_classification(
    client: OpenAI,
    title: str,
    abstract: str,
    candidates_df: pd.DataFrame,
) -> dict:
    prompt = generate_classification_prompt(
        title=title,
        abstract=abstract,
        candidates_df=candidates_df,
        max_secondary=MAX_SECONDARY,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        print(f"❌ Errore API / parsing JSON: {e}")
        return {}


def classify_single_patent(
    row: pd.Series,
    nace_embeddings_df: pd.DataFrame,
    client: OpenAI,
    valid_codes: set[str],
) -> tuple[dict, pd.DataFrame]:
    patent_text = build_patent_text(
        title=row["title"],
        abstract=row["abstract"],
    )

    top_k_df = retrieve_top_k_nace(
        patent_text=patent_text,
        nace_embeddings_df=nace_embeddings_df,
        client=client,
        k=TOP_K_CANDIDATES,
    )

    raw_result = call_llm_for_classification(
        client=client,
        title=row["title"],
        abstract=row["abstract"],
        candidates_df=top_k_df,
    )

    clean_result = validate_classification_result(
        result=raw_result,
        valid_codes=valid_codes,
        max_secondary=MAX_SECONDARY,
    )

    return clean_result, top_k_df


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------


def main():
    ensure_directories()

    start_time = time.time()

    print("🔹 Caricamento client OpenAI...")
    client = get_openai_client()

    print("🔹 Caricamento brevetti...")
    patents_df = pd.read_csv(PATENTS_SAMPLE_PATH)

    if MAX_ROWS is not None:
        patents_df = patents_df.head(MAX_ROWS).copy()
        print(f"🔹 Modalità test: {len(patents_df)} brevetti")

    patents_df = patents_df.reset_index(drop=True)
    patents_df["row_id"] = patents_df.index.astype(int)

    print("🔹 Caricamento embedding NACE...")
    nace_embeddings_df = pd.read_parquet(NACE_EMBEDDINGS_PATH)

    valid_codes = set(nace_embeddings_df["code"].astype(str).str.strip())

    existing_df = load_existing_results(OUTPUT_PATH) if RESUME_IF_EXISTS else None
    done_ids = set(existing_df["row_id"]) if existing_df is not None else set()

    results = []

    for _, row in patents_df.iterrows():
        row_id = row["row_id"]

        if row_id in done_ids:
            continue

        processed = len(results) + (len(existing_df) if existing_df is not None else 0)

        if processed % LOG_EVERY == 0:
            total = len(patents_df)
            print(f"📊 Progress: {processed}/{total} ({processed/total:.1%})")

        result, top_k_df = classify_single_patent(
            row=row,
            nace_embeddings_df=nace_embeddings_df,
            client=client,
            valid_codes=valid_codes,
        )

        result["row_id"] = row_id
        result["id"] = str(row["id"]).strip()
        result["title"] = str(row["title"]).strip()
        result["abstract"] = str(row["abstract"]).strip()
        result["year"] = str(row.get("year", "")).strip()
        result["primary_code"] = str(result.get("primary_code", "")).strip()
        result["secondary_codes"] = serialize_secondary_codes(
            result.get("secondary_codes", [])
        )

        # diagnostica retrieval
        top_k_codes = top_k_df["code"].astype(str).tolist()
        result["top_k_codes"] = json.dumps(top_k_codes, ensure_ascii=False)

        result["top1_code"] = top_k_codes[0] if len(top_k_codes) > 0 else ""
        result["top1_similarity"] = (
            float(top_k_df.iloc[0]["similarity"]) if len(top_k_df) > 0 else None
        )

        result["top2_code"] = top_k_codes[1] if len(top_k_codes) > 1 else ""
        result["top2_similarity"] = (
            float(top_k_df.iloc[1]["similarity"]) if len(top_k_df) > 1 else None
        )

        results.append(result)

        processed = len(results) + (len(existing_df) if existing_df is not None else 0)

        if processed % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0

            total = len(patents_df)
            remaining = total - processed

            eta_seconds = remaining / rate if rate > 0 else 0

            print(
                f"📊 Progress: {processed}/{total} "
                f"({processed/total:.1%}) | "
                f"⏱ elapsed: {timedelta(seconds=int(elapsed))} | "
                f"🚀 rate: {rate:.2f} patents/sec | "
                f"⌛ ETA: {timedelta(seconds=int(eta_seconds))}"
            )

        if len(results) % CHECKPOINT_EVERY == 0:
            temp_df = pd.DataFrame(results)

            if existing_df is not None:
                temp_df = pd.concat([existing_df, temp_df], ignore_index=True)

            existing_df = reorder_columns(temp_df)
            save_checkpoint(existing_df, OUTPUT_PATH)

        sleep(SLEEP_SECONDS)

    results_df = pd.DataFrame(results)

    final_df = (
        pd.concat([existing_df, results_df], ignore_index=True)
        if existing_df is not None
        else results_df
    )

    final_df = reorder_columns(final_df)
    final_df = final_df.drop_duplicates(subset=["row_id"], keep="last")
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Completato. File salvato in: {OUTPUT_PATH}")
    print(f"📦 Numero record classificati in questa run: {len(results_df)}")
    print(f"📁 Totale record nel file finale: {len(final_df)}")


if __name__ == "__main__":
    main()
