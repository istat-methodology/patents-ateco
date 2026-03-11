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
    "abstract",
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


def load_existing_results(path: str) -> pd.DataFrame | None:
    """
    Carica il file di output esistente per il resume.
    Il resume è basato su 'id' del brevetto.
    """
    if not os.path.exists(path):
        return None

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

        df = reorder_columns(df)

        if "id" not in df.columns:
            raise ValueError("La colonna 'id' non è presente nel file di output.")

        df["id"] = df["id"].astype(str).str.strip()
        df = df.dropna(subset=["id"]).copy()

        # deduplica per id
        df = df.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)

        print(f"🟡 Resume da file esistente: {len(df)} righe")

        if len(df) > 0:
            print(f"🟡 Ultimo id presente nel file: {df['id'].iloc[-1]}")

        return df

    except Exception as e:
        print(f"⚠️ Errore caricando output esistente: {e}")
        return None


def append_checkpoint(rows: list[dict], path: str) -> None:
    """
    Salva solo le nuove righe in append.
    In questo modo il resume riparte da ciò che è davvero già scritto nel CSV.
    """
    if not rows:
        return

    df = pd.DataFrame(rows)
    df = reorder_columns(df)

    file_exists = os.path.exists(path)
    write_header = not file_exists or os.path.getsize(path) == 0

    df.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
    )

    print(f"💾 Checkpoint salvato: +{len(df)} righe")


def save_final_snapshot(path: str) -> pd.DataFrame | None:
    """
    Rilegge il file finale, deduplica per id e lo risalva pulito.
    """
    final_df = load_existing_results(path)
    if final_df is None:
        return None

    final_df = reorder_columns(final_df)
    final_df = final_df.drop_duplicates(subset=["id"], keep="last").reset_index(
        drop=True
    )
    final_df.to_csv(path, index=False)

    print(f"💾 Snapshot finale salvato: {len(final_df)} righe")
    return final_df


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


def build_output_row(row: pd.Series, result: dict, top_k_df: pd.DataFrame) -> dict:
    output_row = dict(result)

    output_row["row_id"] = int(row["row_id"])
    output_row["id"] = str(row["id"]).strip()
    output_row["title"] = str(row["title"]).strip()
    output_row["abstract"] = str(row["abstract"]).strip()
    output_row["year"] = str(row.get("year", "")).strip()
    output_row["primary_code"] = str(output_row.get("primary_code", "")).strip()
    output_row["secondary_codes"] = serialize_secondary_codes(
        output_row.get("secondary_codes", [])
    )

    # diagnostica retrieval
    top_k_codes = top_k_df["code"].astype(str).tolist()
    output_row["top_k_codes"] = json.dumps(top_k_codes, ensure_ascii=False)

    output_row["top1_code"] = top_k_codes[0] if len(top_k_codes) > 0 else ""
    output_row["top1_similarity"] = (
        float(top_k_df.iloc[0]["similarity"]) if len(top_k_df) > 0 else None
    )

    output_row["top2_code"] = top_k_codes[1] if len(top_k_codes) > 1 else ""
    output_row["top2_similarity"] = (
        float(top_k_df.iloc[1]["similarity"]) if len(top_k_df) > 1 else None
    )

    return output_row


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
    patents_df["id"] = patents_df["id"].astype(str).str.strip()

    total = len(patents_df)
    current_ids = set(patents_df["id"].tolist())

    print("🔹 Caricamento embedding NACE...")
    nace_embeddings_df = pd.read_parquet(NACE_EMBEDDINGS_PATH)
    valid_codes = set(nace_embeddings_df["code"].astype(str).str.strip())

    existing_df = load_existing_results(OUTPUT_PATH) if RESUME_IF_EXISTS else None
    done_ids = (
        set(existing_df["id"].astype(str).str.strip())
        if existing_df is not None
        else set()
    )

    skipped = 0
    newly_processed = 0
    buffer: list[dict] = []

    print(
        f"🔹 Record già presenti nel file di output: {len(done_ids.intersection(current_ids))}"
    )
    print(f"🔹 Totale record input da considerare: {total}")

    try:
        for _, row in patents_df.iterrows():
            patent_id = str(row["id"]).strip()

            if patent_id in done_ids:
                skipped += 1
                continue

            result, top_k_df = classify_single_patent(
                row=row,
                nace_embeddings_df=nace_embeddings_df,
                client=client,
                valid_codes=valid_codes,
            )

            output_row = build_output_row(row, result, top_k_df)
            buffer.append(output_row)

            done_ids.add(patent_id)
            newly_processed += 1

            completed = len(done_ids.intersection(current_ids))

            if completed % LOG_EVERY == 0 or newly_processed == 1 or completed == total:
                elapsed = time.time() - start_time
                rate = newly_processed / elapsed if elapsed > 0 else 0
                remaining = total - completed
                eta_seconds = remaining / rate if rate > 0 else 0

                print(
                    f"📊 Progress: {completed}/{total} "
                    f"({completed/total:.1%}) | "
                    f"nuovi: {newly_processed} | "
                    f"skipped: {skipped} | "
                    f"⏱ elapsed: {timedelta(seconds=int(elapsed))} | "
                    f"🚀 rate: {rate:.2f} patents/sec | "
                    f"⌛ ETA: {timedelta(seconds=int(eta_seconds))}"
                )

            if len(buffer) >= CHECKPOINT_EVERY:
                append_checkpoint(buffer, OUTPUT_PATH)
                buffer = []

            sleep(SLEEP_SECONDS)

    except KeyboardInterrupt:
        print("\n⚠️ Interruzione manuale rilevata.")
        if buffer:
            append_checkpoint(buffer, OUTPUT_PATH)
            buffer = []
        print(
            "🟡 Buffer salvato. Alla prossima esecuzione il resume ripartirà dal CSV."
        )
        raise

    except Exception as e:
        print(f"\n❌ Errore durante l'elaborazione: {e}")
        if buffer:
            append_checkpoint(buffer, OUTPUT_PATH)
            buffer = []
        print(
            "🟡 Buffer salvato prima dell'uscita. Il resume userà il CSV già scritto."
        )
        raise

    if buffer:
        append_checkpoint(buffer, OUTPUT_PATH)

    final_df = save_final_snapshot(OUTPUT_PATH)

    print(f"✅ Completato. File salvato in: {OUTPUT_PATH}")
    print(f"📦 Numero record classificati in questa run: {newly_processed}")
    print(f"⏭️ Record già presenti e saltati: {skipped}")

    if final_df is not None:
        print(f"📁 Totale record nel file finale: {len(final_df)}")


if __name__ == "__main__":
    main()
