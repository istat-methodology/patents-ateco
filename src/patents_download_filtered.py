import os

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

from utils.config import RAW_DIR, ensure_directories

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_NAME = "istat-ai/ai-patents"
SPLIT = "train"

MIN_ABSTRACT_LEN = 30

OUTPUT_PATH = RAW_DIR / "patents_filtered.parquet"


def get_year(row):
    priority_date = pd.to_datetime(row.get("priority date"), errors="coerce")
    grant_date = pd.to_datetime(row.get("grant date"), errors="coerce")

    if pd.notna(priority_date):
        return int(priority_date.year)

    if pd.notna(grant_date):
        return int(grant_date.year)

    return None


def is_valid_record(row):

    if row.get("ita_only") != 0:
        return False

    abstract = row.get("abstract")
    if abstract is None:
        return False

    abstract = str(abstract).strip()

    if len(abstract) <= MIN_ABSTRACT_LEN:
        return False

    year = get_year(row)
    if year is None:
        return False

    return True


def stream_dataset():
    return load_dataset(
        DATASET_NAME,
        split=SPLIT,
        streaming=True,
        token=HF_TOKEN,
    )


def download_filtered_dataset():

    records = []

    total = 0
    valid = 0

    for row in stream_dataset():

        total += 1

        if not is_valid_record(row):
            continue

        year = get_year(row)

        records.append(
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "abstract": str(row.get("abstract")).strip(),
                "grant_date": row.get("grant date"),
                "year": year,
            }
        )

        valid += 1

        if valid % 10000 == 0:
            print(f"Valid records collected: {valid}")

    df = pd.DataFrame(records)

    print("\nDownload completed")
    print(f"Total rows scanned: {total}")
    print(f"Valid rows kept: {valid}")

    return df


if __name__ == "__main__":

    ensure_directories()

    print("Downloading and filtering patents dataset...")

    df = download_filtered_dataset()

    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nFile saved to: {OUTPUT_PATH}")
    print(f"Final dataset size: {len(df)}")
