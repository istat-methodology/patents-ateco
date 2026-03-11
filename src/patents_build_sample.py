import os

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

from utils.config import PATENTS_SAMPLE_PATH, ensure_directories

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_NAME = "istat-ai/ai-patents"
SPLIT = "train"
SAMPLE_FRAC = 0.20
MIN_ABSTRACT_LEN = 30


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
    return load_dataset(DATASET_NAME, split=SPLIT, streaming=True, token=HF_TOKEN)


def count_valid_records_by_year():
    counts = {}

    for row in stream_dataset():
        if not is_valid_record(row):
            continue

        year = get_year(row)
        counts[year] = counts.get(year, 0) + 1

    return counts


def build_targets(counts, frac=0.20):
    targets = {}
    for year, count in counts.items():
        target = max(1, int(round(count * frac)))
        targets[year] = target
    return targets


def collect_sample(targets):
    collected = []
    selected_per_year = {year: 0 for year in targets}

    for row in stream_dataset():
        if not is_valid_record(row):
            continue

        year = get_year(row)

        if selected_per_year[year] >= targets[year]:
            continue

        collected.append(
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "grant_date": row.get("grant date"),
                "abstract": row.get("abstract"),
                "year": year,
            }
        )

        selected_per_year[year] += 1

        if all(selected_per_year[y] >= targets[y] for y in targets):
            break

    return pd.DataFrame(collected), selected_per_year


if __name__ == "__main__":
    ensure_directories()

    print("Conteggio record validi per anno...")
    counts = count_valid_records_by_year()

    print("Distribuzione completa:")
    for year in sorted(counts):
        print(year, counts[year])

    targets = build_targets(counts, frac=SAMPLE_FRAC)

    print("\nTarget sample 20% per anno:")
    for year in sorted(targets):
        print(year, targets[year])

    print("\nRaccolta sample stratificato...")
    sample_df, _ = collect_sample(targets)

    sample_df.to_csv(PATENTS_SAMPLE_PATH, index=False)

    print(f"\nFile salvato: {PATENTS_SAMPLE_PATH}")
    print(f"Numero record salvati: {len(sample_df)}")

    print("\nDistribuzione sample:")
    print(sample_df["year"].value_counts().sort_index())
