import pandas as pd

from utils.config import (
    PATENTS_FILTERED_PATH,
    PATENTS_SAMPLE_PATH,
    ensure_directories,
)

SAMPLE_FRAC = 0.20


def load_filtered_patents() -> pd.DataFrame:
    df = pd.read_parquet(PATENTS_FILTERED_PATH)

    required_columns = {"id", "title", "abstract", "grant_date", "year"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Il file filtrato non contiene le colonne richieste: {sorted(missing)}"
        )

    return df


def count_records_by_year(df: pd.DataFrame) -> dict[int, int]:
    counts = df["year"].value_counts().sort_index().to_dict()
    return {int(year): int(count) for year, count in counts.items()}


def build_targets(counts: dict[int, int], frac: float = SAMPLE_FRAC) -> dict[int, int]:
    targets = {}
    for year, count in counts.items():
        target = max(1, int(round(count * frac)))
        targets[year] = target
    return targets


def collect_sample(
    df: pd.DataFrame,
    targets: dict[int, int],
) -> tuple[pd.DataFrame, dict[int, int]]:
    sampled_parts = []
    selected_per_year = {}

    for year in sorted(targets):
        year_df = df[df["year"] == year]
        n_target = targets[year]

        sampled_year_df = year_df.sample(n=n_target, random_state=42)

        sampled_parts.append(sampled_year_df)
        selected_per_year[year] = len(sampled_year_df)

    sample_df = pd.concat(sampled_parts, ignore_index=True)
    sample_df = sample_df.sort_values(["year", "id"]).reset_index(drop=True)

    return sample_df, selected_per_year


if __name__ == "__main__":
    ensure_directories()

    print("Caricamento dataset filtrato...")
    patents_df = load_filtered_patents()

    print(f"File sorgente: {PATENTS_FILTERED_PATH}")
    print(f"Numero record disponibili: {len(patents_df)}")

    counts = count_records_by_year(patents_df)

    print("\nDistribuzione completa:")
    for year in sorted(counts):
        print(year, counts[year])

    targets = build_targets(counts, frac=SAMPLE_FRAC)

    print(f"\nTarget sample {int(SAMPLE_FRAC * 100)}% per anno:")
    for year in sorted(targets):
        print(year, targets[year])

    print("\nRaccolta sample stratificato...")
    sample_df, selected_per_year = collect_sample(patents_df, targets)

    sample_df.to_csv(PATENTS_SAMPLE_PATH, index=False)

    print(f"\nFile salvato: {PATENTS_SAMPLE_PATH}")
    print(f"Numero record salvati: {len(sample_df)}")

    print("\nDistribuzione sample:")
    print(sample_df["year"].value_counts().sort_index())

    print("\nRecord selezionati per anno:")
    for year in sorted(selected_per_year):
        print(year, selected_per_year[year])
