import pandas as pd

from utils.config import (
    NACE_PREPROCESSED_PATH,
    NACE_SOURCE_PATH,
    ensure_directories,
)


def load_nace(path: str | None = None, level: int = 4) -> pd.DataFrame:
    """
    Carica la classificazione NACE da Excel e restituisce
    un DataFrame pulito al livello richiesto.

    Il campo 'text' viene costruito in forma semantica per retrieval:
    - NACE code
    - Division
    - Class
    - Includes
    - Includes also
    """
    source_path = path or NACE_SOURCE_PATH

    full_nace = pd.read_excel(source_path).fillna("")
    full_nace = full_nace.replace("\n", " ", regex=True)

    # Normalizzazione campi nel dataframe completo
    for col in ["CODE", "NAME", "PARENT_CODE", "Includes", "IncludesAlso"]:
        if col in full_nace.columns:
            full_nace[col] = full_nace[col].astype(str).str.strip()

    # Costruiamo una mappa delle divisioni (LEVEL == 2)
    divisions_df = full_nace[full_nace["LEVEL"] == 2][["CODE", "NAME"]].copy()
    division_map = dict(zip(divisions_df["CODE"], divisions_df["NAME"]))

    # Filtriamo il livello richiesto (es. 4)
    nace = full_nace[full_nace["LEVEL"] == level].copy()

    # Codice divisione = prime 2 cifre del codice classe
    nace["division_code"] = nace["CODE"].astype(str).str[:2]
    nace["division_title"] = nace["division_code"].map(division_map).fillna("")

    def build_nace_text(row) -> str:
        parts = [
            f"NACE code: {row['CODE']}.",
            f"Division: {row['division_title']}.",
            f"Class: {row['NAME']}.",
        ]

        if row["Includes"]:
            parts.append(f"Includes: {row['Includes']}.")
        if row["IncludesAlso"]:
            parts.append(f"Includes also: {row['IncludesAlso']}.")

        text = " ".join(parts)
        text = " ".join(text.split())
        return text

    nace["text"] = nace.apply(build_nace_text, axis=1)

    nace = nace[
        ["CODE", "NAME", "PARENT_CODE", "division_code", "division_title", "text"]
    ].reset_index(drop=True)

    nace.columns = [
        "code",
        "title",
        "parent_code",
        "division_code",
        "division_title",
        "text",
    ]

    nace["code"] = nace["code"].astype(str).str.strip()
    nace["title"] = nace["title"].astype(str).str.strip()
    nace["parent_code"] = nace["parent_code"].astype(str).str.strip()
    nace["division_code"] = nace["division_code"].astype(str).str.strip()
    nace["division_title"] = nace["division_title"].astype(str).str.strip()
    nace["text"] = nace["text"].astype(str).str.strip()

    return nace


def save_nace_preprocessed(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Salva il DataFrame preprocessato in parquet."""
    target_path = output_path or NACE_PREPROCESSED_PATH
    df.to_parquet(target_path, index=False)


def main() -> None:
    ensure_directories()

    nace_df = load_nace()
    save_nace_preprocessed(nace_df)

    print(f"NACE preprocessato salvato in: {NACE_PREPROCESSED_PATH}")
    print(f"Numero classi: {len(nace_df)}")
    print(nace_df.head())


if __name__ == "__main__":
    main()
