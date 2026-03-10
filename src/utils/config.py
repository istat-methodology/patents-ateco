from pathlib import Path

# Root del progetto: patents-ateco/
BASE_DIR = Path(__file__).resolve().parents[2]

# Cartelle dati
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Risorse
RESOURCES_DIR = BASE_DIR / "resources"
CLASSIFICATION_DIR = RESOURCES_DIR / "classification"

# File input principali
PATENTS_SAMPLE_PATH = RAW_DIR / "patents_sample_20pct_stratified_by_year.csv"
NACE_SOURCE_PATH = CLASSIFICATION_DIR / "NACE_Rev2_1_Structure_Explanatory_Notes.xlsx"

# File intermedi
NACE_PREPROCESSED_PATH = INTERIM_DIR / "nace_level4_preprocessed.parquet"
NACE_EMBEDDINGS_PATH = INTERIM_DIR / "nace_level4_embeddings.parquet"
PATENTS_CANDIDATES_PATH = INTERIM_DIR / "patents_candidates_topk.parquet"

# File output
PREDICTIONS_TEST_PATH = PROCESSED_DIR / "patents_ateco_predictions_test.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "patents_ateco_predictions.csv"


def ensure_directories() -> None:
    """Crea le cartelle di progetto se non esistono."""
    for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, CLASSIFICATION_DIR]:
        path.mkdir(parents=True, exist_ok=True)
