import os
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
PATENTS_FILTERED_PATH = RAW_DIR / "patents_filtered.parquet"

NACE_SOURCE_PATH = CLASSIFICATION_DIR / "NACE_Rev2_1_Structure_Explanatory_Notes.xlsx"


# File intermedi
NACE_PREPROCESSED_PATH = INTERIM_DIR / "nace_level4_preprocessed.parquet"
NACE_EMBEDDINGS_PATH = INTERIM_DIR / "nace_level4_embeddings.parquet"
PATENTS_CANDIDATES_PATH = INTERIM_DIR / "patents_candidates_topk.parquet"

PATENT_OPEN_EMBEDDINGS_PATH = INTERIM_DIR / "patent_filtered_open_embeddings.parquet"
NACE_OPEN_EMBEDDINGS_PATH = INTERIM_DIR / "nace_open_embeddings.parquet"

OPEN_EMBEDDING_MODEL_NAME = os.getenv(
    "OPEN_EMBEDDING_MODEL_NAME",
    "BAAI/bge-m3",
)

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

raw_top_k = os.getenv("SIMILARITY_TOP_K", "20")
SIMILARITY_TOP_K = None if raw_top_k.lower() == "none" else int(raw_top_k)

RECOMPUTE_EMBEDDINGS = os.getenv("RECOMPUTE_EMBEDDINGS", "0") == "1"


# File output
PREDICTIONS_TEST_PATH = PROCESSED_DIR / "patents_ateco_predictions_test.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "patents_ateco_predictions.csv"


# Analysis
PATENT_CODE_SIMILARITY_PATH = INTERIM_DIR / "patents_filtered_code_similarity.parquet"

# Analysis outputs
ANALYSIS_DIR = PROCESSED_DIR / "analysis"

SIMILARITY_GLOBAL_STATS_PATH = ANALYSIS_DIR / "similarity_global_stats.csv"
SIMILARITY_RANK_STATS_PATH = ANALYSIS_DIR / "similarity_rank_stats.csv"
PATENT_TOP12_GAP_PATH = ANALYSIS_DIR / "patent_top1_top2_gap.parquet"
GAP_STATS_PATH = ANALYSIS_DIR / "gap_stats.csv"
TOP1_SIMILARITY_STATS_PATH = ANALYSIS_DIR / "top1_similarity_stats.csv"
TOP1_CODE_DISTRIBUTION_PATH = ANALYSIS_DIR / "top1_code_distribution.csv"
TOP1_CODE_CONCENTRATION_STATS_PATH = ANALYSIS_DIR / "top1_code_concentration_stats.csv"


def ensure_directories() -> None:
    """Crea le cartelle di progetto se non esistono."""
    for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, CLASSIFICATION_DIR]:
        path.mkdir(parents=True, exist_ok=True)
