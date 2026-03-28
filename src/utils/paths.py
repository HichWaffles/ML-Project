from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "retail_customers_COMPLETE_CATEGORICAL.csv"
TRAIN_TEST_DIR = DATA_DIR / "train_test"
SEGMENTS_DIR = DATA_DIR / "segments"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
GEOIP_DB_PATH = DATA_DIR / "GeoLite2-City.mmdb"
