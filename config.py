from pathlib import Path

# Absolute project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Directories
# Detect if running in Kubernetes (PV is mounted at /data)
K8S_DATA_DIR = Path("/data")

if K8S_DATA_DIR.exists():
    # Running inside Kubernetes
    DATA_DIR = K8S_DATA_DIR
else:
    # Running locally
    DATA_DIR = PROJECT_ROOT / "data"

FE_DIR = DATA_DIR / "feature_engineering"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Feature engineering artifacts
FEATURE_NAMES_PATH = FE_DIR / "feature_names.json"
PREPROCESSOR_PATH = FE_DIR / "preprocessor.joblib"
METADATA_PATH = FE_DIR / "metadata.json"

# Artifacts
MODEL_PATH = MODELS_DIR / "xgboost" / "best_model.joblib"
BEST_PARAMS_PATH = MODELS_DIR / "xgboost" / "best_params.json"

