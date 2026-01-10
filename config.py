from pathlib import Path

# Absolute project root (local mode)
PROJECT_ROOT = Path(__file__).resolve().parent

# Priority 1 — Docker/Kubernetes image paths (/app)
DOCKER_DATA = Path("/app/data")
DOCKER_MODELS = Path("/app/models")

# Priority 2 — Kubernetes PV mount (/data)
K8S_DATA = Path("/data")
K8S_MODELS = Path("/data/models")  # optional

# Priority 3 — Local paths
LOCAL_DATA = PROJECT_ROOT / "data"
LOCAL_MODELS = PROJECT_ROOT / "models"

# ---------------------------------------
# Select DATA_DIR
# ---------------------------------------

if DOCKER_DATA.exists():
    # Running inside Docker container
    DATA_DIR = DOCKER_DATA
    MODELS_DIR = DOCKER_MODELS
elif K8S_DATA.exists():
    # Running in Kubernetes with PV
    DATA_DIR = K8S_DATA
    MODELS_DIR = K8S_MODELS
else:
    # Running locally
    DATA_DIR = LOCAL_DATA
    MODELS_DIR = LOCAL_MODELS

# ---------------------------------------
# Directories
# ---------------------------------------

FE_DIR = DATA_DIR / "feature_engineering"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------
# Feature Engineering Artifacts
# ---------------------------------------

FEATURE_NAMES_PATH = FE_DIR / "feature_names.json"
PREPROCESSOR_PATH = FE_DIR / "preprocessor.joblib"
METADATA_PATH = FE_DIR / "metadata.json"

# ---------------------------------------
# Model Artifacts
# ---------------------------------------

MODEL_PATH = MODELS_DIR / "xgboost" / "best_model.joblib"
BEST_PARAMS_PATH = MODELS_DIR / "xgboost" / "best_params.json"
