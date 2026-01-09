import sys
from pathlib import Path

# Add project root to sys.path so imports like `import app.main` work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
