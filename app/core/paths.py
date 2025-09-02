from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
VERSIONS_DIR = DATA_ROOT / "versions"
ACTIVE_DIR = DATA_ROOT / "active"

for p in [DATA_ROOT, VERSIONS_DIR, ACTIVE_DIR]:
    p.mkdir(parents=True, exist_ok=True)