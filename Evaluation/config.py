from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = EVAL_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
