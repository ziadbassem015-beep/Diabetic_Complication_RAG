"""Utility helpers for the evaluation framework."""
from pathlib import Path
import ast
import json
from typing import List, Dict, Any
from config import ROOT, OUTPUT_DIR


def repo_files(pattern="**/*.py") -> List[Path]:
    excluded = {'.venv', 'venv', '.git', 'node_modules', 'notebooks', 'database/migrations', 'Evaluation/outputs'}
    files: List[Path] = []
    for p in ROOT.rglob('*.py'):
        parts = set(p.parts)
        if parts & excluded:
            continue
        try:
            if p.stat().st_size > 1_000_000:  # skip very large files
                continue
        except Exception:
            continue
        files.append(p)
    return files


def read_file(path: Path) -> str:
    # Read in binary then decode with fallbacks to avoid blocking or codec issues.
    try:
        with path.open('rb') as f:
            data = f.read()
    except Exception:
        # fallback to text read with replace
        try:
            return path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            return ''
    for enc in ('utf-8', 'utf-8-sig', 'latin-1'):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode('utf-8', errors='replace')


def save_json(name: str, data: Dict[str, Any]):
    out = OUTPUT_DIR / f"{name}.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    return out


def parse_imports(source: str) -> List[str]:
    try:
        tree = ast.parse(source)
    except Exception:
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(module)
    return [i for i in imports if i]
