from __future__ import annotations
from pathlib import Path
import sys
import json
from typing import List, Dict, Any

EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_ROOT))

from config import ROOT, OUTPUT_DIR
from utils import repo_files, read_file, save_json

METRICS_TERMS = ['prometheus', 'opentelemetry', 'trace', 'meter', 'counter', 'gauge', 'histogram', 'metrics', 'telemetry']


def evaluate() -> Dict[str, Any]:
    logging_files: List[str] = []
    metrics_files: List[str] = []
    findings: List[Dict[str, Any]] = []

    for f in repo_files():
        text = read_file(f)
        lower = text.lower()
        if 'import logging' in lower or 'logging.' in lower or 'logger.' in lower:
            logging_files.append(str(f.relative_to(ROOT)))
        if any(term in lower for term in METRICS_TERMS):
            metrics_files.append(str(f.relative_to(ROOT)))

    if not logging_files:
        findings.append({'issue': 'No logging imports or usage detected', 'severity': 'MEDIUM', 'recommendation': 'Add structured logging to key services and modules.'})
    if not metrics_files:
        findings.append({'issue': 'No metrics instrumentation detected', 'severity': 'MEDIUM', 'recommendation': 'Add metrics hooks for critical application paths or use telemetry libraries.'})

    score = 0
    if logging_files:
        score += 5
    if metrics_files:
        score += 5
    score = max(0, min(10, score))

    result = {
        'score': score,
        'logging_file_count': len(logging_files),
        'metrics_file_count': len(metrics_files),
        'logging_files': sorted(logging_files),
        'metrics_files': sorted(metrics_files),
        'findings': findings,
    }
    save_json('observability', result)
    return result


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
