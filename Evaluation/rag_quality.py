from __future__ import annotations
from pathlib import Path
import sys
import json
from typing import List, Dict, Any

EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_ROOT))

from config import ROOT, OUTPUT_DIR
from utils import repo_files, read_file, save_json


def evaluate() -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    matches: List[Dict[str, Any]] = []
    retrieval_paths: List[str] = []

    for f in repo_files():
        text = read_file(f)
        lower = text.lower()
        if 'match_memory' in text:
            matches.append({'file': str(f.relative_to(ROOT)), 'context': 'match_memory usage'})
        if 'retrieve_memory' in text:
            retrieval_paths.append(str(f.relative_to(ROOT)))
        if 'search_memory' in text:
            retrieval_paths.append(str(f.relative_to(ROOT)))

    if not matches:
        findings.append({'issue': 'No match_memory retrieval found in the codebase', 'severity': 'HIGH', 'recommendation': 'Add semantic memory retrieval via match_memory or equivalent RPC call.'})
    if not retrieval_paths:
        findings.append({'issue': 'No memory retrieval entrypoints detected', 'severity': 'MEDIUM', 'recommendation': 'Ensure RAG retrieval is routed through service or pipeline functions.'})
    if matches and retrieval_paths:
        findings.append({'issue': 'RAG retrieval pipeline contains match_memory and retrieval hooks', 'severity': 'LOW'})

    score = 10
    if not matches:
        score -= 6
    if not retrieval_paths:
        score -= 4
    score = max(0, min(10, score))

    result = {
        'score': score,
        'matches_found': len(matches),
        'retrieval_files': sorted(set(retrieval_paths)),
        'findings': findings,
    }
    save_json('rag_quality', result)
    return result


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
