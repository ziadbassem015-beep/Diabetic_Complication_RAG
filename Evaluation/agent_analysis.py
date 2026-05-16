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
    files_checked = 0

    for f in repo_files():
        if 'multi_agent' not in str(f.relative_to(ROOT)):
            continue
        files_checked += 1
        text = read_file(f)
        if 'DiagnosticGraph' in text and 'initialize' not in text:
            findings.append({
                'file': str(f.relative_to(ROOT)),
                'issue': 'DiagnosticGraph missing initialize()',
                'severity': 'MEDIUM',
                'recommendation': 'Implement DiagnosticGraph.initialize() or confirm the graph is initialized before use.'
            })

    if files_checked == 0:
        findings.append({'issue': 'No multi-agent files were discovered for analysis', 'severity': 'HIGH', 'recommendation': 'Verify the multi_agent module exists and is part of the repository.'})

    score = 10
    score -= 3 * len([f for f in findings if f['severity'] in ('HIGH', 'MEDIUM')])
    score = max(0, min(10, score))

    result = {
        'score': score,
        'files_checked': files_checked,
        'findings': findings,
    }
    save_json('agent_analysis', result)
    return result


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
