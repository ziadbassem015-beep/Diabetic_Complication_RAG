from __future__ import annotations
from pathlib import Path
import sys
import json
from typing import List, Dict, Any

EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_ROOT))

from config import ROOT, OUTPUT_DIR
from utils import repo_files, read_file, save_json

UNSAFE_TERMS = ['diagnose', 'diagnosis', 'diagnosed', 'cure', 'treatment plan', 'prescribe']
DISCLAIMER_TERMS = ['disclaimer', 'consult a physician', 'medical professional', 'licensed', 'not medical advice']


def has_disclaimer(text: str) -> bool:
    low = text.lower()
    return any(term in low for term in DISCLAIMER_TERMS)


def evaluate() -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    unsafe_hits: List[Dict[str, Any]] = []

    for f in repo_files():
        text = read_file(f)
        lower = text.lower()
        if any(term in lower for term in UNSAFE_TERMS):
            if not has_disclaimer(text):
                snippet = next((line for line in text.splitlines() if any(term in line.lower() for term in UNSAFE_TERMS)), '')
                unsafe_hits.append({
                    'file': str(f.relative_to(ROOT)),
                    'snippet': snippet.strip(),
                })

    for hit in unsafe_hits:
        findings.append({
            'file': hit['file'],
            'issue': 'Potential unsafe medical wording without disclaimer',
            'severity': 'MEDIUM',
            'description': hit['snippet'],
            'recommendation': 'Add clear medical disclaimers around clinical advice and avoid diagnostic language without supervision.'
        })

    score = 10 - 2 * len(unsafe_hits)
    score = max(0, min(10, score))

    result = {
        'score': score,
        'unsafe_hits': unsafe_hits,
        'findings': findings,
    }
    save_json('prompt_safety', result)
    return result


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
