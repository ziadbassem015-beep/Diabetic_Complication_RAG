from __future__ import annotations
from pathlib import Path
import sys
import json
import hashlib
import difflib
from typing import List, Dict, Any

EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_ROOT))

from config import ROOT, OUTPUT_DIR
from utils import repo_files, read_file, save_json


def compute_hash(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def normalize_text(text: str) -> str:
    return ''.join(line.strip() for line in text.splitlines() if line.strip())


def evaluate() -> Dict[str, Any]:
    files = repo_files()
    hashes: Dict[str, List[Path]] = {}
    exact_duplicates: List[Dict[str, Any]] = []
    similar_pairs: List[Dict[str, Any]] = []

    for f in files:
        try:
            content = f.read_bytes()
        except Exception:
            continue
        fingerprint = compute_hash(content)
        hashes.setdefault(fingerprint, []).append(f)

    for fingerprint, paths in hashes.items():
        if len(paths) > 1:
            exact_duplicates.append({
                'hash': fingerprint,
                'files': [str(p.relative_to(ROOT)) for p in sorted(paths)],
                'count': len(paths),
            })

    python_files = [f for f in files if f.suffix == '.py']
    for i, a in enumerate(python_files):
        for b in python_files[i + 1:]:
            if a.samefile(b):
                continue
            if a.read_bytes() == b.read_bytes():
                continue
            text_a = normalize_text(read_file(a))
            text_b = normalize_text(read_file(b))
            if not text_a or not text_b:
                continue
            ratio = difflib.SequenceMatcher(None, text_a, text_b).ratio()
            if ratio >= 0.75:
                similar_pairs.append({
                    'file_a': str(a.relative_to(ROOT)),
                    'file_b': str(b.relative_to(ROOT)),
                    'similarity': round(ratio, 3),
                })

    findings: List[Dict[str, Any]] = []
    for dup in exact_duplicates:
        findings.append({
            'issue': 'Exact duplicate files detected',
            'severity': 'MEDIUM',
            'files': dup['files'],
            'count': dup['count'],
            'recommendation': 'Remove or consolidate exact duplicate files to reduce maintenance and import confusion.'
        })
    for pair in similar_pairs:
        findings.append({
            'issue': 'High similarity pair detected',
            'severity': 'LOW',
            'files': [pair['file_a'], pair['file_b']],
            'similarity': pair['similarity'],
            'recommendation': 'Review similar files for potential consolidation or shared helper extraction.'
        })

    score = 10
    score -= 4 * len(exact_duplicates)
    score -= 1 * len(similar_pairs)
    score = max(0, min(10, score))

    result = {
        'score': score,
        'exact_duplicate_groups': exact_duplicates,
        'similar_pairs': sorted(similar_pairs, key=lambda item: -item['similarity'])[:25],
        'findings': findings,
    }
    save_json('duplicate_detector', result)
    return result


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
