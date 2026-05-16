"""Compute a simple architecture score based on repository structure and layering."""
from pathlib import Path
import json
from evaluation.config import EVAL_ROOT, ROOT
from evaluation.utils import repo_files, read_file, save_json, parse_imports


def run() -> dict:
    score = 0
    findings = []

    # Check presence of service layer
    svc = list((ROOT / 'core' / 'services').glob('*.py'))
    repo = list((ROOT / 'core' / 'repositories').glob('*.py'))
    client = (ROOT / 'core' / 'database' / 'client.py')

    if svc:
        score += 3
    else:
        findings.append('Missing core/services implementations')

    if repo:
        score += 3
    else:
        findings.append('Missing core/repositories implementations')

    if client.exists():
        score += 2
    else:
        findings.append('Missing centralized DB client')

    # Check for direct supabase usage
    direct_db = []
    for f in repo_files():
        text = read_file(f)
        if 'supabase' in text and 'core/database/client' not in text:
            direct_db.append(str(f))

    if direct_db:
        findings.append('Direct supabase usage outside client: ' + ','.join(direct_db[:5]))
        score -= 1

    score = max(0, min(10, score))
    result = {'score': score, 'findings': findings}
    save_json('architecture_score', result)
    return result


if __name__ == '__main__':
    print(run())
