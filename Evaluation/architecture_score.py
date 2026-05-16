from __future__ import annotations
from pathlib import Path
import sys
import json
from typing import List, Dict, Any

EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_ROOT))

from config import ROOT, OUTPUT_DIR
from utils import repo_files, read_file, save_json, parse_imports


def evaluate() -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    score = 10

    services = list((ROOT / 'core' / 'services').glob('*.py'))
    repos = list((ROOT / 'core' / 'repositories').glob('*.py'))
    client_file = ROOT / 'core' / 'database' / 'client.py'

    if not services:
        findings.append({'issue': 'Missing core/services implementations', 'severity': 'HIGH'})
        score -= 4

    if not repos:
        findings.append({'issue': 'Missing core/repositories implementations', 'severity': 'HIGH'})
        score -= 4

    if not client_file.exists():
        findings.append({'issue': 'Missing centralized DB client', 'severity': 'HIGH'})
        score -= 4

    gest_repo = ROOT / 'core' / 'repositories' / 'gestational_repo.py'
    heart_repo = ROOT / 'core' / 'repositories' / 'heart_risk_repo.py'
    if not gest_repo.exists():
        findings.append({'issue': 'Missing GestationalRepository', 'severity': 'MEDIUM'})
        score -= 1
    if not heart_repo.exists():
        findings.append({'issue': 'Missing HeartRiskRepository', 'severity': 'MEDIUM'})
        score -= 1

    for f in repo_files():
        rel = f.relative_to(ROOT)
        text = read_file(f)
        is_service = rel.parts[:2] == ('core', 'services')
        is_repo = rel.parts[:2] == ('core', 'repositories')
        is_client = rel == Path('core/database/client.py')
        is_ui = rel.name in ('app.py', 'main.py') or rel.parts[0] == 'multi_agent'

        if is_ui or is_service:
            if 'from core.database.client' in text or 'get_supabase' in text or 'from supabase' in text or 'supabase.' in text:
                findings.append({
                    'file': str(rel),
                    'issue': 'Service/UI layer uses database client or direct Supabase access',
                    'severity': 'HIGH',
                    'recommendation': 'Only repositories should import core.database.client and access Supabase through the client abstraction.'
                })
                score -= 3

        if is_repo and not is_client:
            if 'from supabase' in text or 'supabase.' in text:
                findings.append({
                    'file': str(rel),
                    'issue': 'Repository uses Supabase directly instead of core.database.client',
                    'severity': 'HIGH',
                    'recommendation': 'Refactor repository code to use get_supabase() from core.database.client only.'
                })
                score -= 3

        if not is_client and 'from supabase' in text and not is_repo:
            findings.append({
                'file': str(rel),
                'issue': 'Direct Supabase import found outside database client module',
                'severity': 'HIGH',
                'recommendation': 'Move direct Supabase client creation into core.database.client and reference it through repositories only.'
            })
            score -= 4

    score = max(0, min(10, score))
    result = {
        'score': score,
        'findings': findings,
        'summary': {
            'service_files': len(services),
            'repository_files': len(repos),
            'db_client_exists': client_file.exists(),
        }
    }
    save_json('architecture_score', result)
    return result


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
