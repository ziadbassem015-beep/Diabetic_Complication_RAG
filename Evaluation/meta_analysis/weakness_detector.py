"""Detect weaknesses from raw findings and module results."""
from typing import List, Dict, Any


def detect_weaknesses(risk_summary: List[Dict[str, Any]], module_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    weaknesses = []
    # Look for direct DB access hints in module_results findings
    for m in module_results:
        res = m.get('result') or {}
        if isinstance(res, dict):
            findings = res.get('findings') or res.get('details') or []
            for f in findings:
                text = ''
                if isinstance(f, str):
                    text = f
                elif isinstance(f, dict):
                    text = ' '.join(str(v) for v in f.values())
                if 'Direct supabase' in text or 'supabase' in text.lower():
                    weaknesses.append({'title': 'Direct DB access', 'description': text, 'severity': 'HIGH', 'files': [m.get('path')]})

    # Convert risk_summary items into structured weaknesses
    for item in (risk_summary or []):
        title = item.get('title') or item.get('file') or 'Finding'
        desc = item.get('description') or item.get('issue') or str(item)
        sev = item.get('severity') or ('MEDIUM' if 'issue' in item else 'LOW')
        weaknesses.append({'title': title, 'description': desc, 'severity': sev, 'files': item.get('affected_files') or [item.get('file')]})

    # dedupe by title
    seen = set()
    dedup = []
    for w in weaknesses:
        key = (w.get('title'), w.get('description'))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(w)

    return dedup
