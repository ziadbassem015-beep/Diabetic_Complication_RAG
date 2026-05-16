"""Generate prioritized improvement recommendations based on interpreted risks."""
from typing import List, Dict, Any


def recommend_improvements(interpreted_risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    recs = []
    for r in interpreted_risks:
        sev = r.get('severity', 'MEDIUM')
        title = r.get('title')
        if 'DB' in title or 'DB' in (r.get('description') or '') or 'Direct DB' in title:
            recs.append({'priority': 'P0', 'action': 'Remove direct DB access from application code; use repository layer.', 'rationale': r.get('description'), 'files': r.get('files')})
            continue
        if sev == 'CRITICAL' or sev == 'HIGH':
            recs.append({'priority': 'P0', 'action': f'Fix: {title}', 'rationale': r.get('description'), 'files': r.get('files')})
        elif sev == 'MEDIUM':
            recs.append({'priority': 'P1', 'action': f'Improve: {title}', 'rationale': r.get('description'), 'files': r.get('files')})
        else:
            recs.append({'priority': 'P2', 'action': f'Review: {title}', 'rationale': r.get('description'), 'files': r.get('files')})

    # sort by priority
    order = {'P0': 0, 'P1': 1, 'P2': 2}
    recs.sort(key=lambda x: order.get(x['priority'], 3))
    return recs
