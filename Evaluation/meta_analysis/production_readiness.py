"""Compute a production readiness interpretation and classification."""
from typing import Dict, Any


def assess_production_readiness(category_scores: Dict[str, float], interpreted_risks: list) -> Dict[str, Any]:
    # Basic readiness: average of key categories with penalties for high-severity risks
    key = ['architecture', 'safety', 'observability', 'agents']
    vals = [category_scores.get(k, 0.0) for k in key]
    base = sum(vals) / len(vals) if vals else 0.0

    penalty = 0.0
    for r in interpreted_risks:
        if r.get('severity') in ('CRITICAL', 'HIGH'):
            penalty += 1.5
        elif r.get('severity') == 'MEDIUM':
            penalty += 0.5

    final = max(0.0, base - penalty)

    classification = 'Production-ready' if final >= 8.0 and penalty < 3 else ('Prototype' if final >= 5.0 else 'Research / Experimental')

    return {'score': round(final, 2), 'classification': classification, 'penalty': round(penalty, 2), 'base': round(base, 2)}
