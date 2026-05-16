"""Interpret weaknesses and scores into human-readable risk statements."""
from typing import List, Dict, Any


def interpret_risks(weaknesses: List[Dict[str, Any]], score_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    interpreted = []
    for w in weaknesses:
        sev = w.get('severity', 'MEDIUM')
        reason = ''
        if sev == 'CRITICAL' or sev == 'HIGH':
            reason = 'Immediate attention required: may cause incorrect clinical outputs or data leaks.'
        elif sev == 'MEDIUM':
            reason = 'Address soon: impacts reliability or maintainability.'
        else:
            reason = 'Monitor: minor or informational.'

        confidence = 0.6
        if sev == 'CRITICAL':
            confidence = 0.95
        elif sev == 'HIGH':
            confidence = 0.85
        elif sev == 'MEDIUM':
            confidence = 0.7

        interpreted.append({
            'title': w.get('title'),
            'description': w.get('description'),
            'severity': sev,
            'files': w.get('files'),
            'reason': reason,
            'confidence': confidence,
        })

    # Add any score anomalies as low-level risks
    anomalies = score_analysis.get('anomalies', {}) if score_analysis else {}
    for k, v in anomalies.items():
        interpreted.append({'title': f'Score anomaly: {k}', 'description': v, 'severity': 'LOW', 'files': [], 'reason': 'Statistical deviation', 'confidence': 0.6})

    return interpreted
