"""Analyze raw category scores and compute derived metrics."""
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ScoreAnalysis:
    raw: Dict[str, float]
    mean: float
    weighted: float
    anomalies: Dict[str, str]


def analyze_scores(scores: Dict[str, float]) -> ScoreAnalysis:
    if not scores:
        return ScoreAnalysis(raw={}, mean=0.0, weighted=0.0, anomalies={})
    vals = list(scores.values())
    mean = sum(vals) / len(vals)

    # simple weighting: safety & architecture more important
    weights = {
        'architecture': 1.5,
        'safety': 1.7,
        'agents': 1.2,
        'rag': 1.1,
        'prompts': 1.0,
        'observability': 1.0,
    }
    total_w = 0.0
    acc = 0.0
    for k, v in scores.items():
        w = weights.get(k, 1.0)
        acc += v * w
        total_w += w
    weighted = acc / total_w if total_w else mean

    anomalies: Dict[str, str] = {}
    # detect large deviations from mean
    for k, v in scores.items():
        if mean - v >= 3.0:
            anomalies[k] = 'Significantly below average'
        if v - mean >= 3.0:
            anomalies[k] = 'Significantly above average'

    return ScoreAnalysis(raw=scores, mean=round(mean, 2), weighted=round(weighted, 2), anomalies=anomalies)
