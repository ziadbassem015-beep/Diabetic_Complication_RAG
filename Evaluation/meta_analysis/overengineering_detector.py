"""Detect signs of overengineering based on scores and code complexity heuristics."""
from typing import Dict, Any
from pathlib import Path


def detect_overengineering(scores: Dict[str, float], root: Path) -> Dict[str, Any]:
    # Heuristic: high scores but many agent files and large graph file indicate overengineering
    agents_count = 0
    graph_complexity = 0
    try:
        agents_file = root.parent / 'multi_agent' / 'agents.py'
        if agents_file.exists():
            content = agents_file.read_text(encoding='utf-8', errors='ignore')
            agents_count = content.count('class ')
    except Exception:
        agents_count = 0

    try:
        graph_file = root.parent / 'multi_agent' / 'graph.py'
        if graph_file.exists():
            content = graph_file.read_text(encoding='utf-8', errors='ignore')
            graph_complexity = content.count('def ') + content.count('class ')
    except Exception:
        graph_complexity = 0

    over = False
    reasons = []
    if scores.get('architecture', 0) > 8 and scores.get('agents', 0) > 8 and scores.get('rag', 0) > 8:
        if agents_count >= 5 or graph_complexity >= 30:
            over = True
            reasons.append(f'High agent count ({agents_count}) and graph complexity ({graph_complexity}) with already-strong scores.')

    return {'overengineered': over, 'agents_count': agents_count, 'graph_complexity': graph_complexity, 'reasons': reasons}
